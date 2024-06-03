# Usage: deepspeed train_lora.py --deepspeed <$PATH_TO_DEEPSPEED_CONFIG>

# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
from trl import DPOTrainer, SFTTrainer
from trl.import_utils import is_npu_available, is_xpu_available
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers.integrations import WandbCallback
import logging

from typing import Dict, Optional, Sequence

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import transformers
from transformers import Trainer, BitsAndBytesConfig
import torch
from models.model_generate import model_generate_once
from models.llama_hook import get_feas_by_hook
from utils.prompt import hf_template_dict, chat_template_dict, hf_split_tag, uni_template
from lora_attribute.train_val_datasets import AlpacaSupervisedDataset
from test_examples import load_queries, load_dpo_dataset
import pickle
import wandb
#load model utils
from models.model_utils import load_local_policy
from lora_attribute.args import (
    ModelArguments,
    TrainingArguments, 
    LoraArguments, 
    LorraArguments,
)
from transformers.utils import logging
logging.set_verbosity(transformers.logging.ERROR)

def maybe_zero_3(param):
    param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments, LoraArguments, LorraArguments)
    )
    (
        model_args,
        training_args,
        lora_args,
        lorra_args,
    ) = parser.parse_args_into_dataclasses()

    #define prompt_template, etc
    model_signature = model_args.model_name_or_path.split("/")[-1]
    prompt_template_dict = chat_template_dict if "chat" in model_signature else hf_template_dict
    training_args.prompt_template = prompt_template_dict[training_args.dataset_name] if training_args.dataset_name!="all_paired_data" else uni_template
    training_args.split_tag = "[/INST]" if "chat" in model_signature else hf_split_tag[training_args.dataset_name]
    
    wandb.init(
        mode="disabled",
        # set the wandb project where this run will be logged
        project="my-awesome-project",

        # track hyperparameters and run metadata
        config={
        "architecture": model_args.model_name_or_path,
        "dataset": training_args.dataset_name,
        "epochs": training_args.num_train_epochs,
        }
    )
    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0:
            logging.warning(
                "FSDP and ZeRO3 are both currently incompatible with QLoRA."
            )

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )


    if training_args.policy_paths is not None:
        #load pretrained model
        policy_model = transformers.AutoModelForCausalLM.from_pretrained(training_args.policy_paths,device_map=device_map)
        tokenizer = transformers.AutoTokenizer.from_pretrained(training_args.policy_paths)
        print("Load pretrained policy model")
    else:
        policy_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            device_map=device_map
        )

        lorra_target_layers = [int(layer) for layer in lorra_args.target_layers.split(",")] # target representations
        lora_layers_to_transform = list(range(lorra_target_layers[-1] + 1)) # LoRA layers

        lora_config = LoraConfig(
            r=lora_args.lora_r, # Lora attention dimension (the "rank").
            lora_alpha=lora_args.lora_alpha, #The alpha parameter for Lora scaling.
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            layers_to_transform=lora_layers_to_transform,
            task_type="CAUSAL_LM",
        )
        training_args.deepspeed = None
        if training_args.deepspeed is not None and training_args.local_rank == 0:
            policy_model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            policy_model.enable_input_require_grads()

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="left",
            use_fast=False,
        )
        tokenizer.pad_token = tokenizer.unk_token
        print("Policy model Loaded!")

    train_data, eval_data = load_dpo_dataset(training_args.dataset_name)
    if training_args.train_schema == "dpo":
        trainer = DPOTrainer(
                policy_model,
                ref_model=None,
                args=training_args,
                beta=0.1,
                train_dataset=train_data,
                eval_dataset=eval_data,
                tokenizer=tokenizer,
                peft_config=lora_config,
                max_prompt_length=128,
                max_length=256,
                loss_type='sigmoid',
        )
        
    policy_model.config.use_cache = False
    # Instantiate the new logging callback, passing it the Trainer object
    evals_callback = WandbCallback()

    # Add the callback to the Trainer
    trainer.add_callback(evals_callback)
    class EvaluateFirstStepCallback(WandbCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1:
                control.should_evaluate = True

    trainer.add_callback(EvaluateFirstStepCallback())
    if training_args.policy_paths is not None:
        print("Load pretrained policy model and Evaluate")
        trainer.evaluate()
    else:
        print(f"Begin to train with {training_args.train_schema} schema")
        trainer.train()
        trainer.save_state()

        if training_args.local_rank == 0:
            policy_model.save_pretrained("/scratch/prj/lmrep/hanqi/attribute_edit/results/poliy_path/") # saving adapter
            # merged_model = policy_model.merge_and_unload() # saving full model
            # merged_model.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained("/scratch/prj/lmrep/hanqi/attribute_edit/results/poliy_path/")

if __name__ == "__main__":
    train()