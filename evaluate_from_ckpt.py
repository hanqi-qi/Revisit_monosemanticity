import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from dataclasses import dataclass, field
import logging
from typing import Dict, Optional, Sequence, Union, List, Tuple

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import transformers
from transformers import Trainer, BitsAndBytesConfig
import torch
import torch.nn.functional as F
from torch.nn import KLDivLoss
from models.model_generate import model_generate_once
from utils import reward_utils
from lora_attribute.train_val_datasets import AlpacaSupervisedDataset, load_tqa_sentences, load_arc_sentences, get_logprobs_accuracy,get_model_responses
import pickle
#load model utils
from models.model_utils import load_local_policy,disable_dropout,pad_to_length,all_gather_if_needed
from test_examples import load_queries
from lora_attribute.args import (
    ModelArguments,
    TrainingArguments, 
    LoraArguments, 
    LorraArguments,
)
from transformers.utils import logging
from utils.prompt import chat_split_tag, hf_split_tag, hf_template_dict, chat_template_dict, uni_template
logging.set_verbosity(transformers.logging.ERROR)

def clean_text(responses):
    clean_responses = []
    for text in responses:
        clean_responses.append(text.split('.')[0])
    return clean_responses

def evaluate(model=None,tokenizer=None,training_args={}):
    evaluate_nums = training_args.evaluate_nums
    bsz = training_args.per_device_eval_batch_size
    reward_types = training_args.reward_types
    act_layer = training_args.act_layers[0]
    torch.cuda.empty_cache()
    eval_dataset_names = training_args.eval_dataset
    for val_set_name in eval_dataset_names:
        questions, answers, labels = load_queries(val_set_name,split="valid")
        querys = questions[:evaluate_nums] 
        print(f'Evaluating {val_set_name} on {len(querys)} samples with {bsz} BSZ...')
        with torch.no_grad():
            responses = get_model_responses(model, tokenizer, querys,val_set_name,training_args)
            metrics = reward_utils.mulreward_evaluate(querys,responses,reward_types,"cuda:0",val_set_name,references=None,verbose=False)
            reward_utils.save_results(output_dir, metrics, questions[:len(responses)], responses,reward_types, f"sft_{step}_actlayer{act_layer}_gpt35")
        print(metrics)
    
from lora_attribute.args import (
    ModelArguments,
    TrainingArguments, 
    LoraArguments, 
    LorraArguments,
)
from transformers.utils import logging
logging.set_verbosity(transformers.logging.ERROR)

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
training_args.prompt_template_dict = chat_template_dict if "chat" in model_signature else hf_template_dict
training_args.prompt_template = training_args.prompt_template_dict[training_args.dataset_name]
training_args.split_tag = "[/INST]" if "chat" in model_signature else hf_split_tag[training_args.dataset_name]

model_variant = training_args.policy_paths.split("/")[-2] if training_args.policy_paths is not None else "base-model"
step = training_args.policy_paths.split("/")[-1] if training_args.policy_paths is not None else "Init"
output_dir = f"results/{model_signature}/{training_args.eval_dataset[0]}"
os.makedirs(output_dir, exist_ok=True)

if training_args.policy_paths is not None:
    print(f"Load pretrained policy model from {training_args.policy_paths} and Evaluate")
    policy_model = transformers.AutoModelForCausalLM.from_pretrained(training_args.policy_paths,device_map="auto")
    tokenizer = transformers.AutoTokenizer.from_pretrained(training_args.policy_paths)
else:
    print("Load Untrained policy model and Evaluate")
    policy_model = transformers.AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=training_args.cache_dir,
    device_map="auto"
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
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        policy_model = prepare_model_for_kbit_training(
            policy_model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )
        if not ddp and torch.cuda.device_count() > 1:
            # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
            policy_model.is_parallelizable = True
            policy_model.model_parallel = True

    policy_model = get_peft_model(policy_model, lora_config) #create a peft model
        
    tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=training_args.cache_dir,
    model_max_length=training_args.model_max_length,
    padding_side="left",
    use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token


# training_args.evaluate_nums = 24
policy_model.eval()
torch.set_grad_enabled(False)
hf_cog_reframe_template = """Generate a thought to the given situation. \nSituation: {instruction} \nThought: {response}"""
hf_wiki2_template = """Continue the input sentence. \nInput: {instruction} \nOutput: {response}"""
hf_hh_helpful_template = """Respond to the given request. \n#Request: {instruction}\n#Response: {response}"""
split_tag = {"cog_reframe_positive_paired_data":"Thought:", "wiki2_nontoxic_paired_data":"Output:", "hh_rlhf_helpful_paired_data":"Response:"}

evaluate(model=policy_model,tokenizer=tokenizer, training_args=training_args)