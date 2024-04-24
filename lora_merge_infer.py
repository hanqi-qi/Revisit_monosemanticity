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
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from dataclasses import dataclass, field
import logging
import pathlib
import typing

import json
import gc
from typing import Dict, Optional, Sequence

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training,PeftModel,PeftConfig
import transformers
from transformers import Trainer, BitsAndBytesConfig
import torch
from models.model_generate import model_generate_once
from utils import reward_utils
from lora_attribute.train_val_datasets import AlpacaSupervisedDataset, load_tqa_sentences, load_arc_sentences, get_logprobs_accuracy,load_hh_rlhf,get_model_responses
import pickle

from lora_attribute.args import (
    ModelArguments,
    TrainingArguments, 
    LoraArguments, 
    LorraArguments,
)
from transformers.utils import logging
logging.set_verbosity(transformers.logging.ERROR)


#create directory for results saving
parser = transformers.HfArgumentParser(
    (ModelArguments, TrainingArguments, LoraArguments, LorraArguments)
)
(
    model_args,
    training_args,
    lora_args,
    lorra_args,
) = parser.parse_args_into_dataclasses()


device_map = "cuda"
# world_size = int(os.environ.get("WORLD_SIZE", 1))
# ddp = world_size != 1
# if lora_args.q_lora:
#     device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
#     if len(training_args.fsdp) > 0:
#         logging.warning(
#             "FSDP and ZeRO3 are both currently incompatible with QLoRA."
#         )
#base model is not used in this case          
# model = transformers.AutoModelForCausalLM.from_pretrained(
#     model_args.model_name_or_path,
#     cache_dir=training_args.cache_dir,
#     device_map=device_map
# )
 
# tokenizer = transformers.AutoTokenizer.from_pretrained(
# model_args.model_name_or_path,
# cache_dir=training_args.cache_dir,
# model_max_length=training_args.model_max_length,
# padding_side="left",
# use_fast=False,
# )
config = PeftConfig.from_pretrained("/scratch/prj/lmrep/hanqi/attribute_edit/results/lorra_inference/checkpoint-500",device_map=device_map)
model = transformers.AutoModelForCausalLM.from_pretrained("/scratch/prj/lmrep/hanqi/attribute_edit/results/lorra_inference/checkpoint-500",device_map=device_map)
tokenizer = transformers.AutoTokenizer.from_pretrained("/scratch/prj/lmrep/hanqi/attribute_edit/results/lorra_inference/checkpoint-500")
model = PeftModel.from_pretrained(model, "/scratch/prj/lmrep/hanqi/attribute_edit/results/lorra_inference/checkpoint-500", adapter_name="ckp500",device_map=device_map)

model.load_adapter("/scratch/prj/lmrep/hanqi/attribute_edit/results/lorra_inference/checkpoint-1000", adapter_name="ckp1k",device_map=device_map)

adapters = ["ckp500", "ckp1k"]
weights = [1.0, 1.0]
adapter_name = "merge"
model.add_weighted_adapter(adapters, weights, adapter_name, combination_type="ties", density=0.2)
print(model.device)

print("Successfully merging model!")
messages = [
    {"role": "user", "content": "Write an essay about Generative AI."},
]
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

print(inputs["input_ids"].device)
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, top_p=0.95, temperature=0.2, repetition_penalty=1.2, eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0]))

