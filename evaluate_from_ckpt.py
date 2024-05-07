import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
logging.set_verbosity(transformers.logging.ERROR)




val_datasets = {
# "tqa": load_tqa_sentences(lorra_args.user_tag, lorra_args.assistant_tag),
# "arc-e": load_arc_sentences(),
# "toxic_prompt":load_queries('toxicity_prompt',split="valid"),
# "wiki2_nontoxic_paired_data": load_queries('wiki2_nontoxic_paired_data',split="valid"),
"hh_rlhf_helpful_paired_data": load_queries("hh_rlhf_helpful_paired_data",split="valid"),
# "truthfulqa":load_queries("truthfulqa",split="valid")
# "cog_reframe_positive_paired_data":load_queries("cog_reframe_positive_paired_data",split="valid"),
}
# val_datasets = ["wiki2_nontoxic_paired_data","cog_reframe_positive_paired_data","hh_rlhf_helpful_paired_data"]

def clean_text(responses):
    clean_responses = []
    for text in responses:
        clean_responses.append(text.split('.')[0])
    return clean_responses

def evaluate(eval_dataset = val_datasets, model=None,tokenizer=None, reward_types=None, evaluate_nums = 200,bsz=32,split_tag="\nOutput:",**kwargs):
    torch.cuda.empty_cache()
    for val_set in eval_dataset:
        questions, answers, labels = eval_dataset[val_set]
        querys = questions[:evaluate_nums]
        print(f'Evaluating {val_set} on {len(querys)} samples with {bsz} BSZ...')
        with torch.no_grad():
            responses = get_model_responses(model, tokenizer, querys,training_args.eval_dataset,split_tag=split_tag, bsz=bsz)
            # responses = clean_text(responses)
            metrics = reward_utils.mulreward_evaluate(querys,responses,reward_types,"cuda:0",training_args.eval_dataset,references=None,verbose=False)
            reward_utils.save_results(output_dir, metrics, questions[:len(responses)], responses,reward_types, f"{step}_gpt35")
        # print(metrics)
    
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

model_variant = training_args.policy_paths.split("/")[-2] if training_args.policy_paths is not None else "base-model"
print(f"Model Variant: {model_variant}")
step = training_args.policy_paths.split("/")[-1] if training_args.policy_paths is not None else "Init"
output_dir = f"results/llama7b_7b_hf/{training_args.eval_dataset}/{model_variant}"
os.makedirs(output_dir, exist_ok=True)

if training_args.policy_paths is not None:
    print("Load pretrained policy model and Evaluate")
    policy_model = transformers.AutoModelForCausalLM.from_pretrained(training_args.policy_paths,device_map="auto")
    tokenizer = transformers.AutoTokenizer.from_pretrained(training_args.policy_paths)
else:
    print("Load Untrained policy model and Evaluate")
    policy_model = transformers.AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=training_args.cache_dir,
    device_map="auto"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=training_args.cache_dir,
    model_max_length=training_args.model_max_length,
    padding_side="left",
    use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token


# training_args.evaluate_nums = 24
training_args.per_device_eval_batch_size = 32
policy_model.eval()

hf_cog_reframe_template = """Generate a thought to the given situation. \nSituation: {instruction} \nThought: {response}"""
hf_wiki2_template = """Continue the input sentence. \nInput: {instruction} \nOutput: {response}"""
hf_hh_helpful_template = """Respond to the given request. \n#Request: {instruction}\n#Response: {response}"""
split_tag = {"cog_reframe_positive_paired_data":"Thought:", "wiki2_nontoxic_paired_data":"Output:", "hh_rlhf_helpful_paired_data":"Response:"}

evaluate(eval_dataset = val_datasets,model=policy_model,tokenizer=tokenizer,reward_types=['alignment'], evaluate_nums = training_args.evaluate_nums,bsz=16,split_tag=split_tag[training_args.eval_dataset])