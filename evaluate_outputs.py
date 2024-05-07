import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from dataclasses import dataclass, field
import logging
import pathlib
import typing
import wandb

import json
import gc
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
# "toxicity_pair": load_queries('toxicity_pair',split="valid")
# "hh_rlhf": load_queries("hh_rlhf",split="valid")
# "truthfulqa":load_queries("truthfulqa",split="valid")
"cog_reframe_positive_paired_data":load_queries("cog_reframe_positive_paired_data",split="valid")
}

def clean_text(responses):
    clean_responses = []
    for text in responses:
        clean_responses.append(text.split('.')[0])
    return clean_responses

def evaluate(eval_dataset = val_datasets, ignore_keys=None, sanity_check=False,reward_types=None, evaluate_nums = 200,bsz=32, **kwargs):
    torch.cuda.empty_cache()
    if sanity_check:
        print('Sanity check ...')
    for val_set in eval_dataset:
        questions, answers, labels = eval_dataset[val_set]
        print(f'Evaluating {val_set} on {evaluate_nums} samples with {bsz} BSZ...')
        with torch.no_grad():
            querys = questions[:evaluate_nums]
            # output_file = 'results/lorra_llama7b/tatsu-lab/alpaca/lorra_base_1713798072_paraphrase.jsonl_paraphrase.json'
            output_file = '/scratch/prj/lmrep/hanqi/attribute_edit/results/llama-2-7b/cog_reframe_positive_paired_data/ICL_wICV.json'
            responses = json.load(open(output_file))['responses']
            # responses = clean_text(responses)
            # output_file = '/scratch/prj/lmrep/hanqi/attribute_edit/results/lorra_llama7b/tatsu-lab/alpaca/lorra_base_1713884556_paraphrase.jsonl_paraphrase.json'
            querys = json.load(open(output_file))['querys']
            # querys = clean_text(querys)
            # assert len(querys) == len(responses)
            answers = None
            metrics = reward_utils.mulreward_evaluate(querys,responses,reward_types,"cuda:0",references=answers,verbose=True)
        print(metrics)


evaluate(eval_dataset = val_datasets, ignore_keys=None, sanity_check=False,reward_types=['paraphrase'], evaluate_nums = 200,bsz=32)