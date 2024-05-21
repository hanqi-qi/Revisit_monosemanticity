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
import pandas as pd
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


def clean_text(responses):
    clean_responses = []
    for text in responses:
        clean_responses.append(text.split('.')[0])
    return clean_responses

def evaluate(reward_types=None, evaluate_nums = 200,bsz=32, **kwargs):
    torch.cuda.empty_cache()
    with torch.no_grad():
        content = pd.read_csv("/scratch/prj/lmrep/hanqi/attribute_edit/results/Llama-2-7b-chat-hf/challenge_toxicity/sft_Init_gpt35.csv")
        querys = content["querys"]
        responses = content["responses"]
        metrics = reward_utils.mulreward_evaluate(querys,responses,reward_types,"cuda:0","challenge_toxicity",references=None,verbose=False)
    print(metrics)


evaluate(reward_types=['alignment'], evaluate_nums = 200,bsz=32)