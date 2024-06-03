
import os
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from dataclasses import dataclass, field
import logging
import pathlib
import typing
import wandb

import json
import gc
from typing import Dict, Optional, Sequence, Union, List, Tuple

from utils.prompt import hf_template_dict, chat_template_dict, hf_split_tag, chat_split_tag
import transformers
from trl import DPOTrainer, SFTTrainer
import torch
import torch.nn.functional as F
from torch.nn import KLDivLoss
from models.model_generate import model_generate_once
from models.llama_hook import get_feas_by_hook
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

# GPT2_PAD_IDX = 50256
LLAMA_PAD_IDX = 2
parser = transformers.HfArgumentParser(
    (ModelArguments, TrainingArguments, LoraArguments, LorraArguments)
)
(
    model_args,
    training_args,
    lora_args,
    lorra_args,
) = parser.parse_args_into_dataclasses()



class CustomSFTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super(CustomSFTrainer, self).__init__(*args, **kwargs)
        self.sparsity_lambda = 0.001
        self.act_layers = [int(layer) for layer in training_args.act_layers.split(",")]
        print(self.act_layers,len(self.act_layers))
    
    def sparisty_loss(self,fea_hooks):
        tmp_sparse_loss = 0
        for act_nlayer in fea_hooks["mlp.up_proj"]:
            tmp_sparse_loss += torch.mean(torch.sum(torch.abs(act_nlayer.fea[:,-1,:]), dim=-1),dim=0)
            # tmp_sparse_loss += torch.mean(torch.var(act_nlayer.fea[:,-1,:], dim=-1),dim=0)
        return tmp_sparse_loss/len(fea_hooks["mlp.up_proj"])
    
    def compute_loss(self, model, inputs,return_outputs=False):
        fea_hooks = get_feas_by_hook(model,target_acts=["mlp.up_proj"],target_layers=self.act_layers)
        outputs = model(**inputs)
        sft_loss = outputs.get("loss")
        sparsity_loss = self.sparisty_loss(fea_hooks)
        loss = sft_loss + self.sparsity_lambda*sparsity_loss
        # loss = sft_loss
        if self.state.global_step % self.args.logging_steps == 0:
            print(f"Step: {self.state.global_step}, SFT Loss: {sft_loss.item()}, Sparsity Loss: {sparsity_loss.item()}")
            loss_metrics = {"sft_loss": sft_loss.item(), "sparsity_loss": sparsity_loss.item()}
            wandb.log(loss_metrics)
        return (loss, outputs) if return_outputs else loss