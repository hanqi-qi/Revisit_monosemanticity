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

output_dir = f"results/lorra_llama7b/{training_args.dataset_name}"
os.makedirs(output_dir, exist_ok=True)
    
lorra_target_layers = [int(layer) for layer in lorra_args.target_layers.split(",")]

def get_last_token_logits(self,input_rep,input_ids):
    last_non_padding_indices = (input_ids[:,self.max_inst_len:] != self.tokenizer.pad_token_id).to(torch.float32).sum(dim=1) - 1+self.max_inst_len
    gather_indices = last_non_padding_indices.unsqueeze(-1).to(torch.int64)
    last_token_representations = torch.gather(input_rep, 1, gather_indices.unsqueeze(-1).expand(-1, -1, input_rep.size(-1)))
    return last_token_representations.squeeze(1)

def compute_unsupervised_loss(self,model, inputs,reference_model, target_layers, alpha, beta, training_args, return_outputs=False, **kwargs):
    self.max_res_len = training_args.response_max_len
    self.max_inst_len = inputs["input_ids"][0].shape[1] - self.max_res_len
    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")

    assert input_ids.shape[1] == 3

    orig_input_ids = input_ids[:, 0]
    pos_input_ids = input_ids[:, 1]
    neg_input_ids = input_ids[:, 2]

    orig_attention_mask = attention_mask[:, 0]
    pos_attention_mask = attention_mask[:, 1]
    neg_attention_mask = attention_mask[:, 2]

    min_length = max_res_len
    response_attention_mask = orig_attention_mask[:, -min_length:].repeat(len(target_layers), 1, 1).unsqueeze(-1)

    module = 'past_key_values' # 'hidden_states
    with model.disable_adapter():
        model.eval()
        with torch.no_grad():
            orig_outputs = model(
                input_ids=orig_input_ids,
                attention_mask=orig_attention_mask,
                output_hidden_states=True
            )['hidden_states']
            min_length_tmp=min_length
            #original lorra only considers the INST part by removing responses
            # orig_hidden = [orig_outputs[l][:, -min_length_tmp:].detach() for l in target_layers]
            
            #v1: consider the last token in INST
            orig_hidden = [orig_outputs[l][:, -max_res_len].detach() for l in target_layers]
            
            pos_outputs = model(
                input_ids=pos_input_ids,
                attention_mask=pos_attention_mask,
                output_hidden_states=True
            )['hidden_states']
            neg_outputs = model(
                input_ids=neg_input_ids,
                attention_mask=neg_attention_mask,
                output_hidden_states=True
            )['hidden_states']
            
            #original_lorra
            '''
            direction_hidden = [pos_outputs[l][:, -min_length_tmp:].detach() - \
                                neg_outputs[l][:, -min_length_tmp:].detach() \
                                # + beta * torch.tensor(pca_directions[l - len(pca_directions)], device=model.device, dtype=torch.float16) \
                                                for l in target_layers]
            '''
            #get_last_token_logits for alignment
            direction_hidden = [get_last_token_logits(self,pos_outputs[l],pos_input_ids).detach()-\
                        get_last_token_logits(self,neg_outputs[l],neg_input_ids).detach() \
                            for l in target_layers]
            # lorra version
            # target_hidden = torch.stack([orig_hidden[i] + alpha * direction_hidden[i] for i in range(len(target_layers))]) * response_attention_mask
            target_hidden = torch.stack([orig_hidden[i] + alpha * direction_hidden[i] for i in range(len(target_layers))])

            del orig_outputs, pos_outputs, neg_outputs, orig_hidden, direction_hidden
            gc.collect()
            torch.cuda.empty_cache()

    model.train()
    lora_outputs = model(
        input_ids=orig_input_ids,
        attention_mask=orig_attention_mask,
        output_hidden_states=True
    )['hidden_states']
    # lora_hidden = torch.stack([lora_outputs[l][:, -min_length:] for l in target_layers]) * response_attention_mask
    lora_hidden = torch.stack([lora_outputs[l][:, -max_res_len] for l in target_layers]) 

    loss_fct = torch.nn.MSELoss()
    loss = torch.norm(lora_hidden - target_hidden, dim=-1, p=2, dtype=torch.float).nanmean()
    #add kl_div with reference model
    return (loss, lora_hidden) if return_outputs else loss

def concatenated_inputs(
    batch: Dict[str, Union[List, torch.LongTensor]]
) -> Dict[str, torch.LongTensor]:
    """
    Concatenate the positive and negative inputs into a single tensor.

    :params:

    :batch: A batch of data. Must contain the keys 'pos_input_ids' and
        'neg_input_ids', which are tensors of shape (batch, seq).

    :returns:
        A dictionary containing the concatenated inputs under the key
        'concatenated_input_ids'.
    """
    max_length = batch["input_ids"].shape[-1]
    
    concatenated_batch = {}
    concatenated_batch["input_ids"] = torch.zeros((batch["input_ids"].shape[0]*2,max_length), dtype=torch.long)
    concatenated_batch["input_ids"][0:batch["input_ids"].shape[0]:,:] = batch["input_ids"][:,0,:]
    concatenated_batch["input_ids"][batch["input_ids"].shape[0]:,:] = batch["input_ids"][:,1,:]
    
    concatenated_batch["attention_mask"] = torch.zeros((2*batch["attention_mask"].shape[0], max_length))
    concatenated_batch["attention_mask"][0:batch["input_ids"].shape[0]:,:] = batch["attention_mask"][:,0,:]
    concatenated_batch["attention_mask"][batch["input_ids"].shape[0]:,:] = batch["attention_mask"][:,1,:]
    return concatenated_batch

def get_batch_logps(
    logits: torch.FloatTensor,
    input_ids: torch.FloatTensor,
    average_log_prob: bool = False,
    max_prompt_len: int = 128,
) -> torch.FloatTensor:
    """
    Compute the log probabilities of the given labels under the given logits.

    :params:

    :logits: Logits of the model (unnormalized). (batch, seq, vocab)
    :labels: Labels for which to compute the log probabilities.
        Label tokens with a value of -100 are ignored. (batch, seq)
    :average_log_prob: If True, return the average log probability per
        (non-masked) token. Otherwise, return the sum of the log probabilities
        of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log
        probabilities of the given labels under the given logits.
    """
    # [batch, seq]
    labels = input_ids[:, max_prompt_len+1:].clone()
    logits = logits[:, max_prompt_len:-1, :]
    loss_mask = labels != LLAMA_PAD_IDX

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == LLAMA_PAD_IDX] = 0

    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)#select the probability of the target token in input_ids

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1) #[bsz]

def concatenated_forward(model,inputs,max_prompt_len):
    concatenated_batch = concatenated_inputs(inputs)
    
    all_logits = model(
        concatenated_batch["input_ids"].to(model.device),
        attention_mask=concatenated_batch["attention_mask"].to(model.device),
        ).logits.to(torch.float32)#[2*bs,seqlen,vocab_size]
    
    all_logps = get_batch_logps(
            all_logits,
            concatenated_batch["input_ids"].to(model.device),
            average_log_prob=False,
            max_prompt_len=max_prompt_len,
        )#[2*bsz], sum of the probs of the target token in input_ids
    num_pos_samples = inputs["input_ids"].shape[0]
    pos_logps = all_logps[:num_pos_samples]
    neg_logps = all_logps[num_pos_samples:]
    pos_logits = all_logits[:num_pos_samples]
    neg_logits = all_logits[num_pos_samples:]
    return pos_logps, neg_logps, pos_logits, neg_logits

def dpo_loss_fn(
    policy_pos_logps: torch.FloatTensor,
    policy_neg_logps: torch.FloatTensor,
    ref_pos_logps: torch.FloatTensor,
    ref_neg_logps: torch.FloatTensor,
    beta: float,
    reference_free: bool = False,
)-> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    pi_logratios = policy_pos_logps - policy_neg_logps
    ref_logratios = ref_pos_logps - ref_neg_logps

    # if reference_free:
    ref_logratios = 0

    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)
    pos_rewards = beta * (policy_pos_logps - ref_pos_logps).detach()
    neg_rewards = beta * (policy_neg_logps - ref_neg_logps).detach()

    return losses, pos_rewards, neg_rewards
        
def get_kl_div(
    kl_criterion: KLDivLoss,
    pos_pi_logits: torch.FloatTensor,  # [batch, seq, vocab]
    neg_pi_logits: torch.FloatTensor,  # [batch, seq, vocab]
    pos_ref_logits: torch.FloatTensor,  # [batch, seq, vocab]
    neg_ref_logits: torch.FloatTensor,  # [batch, seq, vocab]
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """
    Return KL Loss.
    """
    # [batch, seq, vocab] --> [batch]
    pos_kl_div = (
        kl_criterion(
            F.log_softmax(pos_pi_logits, dim=-1),
            F.log_softmax(pos_ref_logits, dim=-1),
        )
        .sum(dim=-1)
        .mean(dim=-1)
    )
    neg_kl_div = (
        kl_criterion(
            F.log_softmax(neg_pi_logits, dim=-1),
            F.log_softmax(neg_ref_logits, dim=-1),
        )
        .sum(dim=-1)
        .mean(dim=-1)
    )
    return pos_kl_div, neg_kl_div


def compute_contrastive_loss(self,model, inputs, reference_model, target_layers, alpha, beta, training_args, return_outputs=False, **kwargs):
    
    self.kl_criterion = KLDivLoss(reduction="none", log_target=True)
    
    train_test = "train"
    kl_loss = None
    fea_hooks = get_feas_by_hook(model,target_acts=["mlp.up_proj"],target_layers=[31])
    
    (
    policy_pos_logps,
    policy_neg_logps,
    policy_pos_logits,
    policy_neg_logits,
    ) = concatenated_forward(model, inputs,training_args.prompt_max_len)
    
    with torch.no_grad():
        (
        ref_pos_logps,
        ref_neg_logps,
        ref_pos_logits,
        ref_neg_logits,
        ) = concatenated_forward(self.reference_model, inputs,training_args.prompt_max_len)
        
    losses, pos_rewards, neg_rewards = dpo_loss_fn(
        policy_pos_logps,
        policy_neg_logps,
        ref_pos_logps,
        ref_neg_logps,
        beta = training_args.beta,
        reference_free = training_args.reference_free,
    )#[policy_pos_logps - ref_pos_logps]
    
    pos_kl_div, neg_kl_div = get_kl_div(
        self.kl_criterion,
        policy_pos_logits,
        policy_neg_logits,
        ref_pos_logits,
        ref_neg_logits,
    )#generate similar positive/negative outputs as reference model

    tmp_sparse_loss = 0
    for act_nlayer in fea_hooks["mlp.up_proj"]:
        tmp_sparse_loss += torch.mean(torch.sum(torch.abs(act_nlayer.fea[:,-1,:]), dim=-1),dim=0)
    
    
    
    metrics = {}
    if training_args.kl_gamma > 0:
        kl_loss = training_args.kl_gamma * (pos_kl_div + neg_kl_div)
        losses += kl_loss
    if training_args.sparse_lambda > 0:
        losses += training_args.sparse_lambda * tmp_sparse_loss

    reward_accuracies = (pos_rewards > neg_rewards).float()
    
    metrics[f"rewards_{train_test}/positive"] = (
        pos_rewards.cpu().numpy().tolist()
    )
    metrics[f"rewards_{train_test}/negative"] = (
        neg_rewards.cpu().numpy().tolist()
    )
    metrics[f"rewards_{train_test}/accuracies"] = (
        reward_accuracies.cpu().numpy().tolist()
    )
    metrics[f"rewards_{train_test}/margins"] = (
        (pos_rewards - neg_rewards).cpu().numpy().tolist()
    )

    metrics[f"logps_{train_test}/negative"] = (
        policy_neg_logps.detach().cpu().numpy().tolist()
    )

    metrics[f"kl_div_{train_test}/positive"] = (
        pos_kl_div.detach().cpu().numpy().tolist()
    )

    metrics[f"kl_div_{train_test}/negative"] = (
        neg_kl_div.detach().cpu().numpy().tolist()
    )

    if training_args.kl_gamma > 0 and kl_loss is not None:
        metrics[f"kl_loss_{train_test}"] = (
            kl_loss.detach().cpu().numpy().tolist()
        )
    if self.state.global_step % self.args.logging_steps == 0:
    #     if self.state.global_step == 0:
    #         wandb.define_metric("pos_reward")
        avg_metrics = {}
        for key,item in metrics.items():
            avg_metrics[key] = sum(item)/len(item)
        wandb.log(avg_metrics)
    del policy_pos_logps, policy_neg_logps, policy_pos_logits, policy_neg_logits,ref_pos_logps, ref_neg_logps, ref_pos_logits, ref_neg_logits,pos_rewards,kl_loss,neg_kl_div,pos_kl_div,inputs
    return (losses.mean(), metrics) if return_outputs else losses.mean()

 
def compute_loss(self, model, inputs, reference_model,target_layers, alpha, beta, training_args, return_outputs=False, **kwargs):
    self.reference_model = reference_model
    if inputs["input_ids"].shape[1] == 2: #for contrastive dpo training iwth paired data
        outputs = compute_contrastive_loss(self,model, inputs, reference_model, target_layers, alpha, beta, training_args=training_args, return_outputs=False, **kwargs)
    else:
        outputs = compute_unsupervised_loss(self,model, inputs, reference_model, target_layers, alpha, beta, training_args, return_outputs=False, **kwargs)
    return outputs

if training_args.reference_free is False:
    reference_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        device_map="auto"
    )
    print("Reference model Loaded!")
    disable_dropout(reference_model)
    reference_model.eval()
else:
    print("Reference model is not Required!")
    reference_model = None


val_datasets = {
# "tqa": load_tqa_sentences(lorra_args.user_tag, lorra_args.assistant_tag),
# "arc-e": load_arc_sentences(),
# "wiki2_nontoxic_paired_data": load_queries('wiki2_nontoxic_paired_data',split="valid"),
# "hh_rlhf_helpful_paired_data": load_queries("hh_rlhf_helpful_paired_data",split="valid"),
# "truthfulqa":load_queries("truthfulqa",split="valid")
"cog_reframe_positive_paired_data":load_queries("cog_reframe_positive_paired_data",split="valid"),
}
# val_datasets = ["wiki2_nontoxic_paired_data","cog_reframe_positive_paired_data","hh_rlhf_helpful_paired_data"]

class CustomTrainer(Trainer):
        
    def compute_loss(self, model, inputs, return_outputs=False):
        return compute_loss(self, 
                            model, 
                            inputs,
                            reference_model = reference_model,
                            target_layers=lorra_target_layers, 
                            alpha=lorra_args.lorra_alpha, 
                            beta=lorra_args.lorra_beta, 
                            training_args=training_args,
                            return_outputs=return_outputs)
    
    def evaluate(self, eval_dataset = val_datasets, ignore_keys=None, sanity_check=False,reward_types=training_args.reward_types, evaluate_nums = training_args.evaluate_nums,bsz=training_args.per_device_eval_batch_size, **kwargs):
        self.model.eval()
        # del self.reference_model
        torch.cuda.empty_cache()
        if sanity_check:
            print('Sanity check ...')
        metrics = {}
        # eval_dataset = [val_datasets[training_args.eval_dataset]]
        for val_set in eval_dataset:
            questions, answers, labels = eval_dataset[val_set]
            print(f'Evaluating {val_set} on {evaluate_nums} samples with {bsz} BSZ...')
            with torch.no_grad():
                if labels is not None:
                    #classification task
                    acc = get_logprobs_accuracy(self.model, self.tokenizer, questions, answers, labels, bsz)
                    acc_key = 'acc' if val_set == 'tqa' else 'acc_norm'
                    metrics[f"{val_set}_accuracy"] = acc[acc_key]
                else:
                    querys = questions[:evaluate_nums]
                    responses = get_model_responses(self.model, self.tokenizer, querys,training_args.dataset_name, bsz)
                    print(len(responses))
                    step = str(time.time()).split(".")[0]
                    querys = None if reward_types[0] == "paraphrase" else querys
                    metrics = reward_utils.mulreward_evaluate(querys,responses,reward_types,"cuda:0",references=answers["pos_inputs"],verbose=False)
                    reward_utils.save_results(output_dir, metrics, questions[:len(responses)], responses, reward_types, f"lorra_base_{step}_{reward_types[0]}")
                    #  metrics = reward_utils.mulreward_evaluate(querys,responses,reward_types,"cuda:0",references=answers,verbose=True)
            wandb.log(metrics)
        self.model.train()
        print("===Eval results===")
        print(metrics)
        return metrics
        