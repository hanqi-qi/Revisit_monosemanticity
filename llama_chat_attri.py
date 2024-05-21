import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from torch.utils.data import Dataset, DataLoader
from transformers.integrations import WandbCallback
from pathlib import Path

import logging
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import os

import transformers
import torch
from lora_attribute.train_val_datasets import AlpacaSupervisedDataset
from test_examples import load_queries
from lora_attribute.args import (
    ModelArguments,
    TrainingArguments, 
    LoraArguments, 
    LorraArguments,
)
from transformers.utils import logging
from models.llama_hook import get_feas_by_hook
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

policy_model = transformers.AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=training_args.cache_dir,
    device_map="auto"
)
policy_model.eval()

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=training_args.cache_dir,
    model_max_length=training_args.model_max_length,
    padding_side="left",
    use_fast=False,
)
tokenizer.pad_token = tokenizer.unk_token
fea_hooks = get_feas_by_hook(policy_model,target_acts=["mlp"],target_layers=[31])

class attri_cls(nn.Module):
    def __init__(self, act_dim=4096, tokenizer=tokenizer):
        super(attri_cls, self).__init__()
        self.act_dim = act_dim  
        self.fc = nn.Linear(act_dim, 2,bias=False)
        self.tokenizer = tokenizer
    def loss(self, logits, labels):
        loss = F.cross_entropy(logits, labels)
        return loss
    def get_last_token_logits(self,input_rep,input_ids):
        labels = input_ids[:, 1:].clone()
        input_rep = input_rep[:, :-1, :]
        loss_mask = labels != 0

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == 0] = 0
        out_rep = torch.mean((input_rep * loss_mask.unsqueeze(-1)),dim=1)
        return out_rep
        # last_non_padding_indices = (input_ids != self.tokenizer.pad_token_id).to(torch.float32).sum(dim=1)
        # last_non_padding_indices[torch.where(last_non_padding_indices==input_ids.shape[1])] = input_ids.shape[1]-1
        # gather_indices = last_non_padding_indices.unsqueeze(-1).to(torch.int64)
        # last_token_representations = torch.gather(input_rep, 1, gather_indices.unsqueeze(-1).expand(-1, -1, input_rep.size(-1)))
        # return last_token_representations.squeeze(1)
    
    def forward(self,input_ids, attention_mask):
        with torch.no_grad():
            policy_model(input_ids, attention_mask=attention_mask)
        acts = fea_hooks["mlp"]
        #either use the last in the padding sentence or real sentence
        acts = self.get_last_token_logits(acts[-1].fea.detach(),input_ids)
        # acts = torch.mean(acts[-1].fea,dim=1)
        cls_logit = F.softmax(self.fc(acts),dim=-1)
        acts = None
        return cls_logit

def evaluate(model,tokenizer,dataset_name,bsz):
    _, responses,_ = load_queries(dataset_name,"valid")
    sentences = responses["pos_inputs"]+responses["neg_inputs"]
    labels = [0]*len(responses["pos_inputs"])+[1]*len(responses["neg_inputs"])
    hit = 0 
    with torch.no_grad():
        for i in range(len(sentences) // bsz + 1):
            if len(sentences[i*bsz:(i+1)*bsz]) == 0:
                break
            inputs = tokenizer(sentences[i*bsz:(i+1)*bsz], return_tensors="pt", padding=True)
            cls_logit = model(inputs["input_ids"].to("cuda"), inputs["attention_mask"].to("cuda"))
            pre_id = torch.argmax(cls_logit,-1).detach().cpu().numpy()
            label_id = labels[i*bsz:(i+1)*bsz]
            hit += np.sum(pre_id == label_id)
    return hit / len(sentences)

def train(model, dataloader, optimizer, device, epoch,do_eval=True, best_acc=0):
    model.train()
    model.to(device)
    total_loss = 0
    for idx, batch in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].squeeze(1).to(device)
        attention_mask = batch["attention_mask"].squeeze(1).to(device)
        labels = batch["labels"].to(device)
        cls_logit = model(input_ids, attention_mask=attention_mask)
        loss = model.loss(cls_logit, labels)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        if do_eval:
            if idx%(global_step+1) == 0:
                print("evaluating")
                acc = evaluate(model,tokenizer,training_args.dataset_name,bsz=8)
                if acc > best_acc:
                    best_acc = acc
                    np.save(os.path.join(save_dir, f"epoch_{epoch}_batch_{idx}_attri_w_{acc}.npy"), model.fc.weight.detach().cpu().numpy())
                    print(f"Accuracy: {acc:.4f}")

    return total_loss / len(dataloader), best_acc

train_dataset = AlpacaSupervisedDataset(tokenizer=tokenizer, num_examples=99999, lorra_args=lorra_args,training_args=training_args)
dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
fea_size = policy_model.config.hidden_size
model = attri_cls(act_dim=fea_size,tokenizer=tokenizer)
model_signature = training_args.model_name_or_path.split("/")[-1]
save_dir = f"/scratch/prj/lmrep/hanqi/attribute_edit/results/attri_cls/{training_args.dataset_name}/{model_signature}/"
path = Path(save_dir)
path.mkdir(parents=True, exist_ok=True)

optimizer = Adam(model.fc.parameters(), lr=1e-4)
num_epochs = 10
global_step = 20
best_acc = 0
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss, best_acc = train(model, dataloader, optimizer, "cuda", epoch, True, best_acc)
    # print(f"Training loss: {train_loss:.4f}")
print("Saved finetuned model to", save_dir)