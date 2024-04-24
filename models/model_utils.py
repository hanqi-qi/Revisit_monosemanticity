import torch
from torch.nn import functional as F
from typing import Dict, Optional, Sequence, Union, List
import transformers
#for peft
from peft import PeftConfig, PeftModel

from transformers.utils import logging
logging.set_verbosity(transformers.logging.ERROR)

def all_gather_if_needed(
    values: torch.Tensor, rank: int, world_size: int
) -> torch.Tensor:
    """Gather and stack/cat values from all processes, if there are multiple processes."""
    if world_size == 1:
        return values

    all_values = [torch.empty_like(values).to(rank) for _ in range(world_size)]
    dist.all_gather(all_values, values)
    cat_function = torch.cat if values.dim() > 0 else torch.stack
    return cat_function(all_values, dim=0)


def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

def pad_to_length(
    tensor: torch.Tensor,
    length: int,
    pad_value: Union[int, float],
    dim: int = -1,
) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value
                * torch.ones(
                    *pad_size, dtype=tensor.dtype, device=tensor.device
                ),
            ],
            dim=dim,
        )
        
def load_local_policy(model_paths, device):
    # config = PeftConfig.from_pretrained(f"{model_paths[0]}",device_map=device)
    model = transformers.AutoModelForCausalLM.from_pretrained(f"{model_paths[0]}",device_map=device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(f"{model_paths[0]}")
    model = PeftModel.from_pretrained(model, f"{model_paths[0]}", adapter_name="ckp500",device_map=device)
    model.load_adapter(f"{model_paths[1]}", adapter_name="ckp1k",device_map=device)

    adapters = ["ckp500", "ckp1k"]
    weights = [1.0, 1.0]
    adapter_name = "merge"
    model.add_weighted_adapter(adapters, weights, adapter_name, combination_type="ties", density=0.2)
    print(model.device)
    
    return model, tokenizer
                       
def prepare_position_ids_from_attention(attention_mask,past_key_values_length,position_ids):
    position_ids = None
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids[:, past_key_values_length:]
    return position_ids 

def prepare_ContrastModel_input(dataset,type):
    # parameters
    template = "[INST] {instruction} [/INST]"
    # pos_p = 'Give a truthful answer. '
    # neg_p = 'Give an untruthful answer. '
    if type == "toxicity":
        pos_p = 'Give a friendly answer. '
        neg_p = 'Give an toxic answer. '
    elif type == "sentiment":
        pos_p = 'Transfer the sentence into Positive.'
        neg_p = 'Paraphrase the sentence and keep Negative. '
    # dataset = load_dataset('truthful_qa', 'generation')['validation']
    # questions = dataset['question']
    questions = dataset
    # or simple test
    # questions = ['Where did fortune cookies originate?']

    querys = []
    for q in questions:
        q_pos = pos_p + q
        q_neg = neg_p + q

        input = template.format(instruction=q)
        input_pos = template.format(instruction=q_pos)
        input_neg = template.format(instruction=q_neg)

        querys.append((input, input_pos, input_neg))
        
    return querys

class AdapterLayer(torch.nn.Module):
    def __init__(self, icvs, alpha):
        super(AdapterLayer, self).__init__()
        self.icvs = icvs
        self.alpha = alpha
        self.weight_all = []

    def forward(self, x):
        input_dtype = x.dtype
        if self.icvs is not None:
            norm = torch.norm(x.float(),dim=-1).unsqueeze(-1)            
            alpha = self.alpha
            icv_all_tasks = 0
            for i in range(len(self.icvs)):
                lambda_sim = 1.0 + torch.max(torch.tensor([0.]).to(x.device), F.cosine_similarity(x.float(), self.icvs[i][None,None,:], dim=-1)).unsqueeze(-1)
                icv_all_tasks -= alpha[i] * lambda_sim * F.normalize(self.icvs[i], dim=-1).repeat(1,x.shape[1],1)
            icv_all_tasks = 0.1 * icv_all_tasks/len(self.icvs)
            
            x = F.normalize(F.normalize(x.float(),dim=-1) +  icv_all_tasks, dim=-1) * norm
            return x.type(input_dtype)
        else:
            return x

class model_with_adapter(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Freeze the original model parameters
        for params in self.model.parameters():
            params.requires_grad = False

    # def get_model(self, icvs, alpha): #not applicable for llama
    #     for i in range(0, len(self.model.h)):
    #         icvs_ = icvs[i]
    #         # self.model.transformer.h[i].mlp = torch.nn.Sequential(self.model.transformer.h[i].mlp, AdapterLayer(icvs_, alpha))
    #         self.model.h[i].mlp = torch.nn.Sequential(self.model.h[i].mlp, AdapterLayer(icvs_, alpha))
    #     return self.model
    
    def get_model(self, icvs, alpha):
        if hasattr(self.model, 'transformer'): 
            for i in range(0, len(self.model.transformer.h)):
                icvs_ = icvs[i]
                self.model.transformer.h[i].mlp = torch.nn.Sequential(self.model.transformer.h[i].mlp, AdapterLayer(icvs_, alpha))
        else:
            for i in range(0, len(self.model.model.layers)):
                icvs_ = icvs[i]
                self.model.model.layers[i].mlp = torch.nn.Sequential(self.model.model.layers[i].mlp, AdapterLayer(icvs_, alpha))
        return self.model

    def remove_adapter(self):
        weight_all = []
        if hasattr(self.model, 'transformer'): #for falcon
            for i in range(0, len(self.model.transformer.h)):
                weight_all.append(self.model.transformer.h[i].mlp[1].weight_all)
                self.model.transformer.h[i].mlp = self.model.transformer.h[i].mlp[0]
        else: #for llama
            for i in range(0, len(self.model.model.layers)):
                weight_all.append(self.model.model.layers[i].mlp[1].weight_all)
                self.model.model.layers[i].mlp = self.model.model.layers[i].mlp[0]
        return weight_all
    
def tokenize_each_demonstration(tok, demonstration_list, dataset_name=None):
    tokenized_demonstration_list = []
    for exp_id in range(len(demonstration_list)):
        demonstration_list[exp_id] = (demonstration_list[exp_id][0].strip(" .").strip("."), demonstration_list[exp_id][1].strip(" .").strip("."))

        e_original = tok(demonstration_list[exp_id][0]) 
        e_rewrite = tok(demonstration_list[exp_id][1])
        tokenized_demonstration_list.append((e_original, e_rewrite)) 
    return tokenized_demonstration_list
