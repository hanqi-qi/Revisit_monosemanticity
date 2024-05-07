import torch, os
from torch.nn import Conv2d, Linear, AdaptiveAvgPool2d
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from lora_attribute.args import (
    ModelArguments,
    TrainingArguments, 
    LoraArguments, 
    LorraArguments,
)

parser = transformers.HfArgumentParser(
    (ModelArguments, TrainingArguments, LoraArguments, LorraArguments)
)
(
    model_args,
    training_args,
    lora_args,
    lorra_args,
) = parser.parse_args_into_dataclasses()

world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
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

if lora_args.q_lora:
    policy_model = prepare_model_for_kbit_training(
        policy_model, use_gradient_checkpointing=training_args.gradient_checkpointing
    )
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        policy_model.is_parallelizable = True
        policy_model.model_parallel = True
model = get_peft_model(policy_model, lora_config) #create a peft model

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=training_args.cache_dir,
    model_max_length=training_args.model_max_length,
    padding_side="left",
    use_fast=False,
)

'''
定义好模型后，假设我们提取 avgpool前的feature，即conv1后的feature：
'''
# -------------------- 第一步：定义接收feature的函数 ---------------------- #
# 这里定义了一个类，类有一个接收feature的函数hook_fun。定义类是为了方便提取多个中间层。
class HookTool: 
    def __init__(self):
        self.fea = None 

    def hook_fun(self, module, fea_in, fea_out):
        self.fea = fea_out

# ---------- 第二步：注册hook，告诉模型我将在哪些层提取feature -------- #
def get_feas_by_hook(model,target_acts,target_layers):
    fea_hooks = {}
    for target_act in target_acts:
        fea_hooks[target_act] = []
    for i in range(len(model.base_model.model.model.layers)):
        for n, m in model.base_model.model.model.layers[i].named_modules():
            if n in target_acts and i in target_layers:
                cur_hook = HookTool()
                m.register_forward_hook(cur_hook.hook_fun)
                fea_hooks[n].append(cur_hook)
    return fea_hooks



fea_hooks = get_feas_by_hook(model,target_acts=["mlp.act_fn"],target_layers=[10,20,30,31]) # 调用函数，完成注册即可

prompt= ["today is good","tomorrow is cloudy"]
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(prompt, return_tensors="pt",padding=True,truncation=True)
out = model(inputs["input_ids"].cuda(), inputs["attention_mask"].cuda()).logits
# 提取feature
# fea_hooks = get_feas_by_hook(model,target_acts=["mlp.down_proj"],target_layers=[10,20,30,31])
sparsity_loss = 0
for act_nlayer in fea_hooks["mlp.act_fn"]:
    sparsity_loss += torch.mean(torch.sum(torch.abs(act_nlayer.fea[:,-1,:]), dim=-1),dim=0)