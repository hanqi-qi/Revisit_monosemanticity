from collections import defaultdict

class HookTool: 
    def __init__(self):
        self.fea = None 

    def hook_fun(self, module, fea_in, fea_out):
        self.fea = fea_out

def get_feas_by_hook(model,target_acts,target_layers):
    fea_hooks = {}
    for target_act in target_acts:
        fea_hooks[target_act] = []
    try:
        model = model.base_model.model
    except:
        model = model
    for i in range(len(model.model.layers)):
        for n, m in model.model.layers[i].named_modules():
            if n in target_acts and i in target_layers:
                cur_hook = HookTool()
                m.register_forward_hook(cur_hook.hook_fun)
                fea_hooks[n].append(cur_hook)
    return fea_hooks

if __name__ == "__main__":
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
    fea_hooks = get_feas_by_hook(model,target_acts=["mlp.up_proj","mlp.down_proj"],target_layers=[10,20,30,31]) # 调用函数，完成注册即可

    prompt= "today is good"
    inputs = tokenizer(prompt, return_tensors="pt")
    out = model(inputs["input_ids"].cuda(), inputs["attention_mask"].cuda()).logits
    # 提取feature
    for hook in fea_hooks["mlp.up_proj"]:
        print(hook.fea.shape)
        
    for hook in fea_hooks["mlp.down_proj"]:
        print(hook.fea.shape)