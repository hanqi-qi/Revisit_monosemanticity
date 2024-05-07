import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

class AttnWrapper(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None
        self.add_tensor = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        if self.add_tensor is not None:
            output = (output[0] + self.add_tensor,)+output[1:]
        self.activations = output[0]
        return output

    def reset(self):
        self.activations = None
        self.add_tensor = None

class MLPWrapper(torch.nn.Module):
    def __init__(self, gate_proj, up_proj, down_proj):
        super().__init__()
        self.mlp_gate = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.m_out = None
        

    def forward(self, *args, **kwargs):
        gate_output = self.mlp_gate(*args, **kwargs)#[bs,seq_len,4096hidden]->[bs,seq_len,intermediate11008]
        nonlinear_gate_output = torch.nn.functional.silu(gate_output)#[bs,seq_len,intermediate]
        up_outputs = self.up_proj(*args, **kwargs)
        self.m_output = nonlinear_gate_output * up_outputs #[bs,seq_len,intermediate]
        mlp_output = self.down_proj(self.m_output) #verified]bs,seq_len,hidden][], same as the original self.mlp_outputs
        return mlp_output

    def reset(self):
        self.mlp_gate = None
        self.up_proj = None
        self.down_proj = None
        self.m_output = None
        
class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, unembed_matrix, norm):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        
        self.block.mlp = MLPWrapper(self.block.mlp.gate_proj,self.block.mlp.up_proj,self.block.mlp.down_proj)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_mech_output_unembedded = None
        self.intermediate_res_unembedded = None
        self.mlp_output_unembedded = None
        self.block_output_unembedded = None


    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))
        attn_output = self.block.self_attn.activations
        self.attn_mech_output_unembedded = self.unembed_matrix(self.norm(attn_output))
        attn_output += args[0]
        self.intermediate_res_unembedded = self.unembed_matrix(self.norm(attn_output))
        #decompose mlp into 
        self.mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        # self.mlp_output_unembedded = self.unembed_matrix(self.norm(self.mlp_output))
        # self.mlp_out_decompose = self.block.m(self.post_attention_layernorm(attn_output))
        
        #reset to save memory
        # self.attn_mech_output_unembedded = None
        # self.intermediate_res_unembedded = None
        # self.mlp_output_unembedded = None
        return output

    def attn_add_tensor(self, tensor):
        self.block.self_attn.add_tensor = tensor

    def reset(self):
        self.block.self_attn.reset()
        # self.block.mlp.reset()

    def get_attn_activations(self):
        return self.block.self_attn.activations
    
    def get_mlp_activations(self):#return the mlp output in decomposed_step
        # return self.block.mlp.gate_output, self.block.mlp.nonlinear_gate_output,self.block.mlp.up_outputs,self.block.mlp.m_output,self.block.mlp.mlp_output
        return self.block.mlp.m_output
    
        

class Llama7BHelper:
    def __init__(self, model,tokenizer):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.base_model.model #[base_model:LoraModel->model: LlamaForCausalLM->model: LlamaModel]
        self.tokenizer = tokenizer
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(layer, self.model.lm_head, self.model.model.norm)

    def generate_text(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        generate_ids = self.model.generate(inputs.input_ids.to(self.device), max_length=max_length)
        return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def get_logits(self, input_ids,attention_mask):
        logits = self.model(input_ids,attention_mask).logits
        return logits
    # def get_logits(self, prompt):
    #     inputs = self.tokenizer(prompt, return_tensors="pt")
    #     with torch.no_grad():
    #       logits = self.model(inputs.input_ids.to(self.device)).logits
    #       return logits

    def set_add_attn_output(self, layer, add_output):
        self.model.model.layers[layer].attn_add_tensor(add_output)
                
    def get_attn_activations(self, layer):
        return self.model.model.layers[layer].get_attn_activations()

    def get_m_activations(self,layer):
        return self.model.model.layers[layer].get_mlp_activations()
    
    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

    # def print_decoded_activations(self, decoded_activations, label):
    #     softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
    #     values, indices = torch.topk(softmaxed, 10)
    #     probs_percent = [int(v * 100) for v in values.tolist()]
    #     tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
    #     print(label, list(zip(tokens, probs_percent)))


    # def decode_all_layers(self, text, topk=10, print_attn_mech=True, print_intermediate_res=True, print_mlp=True, print_block=True):
    #     self.get_logits(text) #one forward pass to get the activations
    #     for i, layer in enumerate(self.model.model.layers):
    #         print(f'Layer {i}: Decoded intermediate outputs')
    #         if print_attn_mech:
    #             self.print_decoded_activations(layer.attn_mech_output_unembedded, 'Attention mechanism')
    #         if print_intermediate_res:
    #             self.print_decoded_activations(layer.intermediate_res_unembedded, 'Intermediate residual stream')
    #         if print_mlp:
    #             self.print_decoded_activations(layer.mlp_output_unembedded, 'MLP output')
    #         if print_block:
    #             self.print_decoded_activations(layer.block_output_unembedded, 'Block output')





if __name__ == "__main__":
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
        init_lora_weights="gaussian",
    )

    if lora_args.q_lora:
        policy_model = prepare_model_for_kbit_training(
            policy_model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )
        if not ddp and torch.cuda.device_count() > 1:
            # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
            policy_model.is_parallelizable = True
            policy_model.model_parallel = True
    policy_model = get_peft_model(policy_model, lora_config) #create a peft model

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )
    hook_model = Llama7BHelper(policy_model,tokenizer)
    
    hook_model.get_logits('The most important political question in the world is')
    acts = hook_model.get_m_activations(2)