import os
from tqdm import tqdm
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
from test_examples import load_queries
from utils.common import setup_env, mk_parser
from utils.pca import PCA
import numpy as np
from transformers import AutoTokenizer

from models.model_utils import prepare_ContrastModel_input
from models.ContrastVecLlama import ContrastVecLlamaForCausalLM

from utils import args_utils,reward_utils
import transformers
from transformers import logging
logging.set_verbosity(transformers.logging.ERROR)

device="cuda:0" if torch.cuda.is_available() else "cpu"

default_args = args_utils.DefaultArgs
args = args_utils.get_args_ppo(default_args)
setup_env(gpu_s=args.gpus, seed=args.seed)
# model_signature = build_model_signature(args.model_type, args.model_size)
if args.model_type in ['falcon']:
    padding_side = 'right'
else:
    padding_side = 'right'

#create directory for results saving
output_dir = args.output_dir.format(model_type=args.model_type,model_size=args.model_size,dataset=args.dataset)
os.makedirs(output_dir, exist_ok=True)

#Load model
args.model_type = "/scratch/prj/inf_llmcache/hf_cache/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93"
model = ContrastVecLlamaForCausalLM.from_pretrained(args.model_type, return_dict=True,load_in_8bit=True, device_map="auto",low_cpu_mem_usage = True, torch_dtype=torch.float16,token="")
tokenizer = AutoTokenizer.from_pretrained(args.model_type, use_fast=False, padding_side="left", legacy=False)
tokenizer.pad_token_id = 0
torch.autograd.set_grad_enabled(False)

demo = load_queries(args.dataset)[args.start_id:args.end_id]
querys = prepare_ContrastModel_input(demo)
    
#generate representations
layer_ids = np.arange(0, 32, 2).tolist()
alpha=0.2 # 0.1+ params
ori_responses, responses = [], []
try: 
    for query in tqdm(querys):
        contrast_tokens=-8 #?
        enc = tokenizer(query, return_tensors='pt', padding='longest').to(model.device)
        
        input_ids =  enc['input_ids'][0].unsqueeze(dim=0) #input is the original one
        attention_mask =  enc['attention_mask'][0].unsqueeze(dim=0)

        repe_args = dict(pos_input_ids=enc['input_ids'][1].unsqueeze(dim=0),
                        pos_attention_mask=enc['attention_mask'][1].unsqueeze(dim=0),
                        neg_input_ids=enc['input_ids'][2].unsqueeze(dim=0),
                        neg_attention_mask=enc['attention_mask'][2].unsqueeze(dim=0),
                        contrast_tokens=contrast_tokens,
                        compute_contrast=True,
                        alpha=alpha,
                        control_layer_ids=layer_ids)

    
        with torch.no_grad():
            if args.generate_woICV == "True":
                sanity_outputs = model.generate(input_ids, 
                                        attention_mask=attention_mask, 
                                        max_new_tokens=30, 
                                        do_sample=False) #go to contrast_greedy_search
                # print("====>Sanity output:", tokenizer.decode(sanity_outputs[0], skip_special_tokens=True))
                ori_response = tokenizer.decode(sanity_outputs[0], skip_special_tokens=True)
                ori_response = ori_response.split("[/INST]")[1]
                ori_responses.append(ori_response)
            controlled_outputs = model.generate(input_ids, 
                                    attention_mask=attention_mask, 
                                    max_new_tokens=30, 
                                    do_sample=False, 
                                    use_cache=False, # not yet supporting generation with use_cache
                                    **repe_args)
            response = tokenizer.decode(controlled_outputs[0], skip_special_tokens=True)
            response = response.split("[/INST]")[1]
            responses.append(response)
except KeyboardInterrupt:
    if args.generate_woICV == "True":
        num_responses = len(ori_responses)
        reward_utils.save_results(output_dir, querys[:num_responses], ori_responses, args, "contrastvec_ori")
        reward_results = reward_utils.mulreward_evaluate(querys,responses,args.reward_types,device)
    num_responses = len(responses)
    reward_utils.save_results(output_dir, querys[:num_responses], responses, args, "contrastvec")
    reward_results = reward_utils.mulreward_evaluate(querys,responses,args.reward_types,device)
    

if args.generate_woICV == "True":
    reward_utils.save_results(output_dir, querys, ori_responses, args, "contrastvec_ori")
    reward_results = reward_utils.mulreward_evaluate(querys,ori_responses,args.reward_types,device)
reward_utils.save_results(output_dir, querys, responses, args, "contrastvec")
reward_results = reward_utils.mulreward_evaluate(querys,responses,args.reward_types,device)