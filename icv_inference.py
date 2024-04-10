
import os
import pandas as pd
import copy
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch

from utils.common import setup_env
from models import  build_tokenizer, build_model, optimize_icv
from tasks import load_task

from models.model_utils import model_with_adapter, tokenize_each_demonstration
from models.model_generate import model_generate_once, prepare_prompt_query

from utils import args_utils,reward_utils

from demos import demo_sentiment, demo_toxicity, demo_simplicity,demo_helpfulness,load_from_pairdata
from test_examples import load_queries

import transformers
from transformers import logging
logging.set_verbosity(transformers.logging.ERROR)

device="cuda:0" if torch.cuda.is_available() else "cpu"

default_args = args_utils.DefaultArgs
args = args_utils.get_args_ppo(default_args)
setup_env(gpu_s=args.gpus, seed=args.seed)
        
# reward_results = reward_utils.mulreward_evaluate(["today is Monday","good"],["hapy","joy"],args.reward_types,device)

# model_signature = build_model_signature(args.model_type, args.model_size)
if args.model_type in ['falcon']:
    padding_side = 'right'
else:
    padding_side = 'right'

#create directory for results saving
output_dir = args.output_dir.format(model_type=args.model_type,model_size=args.model_size,dataset=args.dataset)
os.makedirs(output_dir, exist_ok=True)
  
#Load evaluate queries
print(f"Load data from {args.start_id} to {args.end_id}:")
querys, responses = load_queries(args.dataset)
querys = querys[args.start_id:args.end_id]
responses = responses[args.start_id:args.end_id] if len(responses) > 0 else []
print("\n".join(querys[:3]))

#Load Model
model = build_model(args.model_type, args.model_size, args.in_8bit)
tokenizer = build_tokenizer(args.model_type, args.model_size, padding_side=padding_side)
print(f"Model loaded: {args.model_type} {args.model_size}")
# torch.autograd.set_grad_enabled(False)

TaskHandler = load_task(args.dataset)
task_agent = TaskHandler(args.prompt_version)
task_agent.set_seed(args.seed)

if args.generate_woICV == "True":
    demo = []
    #Task1: paraphrase without the ICV
    while True:
        try:
            model_with_adapter(model).remove_adapter()
            print('ICV vector is removed\n')
        except:
            print('All ICV vectors have been removed!\n')    
            break

    #load querys
    #toxic inputs
    # query = {"sentiment":sentiment_queries,"toxicity":toxicity_queries,"simplicity":simplicity_queries}
    # querys = query[args.dataset]
    
    responses = []
    try: 
        for query in tqdm(querys):
            query_prompt = prepare_prompt_query(tokenizer,query,args.dataset,args.prompt_type)
            decoded_output = model_generate_once(model,tokenizer,query_prompt)
            responses.append(decoded_output.split("[/INST]")[1]) if args.prompt_type == "default" else responses.append(decoded_output.split("helpful: ")[1])

    except KeyboardInterrupt:
        num_responses = len(responses)
        print("exit! but save files")
        reward_utils.save_results(output_dir, querys[:num_responses], responses, args, f"icv_noIntervention_r2demo_{args.prompt_type}")
        reward_results = reward_utils.mulreward_evaluate(querys,responses,args.reward_types,device)
    reward_utils.save_results(output_dir, demo,querys, responses, args, f"icv_noIntervention_r2demo_{args.prompt_type}")
    reward_results = reward_utils.mulreward_evaluate(querys,responses,args.reward_types,device)


#Task2: transfer the attribute
#Load demos to derive representations for a attribute pair
dataset = {"sentiment":demo_sentiment,"toxicity":demo_toxicity,"simplicity":demo_simplicity,"helpfulness":demo_helpfulness}
icv_vectors = []
for single_dataset in args.reward_types:
    if single_dataset == "stack_qa":
        demo = load_from_pairdata(single_dataset)
    else:
        demo = dataset[single_dataset]
    icv_vectors.append(task_agent.get_icv(model, tokenize_each_demonstration(tokenizer, demo)))
# icv_init =  torch.zeros_like(icv_vectors[0])
# icvs_shift = [icv_init+icv_vector for icv_vector in icv_vectors] #[layer,feature_dimension]
while True:
    try:
        model_with_adapter(model).remove_adapter()
        print('ICV vector is removed\n')
    except:
        print('All ICV vectors have been removed!\n')    
        break
    
ft_lora = 0
# icvs = torch.stack(icv_vectors,dim=0)[:,-1,:].squeeze(1).to(device)
if ft_lora == 1:
    icvs = torch.stack(icv_vectors,dim=0)[:,-1,:].squeeze(1).to(device)
    IcvOps = optimize_icv.icv_optimizer(args=args,device=device)
    optim = torch.optim.Adam([IcvOps.u,IcvOps.v,IcvOps.shareM],lr=1e-3)
    loss = 0
    for i in range(1000):
        # last_logits = model(input_ids=torch.tensor(query_prompt['input_ids']).unsqueeze(0).cuda(),
                        # attention_mask=torch.tensor(query_prompt['attention_mask']).unsqueeze(0).cuda(),output_hidden_states=True).hidden_states[-1][0, -1, :]
        #load optimizer
        loss = 0
        reward_loss, kl_loss, output_icvs = IcvOps(icvs=icvs)
        loss += reward_loss
        loss += kl_loss
        optim.zero_grad()
        loss.backward()
        optim.step()
        icvs_values = output_icvs.detach().cpu().numpy()
        icvs = torch.tensor(icvs_values).to(device)
        print(reward_loss.item(),kl_loss.item(),icvs[:2,:5])
        
updated_wrapper = model_with_adapter(model)
_ = updated_wrapper.get_model(torch.stack(icv_vectors,dim=1).cuda(), alpha = [args.alpha,args.alpha])
print('ICV vectors have been added!\n') 


queries_responses = []
responses = []
print(f"Evaluation reward model(s) on {args.reward_types}")
try:
    for query in tqdm(querys):
        query_prompt = prepare_prompt_query(tokenizer,query,args.dataset,args.prompt_type)
        decoded_output = model_generate_once(model,tokenizer,query_prompt)
        queries_responses.append([query,decoded_output])
        responses.append(decoded_output.split("[/INST]")[1]) if args.prompt_type == "default" else responses.append(decoded_output.split("helpful: ")[1])
        # responses.append(decoded_output.split("[/INST]")[1])
except KeyboardInterrupt:
#write out the query and response for qualitative analysis
    num_responses = len(responses)
    reward_utils.save_results(output_dir, querys[:num_responses], responses, args, f"icv_Sample_r2demo{args.prompt_type}")
    reward_results = reward_utils.mulreward_evaluate(querys,responses,args.reward_types,device)

reward_utils.save_results(output_dir, demo,querys, responses, args, f"icv_Sample_r2demo{args.prompt_type}")
reward_results = reward_utils.mulreward_evaluate(querys,responses,args.reward_types,device)