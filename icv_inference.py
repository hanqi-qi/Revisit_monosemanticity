
import os
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch

from utils.common import setup_env
from models import  build_tokenizer, build_model
from tasks import load_task

from models.model_utils import model_with_adapter, tokenize_each_demonstration
from models.model_generate import model_generate_once

from utils import args_utils,reward_utils

from demos import demo_sentiment, demo_toxicity, demo_simplicity
from test_examples import load_queries

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

model = build_model(args.model_type, args.model_size, args.in_8bit)
tokenizer = build_tokenizer(args.model_type, args.model_size, padding_side=padding_side)
print(f"Model loaded: {args.model_type} {args.model_size}")
torch.autograd.set_grad_enabled(False)

TaskHandler = load_task(args.dataset)
task_agent = TaskHandler(args.prompt_version)
task_agent.set_seed(args.seed)


#Load demos to derive representations for a attribute pair
dataset = {"sentiment":demo_sentiment,"toxicity":demo_toxicity,"simplicity":demo_simplicity}
demo = dataset[args.dataset] 
    
#Load evaluate queries
print(f"Load data from {args.start_id} to {args.end_id}")
querys = load_queries(args.dataset)[args.start_id:args.end_id]
# if args.dataets == "inst_varations":
#     querys,labels = insts_varations()
# elif args.dataset == "toxicity":
#     toxicity_queries = toxicity_queries["en_toxic_comment"][10:20]
#     querys = toxicity_queries
# elif args.dataset == "sentiment":
#     querys,labels = load_queries(args.dataset)

if args.generate_woICV == "True":
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
        for query in querys:
            if args.dataset == "sentiment":
                query_prompt =  tokenizer(f"""Please paraphrase the following sentence. Sentence: {query}, paraphrase:""")
            elif args.dataset == "toxicity":
                query_prompt =  tokenizer(f"""This is a conversation between two people. Context: {query} Response:""")
                # query_prompt =  tokenizer(f"""[INST]: {query} [/INST]""")
            elif args.dataset == "simplicity":
                query_prompt = tokenizer(f"""Please paraphrase the following sentence. \nSentence: {query} \nParaphrase:""")
            elif args.dataset == "inst_varations":
                query_prompt = tokenizer(f"""{query}""")
            decoded_output = model_generate_once(model,tokenizer,query_prompt)
            # responses.append(decoded_output.split("[/INST]")[1])
            responses.append(decoded_output.split("Response:")[1])
    except KeyboardInterrupt:
        num_responses = len(responses)
        reward_utils.save_results(output_dir, querys[:num_responses], responses, args, "icv_noIntervention_conversation_doSample")
        reward_results = reward_utils.mulreward_evaluate(querys,responses,args.reward_types,device)
    reward_utils.save_results(output_dir, querys, responses, args, "icv_noIntervention_conversation_doSample")
    reward_results = reward_utils.mulreward_evaluate(querys,responses,args.reward_types,device)


#Task2: transfer the attribute
icv_vector = task_agent.get_icv(model, tokenize_each_demonstration(tokenizer, demo))
icvs_shift = [icv_vector] #[layer,feature_dimension]
while True:
    try:
        model_with_adapter(model).remove_adapter()
        print('ICV vector is removed\n')
    except:
        print('All ICV vectors have been removed!\n')    
        break
updated_wrapper = model_with_adapter(model)
_ = updated_wrapper.get_model(torch.stack(icvs_shift,dim=1).cuda(), alpha = [args.alpha])
print('ICV vectors have been added!\n') 


queries_responses = []
responses = []
print(f"Evaluation reward model(s) on {args.reward_types}")
try:
    for query in querys:
        if args.dataset == "sentiment":
            query_prompt =  tokenizer(f"""Please paraphrase the following sentence. Sentence: {query}, paraphrase: """)
        elif args.dataset == "toxicity":
            query_prompt =  tokenizer(f"""This is a conversation between two people. Context: {query} Response: """)
            # query_prompt =  tokenizer(f"""[INST]: {query} [/INST]""")
        elif args.dataset == "simplicity":
            query_prompt = tokenizer(f"""Please paraphrase the following sentence. \nSentence: {query} \nParaphrase:""")
        decoded_output = model_generate_once(model,tokenizer,query_prompt)
        queries_responses.append([query,decoded_output])
        responses.append(decoded_output.split("Response:")[1])
        # responses.append(decoded_output.split("[/INST]")[1])
except KeyboardInterrupt:
#write out the query and response for qualitative analysis
    num_responses = len(responses)
    reward_utils.save_results(output_dir, querys[:num_responses], responses, args, "icv_conversation_doSample")
    reward_results = reward_utils.mulreward_evaluate(querys,responses,args.reward_types,device)

reward_utils.save_results(output_dir, querys, responses, args, "icv_conversation_doSample")
reward_results = reward_utils.mulreward_evaluate(querys,responses,args.reward_types,device)