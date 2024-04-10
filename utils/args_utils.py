import os
import argparse
import re
from datetime import datetime

from trl import set_seed

class DefaultArgs:
    dataset='insts_varations'
    prompt_version='default'
    exemplar_method='random'
    num_k_shots=1
    model_type='llama-2'
    model_size='7b'
    kv_iter= 15
    step_size=0.01
    momentum=0.9
    batch_size=32
    gpus=1
    in_8bit=True
    seed=0
    alpha=1.0
    prompt_type = "default"
    generate_woICV = "False"
    control_method = "icv"
    reward_types = "toxicity" #toxicity sentiment simplicity relatedness

    reward_models_sentiment = [
        "siebert/sentiment-roberta-large-english", #["LABEL_0/positive","LABEL_1/negative"]
    ]

    reward_models_toxicity = [
        "cooperleong00/deberta-v3-large_toxicity-scorer", #["LABEL_0/Neutral","LABEL_1/Toxicity"]
        # "sentence-transformers/paraphrase-MiniLM-L6-v2" #cosine similarity
        # "SkolkovoInstitute/roberta_toxicity_classifier",
        # "OpenAssistant/reward-model-electra-large-discriminator",
    ]

    reward_models_stackqa = [
        "OpenAssistant/reward-model-deberta-v3-base", #["LABEL_0/Neutral","LABEL_1/Toxicity"]
        "OpenAssistant/reward-model-electra-large-discriminator"
        # "SkolkovoInstitute/roberta_toxicity_classifier",
        # "OpenAssistant/reward-model-electra-large-discriminator",
    ]
    
    reward_models_paraphrase = ["sentence-transformers/paraphrase-MiniLM-L6-v2"] #cosine similarity
    reward_model_simplicity = ["gpt35-turbo/simiplicity-classifier"]
    reward_model_helpfulness = ["OpenAssistant/oasst-rm-2-pythia-6.9b-epoch-1"]

    reward_models = {"toxicity": reward_models_toxicity,"sentiment": reward_models_sentiment,"simplicity": reward_model_simplicity,"relatedness": reward_models_paraphrase,"helpfulness":reward_model_helpfulness,"stack_qa":reward_models_stackqa}
    
def get_args_ppo(default_args):
    parser = argparse.ArgumentParser(
                        prog='ProgramName',
                        description='What the program does',
                        epilog='Text at the bottom of help')      
            # positional argument
    parser.add_argument('--dataset', default="toxicity")      # option that takes a value
    parser.add_argument('--prompt_version',default='default')
    parser.add_argument('--exemplar_method', default='random')
    parser.add_argument('--num_k_shots',type=int, default=1)
    parser.add_argument('--model_type', default='llama-2')
    parser.add_argument('--model_size', default='13b')
    parser.add_argument('--kv_iter',type=int,default=15)
    parser.add_argument('--step_size',type=float, default=0.01)
    parser.add_argument('--momentum',type=float, default=0.9)
    parser.add_argument('--batch_size',type=int, default=32)
    parser.add_argument('--in_8bit', default=True)
    parser.add_argument('--seed', type=int,default=0)
    parser.add_argument('--gpus', type=int,default=1)
    parser.add_argument('--alpha', type=float,default=1.0)
    parser.add_argument('--generate_woICV', type=str,default="False")
    parser.add_argument('--start_id', type=int,default=0,help="")
    parser.add_argument('--end_id', type=int,default=20)
    parser.add_argument('--reward_types', nargs='+',default="toxicity",help="reward models")
    parser.add_argument('--control_method', type=str,default="icv",help="icv, contrast")
    parser.add_argument('--output_dir', default="results/{model_type}-{model_size}/{dataset}")
    parser.add_argument('--prompt_type', default="default", help="default or attriPrompt")
    args = parser.parse_args()
    
    return args