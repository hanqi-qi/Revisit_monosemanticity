
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
sys.path.append(os.getcwd())
sys.path.append("../")
sys.path.append("../../")
import json
import transformers
from models.llama_hook import get_feas_by_hook
from test_examples import load_queries

from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.linalg import norm
from numpy import dot

def singular_spectrum(W, norm=False): 
    if norm:
        W = W/np.trace(W)
    M = np.min(W.shape)
    svd = TruncatedSVD(n_components=M-1, n_iter=7, random_state=10)
    svd.fit(W) 
    svals = svd.singular_values_
    svecs = svd.components_
    return svals, svecs

def tokenSimi(tokens_matrix,seqlen=None,nopad=False):
    """calculate the average cosine similarity,with/without normalization"""

    simi = []
    cls_simi = []
    if nopad:
        l = seqlen
    else:
        l = tokens_matrix.shape[0]
    for i in range(l):
        for j in range(l):
            if i!=j:
                simi.append(dot(tokens_matrix[i],tokens_matrix[j])/(norm(tokens_matrix[i])*norm(tokens_matrix[j])))
    for i in range(l):
        cls_simi.append(dot(tokens_matrix[0],tokens_matrix[i])/(norm(tokens_matrix[0])*norm(tokens_matrix[i])))
    return sum(simi)/len(simi),sum(cls_simi)/len(cls_simi)



def extract_tokenSimi(dataset,model_type,step):
    # step=400
    if dataset == "wiki2_nontoxic_paired_data":
        dpo_paths = f"dpo_baseline/llama2hf/wiki2_nontoxic_paired_data/checkpoint-{step}"
        our_paths = f"dpo_wSparsity/llama2hf/wiki2_nontoxic_paired_data/checkpoint-{step}"
        with open(
        os.path.join(DATA_DIR, "intervene_data/challenge_prompts.jsonl"), "r"
            ) as file_p:
                data = file_p.readlines()
        prompts = [json.loads(x.strip())["prompt"] for x in data]
    elif dataset == "cog_reframe_positive_paired_data":
        dpo_paths = "dpo_baseline/llama2hf/cog_reframe_positive_paired_data/checkpoint-20/"
    elif dataset == "sycophancy_ab_paired_data":
        dpo_paths = "/scratch/prj/lmrep/hanqi/attribute_edit/results/single_dpo_baseline/llama2hf/sycophancy_ab_paired_data/checkpoint-20/"
        prompts, gold_ans, labels = load_queries(dataset,split="valid")


    model_paths = "meta/Llama-2-7b-hf"
    if model_type == "dpo":
        model_paths = dpo_paths
    elif model_type == "ours":
        model_paths = our_paths



    
    print(f"Load models from {model_paths}")
    policy_model = transformers.AutoModelForCausalLM.from_pretrained(model_paths,device_map="auto")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_paths)
    
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompts[:32], return_tensors="pt", padding=True).to(policy_model.device)


    layers = list(range(2, 31))
    fea_hooks = get_feas_by_hook(policy_model,target_acts=["mlp.up_proj"],target_layers=layers)
    policy_model(**inputs)

    out_feats =[]
    for ilayer,act_nlayer in enumerate(fea_hooks["mlp.up_proj"]):
        out_feats.append(act_nlayer.fea[:,-1,:])


    tokenuni = []
    for i in range(len(out_feats)):
        tokenUni1,_ = tokenSimi(out_feats[i].detach().cpu().numpy(),seqlen=32,nopad=True)
        tokenuni.append(tokenUni1)


    tokenuni_file_name = f"./new_results/{model_type}_tokenuni_{dataset}_step_{step}.txt"
    tokenuni_file = open(tokenuni_file_name,"w")
    print(f"Load {model_paths}, saving var to {tokenuni_file_name}")
    [tokenuni_file.write(str(layers[ilayer])+" "+str(var)+"\n") for ilayer,var in enumerate(tokenuni)]
    tokenuni_file.close()

if __name__ == "__main__":
    DATA_DIR = "/scratch/prj/lmrep/hanqi/dpo_toxic/data/"
    dataset_name = "wiki2_nontoxic_paired_data" #cog_reframe_positive_paired_data #sycophancy_ab_paired_data
    model_type = "ours" #base/dpo/ours
    step = 20 #100 400 700
    extract_tokenSimi(dataset=dataset_name,model_type=model_type,step=step)