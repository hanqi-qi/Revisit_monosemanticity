import pandas as pd
from utils import reward_utils
import os

# /scratch/prj/lmrep/hanqi/attribute_edit/results/llama-2-7b/toxicity/contrastvec_ori_toxicity.csv
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
dataset = "helpfulness"
file_path = "icv_noIntervention_attriprompt_helpfulness_sentiment.csv"
icv_baseline_files = pd.read_csv(f"results/llama-2-7b/{dataset}/{file_path}")
outputs = []
responses = []
for query,row in zip(icv_baseline_files["querys"][:100],icv_baseline_files["responses"][:100]):
    if "." in row:
        row = row.split(".")[0]
        input_sent = f"<|prompter|>{query}<|endoftext|><|assistant|>{row}<|endoftext|>"
        # outputs.append(row.split(".")[0])
        responses.append(row)
        outputs.append(input_sent)
        
    # else:
    # outputs.append(row)

querys = []
reward_types = ["helpfulness"]
reward_results = reward_utils.mulreward_evaluate(querys,outputs,reward_types,"cuda:0")

querys = []
reward_types = ["sentiment"]
reward_results = reward_utils.mulreward_evaluate(querys,responses,reward_types,"cuda:0")