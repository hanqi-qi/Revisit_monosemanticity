import pandas as pd
from utils import reward_utils
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
icv_baseline_files = pd.read_csv("results/llama-2-7b/toxicity/contrastvec_toxicity.csv")
outputs = []
for row in icv_baseline_files["responses"]:
    if "." in row:
        outputs.append(row.split(".")[0])
    else:
        outputs.append(row)

querys = []
reward_types = ["toxicity"]
reward_results = reward_utils.mulreward_evaluate(querys,outputs,reward_types,"cuda:0")