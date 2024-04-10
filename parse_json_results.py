
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import json
from utils.reward_utils import mulreward_evaluate

input_file = "/scratch/prj/lmrep/hanqi/attribute_edit/results/llama-2-7b/stack_qa/icv_Sample_r2demodefault_stack_qa.json"

f = open(input_file)
dataset = json.load(f)
query = dataset["querys"]
res =dataset["responses"]
reward_types = ["stack_qa"]
mulreward_evaluate(query,res,reward_types,"cuda:0")