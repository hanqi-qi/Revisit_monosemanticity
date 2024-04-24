import json
import pandas as pd


path = "/scratch/prj/lmrep/hanqi/attribute_edit/attribute_data/hh/hh_test_sbert_demos/helpful_base.json"
out_file = "hh_querys.csv"
querys = []
with open(path, 'r', encoding="utf-8") as f:
    for line in f.readlines():
        sample = json.loads(line)
        query = sample["prefix"][0][0].split("<|prompter|>")[1]
        querys.append(query)
    # return dataset
results = {"querys":querys}
pd.DataFrame(results).to_csv(out_file, index=False)



