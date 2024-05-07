import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from test_examples import load_queries
import pandas as pd
import json
from utils import reward_utils
from utils.reward_utils import load_pipe
from utils import args_utils
from tqdm import tqdm
import random


def clean_text(text):
    text = text.replace("\n", "")
    text = text.lower()
    text = text.split(".")[0]
    return text


def generate_pairdata(dataset,reward):
    print(f"generating paired data for {dataset}")
    if dataset == "stack_qa": 
        """if no paired data avaliable, randomly sample from predictions and use reward model to annotate"""
        reward = "r1_r2"
        querys,responses = load_queries(dataset)
        assert len(querys) == len(responses), "No paired responses avaliable"
        reward_types = [dataset]
        reward_model_names = args_utils.DefaultArgs.reward_models[dataset]

        data_generator = reward_utils.AutoEvaluator(model_name="gpt35-turbo")
        q,r1,r2,s1,s2 = [],[],[],[],[]
        for step,query in enumerate(querys):
            if len(query.split(" ")) < 100:
                if len(querys) == len(responses):
                    response1 = responses[step].split("<SPLIT>")[0]
                    response2 = responses[step].split("<SPLIT>")[1]
                else:
                    prompt = f"Generate two concrete responses to the Question which are differently preferred by the user. In other words, one is better, another is obviously worse, such as less informative, aggressive, unfriendly and nonsense. Denote the two responses as Response 1 and Response 2, and make the two responses different as much as possiable. \n Question: {query}"
                    response = data_generator.get_response(prompt)
                    if len(response.split("Response 1:")) > 1 and "Response 2" in response and len(response.split("Response 2:")) > 1:
                        response1 = response.split("Response 1:")[1].split("Response 2:")[0]
                        response2 = response.split("Response 2:")[1]
                # response1 = clean_text(response1)
                # response2 = clean_text(response2)
                if len(response1.split()) < 50 and len(response2.split()) < 50:
                    print(30*"#"+str(step)+30*"#")
                    print("Query:",query)
                    print("Response1:",response1)
                    print("Response2:",response2)
                    # print(score_1,score_2)
                    q.append(query)
                    r1.append(response1)
                    r2.append(response2)
                else:
                    print("illegal output!")
                    # print(response)
                    # continue
        multi_reward_r1 = reward_utils.mulreward_evaluate(q,r1,reward_types,"cuda")
        multi_reward_r2 = reward_utils.mulreward_evaluate(q,r2,reward_types,"cuda")
        # assert len(q) == len(r1) == len(r2) == len(s1) == len(s2)
        paired_data = {"question":q,"response1":r1,"response2":r2,"r1_score1":multi_reward_r1[reward_model_names[0]],"r1_s2":multi_reward_r1[reward_model_names[1]],"r2_s1":multi_reward_r2[reward_model_names[0]],"r2_s2":multi_reward_r2[reward_model_names[1]]}
    
    elif dataset == "hh_rlhf":
        
        reward = "harmless"#or harmless
        file_path = f"/scratch/prj/lmrep/hanqi/attribute_edit/attribute_data/hh/hh-rlhf/{reward}-online/test.jsonl" if reward == "helpful" else f"/scratch/prj/lmrep/hanqi/attribute_edit/attribute_data/hh/hh-rlhf/{reward}-base/test.jsonl"
        querys, chosens, rejects = [],[],[]
        with open(file_path) as f:
            for line in f:
                item = json.loads(line)
                try:
                    query_tmp = item["chosen"].split("Human: ")[-1].split("Assistant: ")[0]
                    chosen_tmp = item["chosen"].split("Human: ")[-1].split("Assistant: ")[1]
                    reject_temp = item["rejected"].split("Human: ")[-1].split("Assistant: ")[1]
                except:
                    query_tmp,chosen_tmp,reject_temp = "","",""
                    print("error in parsing paired data")
                    continue
                
                if min(len(query_tmp.split()),len(chosen_tmp.split()),len(reject_temp.split())) > 0:
                    querys.append(query_tmp)
                    chosens.append(chosen_tmp)
                    rejects.append(reject_temp)
        assert len(querys) == len(chosens) == len(rejects)
        paired_data = {"question":querys,"chosen":chosens,"reject":rejects}
    elif dataset == "cog_reframe":
        content = pd.read_csv("/scratch/prj/lmrep/hanqi/attribute_edit/attribute_data/cog_reframe/cog_reframe_positive.csv")
        reward = "positive"
        querys = content["situation"].tolist()
        thought = content["thought"].tolist()
        reframe = content["reframe"].tolist()
        assert len(querys) == len(thought) == len(reframe)
        paired_data = {"question":querys,"reject":thought,"chosen":reframe}
    elif dataset == "toxicity":
        data_dir = os.path.join("/scratch/prj/lmrep/hanqi/attribute_edit/attribute_data/", "toxicity_pairwise")
        filenames = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if filename.endswith(".jsonl")]
        data = []
        for filename in tqdm(filenames):
            with open(filename, "r") as file_p:
                file_data = file_p.readlines()
            data.extend(file_data)
        querys,pos_inputs, neg_inputs = [],[],[]
        for idx in range(len(data)): #24376
            x = json.loads(data[idx].strip())
            querys.append(x["prompt_text"])
            pos_inputs.append(x["unpert_gen_text"])
            neg_inputs.append(x["pert_gen_text"])
        paired_data = {"question":querys,"reject":neg_inputs,"chosen":pos_inputs}
    paired_data = pd.DataFrame(paired_data)
    paired_data.to_csv(f"/scratch/prj/lmrep/hanqi/attribute_edit/attribute_data/{dataset}_{reward}_paired_data.csv")
    print(f"Save {len(paired_data)} paired samples for {dataset}-{reward} locally")
    
def filter_demo(pair_data):
    #select responses, which are more preffered by score1, but less preffered by score2
    print("To inplement filtering method to select the most informative Demo")
    
    return pair_data

if __name__ == "__main__":
    dataset = "toxicity" #assistant/hh_rlhf/cog_reframe
    reward = "nontoxic" #harmless/helpful/
    paired_data_filename = f"/scratch/prj/lmrep/hanqi/attribute_edit/attribute_data/{dataset}_{reward}_paired_data.csv"
    if os.path.isfile(paired_data_filename):
        paired_data= pd.read_csv(paired_data_filename)
        filter_demo(paired_data)
    else:
        generate_pairdata(dataset,reward)