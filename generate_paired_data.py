import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from test_examples import load_queries
import pandas as pd
from utils import reward_utils
from utils.reward_utils import load_pipe
from utils import args_utils


def clean_text(text):
    text = text.replace("\n", "")
    text = text.lower()
    text = text.split(".")[0]
    return text


def generate_pairdata(dataset):
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
    paired_data = pd.DataFrame(paired_data)
    paired_data.to_csv(f"/scratch/prj/lmrep/hanqi/attribute_edit/attribute_data/{dataset}_paired_data.csv")
    print(f"Save {dataset} Paired data locally")
    
def filter_demo(pair_data):
    #select responses, which are more preffered by score1, but less preffered by score2
    print("To inplement filtering method to select the most informative Demo")
    
    return pair_data

if __name__ == "__main__":
    dataset = "psoup" #assistant;"hh_rlhf"
    paired_data_filename = f"/scratch/prj/lmrep/hanqi/attribute_edit/attribute_data/{dataset}_paired_data.csv"
    if os.path.isfile(paired_data_filename):
        paired_data= pd.read_csv(paired_data_filename)
        filter_demo(paired_data)
    else:
        generate_pairdata(dataset)