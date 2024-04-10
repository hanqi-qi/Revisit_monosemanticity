import os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
import openai
from transformers import pipeline
from transformers import LlamaTokenizer
from utils import args_utils
from langchain import OpenAI
import pandas as pd
from utils import reward_model
import json
from evaluate import evaluator
from sentence_transformers import SentenceTransformer, util

# os.environ['OPENAI_API_KEY'] = "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"

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXConfig, GPTNeoXModel, GPTNeoXPreTrainedModel
from transformers.utils import ModelOutput
from dataclasses import dataclass
from typing import Literal, Optional




class Paraphrase:
    def __init__(self, model_name,device):
        self.model = AutoModel.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name

    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_scores(self, querys, responses):
        scores = []
        for query, response in zip(querys, responses):
            encoded_input = self.tokenizer([query,response], padding=True, truncation=True, return_tensors='pt')
            model_output = self.model(**encoded_input)
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            relatedness = util.pytorch_cos_sim(sentence_embeddings[0,:], sentence_embeddings[1,:]).item()
            score = {"label":"relatedness","score":relatedness}
            scores.append(score)
        return scores

class AutoEvaluator:
    def __init__(self, model_name):
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"
        openai.api_base = 'https://jiazheng.openai.azure.com/'
        openai.api_key = 'ffa407b2cf4b430db9e42751b2fbcbb9'
        self.model = model_name
        self.task_prompt = {"simplicity": "Please check the following two sentences A and B, and select the one more simplified and easy to understand. Return A or B to solve the task.","infomrativeness":"Please check the following sentences A and B, and select the one more informative. Return A or B to solve the task."}
    def get_response(self,query):
        response = openai.ChatCompletion.create(
            engine="gpt35", # The deployment name you chose when you deployed the GPT-3.5-Turbo or GPT-4 model.
            messages=[
            {"role": "system", "content": "You are a intelligent assistant."},
            {"role": "user", "content": query}
        ],
        )
        reply = response.choices[0].message.content
        return reply
    def get_scores(self, querys, responses, task):
        scores = []
        instruction = self.task_prompt[task]
        for query, response in zip(querys, responses):
            input_prompt = instruction + f"\nA: {query}\nB: {response}"
            # response = self.model(input_prompt)
            response = openai.ChatCompletion.create(
                engine="gpt35", # The deployment name you chose when you deployed the GPT-3.5-Turbo or GPT-4 model.
                messages=[
                {"role": "system", "content": "You are a intelligent assistant."},
                {"role": "user", "content": input_prompt}
            ],
            )
            reply = response.choices[0].message.content
            if "B" in reply:
                score = {"label":task,"score":1.0}
            else:
                score = {"label":task,"score":0.0}
            scores.append(score) 
        return scores 
    
    
class Tokenizer:
    
    @staticmethod
    def load_tokenizer(base_model_name):
        tokenizer_name = Tokenizer.load_tokenizer_name(base_model_name)
        tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_name,
            add_eos_token=True,
            padding_side="left",
            local_files_only=args_utils.LOCAL_FILES_ONLY
        )
        tokenizer.pad_token_id = 0
        return tokenizer

    @staticmethod
    def load_tokenizer_name(model_name):
        if "llama-7b" in model_name or "lora-7b" in model_name:
            return "decapoda-research/llama-7b-hf"
        return model_name
    
    
def save_results(output_dir,demo,querys, responses, args, variant):
    attri = "_".join([reward for reward in args.reward_types])
    filename = f"{output_dir}/{variant}_{attri}.json"
    assert len(querys) == len(responses)
    results = {"querys":querys.tolist(),"responses":responses,"demo":demo}
    # result = pd.DataFrame({"querys":querys,"responses":responses})
    # result.to_csv(filename)
    with open(f"{filename}", "w") as outfile: 
        json.dump(results, outfile)
    print(f"Results have been saved to {filename}!")

def transform_text_assistant(reward_pipe, post, response):
    if reward_pipe.model.name_or_path.startswith("OpenAssistant/"):
        return post + reward_pipe.tokenizer.sep_token + response
    if reward_pipe.model.name_or_path.startswith("theblackcat102"):
        return post + reward_pipe.tokenizer.sep_token + response
    raise ValueError(reward_pipe)

def load_pipe(reward_model, device):
    print(f"Load reward model: {reward_model}")
    if "sentiment" in reward_model or "toxicity" in reward_model:
        pipe = pipeline("text-classification", model=reward_model, device=device,
                        tokenizer=Tokenizer.load_tokenizer_name(reward_model))
        if "toxicity" in reward_model:
            pipe.model.config.id2label = {0: "NEUTRAL", 1: "TOXICITY"} 
    elif "paraphrase" in reward_model:
        pipe = Paraphrase(reward_model,device)
    elif "oasst" in reward_model:
        pipe = AutoModelForSequenceClassification.from_pretrained(reward_model).to(device)
    elif "gpt35-turbo" in reward_model:
        # model = OpenAI(temperature=0.8,
        #                 max_tokens=250,
        #                 model_name="text-davinci-003",
        #                 model_kwargs={"stop": "\n"},
        #                 openai_api_key=os.environ['OPENAI_API_KEY']
        # )
    # try:
        pipe = AutoEvaluator("gpt35-turbo")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(reward_model).to(device)
        tokenier = AutoTokenizer.from_pretrained(reward_model)
        pipe = [model,tokenier]
    return pipe

def load_pipes(reward_models, device):
    pipes = [ load_pipe(reward_model, device) for reward_model in reward_models
        ]
    return pipes

def print_reward(response,rewards):
    for idx in range(len(response)):
        print(f"Response: {response[idx]}")
        for key in rewards:
            print(f"{key}: {rewards[key][idx]}")
        print("\n")
    
reward_categories = {"toxicity":["NEUTRAL","TOXICITY"],"sentiment":["POSITIVE","NEGATIVE"],"simplicity":["simplicity"],"relatedness":["relatedness"],"helpfulness":["helpfulness"]}
def transform_reward(reward,reward_types,responses,verbose=False):
     #use specified reward label in reward categories or reward type instead
    reward_labels, avg_rewards = [],[] 
    reward_results = {}
    for reward_type, rew in zip(reward_types, reward):
        if reward_type not in reward_categories.keys():
            reward_label = reward_type
        else:
            reward_label = reward_categories[reward_type][0]
        avg_reward = []
        print("Evaluate reward type:",reward_type)
        if type(rew[0]) is float:
            avg_reward = [rew_item for rew_item in rew]
        else:
            for r in tqdm(rew):
                if r["label"] == reward_label:
                    avg_reward.append(r["score"])
                else:
                    avg_reward.append(1-r["score"])
        reward_results[reward_label] = avg_reward
        # avg_rewards.append(avg_reward)
        reward_labels.append(reward_label)
    if verbose:
        print_reward(responses,reward_results)
    return reward_results

def mulreward_evaluate(querys,responses,reward_types,device):
    reward_model_names = []
    for reward_type in reward_types:
        reward_model_names.extend(args_utils.DefaultArgs.reward_models[reward_type])
    reward_pipes = load_pipes(reward_model_names, device=device)
    # reward_pipes = []
    scores = []
    for reward_name,reward_pipe in zip(reward_model_names,reward_pipes):
        if "paraphrase" in reward_name:
            scores.append(reward_pipe.get_scores(responses,querys))
        elif "toxicity" in reward_name or "sentiment" in reward_name:
            scores.append(reward_pipe(responses))
        elif "simiplicity" in reward_name:
            scores.append(reward_pipe.get_scores(querys, responses, task="simplicity"))
        elif "oasst" in reward_name:
            tokenizer = AutoTokenizer.from_pretrained(reward_name)
            tokenizer.truncation_side = "left"
            input_content = tokenizer(
            responses,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(device)
            with torch.no_grad():
                scores.append(reward_pipe(**input_content).logits.view(-1).tolist())
        elif "OpenAssistant" in reward_name:
            qa = [q+r for q,r in zip(querys,responses)]
            r_score = []
            for input in qa:
                try:
                    inputs = reward_pipe[1](input, return_tensors='pt',padding=True).to(device)
                    r_score.append(reward_pipe[0](**inputs).logits[:,0].cpu().detach().item())
                except:
                    print("Error in OpenAssistant")
                    print(input)
                    # r_score.append(0.5)
            scores.append(r_score)
    reward_types = reward_model_names if len(reward_types) == 1 else reward_types
    reward_results = transform_reward(scores,reward_types,responses,verbose=False)
    for reward_key in reward_results.keys():
        reward_value = reward_results[reward_key]
        print(f"The avg score for {reward_key} is {sum(reward_value)/len(reward_value)}")
    return reward_results

        