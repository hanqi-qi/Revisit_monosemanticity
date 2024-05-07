import os
from tqdm import tqdm
import random
from torch.utils.data import Dataset
from datasets import load_dataset
import transformers
from typing import Dict,Iterator
import torch
import json
import pandas as pd
import numpy as np
from models.model_generate import model_generate_once,prepare_prompt_query,model_generate_batch
from test_examples import _build_stackqa_dataset,load_queries
import torch.nn.functional as F

from transformers.utils import logging
logging.set_verbosity(transformers.logging.ERROR)

uni_template = """<s>[INST] <<SYS>>
Generate a {type} response to the given input.
<</SYS>>

{instruction} [/INST]
{response}"""

orig_template_pair = "[INST] <<SYS>> Paraphrase the sentence <</SYS>>[/INST] [INST] {instruction} [/INST] {assistant_tag} {response}"
pos_template_pair = "[INST] <<SYS>> Paraphrase the sentence that is {type} <</SYS>>[/INST] [INST] {instruction} [/INST] {assistant_tag} {response}"
neg_template_pair = "[INST] <<SYS>> Paraphrase the sentence that is {type} <</SYS>>[/INST] [INST] {instruction} [/INST] {assistant_tag} {response}"

orig_template = "{user_tag} {instruction} {assistant_tag} {response}"
control_templates = [
    # "Pretend you're a {type} person giving a response.", 
    # "Make your response as {type} as possible.",
    "Give a response that is {type}.",
    # "Generate a response in a {type} way.",
]
pos_template = "{user_tag} {instruction} {type} {assistant_tag} {response}"
neg_template = "{user_tag} {instruction} {type} {assistant_tag} {response}"

hf_cog_reframe_template = """Generate a thought to the given situation. \nSituation: {instruction} \nThought: {response}"""
hf_wiki2_template = """Continue the input sentence. \nInput: {instruction} \nOutput: {response}"""
hf_hh_helpful_template = """Respond to the given request. \n#Request: {instruction}\n#Response: {response}"""

hf_template_dict = {
    "cog_reframe_positive_paired_data":hf_cog_reframe_template, 
    "wiki2_nontoxic_paired_data":hf_wiki2_template,
    "hh_rlhf_helpful_paired_data":hf_hh_helpful_template
}

max_res_len = 64

#define reward type according to different datasets. Will use in prompt_template
dataset_dict = {"wiki2_nontoxic_paired_data":0,"cog_reframe_positive_paired_data":1,"hh_rlhf_helpful_paired_data":2}
attri_dict = {"pos_type":["non-toxic","postive","helpful"],"neg_type":["toxic","negative","useless"]}#can be extended to more key words in pos_type and neg_type

def get_truncated_outputs_multask(all_outputs, prefixes, num_examples, user_tag, assistant_tag, pos_type, neg_type, control_template, attri_ids):
    orig_s, pos_s, neg_s = [], [], []
    for pos_input,neg_input, p, attri_id in zip(all_outputs["pos_inputs"],all_outputs["neg_inputs"],prefixes,attri_ids):
        orig_s.append(uni_template.format(
            type="", instruction=p, response=pos_input))#type is not used in this settings
        pos_s.append(uni_template.format(
            type="", instruction=p, response=pos_input))
        neg_s.append(uni_template.format(
            type="", instruction=p, response=neg_input))
        if len(pos_input) > num_examples:
            break
    return orig_s, pos_s, neg_s
            
def get_truncated_outputs(all_outputs, prefixes, num_examples, user_tag, assistant_tag, pos_type, neg_type, dataset_name):
    orig_s, pos_s, neg_s = [], [], []
    if type(all_outputs) == dict:
        print("Processing Paired data")
        for pos_input,neg_input, p in zip(all_outputs["pos_inputs"],all_outputs["neg_inputs"],prefixes):
            # orig_s.append(orig_template.format(
            #     user_tag=user_tag, assistant_tag=assistant_tag,
            #     instruction=p, response=pos_input))#type is not used in this settings
            # pos_s.append(pos_template.format(
            #     user_tag=user_tag, assistant_tag=assistant_tag,
            #     instruction=p, type=control_template.format(type=pos_type), response=pos_input))
            # neg_s.append(neg_template.format(
            #     user_tag=user_tag, assistant_tag=assistant_tag,
            #     instruction=p, type=control_template.format(type=neg_type), response=neg_input))
            orig_s.append(hf_template_dict[dataset_name].format(
                instruction=p.strip(), response=pos_input))#type is not used in this settings
            pos_s.append(hf_template_dict[dataset_name].format(
                instruction=p.strip(), response=pos_input))
            neg_s.append(hf_template_dict[dataset_name].format(
                instruction=p.strip(), response=neg_input))
            #control_template is used, and type is used.
            if len(pos_input) > num_examples:
                break
    else:
        for s, p in zip(all_outputs, prefixes):
            orig_s.append(orig_template.format(
                user_tag=user_tag, assistant_tag=assistant_tag,
                instruction=p, response=s))
            pos_s.append(pos_template.format(
                user_tag=user_tag, assistant_tag=assistant_tag,
                instruction=p, type=control_template.format(type=pos_type), response=s))
            neg_s.append(neg_template.format(
                user_tag=user_tag, assistant_tag=assistant_tag,
                instruction=p, type=control_template.format(type=neg_type), response=s))

            if len(pos_s) > num_examples:
                break
            
    return orig_s, pos_s, neg_s

class AlpacaSupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                tokenizer: transformers.PreTrainedTokenizer, 
                num_examples,
                lorra_args,
                training_args,
                ):
        super(AlpacaSupervisedDataset, self).__init__()
        self.dataset_name = training_args.dataset_name
        instructions,outputs,attri_ids = self.load_ds(self.dataset_name,'train',use_label=training_args.use_label) #
        self.user_tag = lorra_args.user_tag
        self.assistant_tag = lorra_args.assistant_tag
        self.prompt_max_len = training_args.prompt_max_len
        self.response_max_len = training_args.response_max_len
        orig_s, pos_s, neg_s = None, None, None
        if self.dataset_name != "all_paired_data" and training_args.use_label == "False":
            orig_s, pos_s, neg_s = get_truncated_outputs(outputs, 
                                                        instructions, 
                                                        num_examples, 
                                                        self.user_tag,
                                                        self.assistant_tag, 
                                                        lorra_args.pos_type, 
                                                        lorra_args.neg_type,
                                                        training_args.dataset_name)
        elif training_args.use_label == "True":
            orig_s,attri_ids = instructions,attri_ids
        else:
            orig_s, pos_s, neg_s = get_truncated_outputs_multask(outputs, 
                                                        instructions, 
                                                        num_examples, 
                                                        self.user_tag,
                                                        self.assistant_tag, 
                                                        lorra_args.pos_type, 
                                                        lorra_args.neg_type,
                                                        lorra_args.control_template,
                                                        attri_ids)
        self.prompt_s = instructions
        self.orig_s = orig_s
        self.pos_s = pos_s
        self.neg_s = neg_s
        self.attri_ids = attri_ids
        self.max_res_len = lorra_args.max_res_len

        self.tokenizer = tokenizer
        
    def load_ds(self,dataset_name,split="train",use_label=False):
        attri_ids = None
        if dataset_name == "all_paired_data":
            instructions,outputs,attri_ids = [],{"pos_inputs":[],"neg_inputs":[]},[]
            for dataset in dataset_dict.keys():
                questions,pair_responses,_ = load_queries(dataset,split)
                instructions.extend(questions)
                outputs["pos_inputs"].extend(pair_responses["pos_inputs"])
                outputs["neg_inputs"].extend(pair_responses["neg_inputs"])
                attri_ids.extend(len(questions)*[dataset_dict[dataset]])
        else:
            if dataset_name == "tatsu-lab/alpaca":
                ds = load_dataset('tatsu-lab/alpaca')
                ds = ds.filter(lambda x: x['input'] == '')
                instructions = ds['train']['instruction']
                outputs = ds['train']['output']
            if dataset_name == 'stack_qa':
                ds = _build_stackqa_dataset(split="train",dataset_id="lvwerra/stack-exchange-paired")
                instructions = ds['train']['question']
                outputs = ds['train']['response_j']
            if "paired_data" in dataset_name:
                instructions,outputs,_ = load_queries(dataset_name,split)
                print(f"There are {len(instructions)} samples in the {dataset_name} dataset")#660715
            attri_ids = [9999]*len(instructions) if attri_ids is None else attri_ids
        assert len(instructions) == len(outputs["pos_inputs"]) == len(outputs["neg_inputs"]) == len(attri_ids)
        df = pd.DataFrame({"instruction":instructions,"pos_inputs":outputs["pos_inputs"],"neg_inputs":outputs["neg_inputs"],"attri_ids":attri_ids})
        shuffle_df = df.sample(frac=1)
        print(f"Generate {len(instructions)} samples in total for training")
        if use_label == "True":
            labelled_pd = pd.DataFrame({"sentence":outputs["pos_inputs"]+outputs["neg_inputs"],"label":len(df)*[0]+len(df)*[1]})
            shuffle_df = labelled_pd.sample(frac=1)
            return shuffle_df["sentence"], None, shuffle_df["label"]
        else:
            return shuffle_df["instruction"], {"pos_inputs":shuffle_df["pos_inputs"],"neg_inputs":shuffle_df["neg_inputs"]}, shuffle_df["attri_ids"]

    def __len__(self):
        return len(self.orig_s)
           
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        flag = False
        data_processing = "with_label"
        if data_processing == "lorra":
            assistant_tag = self.assistant_tag
            orig_s, pos_s, neg_s, attri_id = self.orig_s[i], self.pos_s[i], self.neg_s[i], self.attri_ids[i]
            self.tokenizer.padding_side = "left"
            tokenized_inputs = self.tokenizer(
                [orig_s.split(assistant_tag)[0], 
                pos_s.split(assistant_tag)[0],
                neg_s.split(assistant_tag)[0]],
                padding="max_length",
                truncation=True,
                max_length=self.prompt_max_len,
                return_tensors="pt",
            )
            self.tokenizer.padding_side = "right"
            response_tokenized_inputs = self.tokenizer(
                [assistant_tag + orig_s.split(assistant_tag)[1],
                 assistant_tag + pos_s.split(assistant_tag)[1],
                 assistant_tag + neg_s.split(assistant_tag)[1]],
                padding="max_length",
                truncation=True,
                max_length=self.response_max_len,
                return_tensors="pt",
            )
            combined_input_ids = torch.cat([tokenized_inputs["input_ids"], response_tokenized_inputs["input_ids"]], dim=1)
            combined_attention_mask = torch.cat([tokenized_inputs["attention_mask"], response_tokenized_inputs["attention_mask"]], dim=1)
            
            return dict(
                input_ids=combined_input_ids,
                attention_mask=combined_attention_mask,
                labels = torch.tensor(attri_id,dtype=torch.int64)
            )
        elif data_processing == "with_label":
            sentence, label_id= self.orig_s[i], self.attri_ids[i]
            label_ids = torch.tensor(label_id,dtype=torch.int64)
            self.tokenizer.padding_side = "right"
            prompt_tokenized = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=self.prompt_max_len,
            return_tensors="pt",
            )
            return dict(
                input_ids=prompt_tokenized["input_ids"],
                attention_mask=prompt_tokenized["attention_mask"],
                labels = label_ids
            )
        elif data_processing == "paired_data":
            # print("Train with paired data")
            """construct combined data as lorra. Prefix: prompt*2, Response:[pos,neg]"""
            orig_s, pos_s, neg_s, attri_id = self.orig_s[i], self.pos_s[i], self.neg_s[i], self.attri_ids[i]
            self.tokenizer.padding_side = "left"
            print("pos_s:",orig_s)
            print("assistant_tag:",self.assistant_tag)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            prompt_tokenized = self.tokenizer(
            [pos_s.split(self.assistant_tag)[0]+self.assistant_tag, neg_s.split(self.assistant_tag)[0]+self.assistant_tag],
            padding="max_length",
            truncation=True,
            max_length=self.prompt_max_len,
            return_tensors="pt",
            )
            self.tokenizer.padding_side = "right"
            
            response_tokenized_inputs = self.tokenizer(        
            [pos_s.split(self.assistant_tag)[1],neg_s.split(self.assistant_tag)[1]],
            padding="max_length",
            truncation=True,
            max_length=self.response_max_len,
            return_tensors="pt",
            )

            combined_input_ids = torch.cat([prompt_tokenized["input_ids"], response_tokenized_inputs["input_ids"]], dim=1)
            # combined_input_ids = response_tokenized_inputs["input_ids"]
            # combined_attention_mask = response_tokenized_inputs["attention_mask"]
            combined_attention_mask = torch.cat([prompt_tokenized["attention_mask"], response_tokenized_inputs["attention_mask"]], dim=1)

            prompt_shape = prompt_tokenized["input_ids"].shape[1]
            label_ids = combined_input_ids.detach().clone()
            label_ids[:, :prompt_shape] = -100

            return dict(
                input_ids=combined_input_ids,
                attention_mask=combined_attention_mask,
                # label_ids=label_ids,
                labels = torch.tensor(attri_id,dtype=torch.int64)
            )
        

################## Val Datasets ##################

def prepare_inputs(tokenized_text, device):
    # put the text on the device
    tokenized_text = {k: v.to(device) for k, v in tokenized_text.items()}
    position_ids = get_position_ids(tokenized_text['attention_mask'])
    # tokenized_text['position_ids'] = position_ids
    return tokenized_text

def get_position_ids(attention_mask):
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids

def prepare_decoder_only_inputs(prompts, targets, tokenizer, device):
    tokenizer.padding_side = "left"
    prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    tokenizer.padding_side = "right"
    target_inputs = tokenizer(targets, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)
    inputs = {k: torch.cat([prompt_inputs[k], target_inputs[k]], dim=1) for k in prompt_inputs}
    inputs = prepare_inputs(inputs, device)
    labels = inputs["attention_mask"].clone()
    labels[:, :prompt_inputs["input_ids"].shape[1]] = 0
    labels[labels == tokenizer.pad_token_id] = 0
    return inputs, labels

def get_logprobs(logits, input_ids, attention_mask, **kwargs):
    # TODO: comments this in release
    logprobs = F.log_softmax(logits, dim=-1)[:, :-1]
    logprobs = torch.gather(logprobs, -1, input_ids[:, 1:, None]) #[bs,seq,1]
    logprobs = logprobs * attention_mask[:, 1:, None]
    # check for nans
    assert logprobs.isnan().sum() == 0 
    return logprobs.squeeze(-1)

def get_model_responses(model, tokenizer, questions, dataset_name, split_tag = "\nOutput",bsz=16):
    output_responses = []
    prompt_type = 'default'
    with torch.no_grad():
        for i in range(len(questions) // bsz + 1):
            if len(questions[i*bsz:(i+1)*bsz]) == 0:
                break
            decoded_output = model_generate_batch(model, tokenizer, questions[i*bsz:(i+1)*bsz],dataset_name=dataset_name)
            if "toxic" in dataset_name:
                output_responses.extend([out_seq for out_seq in decoded_output]) 
            else:
                output_responses.extend([out_seq.split(split_tag)[1] for out_seq in decoded_output]) 
            # 
            print(f"{i} batch prediction")
    return output_responses
    
def get_logprobs_accuracy(model, tokenizer, questions, answers, labels, bsz):
    output_logprobs = []
    for i in range(len(questions) // bsz + 1):
        q_batch = questions[i*bsz:(i+1)*bsz].tolist()
        a_batch = answers[i*bsz:(i+1)*bsz].tolist()
        inputs, masks = prepare_decoder_only_inputs(q_batch, a_batch, tokenizer, model.device)
        with torch.no_grad():
            logits = model(**inputs).logits #[bs,seq_len,|vocab_size|]
            logprobs = get_logprobs(logits, inputs['input_ids'], masks).sum(-1).detach().cpu().numpy()
        output_logprobs.extend(logprobs)
    i = 0
    cors, cors_norm = [], []
    for l in labels:
        log_probs = output_logprobs[i:i+len(l)]
        completion_len = answers[i:i+len(l)]
        completions_len = np.array([float(len(i)) for i in completion_len])
        cors.append(np.argmax(log_probs) == l.index(1))
        cors_norm.append(np.argmax(log_probs / completions_len) == l.index(1))
        i += len(l)
    return {'acc': np.mean(cors), 'acc_norm': np.mean(cors_norm)}


def load_tqa_sentences(user_tag, assistant_tag):
    dataset = load_dataset('truthful_qa', 'multiple_choice')['validation']
    questions, answers = [],[]
    labels = []
    for d in dataset:
        q = d['question']
        for i in range(len(d['mc1_targets']['labels'])):
            a = d['mc1_targets']['choices'][i]
            questions.append(f'{user_tag} ' + q + ' ')
            answers.append(f'{assistant_tag} ' + a)

        labels.append(d['mc1_targets']['labels'])
    return np.array(questions), np.array(answers), labels

def load_stackqa_sentences():
    ds = _build_stackqa_dataset(split="test",dataset_id="lvwerra/stack-exchange-paired")
    instructions = ds['question']
    outputs = ds['train']['response_j']
    labels = None
    return instructions, outputs, labels


def load_arc_sentences(challenge=False):
    config = 'ARC-Challenge' if challenge else 'ARC-Easy'
    dataset = load_dataset('ai2_arc', config)['validation']

    questions, answers = [],[]
    labels = []
    for d in dataset:
        q = d['question']
        choices = d['choices']['text']
        label = [d['answerKey'] == c for c in d['choices']['label']]
        for a in choices:
            questions.append(f'Question: ' + q + '\nAnswer:')
            answers.append(a)
        labels.append(label)
    return np.array(questions), np.array(answers), labels