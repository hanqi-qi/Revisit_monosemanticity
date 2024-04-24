import os
from tqdm import tqdm
import random
from torch.utils.data import Dataset
from datasets import load_dataset
import transformers
from typing import Dict,Iterator
import torch
import json
import numpy as np
from models.model_generate import model_generate_once,prepare_prompt_query,model_generate_batch
from test_examples import _build_stackqa_dataset,load_queries
import torch.nn.functional as F

from transformers.utils import logging
logging.set_verbosity(transformers.logging.ERROR)


orig_template = "{user_tag} {instruction} {assistant_tag} {response}"
control_templates = [
    # "Pretend you're a {type} person giving a response.", 
    # "Make your response as {type} as possible.",
    # "Give a response that is {type}.",
    # "Generate a response in a {type} way.",
]
pos_template = "{user_tag} {instruction} {type} {assistant_tag} {response}"
neg_template = "{user_tag} {instruction} {type} {assistant_tag} {response}"

max_res_len = 64

def get_truncated_outputs(all_outputs, prefixes, num_examples, user_tag, assistant_tag, pos_type, neg_type, control_template):
    orig_s, pos_s, neg_s = [], [], []
    if type(all_outputs) == dict:
        for pos_s,neg_s, p in zip(all_outputs["pos_inputs"],all_outputs["neg_inputs"],prefixes):
            orig_s.append(orig_template.format(
                user_tag=user_tag, assistant_tag=assistant_tag,
                instruction=p, response=pos_s))
            pos_s.append(pos_template.format(
                user_tag=user_tag, assistant_tag=assistant_tag,
                instruction=p, type=control_template.format(type=pos_type), response=pos_s))
            neg_s.append(neg_template.format(
                user_tag=user_tag, assistant_tag=assistant_tag,
                instruction=p, type=control_template.format(type=neg_type), response=neg_s))

            if len(pos_s) > num_examples:
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
                dataset_name
                ):
        super(AlpacaSupervisedDataset, self).__init__()
        self.dataset_name = dataset_name
        instructions,outputs = self.load_ds(dataset_name,'train') #
        # instructions = instructions[:num_examples]
        # outputs = {"pos_inputs":outputs["pos_inputs"][:num_examples], "neg_inputs":outputs["neg_inputs"][:num_examples]}
        print(f"Loaded {len(instructions)} samples for training")
        if type(outputs)==list:
            print("No availiable paired data, only contrastive insts")
            self.user_tag = lorra_args.user_tag
            self.assistant_tag = lorra_args.assistant_tag
            orig_s, pos_s, neg_s = get_truncated_outputs(outputs, 
                                                        instructions, 
                                                        num_examples, 
                                                        self.user_tag,
                                                        self.assistant_tag, 
                                                        lorra_args.pos_type, 
                                                        lorra_args.neg_type,
                                                        lorra_args.control_template)
        elif type(outputs)==dict and "pos_inputs" in outputs.keys() and "neg_inputs" in outputs.keys():
            print("Paired data availiable")
            orig_s = outputs["pos_inputs"]
            pos_s = orig_s
            neg_s = outputs["neg_inputs"]
        self.prompt_s = instructions
        self.orig_s = orig_s
        self.pos_s = pos_s
        self.neg_s = neg_s
        self.max_res_len = lorra_args.max_res_len

        self.tokenizer = tokenizer
        
    def load_ds(self,dataset_name,split="train"):
        if dataset_name == 'stack_qa':
            ds = _build_stackqa_dataset(split="train",dataset_id="lvwerra/stack-exchange-paired")
            instructions = ds['train']['question']
            outputs = ds['train']['response_j']
        if dataset_name == "hh_rlhf" or dataset_name == "toxicity_pair":
            instructions,outputs,_ = load_queries(dataset_name,split)
            print(f"There are {len(instructions)} samples in the {dataset_name} dataset")#660715
        if dataset_name == "tatsu-lab/alpaca":
            ds = load_dataset('tatsu-lab/alpaca')
            ds = ds.filter(lambda x: x['input'] == '')
            instructions = ds['train']['instruction']
            outputs = ds['train']['output']
        return instructions, outputs

    def __len__(self):
        return len(self.orig_s)
    
    def get_pplm_batch_iterator(self,
        training_args,
        split: str = "train",
        device: str = "cuda",
    ) -> Iterator[Dict]:
        """
        Get an iterator over batches of data.

        :params:

        :split: Which split to use.
        :batch_size: Batch size.
        :valid_size: Validation size.
        """
        tokenizer = self.tokenizer
        assert split in ["train", "valid"]
        DATA_DIR = "/scratch/prj/lmrep/hanqi/attribute_edit/attribute_data"
        data_dir = os.path.join(DATA_DIR, "toxicity_pairwise")
        batch_size = training_args.batch_size
        if split == "valid":
            batch_size = training_args.eval_batch_size
        max_prompt_length = training_args.max_prompt_length
        max_new_tokens = training_args.max_new_tokens
        valid_size = training_args.evaluate_num

        filenames = [
            os.path.join(data_dir, filename)
            for filename in os.listdir(data_dir)
            if filename.endswith(".jsonl")
        ]

        data = []
        for filename in tqdm(filenames):
            with open(filename, "r") as file_p:
                file_data = file_p.readlines()

            data.extend(file_data)

        random.shuffle(file_data)
        if split == "train":
            data = data[:-valid_size]
        else:
            data = data[-valid_size:]
        data_size = len(data)

        for idx in range(0, data_size, batch_size):
            batch = data[idx : idx + batch_size]
            batch = [json.loads(x.strip()) for x in batch]

            prompt_text = [x["prompt_text"] for x in batch]
            gold_text = [x["unpert_gen_text"] for x in batch]

            prompt_tokenized = tokenizer(
                prompt_text,
                max_length=max_prompt_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            prompt_input_ids = prompt_tokenized["input_ids"]
            prompt_attention_mask = prompt_tokenized["attention_mask"]

            tokenizer.padding_side = "right"
            gold_tokenized = tokenizer(
                gold_text,
                max_length=max_new_tokens,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            pos_input_id = gold_tokenized["input_ids"].long()

            pplm_text = [x["pert_gen_text"] for x in batch]
            pplm_tokenized = tokenizer(
                pplm_text,
                max_length=max_new_tokens,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            tokenizer.padding_side = "left"

            pos_input_ids = torch.concat(
                [prompt_input_ids, gold_tokenized["input_ids"]], dim=1
            )
            neg_input_ids = torch.concat(
                [prompt_input_ids, pplm_tokenized["input_ids"]], dim=1
            )

            prompt_shape = prompt_input_ids.shape[1]
            pos_labels = pos_input_ids.detach().clone()
            pos_labels[:, :prompt_shape] = -100
            neg_labels = neg_input_ids.detach().clone()
            neg_labels[:, :prompt_shape] = -100

            yield {
                "prompt_input_ids": prompt_input_ids,
                "prompt_attention_mask": prompt_attention_mask,
                "gold_text": gold_text,
                "gold_input_ids": pos_input_id,
                "pos_text": gold_text,
                "pos_input_ids": pos_input_ids,
                "pos_attention_mask": pos_input_ids != tokenizer.pad_token_id,
                "pos_labels": pos_labels,
                "neg_text": pplm_text,
                "neg_input_ids": neg_input_ids,
                "neg_attention_mask": neg_input_ids != tokenizer.pad_token_id,
                "neg_labels": neg_labels,
            }
        
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if "pair" not in self.dataset_name:
            assistant_tag = self.assistant_tag
            orig_s, pos_s, neg_s = self.orig_s[i], self.pos_s[i], self.neg_s[i]
            self.tokenizer.padding_side = "left"
            tokenized_inputs = self.tokenizer(
                [orig_s.split(assistant_tag)[0], 
                pos_s.split(assistant_tag)[0],
                neg_s.split(assistant_tag)[0]],
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            self.tokenizer.padding_side = "right"
            response_tokenized_inputs = self.tokenizer(
                [assistant_tag + orig_s.split(assistant_tag)[1],
                 assistant_tag + pos_s.split(assistant_tag)[1],
                 assistant_tag + neg_s.split(assistant_tag)[1]],
                padding="max_length",
                truncation=True,
                max_length=self.max_res_len,
                return_tensors="pt",
            )
            combined_input_ids = torch.cat([tokenized_inputs["input_ids"], response_tokenized_inputs["input_ids"]], dim=1)
            combined_attention_mask = torch.cat([tokenized_inputs["attention_mask"], response_tokenized_inputs["attention_mask"]], dim=1)
            
            return dict(
                input_ids=combined_input_ids,
                attention_mask=combined_attention_mask
            )

        else:
            # print("Train with paired data")
            """construct combined data as lorra. Prefix: prompt*2, Response:[pos,neg]"""
            prompt_text, gold_text, pos_s, neg_s = self.prompt_s[i], self.orig_s[i], self.pos_s[i], self.neg_s[i]
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token
            prompt_tokenized = self.tokenizer(
            [prompt_text]*2,
            padding="max_length",
            truncation=True,
            max_length=100,
            return_tensors="pt",
            )
            self.tokenizer.padding_side = "right"
            response_tokenized_inputs = self.tokenizer(        
            [pos_s,neg_s],
            padding="max_length",
            truncation=True,
            max_length=self.max_res_len,
            return_tensors="pt",
            )

            combined_input_ids = torch.cat([prompt_tokenized["input_ids"], response_tokenized_inputs["input_ids"]], dim=1)
            combined_attention_mask = torch.cat([prompt_tokenized["attention_mask"], response_tokenized_inputs["attention_mask"]], dim=1)

            prompt_shape = prompt_tokenized["input_ids"].shape[1]
            label_ids = combined_input_ids.detach().clone()
            label_ids[:, :prompt_shape] = -100

            return dict(
                input_ids=combined_input_ids,
                attention_mask=combined_attention_mask,
                label_ids=label_ids,
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

def get_model_responses(model, tokenizer, questions, dataset_name, bsz):
    output_responses = []
    prompt_type = 'default'
    with torch.no_grad():
        for i in range(len(questions) // bsz + 1):
            if len(questions[i*bsz:(i+1)*bsz]) == 0:
                break
            decoded_output = model_generate_batch(model, tokenizer, questions[i*bsz:(i+1)*bsz])
            output_responses.extend([out_seq.split("[/INST]")[1] for out_seq in decoded_output]) if prompt_type == "default" else output_responses.extend([out_seq.split("[helpful]")[1] for out_seq in decoded_output])
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