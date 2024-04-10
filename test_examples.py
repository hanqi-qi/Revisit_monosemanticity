from datasets import load_dataset,ReadInstruction
import pandas as pd
from attribute_data import emotion_dataset 
import numpy as np
import os
import json


MIN_SIZE = 10
MAX_SIZE_NEWS = 1500

def load_queries(dataset):
    responses = []
    if dataset == "toxicity":
        # toxicity_queries = load_dataset("s-nlp/paradetox")["train"]
        # positive_queries = toxicity_queries["en_toxic_comment"][5:]
        # negative_queries = toxicity_queries["en_neutral_comment"][5:]
        data = load_dataset("s-nlp/paradetox")["train"]
        querys = data["en_toxic_comment"][5:]
    elif dataset == "psoup":
        data_path = "/scratch/prj/cllm/datasets/psoups/data/psoups/alpaca_gpt4_P1B_10k.json"
        data_pd = load_dataset("json", data_files=data_path, split = 'train')
        querys = data_path["query"]
        responses = data_path["response"]
    elif dataset == "sentiment":
        data = emotion_dataset.emotion_dict
        querys = data["sadness"]
    elif dataset == "helpfulness":
        querys = pd.read_csv("attribute_data/hh/hh_querys.csv")["querys"].tolist()
    elif dataset == "news-summary":
        dataset_id = "argilla/news-summary"
        querys = _build_news_dataset(split="test", dataset_id=dataset_id)
    elif dataset == "stack_qa":
        filename = "/scratch/prj/cllm/datasets/stack_qa/test_deduplicate.csv"
        # filename = "none"
        if os.path.isfile(filename):
            print("Load from local files")
            query = pd.read_csv(filename)["question"]
            response1 = pd.read_csv(filename)["response_j"]
            response2 = pd.read_csv(filename)["response_k"]
            responses = [r1 + "<SPLIT>"+ r2 for r1,r2 in zip(response1,response2)]
        else:
            print("Load from huggingface")
            dataset_id = "lvwerra/stack-exchange-paired"
            querys = _build_stackqa_dataset(split="train",dataset_id=dataset_id)
    elif dataset == "hh_rlhf":
        filename = "/scratch/prj/cllm/datasets/hh_rlhf/train_deduplicated.json"
        if os.path.isfile(filename):
            print("Load from local files")
            querys = pd.read_csv(filename)["question"]
            response1 = pd.read_csv(filename)["response1"]
            response2 = pd.read_csv(filename)["response2"]
            responses = [r1 + "<SPLIT>"+ r2 for r1,r2 in zip(response1,response2)]
        else:
            print("Load from huggingface")
            dataset_id = "Anthropic/hh-rlhf"
            querys = _build_rlhf_dataset(dataset_name=dataset_id,split="validation")
    return querys,responses

def _build_rlhf_dataset(dataset_name, split="train", max_size=100):

    split = {"train": "train", "validation": "test"}[split]
    dataset_name = f"/scratch/prj/cllm/datasets/hh_rlhf/{split}_deduplicate.csv"
    if os.path.isfile(dataset_name):
        ds = pd.read_csv(dataset_name)
        print("Read files locally")
    else:
        ds = load_dataset("Anthropic/hh-rlhf",cache_dir="/scratch/prj/cllm/datasets/hh_rlhf")
    ds_filtered = ds.filter(
        lambda x: x["chosen"] is not None and MIN_SIZE <
        len(x["chosen"].split("Assistant: ")[0]) < max_size,
        batched=False
    )
    def organize_data(ds_filtered):
        querys, ac_response,re_response = [],[],[]
        for sample in ds_filtered:
            text = sample["chosen"].replace("\n", " ")
            prefix = " ".join(text.split("Human: ")[:-1])
            last_query = text.split("Human: ")[-1].split("Assistant: ")[0]
            try:
                ac_r = text.split("Human: ")[-1].split("Assistant: ")[1]
                re_r = sample["rejected"].replace("\n", " ").split("Human: ")[-1].split("Assistant: ")[1]
                querys.append(last_query)
                ac_response.append(ac_r)
                re_response.append(re_r)
            except:
                print("illegal output")
                print(text)
        return {"query": querys, "chosen": ac_response,"rejected":re_response}

    ds_mapped_train = organize_data(ds_filtered["train"])
    ds_mapped_test = organize_data(ds_filtered["test"])
    with open(f"/scratch/prj/cllm/datasets/hh_rlhf/train_deduplicated.json", "w") as outfile: 
        json.dump(ds_mapped_train, outfile)
    with open(f"/scratch/prj/cllm/datasets/hh_rlhf/test_deduplicated.json", "w") as outfile: 
        json.dump(ds_mapped_test, outfile)
    
    
def load_Psoup_queries(dataset):
    original_columns = dataset.column_names
    num_proc = 24

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "response": [],
        }
        for instruction, response in zip(examples["instruction"], examples["output"]):
            # query = f"<|user|>\n{instruction} {input_ctxt} {PREF_PROMPTS[PREF]}\n<|assistant|>\n"
            query = instruction
            # tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["output"].append(response)
            # new_examples["input_ids"].append(tokenized_question["input_ids"])
        return new_examples

    ds = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) < 300, batched=False)

    ds.set_format(type="pandas")
    return ds

def _build_stackqa_dataset(
    split="test",
    dataset_id="argilla/news-summary",
):
    dataset_name = f"/scratch/prj/cllm/datasets/stack_qa/{split}_deduplicate.csv"
    load_local = "False"
    if os.path.isfile(dataset_name):
        print("Load dataset Locally")
        ds = load_dataset("csv", data_files=dataset_name)
        load_local = "True"
        ds = ds
    else:
        ds = load_dataset("lvwerra/stack-exchange-paired",cache_dir="/scratch/prj/cllm/stackqa")
        print("Load dataset from Huggingface")
        def remove_duplicate(duplicated_dataset):
            initial_list = duplicated_dataset.map(lambda x: {"id": x['qid']})
            _ , unique_indices = np.unique(initial_list["qid"], return_index=True, axis=0)
            filtered_dataset = duplicated_dataset.select(unique_indices.tolist())
            return filtered_dataset
        
        dtrainf = ds["train"].filter(lambda x: len(x["question"]) < 300, batched=False)
        dtrainfs = dtrainf.select(range(40000))
        dtrainf_deduplicated = remove_duplicate(dtrainfs)

        dtestf = ds["test"].filter(lambda x: len(x["question"]) < 300, batched=False)
        dtestfs = dtestf.select(range(1000))
        dtestf_deduplicated = remove_duplicate(dtestfs)

        ds = {"train": dtrainf_deduplicated, "test": dtestf_deduplicated}
        # ds["train"].to_csv("/scratch/prj/cllm/stackqa/train.csv")
        # ds["test"].to_csv("/scratch/prj/cllm/stackqa/test.csv")
    
    return ds
        
def _build_news_dataset(
    split="test",
    dataset_id="argilla/news-summary",
):
    # As the train set is very small and the validation set is larger, we swap them.
    # split = {"train": "test", "validation": "train"}[split]
    # ds = load_dataset("argilla/news-summary", name="comparisons", split=split)
    ds = load_dataset(dataset_id, split="test[:1000]",cache_dir="/scratch/prj/cllm/stackqa")
    ds.to_csv("/scratch/prj/cllm/stackqa/test.csv")
    ds_filtered = ds.filter(
        lambda x: x["question"] is not None and MIN_SIZE < len(x["question"]) < MAX_SIZE_NEWS and x["id"] is
        not None,
        batched=False
    )
    
    def remove_duplicate(duplicated_dataset):
        initial_list = duplicated_dataset.map(lambda x: {"question": x['question']})
        _ , unique_indices = np.unique(initial_list["id"], return_index=True, axis=0)
        filtered_dataset = duplicated_dataset.select(unique_indices.tolist())
        return filtered_dataset

    ds_deduplicated = remove_duplicate(ds_filtered)
    querys = ds_deduplicated.map(lambda x: x["text"])
    return querys

def insts_varations():
    data,labels = [],[]
    emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise","interest", "helpfulness"]
    insts = "consider the following three attributes, which is the leftmost one"
    scenario = "happy, helpful, and insteresting"
    for idx in range(30):
        np.random.shuffle(emotions)
        scenario = ", ".join(emotions[:3])
        answer = emotions[0]
        template_str = f'[INST]{insts}[/INST]:\nScenario: {scenario}\nAnswer: '
        data.append(template_str)
        labels.append(answer)
    return data


simplicity_queries = ['Moth is brave, resourceful, and has a big heart.',
 'For some birds, males sing to attract females to mate; other birds sing to warn other birds of danger.',
 'A restraining order is a legal document that tells someone to stop harming, harassing, or contacting you.',
 'Photosynthesis is when plants use sunlight, water, and air to make food for themselves.',
 'Respiration is when living things, like plants and animals, use food and air to make energy for themselves.',
 'Yoga is a type of exercise that can help your body and your mind.',
 'I was wondering if you could tell me the difference between a dog and a cat.',
 'The earliest civilizations were found in ancient Mesopotamia, which is now modern-day Iraq.',
 'A typical school day starts when you wake up early in the morning, get dressed, and have breakfast with your family.',
 'A judge will listen to your story and look at your evidence. Be honest and clear about what has been happening.'
 ]
