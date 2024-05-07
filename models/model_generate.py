import torch
import transformers
from transformers.utils import logging

logging.set_verbosity(transformers.logging.ERROR)


uni_template_evaluate = """<s>[INST] <<SYS>>
Generate a {type} response to the given input.
<</SYS>>

{instruction} [/INST]
"""

hf_template = """Generate a thought to the given situation. Situation: {instruction} Thought:"""
hf_cog_reframe_template = """Generate a thought to the given situation. \nSituation: {instruction} \nThought: {response}"""
# hf_wiki2_template = """Continue the input sentence. \nInput: {instruction} \nOutput: {response}"""
hf_wiki2_template = """{instruction}"""
hf_hh_helpful_template = """Respond to the given request. \nRequest: {instruction}\nResponse: {response}"""

hf_template_dict = {
    "cog_reframe_positive_paired_data":hf_cog_reframe_template, 
    "wiki2_nontoxic_paired_data":hf_wiki2_template,
    "hh_rlhf_helpful_paired_data":hf_hh_helpful_template
}

def prepare_prompt_query(tokenizer,query,candi_input,dataset,prompt_type):
        prompt = uni_template_evaluate.format(type="",instruction=query)
        query_prompt = tokenizer(prompt)
        # if dataset == "sentiment":
        #     # query_prompt =  tokenizer(f"""Please paraphrase the following sentence. Sentence: {query}, paraphrase:""")
        #     query_prompt =  tokenizer(f"""Transform the sentence \"{query}\" to be positive: """)
        # elif "cog_reframe_positive" in dataset:
        #     # query_prompt = tokenizer(f"""[INST] Given the situation: \"{query}\", transform the negative thought \"{candi_input}\" into a positive thought. [/INST]""")
        #     query_prompt = tokenizer(f"""[INST] Given the situation: \"{query}\", generate a thought about this situation. [/INST]""")
        # elif "wiki2" in dataset:
        #     # query_prompt = tokenizer(f"""[INST] \"{query}\" [/INST]""")
        #     query_prompt = tokenizer(f"""[INST] Continue the sentence \"{query}\" with non-toxicity. [/INST]""")
        # elif "hh_rlhf" in dataset:
        #     tar_type = dataset.split("hh_rlhf_")[1].split("_paired_data")[0]
        #     query_prompt =  tokenizer(f"""[INST] Given the question {query}, generate a {tar_type} response. [/INST]""")
        # elif dataset == "toxicity":
        #     # query_prompt =  tokenizer(f"""This is a conversation between two people. Context: {query} Response:""")
        #     query_prompt =  tokenizer(f"""[INST]: {query} [/INST]""") if prompt_type == "default" else tokenizer(f"""Respond to this conversation \"{query}\" to be detoxified: """)
        # elif dataset == "simplicity":
        #     query_prompt = tokenizer(f"""Please paraphrase the following sentence. \nSentence: {query} \nParaphrase:""")
        # elif dataset == "helpfulness":
        #     #  query_prompt =  tokenizer(f"""[INST]: {query} [/INST]""")
        #     # query_prompt =  tokenizer(f"""Respond to this query \"{query}\" with enough necessary information: """)
        #     query_prompt =  tokenizer(f"""[INST]: {query} [/INST]""") if prompt_type == "default" else tokenizer(f"""Respond to this question \"{query}\", ensuring to be positive and helpful: """)
        # elif dataset == "inst_varations":
        #     query_prompt = tokenizer(f"""{query}""")
        # else:
        #     query_prompt =  tokenizer(f"""[INST]: {query} [/INST]""")
        return query_prompt 

def model_generate_batch(model,tokenizer,prompts,dataset_name="cog_reframe_positive_paired_data"):
    if "toxic" in dataset_name:
        prompts = [hf_template_dict[dataset_name].format(type="",instruction=" ".join(prompt.split(" ")[:-1])) for prompt in prompts]
    else:
        prompts = [hf_template_dict[dataset_name].format(type="",instruction=prompt,response="") for prompt in prompts]
    encodings = tokenizer(prompts, return_tensors="pt", padding=True).to('cuda')
    generation_outputs = model.generate(
        **encodings,
        max_new_tokens=64,
        # do_sample=True,
        # top_p = 0.75,
        pad_token_id=tokenizer.eos_token_id,
    )
    output_sequences = tokenizer.batch_decode(generation_outputs,skip_special_tokens=True)
    del generation_outputs
    return output_sequences
    
def model_generate_once(model,tokenizer,prompt):
    generation_output = model.generate(
                        input_ids=torch.tensor(prompt['input_ids']).unsqueeze(0).cuda(),
                        attention_mask=torch.tensor(prompt['attention_mask']).unsqueeze(0).cuda(),
                        max_new_tokens=64,
                        pad_token_id=tokenizer.eos_token_id,
                        # do_sample=True,
                        # temperature=0.7,
                        # top_p=0.75,
                        # top_k=40,
                        # eos_token_id=[104,193,1001,25,1702,18858,3166],
                    )
    decoded_output = tokenizer.decode(generation_output[0],skip_special_tokens=True)
    # print(decoded_output.split("Response:")[1]) #this is like "continue the story", so the sentiment/attribute is inherent to the previous tokens.
    #Setting `pad_token_id` to `eos_token_id`:104 for open-end generation.
    """OUTPUTS:
    Please paraphrase the following sentence. Sentence: Worst restaurant ever!, paraphrase: (The restaurant) is very expensive and the food is not good."""
    decoded_output = decoded_output.replace("\n"," ")
    # print(decoded_output)
    return decoded_output