from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

chat = [
   {"role": "system", "content": "Generate a response to the given input."},#base model no need!
   {"role": "user", "content": "today is ....."}, #use this for 
]

tokenizer.apply_chat_template(chat, tokenize=False)



uni_template = """<s>[INST] <<SYS>>
Generate a response to the given input.
<</SYS>>

{instruction} [/INST]
{response}"""

continue_template = """<s>[INST] <<SYS>>
Finish the incomplete sentence.
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


#no special tokens for base model
hf_cog_reframe_template = """Generate a thought to the given situation. \nSituation: {instruction} \nThought: {response}"""
hf_wiki2_template = """Continue the input sentence. \nInput: {instruction} \nOutput: {response}"""
hf_hh_helpful_template = """Respond to the given request. \nRequest: {instruction}\nResponse: {response}"""

hf_template_dict = {
    "cog_reframe_positive_paired_data":hf_cog_reframe_template, 
    "wiki2_nontoxic_paired_data":hf_wiki2_template,
    "hh_rlhf_helpful_paired_data":hf_hh_helpful_template
}
chat_template_dict = {
    "cog_reframe_positive_paired_data":uni_template, 
    "wiki2_nontoxic_paired_data":uni_template,
    "hh_rlhf_helpful_paired_data":uni_template,
    "challenge_toxicity":continue_template,
    "sycophancy_ab_paired_data":continue_template
}


#define reward type according to different datasets. Will use in prompt_template
dataset_dict = {"wiki2_nontoxic_paired_data":0,"cog_reframe_positive_paired_data":1,"hh_rlhf_helpful_paired_data":2,"challenge_toxicity":3}
attri_dict = {"pos_type":["non-toxic","postive","helpful"],"neg_type":["toxic","negative","useless"]}#can be extended to more key words in pos_type and neg_type

hf_split_tag = {"cog_reframe_positive_paired_data":"Thought:", "wiki2_nontoxic_paired_data":"Output:", "hh_rlhf_helpful_paired_data":"Response:"}
chat_split_tag = {'cog_reframe_positive_paired_data':"[/INST]", 'wiki2_nontoxic_paired_data':"[/INST]", 'hh_rlhf_helpful_paired_data':"[/INST]",'challenge_toxicity':"[/INST]"}