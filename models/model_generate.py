import torch

def prepare_prompt_query(tokenizer,query,dataset,prompt_type):
        if dataset == "sentiment":
            # query_prompt =  tokenizer(f"""Please paraphrase the following sentence. Sentence: {query}, paraphrase:""")
            query_prompt =  tokenizer(f"""Transform the sentence \"{query}\" to be positive: """) 
        elif dataset == "toxicity":
            # query_prompt =  tokenizer(f"""This is a conversation between two people. Context: {query} Response:""")
            query_prompt =  tokenizer(f"""[INST]: {query} [/INST]""") if prompt_type == "default" else tokenizer(f"""Respond to this conversation \"{query}\" to be detoxified: """)
        elif dataset == "simplicity":
            query_prompt = tokenizer(f"""Please paraphrase the following sentence. \nSentence: {query} \nParaphrase:""")
        elif dataset == "helpfulness":
            #  query_prompt =  tokenizer(f"""[INST]: {query} [/INST]""")
            # query_prompt =  tokenizer(f"""Respond to this query \"{query}\" with enough necessary information: """)
            query_prompt =  tokenizer(f"""[INST]: {query} [/INST]""") if prompt_type == "default" else tokenizer(f"""Respond to this question \"{query}\", ensuring to be positive and helpful: """)
        elif dataset == "inst_varations":
            query_prompt = tokenizer(f"""{query}""")
        else:
            query_prompt =  tokenizer(f"""[INST]: {query} [/INST]""")
        return query_prompt 
    

def model_generate_once(model,tokenizer,prompt):
    generation_output = model.generate(
                        input_ids=torch.tensor(prompt['input_ids']).unsqueeze(0).cuda(),
                        attention_mask=torch.tensor(prompt['attention_mask']).unsqueeze(0).cuda(),
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.75,
                        top_k=40,
                        eos_token_id=[104,193,1001,25,1702,18858,3166],
                    )
    decoded_output = tokenizer.decode(generation_output[0])
    # print(decoded_output.split("Response:")[1]) #this is like "continue the story", so the sentiment/attribute is inherent to the previous tokens.
    #Setting `pad_token_id` to `eos_token_id`:104 for open-end generation.
    """OUTPUTS:
    Please paraphrase the following sentence. Sentence: Worst restaurant ever!, paraphrase: (The restaurant) is very expensive and the food is not good."""
    return decoded_output