import os, sys
sys.path.append("./")
sys.path.append("../")
import torch
import transformers
from transformers.utils import logging

logging.set_verbosity(transformers.logging.ERROR)
from utils.prompt import hf_template_dict, chat_template_dict, hf_split_tag, chat_split_tag



def model_generate_batch(model,tokenizer,prompts,dataset_name="cog_reframe_positive_paired_data",training_args={}):
    # if "toxic" in dataset_name:
    #     prompts = [training_args.prompt_template_dict[dataset_name].format(instruction=" ".join(prompt.split(" ")[:-1]),response="") for prompt in prompts]
    # else:
    prompts = [training_args.prompt_template_dict[dataset_name].format(instruction=prompt,response="") for prompt in prompts]
    encodings = tokenizer(prompts, return_tensors="pt", padding=True).to('cuda')
    torch.cuda.empty_cache()
    with torch.no_grad():
        generation_outputs = model.generate(
            **encodings,
            max_new_tokens=64,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.empty_cache()
    output_sequences = tokenizer.batch_decode(generation_outputs,skip_special_tokens=True)
    generation_outputs,encodings = None, None
    model = None
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