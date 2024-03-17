import torch

def model_generate_once(model,tokenizer,prompt):
    generation_output = model.generate(
                        input_ids=torch.tensor(prompt['input_ids']).unsqueeze(0).cuda(),
                        attention_mask=torch.tensor(prompt['attention_mask']).unsqueeze(0).cuda(),
                        max_new_tokens=30,
                        do_sample=False,
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