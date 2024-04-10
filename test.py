from test_examples import _build_stackqa_dataset

orig_template = "{user_tag} {instruction} {assistant_tag} {response}"
# control_templates = [
#     # "Pretend you're a {type} person giving a response.", 
#     # "Make your response as {type} as possible.",
#     # "Give a response that is {type}.",
#     # "Generate a response in a {type} way.",
# ]
pos_template = "{user_tag} {instruction} {type} {assistant_tag} {response}"
neg_template = "{user_tag} {instruction} {type} {assistant_tag} {response}"

def get_truncated_outputs(all_outputs, prefixes, num_examples, user_tag, assistant_tag, pos_type, neg_type, control_template):
    orig_s, pos_s, neg_s = [], [], []
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

def load_ds(dataset_name="stack_qa"):
    ds = _build_stackqa_dataset(split="train",dataset_id="lvwerra/stack-exchange-paired")
    return ds

ds = load_ds("stack_qa") #
ds = ds.filter(lambda x: x['input'] == '')
instructions = ds['train']['instruction']
outputs = ds['train']['output']
orig_s, pos_s, neg_s = get_truncated_outputs(outputs, 
                                            instructions, 
                                            100, 
                                            "USER",
                                            "ASSISTANT", 
                                            "a better received", 
                                            "a less well received",
                                            "give a {type} answer")



def get_truncated_outputs(all_outputs, prefixes, num_examples, user_tag, assistant_tag, pos_type, neg_type, control_template):
    orig_s, pos_s, neg_s = [], [], []
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