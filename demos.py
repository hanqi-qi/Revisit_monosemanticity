from datasets import load_dataset
import pandas as pd
import numpy as np
#demo for expertise and simplicity
# demo_sentiment = [
#     ("Blockchain technology is like a special kind of computer notebook that everyone can write in.", "Blockchain technology can be likened to a immutable distributed ledger."),
#     ("One creative way to reuse plastic waste is to make new toys and playground equipment from it.", "An innovative approach to repurposing plastic waste involves its transformation into a versatile material suitable for constructing roads."),
#     ("Iron Man is a superhero movie about a man who makes himself a suit of armor to fight bad guys.", "Iron Man epitomizes the superhero genre, portraying the journey of a man who engineers his own suit of armor to combat villains.")
#     ]
dataset_honest = load_dataset('truthful_qa', 'generation')['validation']
demo_honesty = dataset_honest

def load_from_pairdata(dataset):
    if dataset == "stack_qa":
        stackqa_paired_data_filename = f"/scratch/prj/lmrep/hanqi/attribute_edit/attribute_data/{dataset}_paired_data.csv"
        paired_data= pd.read_csv(stackqa_paired_data_filename)
        difference = np.diff(paired_data[['r1_score1', 'r2_s1']], axis=1)
        paired_data['r1_s1_diff'] = difference
        pair_data = paired_data.sort_values(by='r1_s1_diff', ascending=False)
        demo_stackqa =[]
        for i in range(5):
            demo_stackqa.append((pair_data.iloc[i]['response2'], pair_data.iloc[i]['response1']))
    elif dataset == "hh_rlhf_helpful":
        pair_data = pd.read_csv("/scratch/prj/lmrep/hanqi/attribute_edit/attribute_data/hh_rlhf_helpful_paired_data.csv")
        demo_stackqa =[]
        for i in range(len(pair_data)):
            if len(pair_data.iloc[i]['chosen'].split(" "))<50 and len(pair_data.iloc[i]['reject'].split(" "))<50:
                demo_stackqa.append((pair_data.iloc[i]['reject'], pair_data.iloc[i]['chosen']))
        print(f"retrieved {len(demo_stackqa)} pairs of data from hh_rlhf_helpful")
    return demo_stackqa


demo_sentiment = [("Zero stars, I hate it.", "Five stars, I love it."),
                ("it was terrible !", "it was awesome!"),
                ("i did nt like it.", "i love it."),
                ("i would call this the worse denny 's ever ", "i would call this the best denny 's ever "),
                ("i would recommend find another place.", "i would recommend this place again!")]

# demo_sentiment = [("Paraphrase the sentence.", "Paraphrase the sentence to be positive."),
#                 ("Paraphrase the sentence.", "Paraphrase the sentence to be happy."),
#                 ("Paraphrase the sentence.", "Paraphrase the sentence to be joyful."),
#                 ("Paraphrase the sentence.", "Paraphrase the sentence to be glad."),
#                 ("Paraphrase the sentence.", "Paraphrase the sentence to be cheerful.")]

demo_helpfulness = [
    ("I’m really not sure if there is an obvious way to do this.","the most accurate way would be to use a computer in your car to log mileage as you drive."),
    ("That’s a surprisingly tricky question!","State limit tables are useful for understanding how much money you could be held personally responsible for in the event of an accident."),
    ("And if you want to use meat in a dehydrator, I like to use steaks","That’s great, and I’ll also write a recipe for a yummy dish with both apples and carrots!"),
    ("...well I like to make really spicy stuff.","I’d suggest fruits, vegetables and meats. But do you want more specific ideas or something even more specific?"),
    ("Would you like to start?","Start by measuring the area you want to cover (use paving")
]

demo_stackqa = [
    ()
]
# demo_helpfulness = [
#     ("Respond to the question","Response to the question with enough necessary information"),
#     ("Respond to the question","Response to the question to be informative"),
#     ("Respond to the question", "Provide a sufficient amount of relevant information in your response to adequately address the question."),
#     ("Respond to the question","Provide an adequate answer to the inquiry with sufficient relevant details"),
#     ("Would you like to start?","Reply to the query, ensuring it contains the requisite amount of pertinent information.")
# ]

demo_simplicity = [
    ("Blockchain technology is a special kind of computer notebook that everyone can write in.", "Blockchain can be likened to a immutable ledger."),
    ("The creative way to reuse plastic waste is to make new toys from it.", "An innovative approach to repurposing plastic waste involves its transformation into a versatile material suitable"),
    ("Iron Man makes himself a suit of armor to fight bad guys.", "Iron Man portrays the journey of a man who engineers his own suit of armor to combat villains.")
    ]

#from paradetox dataset
# demo_toxicity = [
#     ('he had steel balls too !','he was brave too!'), 
#     ('dude should have been taken to api , he would be right at home with all the other knuckleheads there','It would have been good if he went to api. He would fit in.'),
#     ("im not gonna sell the fucking picture , i just want to contribute to the fucking article .", "I\'m not gonna sell the picture, i just want to contribute to the article."),
#     ('the garbage that is being created by cnn and other news agencies is outrageous .', 'the news that is being created by cnn and other news agencies is outrageous.'),
#     ('the reason they dont exist is because neither is a pathological liar like trump .','The reason they don\'t exist is because neither lies like trump')
#     ]
demo_toxicity = [
    ('Paraphrase the sentence.','Paraphrase the sentence to be polite'), 
    ('Paraphrase the sentence.','Paraphrase the sentence to be harmless'), 
    ('Paraphrase the sentence.','Paraphrase the sentence to be friendly'),
    ('Paraphrase the sentence.','Paraphrase the sentence to be gentle'),
    ('Paraphrase the sentence.','Paraphrase the sentence to be kind'), 
    ]

#demo for expertise and simplicity
demo_sentiment_old = [
    ("Blockchain technology is like a special kind of computer notebook that everyone can write in, but no one can erase or change what has already been written. It's like a chain of blocks that are connected and secure, so it's very hard for someone to cheat or lie about something.", 
    "Blockchain technology can be likened to a immutable distributed ledger, wherein entries can only be appended and not altered or removed. Analogous to a shared notebook circulated among numerous participants, each contributor adds information sequentially, ensuring a transparent and tamper-resistant record."),
    ("One creative way to reuse plastic waste is to make new toys and playground equipment from it. For example, old plastic bottles can be cut up and turned into building blocks for kids to play with. Or, plastic bags can be woven together to make a colorful and durable material for swing seats and other playground structures.", 
    "An innovative approach to repurposing plastic waste involves its transformation into a versatile material suitable for constructing roads, a concept known as \"PlasticRoad.\" Through a process of melting plastic and combining it with other substances, a robust and resilient road surface is engineered, demonstrating the potential for sustainable infrastructure solutions."),
    ("One possible synonym for the word \"suspect\" is \"believe\". For example, \"I believe the suspect is guilty of the crime.\"", "Think of a synonym for the word \"suspect\" as being a synonym for the word \"guess\".  Most people use the word \"guess\" in the place of \"suspect\" when they are speaking to children or in informal settings."),
    ("Iron Man is a superhero movie about a man who makes himself a suit of armor to fight bad guys. The movie is very exciting and funny. The special effects are amazing and the action scenes are awesome. I think kids and grown-ups will really enjoy this movie. It's a great start to the Marvel Cinematic Universe.", 
    "Iron Man epitomizes the superhero genre, portraying the journey of a man who engineers his own suit of armor to combat villains. The film captivates audiences with its exhilarating blend of excitement and humor. Remarkable special effects enhance the awe-inspiring action sequences, promising an enthralling cinematic experience. Suitable for viewers of all ages, Iron Man serves as an exceptional introduction to the expansive Marvel Cinematic Universe."),
    ("An imperative sentence gives a command, it's like telling someone to do something, like \"Turn off the lights when you leave the room.\" A declarative sentence makes a statement, it's like giving information, like \"The lights are off.\"", "An imperative sentence is a type of sentence that gives a command or a request. It usually starts with a verb, and it's purpose is to tell someone to do something. For example: \"Turn off the lights when you leave the room.\" \n\nOn the other hand, a declarative sentence is a type of sentence that makes a statement or expresses an opinion. It usually starts with a subject, and its purpose is to provide information. For example: \"The sky is blue today.\"")]