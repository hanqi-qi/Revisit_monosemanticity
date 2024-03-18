from datasets import load_dataset
from attribute_data import emotion_dataset 
import numpy as np


def load_queries(dataset):
    if dataset == "toxicity":
        # toxicity_queries = load_dataset("s-nlp/paradetox")["train"]
        # positive_queries = toxicity_queries["en_toxic_comment"][5:]
        # negative_queries = toxicity_queries["en_neutral_comment"][5:]
        data = load_dataset("s-nlp/paradetox")["train"]
        query = data["en_toxic_comment"][5:]
    elif dataset == "sentiment":
        data = emotion_dataset.emotion_dict
        query = data["sadness"]
    return query




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
