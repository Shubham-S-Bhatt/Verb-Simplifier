import gensim
from gensim.models import Word2Vec
import json
import numpy as np

# Sample corpus
corpus = [
    "When the fox touches a rabbit, it obliterates the rabbit.",
    "When the fox touches a rabbit, it destroys the rabbit.",
    "The government swiftly promulgates new regulations to curb the spread of the disease.",
    "The architect meticulously delineates the blueprint to ensure every detail is captured.",
    "The storm annihilates everything in its path as it progresses across the land.",
    "The company seeks to mitigate the risks associated with the new project.",
    "The scientist's discovery could potentially revolutionize the field of genetics.",
    "The judge adjudicates the case with impartiality, ensuring justice is served.",
    "The engineer calibrates the instrument to ensure accurate measurements.",
    "The CEO orchestrates the company's strategy to optimize profitability.",
    "The teacher elucidates the complex concept so that all students can understand it.",
    "The artist endeavors to encapsulate the essence of life in her paintings.",
    "The manager organizes the event.",
    "The professor explains the theory.",
    "The student tries to solve the problem.",
    "The manager organizes the event.",
    "The professor explains the theory.",
    "The student tries to solve the problem.",
    "The initiative mitigates the potential risks.",
    "The project revolutionizes the industry.",
    "The officer judges the situation fairly.",
    "The technician adjusts the settings carefully.",
    "The coordinator arranges the schedules.",
    "The supervisor clarifies the instructions.",
    "The athlete attempts to break the record.",
    "The government issues new policies every year to improve public welfare.",
    "The researcher describes the results of the experiment in detail.",
    "The policy reduces the amount of pollution in the city.",
    "The weather changes rapidly in this region."
]

# Preprocess the corpus (tokenization)
tokenized_corpus = [sentence.lower().split() for sentence in corpus]

# Train Word2Vec model
model = Word2Vec(sentences=tokenized_corpus, vector_size=50, window=5, min_count=1, workers=4)

# Words of interest for which we need embeddings
words_of_interest = [
    'obliterates', 'destroys', 'promulgates', 'delineates', 'annihilates',
    'mitigates', 'revolutionize', 'adjudicates', 'calibrates', 'orchestrates',
    'elucidates', 'endeavors', 'organizes', 'explains', 'tries', 'issues', 'describes',
    'reduces', 'changes', 'judges', 'adjusts', 'arranges', 'clarifies', 'attempts', 
    'revolutionizes', 
]

# Extract embeddings for the words of interest
embeddings = {word: model.wv[word].tolist() for word in words_of_interest if word in model.wv}

# Save the embeddings to a JSON file
with open('embeddings.json', 'w') as f:
    json.dump(embeddings, f)

print("Word embeddings have been saved to embeddings.json")
