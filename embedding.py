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
]

# Preprocess the corpus (tokenization)
tokenized_corpus = [sentence.lower().split() for sentence in corpus]

# Train Word2Vec model
model = Word2Vec(sentences=tokenized_corpus, vector_size=50, window=5, min_count=1, workers=4)

# Extract embeddings for words of interest
words_of_interest = [
    'obliterates',  # from "When the fox touches a rabbit, it obliterates the rabbit."
    'destroys',
    'promulgates',   # from "The government swiftly promulgates new regulations to curb the spread of the disease."
    'delineates',    # from "The architect meticulously delineates the blueprint to ensure every detail is captured."
    'annihilates',   # from "The storm annihilates everything in its path as it progresses across the land."
    'mitigates',     # from "The company seeks to mitigate the risks associated with the new project."
    'revolutionize', # from "The scientist's discovery could potentially revolutionize the field of genetics."
    'adjudicates',   # from "The judge adjudicates the case with impartiality, ensuring justice is served."
    'calibrates',    # from "The engineer calibrates the instrument to ensure accurate measurements."
    'orchestrates',  # from "The CEO orchestrates the company's strategy to optimize profitability."
    'elucidates',    # from "The teacher elucidates the complex concept so that all students can understand it."
    'endeavors',     # from "The artist endeavors to encapsulate the essence of life in her paintings."
]

embeddings = {word: model.wv[word].tolist() for word in words_of_interest if word in model.wv}

# Save the embeddings to a JSON file
with open('embeddings.json', 'w') as f:
    json.dump(embeddings, f)

print("Word embeddings have been saved to embeddings.json")