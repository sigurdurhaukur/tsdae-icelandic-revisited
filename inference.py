
"""
author: Sigurdur Haukur Birgisson
description: Inference with a fine-tuned Sentence Transformer model for Icelandic.
Code adapted from the Sentence Transformers library quickstart:
    https://sbert.net/docs/quickstart.html
"""

from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
model_name = "output/tsdae-model" # after training with 1k sentences
model = SentenceTransformer(model_name)

# The sentences to encode
sentences = [
    "Það er mjög fallegt veður í dag.",
    "Hér rignir mikið.",
    "Hundurinn minn er mjög sætur.",
]

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9997, 0.9998],
#         [0.9997, 1.0000, 0.9996],
#         [0.9998, 0.9996, 1.0000]])

# note: This similarity matrix shows that all sentences are very similar to each other,
# indicating that the model has yet to learn meaningful distinctions between them.
# I hope that with longer training and more data, the model will learn to differentiate between sentences better.
