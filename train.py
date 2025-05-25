"""
author: Sigurdur Haukur Birgisson
description: Fine-tune 'mideind/IceBERT' using TSDAE (Transformer-based Sentence Denoising Auto-Encoder).
For sentence embeddings in Icelandic.
code adapted from the Sentence Transformers library example:
    https://sbert.net/examples/sentence_transformer/unsupervised_learning/TSDAE/README.html
"""

from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses
from torch.utils.data import DataLoader
from datasets import load_dataset

# Define your sentence transformer model using CLS pooling
model_name = "mideind/IceBERT" # icelandic pre-trained model

word_embedding_model = models.Transformer(model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), "cls")
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Define a list with sentences (1k - 100k sentences)

# using a dataset from the Hugging Face Hub
ds = load_dataset("mideind/icelandic-common-crawl-corpus-IC3")
train_sentences = ds["train"]["text"][:100]  # Use the first 100 sentences for training

# preprocessing
# chunking the sentences to avoid too long sequences
def split_sentences_on_periods(sentences, max_length=512):
    """Split long sentences on periods, keeping complete sentences together."""
    chunks = []
    
    for sentence in sentences:
        if len(sentence) <= max_length:
            chunks.append(sentence)
        else:
            # Split on periods and recombine into chunks
            parts = sentence.split('.')
            current_chunk = ""
            
            for part in parts:
                # Add back the period (except for the last empty part)
                if part.strip():
                    candidate = current_chunk + part + "."
                    if len(candidate) <= max_length:
                        current_chunk = candidate
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = part + "."
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
    
    return chunks

# Split sentences on periods to avoid too long sequences
train_sentences = split_sentences_on_periods(train_sentences, max_length=256)
print(f"Number of sentences after splitting: {len(train_sentences)}")

# Create the special denoising dataset that adds noise on-the-fly
train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)

# DataLoader to batch your data
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

# Use the denoising auto-encoder loss
train_loss = losses.DenoisingAutoEncoderLoss(
    model, decoder_name_or_path=model_name, tie_encoder_decoder=True
)

# Call the fit method
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    weight_decay=0,
    scheduler="constantlr",
    optimizer_params={"lr": 3e-5},
    show_progress_bar=True,
)

model.save("output/tsdae-model")
