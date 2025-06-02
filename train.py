"""
author: Sigurdur Haukur Birgisson
description: Fine-tune 'mideind/IceBERT' using TSDAE (Transformer-based Sentence Denoising Auto-Encoder).
For sentence embeddings in Icelandic.
code adapted from the Sentence Transformers library example:
    https://sbert.net/examples/sentence_transformer/unsupervised_learning/TSDAE/README.html
"""
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from tqdm import tqdm
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def delete_noise(text, del_ratio=0.6):
    """Apply deletion noise to text, same as original DenoisingAutoEncoderDataset"""
    words = word_tokenize(text)
    n = len(words)
    if n == 0:
        return text
    
    keep_or_not = np.random.rand(n) > del_ratio
    if sum(keep_or_not) == 0:
        keep_or_not[np.random.choice(n)] = True  # guarantee that at least one word remains
    
    words_processed = TreebankWordDetokenizer().detokenize(np.array(words)[keep_or_not])
    return words_processed

def create_denoising_dataset(dataset):
    """Convert HF dataset to denoising format with sentence_0 (noisy) and sentence_1 (clean)"""
    def add_noise_pairs(examples):
        noisy_texts = []
        clean_texts = []
        
        for text in examples['text']:
            noisy_text = delete_noise(text)
            noisy_texts.append(noisy_text)
            clean_texts.append(text)
        
        return {
            'sentence_0': noisy_texts,  # noisy version
            'sentence_1': clean_texts,  # clean version
        }
    
    # Apply noise transformation
    denoising_dataset = dataset.map(
        add_noise_pairs, 
        batched=True, 
        remove_columns=['text'],
        desc="Adding denoising pairs"
    )
    
    return denoising_dataset

# Define your sentence transformer model using CLS pooling
model_name = "mideind/IceBERT"  # icelandic pre-trained model
word_embedding_model = models.Transformer(model_name)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(), "cls"
)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Define a list with sentences (1k - 100k sentences)
# using a dataset from the Hugging Face Hub
train_ds = load_dataset("mideind/icelandic-common-crawl-corpus-IC3", split="train").select(range(100))
test_ds = load_dataset("mideind/icelandic-common-crawl-corpus-IC3", split="test").select(range(100))

# preprocessing
# chunking the sentences to avoid too long sequences
def split_sentences_on_periods(examples, max_length=512):
    """Split long sentences on periods, keeping complete sentences together."""
    chunks = []
    for sentence in examples["text"]:
        if len(sentence) <= max_length:
            chunks.append(sentence)
        else:
            # Split on periods and recombine into chunks
            parts = sentence.split(".")
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

    return {"text": chunks}

# Split sentences on periods to avoid too long sequences
train_ds = train_ds.map(split_sentences_on_periods, 
           fn_kwargs={"max_length": 256}, 
           batched=True, 
           remove_columns=train_ds.column_names
            )
test_ds = test_ds.map(split_sentences_on_periods,
              fn_kwargs={"max_length": 256}, 
              batched=True, 
              remove_columns=test_ds.column_names
                )

print(f"Number of train sentences after splitting: {len(train_ds)}")
print(f"Number of test sentences after splitting: {len(test_ds)}")

# Create the denoising datasets (converts to sentence_0/sentence_1 format)
train_dataset = create_denoising_dataset(train_ds)
test_dataset = create_denoising_dataset(test_ds)

print(f"Train dataset columns: {train_dataset.column_names}")
print(f"Sample train example: {train_dataset[0]}")

# Use the denoising auto-encoder loss
train_loss = losses.DenoisingAutoEncoderLoss(
    model, decoder_name_or_path=model_name, tie_encoder_decoder=True
)

# Define training arguments
args = SentenceTransformerTrainingArguments(
    output_dir="output/tsdae-icelandic-icebert",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_ratio=0.1,
    learning_rate=3e-5,
    weight_decay=0.0,
    lr_scheduler_type="cosine",
    
    # Evaluation settings
    eval_strategy="steps",
    eval_steps=100,
    
    # Saving settings
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    
    # Logging
    logging_steps=100,
    run_name="tsdae-icelandic-icebert",
    report_to=["wandb"],  # Enable wandb logging
    
    # Hub integration
    push_to_hub=True,
    hub_model_id="Sigurdur/tsdae-icelandic-icebert",
    hub_strategy="every_save",  # Push model to hub at every save
    hub_private_repo=True,
    
    # Remove evaluation warning
    remove_unused_columns=False,
)

# Create trainer
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    loss=train_loss,
)

# Train the model
trainer.train()

# Save the final model
model.save("output/tsdae-icelandic-icebert")

# Final push to hub (this will also include wandb logs if configured)
model.push_to_hub(
    "Sigurdur/tsdae-icelandic-icebert", 
    private=True, 
    exist_ok=True, 
    train_datasets=["mideind/icelandic-common-crawl-corpus-IC3"]
)
