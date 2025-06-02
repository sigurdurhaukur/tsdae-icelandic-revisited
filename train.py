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
    # Add input validation
    if not text or len(text.strip()) < 3:
        return text
    
    words = word_tokenize(text)
    n = len(words)
    if n == 0:
        return text
    
    keep_or_not = np.random.rand(n) > del_ratio
    if sum(keep_or_not) == 0:
        keep_or_not[np.random.choice(n)] = True  # guarantee that at least one word remains
    
    words_processed = TreebankWordDetokenizer().detokenize(np.array(words)[keep_or_not])
    
    # Ensure we don't return empty strings
    if not words_processed or len(words_processed.strip()) < 1:
        return text
    
    return words_processed

def create_denoising_dataset(dataset):
    """Convert HF dataset to denoising format with sentence_0 (noisy) and sentence_1 (clean)"""
    def add_noise_pairs(examples):
        noisy_texts = []
        clean_texts = []
        
        for text in examples['text']:
            # Skip empty or very short texts
            if not text or len(text.strip()) < 5:
                continue
                
            noisy_text = delete_noise(text)
            # Double-check the noisy text is valid
            if noisy_text and len(noisy_text.strip()) >= 1:
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
    
    # Filter out any remaining empty entries
    def filter_empty(example):
        return (len(example['sentence_0'].strip()) > 0 and 
                len(example['sentence_1'].strip()) > 0)
    
    denoising_dataset = denoising_dataset.filter(filter_empty)
    
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
def split_sentences_on_token_length(examples, tokenizer, max_tokens=500):
    """Split long sentences based on token length, keeping complete sentences together."""
    chunks = []
    
    for sentence in examples["text"]:
        # Skip empty or very short sentences
        if not sentence or len(sentence.strip()) < 5:
            continue
            
        sentence = sentence.strip()
        
        # Tokenize to check length
        tokens = tokenizer(sentence, add_special_tokens=False)['input_ids']
        
        if len(tokens) <= max_tokens:
            chunks.append(sentence)
        else:
            # Split on periods and recombine based on token length
            parts = sentence.split(".")
            current_chunk = ""
            current_tokens = []
            
            for part in parts:
                part = part.strip()
                if not part:  # Skip empty parts
                    continue
                
                # Tokenize the part
                part_with_period = part + "."
                part_tokens = tokenizer(part_with_period, add_special_tokens=False)['input_ids']
                
                # Check if adding this part would exceed token limit
                if len(current_tokens) + len(part_tokens) <= max_tokens:
                    current_chunk = current_chunk + " " + part_with_period if current_chunk else part_with_period
                    current_tokens.extend(part_tokens)
                else:
                    # Save current chunk if it's valid
                    if current_chunk and len(current_chunk.strip()) > 5:
                        chunks.append(current_chunk.strip())
                    
                    # Start new chunk
                    current_chunk = part_with_period
                    current_tokens = part_tokens.copy()
            
            # Add the final chunk if it's valid
            if current_chunk and len(current_chunk.strip()) > 5:
                chunks.append(current_chunk.strip())

    return {"text": chunks}

# Get tokenizer for preprocessing
tokenizer = models.Transformer(model_name).tokenizer

# Split sentences based on token length to avoid too long sequences
train_ds = train_ds.map(
    split_sentences_on_token_length, 
    fn_kwargs={"tokenizer": tokenizer, "max_tokens": 500}, 
    batched=True, 
    remove_columns=train_ds.column_names
)
test_ds = test_ds.map(
    split_sentences_on_token_length,
    fn_kwargs={"tokenizer": tokenizer, "max_tokens": 500}, 
    batched=True, 
    remove_columns=test_ds.column_names
)

# Filter out any empty texts that might have been created
def filter_valid_texts(example):
    return example['text'] and len(example['text'].strip()) >= 5

train_ds = train_ds.filter(filter_valid_texts)
test_ds = test_ds.filter(filter_valid_texts)

print(f"Number of train sentences after splitting and filtering: {len(train_ds)}")
print(f"Number of test sentences after splitting and filtering: {len(test_ds)}")

# Check if we have enough data
if len(train_ds) == 0 or len(test_ds) == 0:
    raise ValueError("No valid sentences remaining after preprocessing!")

# Create the denoising datasets (converts to sentence_0/sentence_1 format)
train_dataset = create_denoising_dataset(train_ds)
test_dataset = create_denoising_dataset(test_ds)

print(f"Train dataset size after denoising: {len(train_dataset)}")
print(f"Test dataset size after denoising: {len(test_dataset)}")
print(f"Train dataset columns: {train_dataset.column_names}")
print(f"Sample train example: {train_dataset[0]}")

# Validate the datasets
def validate_dataset(dataset, name):
    for i, example in enumerate(dataset.select(range(min(5, len(dataset))))):
        if not example['sentence_0'] or not example['sentence_1']:
            print(f"WARNING: Empty text found in {name} at index {i}")
        if len(example['sentence_0']) > 512 or len(example['sentence_1']) > 512:
            print(f"WARNING: Very long text in {name} at index {i}")

validate_dataset(train_dataset, "train")
validate_dataset(test_dataset, "test")

# Use the denoising auto-encoder loss with proper tokenizer settings
train_loss = losses.DenoisingAutoEncoderLoss(
    model, 
    decoder_name_or_path=model_name, 
    tie_encoder_decoder=True
)

# Ensure the tokenizer has proper settings for attention masks
if hasattr(model.tokenizer, 'pad_token') and model.tokenizer.pad_token is None:
    model.tokenizer.pad_token = model.tokenizer.eos_token

# Define training arguments
args = SentenceTransformerTrainingArguments(
    output_dir="output/tsdae-icelandic-icebert",
    num_train_epochs=1,
    per_device_train_batch_size=8,  # Reduced batch size for safety
    per_device_eval_batch_size=8,   # Reduced batch size for safety
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
    
    # Remove evaluation warning and ensure proper data handling
    remove_unused_columns=False,
    
    # Add data collator settings for safety
    dataloader_drop_last=True,  # Drop incomplete batches
    
    # Use the new non-deprecated argument for including inputs in metrics
    include_for_metrics=["attention_mask"],
)

# Add debugging: check tokenizer limits
print(f"Model max position embeddings: {word_embedding_model.auto_model.config.max_position_embeddings}")
print(f"Tokenizer model max length: {tokenizer.model_max_length}")
print(f"Model vocab size: {len(tokenizer)}")

# Quick test: tokenize a sample to check for issues
sample_text = train_dataset[0]['sentence_0']
tokens = tokenizer(
    sample_text, 
    return_tensors='pt', 
    truncation=True, 
    padding=True, 
    max_length=512,
    return_attention_mask=True
)
print(f"Sample tokenization - input_ids shape: {tokens['input_ids'].shape}")
print(f"Sample tokenization - attention_mask shape: {tokens['attention_mask'].shape}")
print(f"Max token ID in sample: {tokens['input_ids'].max().item()}")
print(f"Min token ID in sample: {tokens['input_ids'].min().item()}")
print(f"Sample text length (tokens): {len(tokens['input_ids'][0])}")
print(f"Attention mask sum (non-padded tokens): {tokens['attention_mask'].sum().item()}")

# Validate token lengths across datasets
def check_token_lengths(dataset, name):
    max_len = 0
    for i, example in enumerate(dataset.select(range(min(10, len(dataset))))):
        for key in ['sentence_0', 'sentence_1']:
            tokens = tokenizer(
                example[key], 
                add_special_tokens=True,
                return_attention_mask=True,
                truncation=True,
                max_length=512
            )
            input_ids = tokens['input_ids']
            attention_mask = tokens['attention_mask']
            
            max_len = max(max_len, len(input_ids))
            
            # Validate attention mask
            if len(input_ids) != len(attention_mask):
                print(f"ERROR: {name} example {i} has mismatched input_ids and attention_mask lengths")
            
            if len(input_ids) > 512:
                print(f"WARNING: {name} example {i} has {len(input_ids)} tokens in {key}")
            
            # Check for valid token IDs
            if max(input_ids) >= len(tokenizer):
                print(f"ERROR: {name} example {i} has invalid token ID {max(input_ids)} >= vocab size {len(tokenizer)}")
                
    print(f"{name} dataset max token length: {max_len}")

check_token_lengths(train_dataset, "Train")
check_token_lengths(test_dataset, "Test")

# Add environment variable to suppress tokenizer parallelism warning
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Create trainer with default data handling
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    loss=train_loss,
    # Remove custom data collator - let SentenceTransformers handle it
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
