import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from model import Collator
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "Qwen/Qwen3-4B-Base"
MAX_LENGTH = 1024
TRAIN_FILE = os.path.join(SCRIPT_DIR, "results", "labelling", "rtp_labeled_mixed_25K_cleaned.jsonl")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load dataset
ds_train = load_dataset("json", data_files=TRAIN_FILE, split="train")

# Create collator
collator = Collator(tokenizer, MAX_LENGTH)

# Get first batch
batch = [ds_train[i] for i in range(min(5, len(ds_train)))]
print("Processing batch...")
result = collator(batch)
print("\nBatch processed successfully!")
print(f"Output keys: {result.keys()}")
print(f"Labels shape: {result['labels_r'].shape}")
