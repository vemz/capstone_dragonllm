import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from model import Collator, LABEL_MAP
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

# Look for unsafe examples
print("Looking for unsafe examples...")
batch_items = []
for i in range(len(ds_train)):
    item = ds_train[i]
    safety_label = item.get('safety_label', 'Safe')
    if safety_label == 'Unsafe':
        batch_items.append(item)
        if len(batch_items) >= 5:
            break

if len(batch_items) == 0:
    print("No unsafe examples found in first 500. Using first 5 examples:")
    batch_items = [ds_train[i] for i in range(min(5, len(ds_train)))]

print(f"\nProcessing {len(batch_items)} examples...")
for item in batch_items:
    label = item.get('safety_label', 'Safe')
    unsafe_idx = item.get('unsafe_token_index', -1)
    print(f"  Label: {label}, unsafe_token_index: {unsafe_idx}")

result = collator(batch_items)
print("\nBatch processed successfully!")
print(f"Output keys: {result.keys()}")
print(f"Labels shape: {result['labels_r'].shape}")
print(f"Sample labels[0]: {result['labels_r'][0][:50]}...")  # Show first 50 tokens
