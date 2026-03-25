import torch
import torch.nn as nn
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from model import SafetyHead, StreamGuardModel
  
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model_name = "Qwen/Qwen3-4B-Base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = StreamGuardModel(model_name).to(device)

model.head_r.load_state_dict(torch.load("head_r3.pth", map_location=device, weights_only=True))
model.eval()

def remove_thinking_traces(text):
    clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    clean_text = clean_text.replace('<sop><|assistant|>', '').strip()
    return clean_text

def extract_roles(messages):
    user_msg = next((m['content'] for m in messages if m['role'] == 'user'), "")
    ast_msg = next((m['content'] for m in messages if m['role'] == 'assistant'), "")
    return user_msg, ast_msg

def evaluate_token_location():

    dataset_loc = load_dataset("HuggingFaceH4/Qwen3GuardTest", split="response_loc")
    
    total_absolute_error = 0
    valid_samples = 0
    missed_detections = 0 
    
    for sample in tqdm(dataset_loc):
        user_msg, ast_msg = extract_roles(sample['message'])
        ground_truth_index = sample['unsafe_start_index']
        
        user_part = f"User: {user_msg}\nAssistant:"
        full_text = f"{user_part} {ast_msg}"
        
        inputs = tokenizer(full_text, return_tensors="pt").to(device)
        len_prompt_tokens = len(tokenizer(user_part, add_special_tokens=False).input_ids)
        start_gen = len_prompt_tokens
        
        with torch.no_grad():
            logits_r, _ = model(inputs.input_ids, inputs.attention_mask)
            preds = logits_r[0].argmax(dim=-1) 
            
        response_preds = preds[start_gen:]
        
        unsafe_indices = (response_preds == 1).nonzero(as_tuple=True)[0]
        
        if len(unsafe_indices) > 0:
            predicted_index = unsafe_indices[0].item()
            total_absolute_error += abs(predicted_index - ground_truth_index)
            valid_samples += 1
        else:
            missed_detections += 1

    if valid_samples > 0:
        mae = total_absolute_error / valid_samples
        print(f"MAE (on {valid_samples} detections) : {mae:.2f} tokens")
    print(f"Missed unsafe detections : {missed_detections} / {len(dataset_loc)}")


def evaluate_classification():
    dataset_think = load_dataset("HuggingFaceH4/Qwen3GuardTest", split="thinking")
    
    tp, tn, fp, fn = 0, 0, 0, 0
    
    for sample in tqdm(dataset_think):
        user_msg, raw_ast_msg = extract_roles(sample['message'])
        clean_ast_msg = remove_thinking_traces(raw_ast_msg)
        
        ground_truth_label = sample['label']
        is_true_unsafe = (ground_truth_label == "Unsafe")
        
        user_part = f"User: {user_msg}\nAssistant:"
        full_text = f"{user_part} {clean_ast_msg}"
        
        inputs = tokenizer(full_text, return_tensors="pt").to(device)
        len_prompt_tokens = len(tokenizer(user_part, add_special_tokens=False).input_ids)
        start_gen = len_prompt_tokens
        
        with torch.no_grad():
            logits_r, _ = model(inputs.input_ids, inputs.attention_mask)
            preds = logits_r[0].argmax(dim=-1)
            
        response_preds = preds[start_gen:]
        is_pred_unsafe = (response_preds == 1).any().item()
        
        if is_true_unsafe and is_pred_unsafe: tp += 1
        elif not is_true_unsafe and not is_pred_unsafe: tn += 1
        elif not is_true_unsafe and is_pred_unsafe: fp += 1
        elif is_true_unsafe and not is_pred_unsafe: fn += 1

    print(f"{tp} | {fp}")
    print(f"{fn} | {tn}")
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Accuracy  : {accuracy:.2%}")
    print(f"Precision : {precision:.2%}")
    print(f"Recall    : {recall:.2%}")
    print(f"F1-Score  : {f1:.2%}")

evaluate_token_location()
evaluate_classification()