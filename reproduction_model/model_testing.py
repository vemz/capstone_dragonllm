import argparse
from contextlib import nullcontext
import os
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import SafetyHead

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Base"
DEFAULT_HEAD_PATH = os.path.join(SCRIPT_DIR, "head_r3.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"{DEVICE}")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float16 if DEVICE.type == "mps" else torch.float32

def normalize_label(value: Any) -> int:
    if isinstance(value, str):
        return 1 if value.strip().lower() == "unsafe" else 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if int(value) == 1 else 0
    return 0

def parse_unsafe_start_index(value: Any) -> int:
    if value is None:
        return -1
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"", "none", "null", "nan"}:
            return -1
        try:
            return int(float(v))
        except ValueError:
            return -1
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value != value:
            return -1
        return int(value)
    return -1

def get_messages(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages_value = item.get("message") or item.get("messages", [])
    if not isinstance(messages_value, list):
        return []
    return [m for m in messages_value if isinstance(m, dict)]

def split_prompt_and_assistant(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    if not messages:
        return [], ""
    assistant_idx = -1
    assistant_content = ""
    for i, msg in enumerate(messages):
        role = str(msg.get("role", "")).strip().lower()
        if role == "assistant":
            assistant_idx = i
            assistant_content = str(msg.get("content", ""))
            break
    if assistant_idx < 0:
        return messages, ""
    return messages[:assistant_idx], assistant_content

def build_input_ids_from_messages(tokenizer, messages: List[Dict[str, Any]], max_length: int) -> Tuple[List[int], int]:
    prompt_messages, _ = split_prompt_and_assistant(messages)
    try:
        full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        prompt_ids = tokenizer.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True)
    except Exception:
        prompt_text = next((str(m.get("content", "")) for m in messages if str(m.get("role", "")).strip().lower() == "user"), "")
        assistant_text = next((str(m.get("content", "")) for m in messages if str(m.get("role", "")).strip().lower() == "assistant"), "")
        raw = f"User: {prompt_text}\nAssistant: {assistant_text}"
        full_ids = tokenizer(raw, add_special_tokens=False).input_ids
        prompt_ids = tokenizer(f"User: {prompt_text}\nAssistant:", add_special_tokens=False).input_ids

    if max_length and max_length > 0 and len(full_ids) > max_length:
        full_ids = full_ids[:max_length]

    response_start = min(len(prompt_ids), len(full_ids))
    return full_ids, response_start

def load_guardtest_split(dataset_name: str, split: str) -> Dataset:
    ds = load_dataset(dataset_name)
    if isinstance(ds, DatasetDict):
        if split not in ds:
            raise ValueError(f"Split '{split}' introuvable. Disponibles: {', '.join(ds.keys())}")
        return ds[split]
    return ds

def load_model(model_name: str, head_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    backbone = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=DTYPE)
    backbone.to(DEVICE)
    backbone.eval()

    hidden_size = backbone.config.hidden_size
    head_r = SafetyHead(hidden_size, num_classes=2)
    head_r.load_state_dict(torch.load(head_path, map_location=DEVICE, weights_only=True))
    head_r.to(DTYPE).to(DEVICE)
    head_r.eval()

    return tokenizer, backbone, head_r

def collate_fn_builder(tokenizer, max_length):
    def collate_fn(batch):
        input_ids_list, attention_masks, response_starts, labels, gt_starts = [], [], [], [], []
        
        for item in batch:
            messages = get_messages(item)
            ids, resp_start = build_input_ids_from_messages(tokenizer, messages, max_length)
            
            input_ids_list.append(torch.tensor(ids, dtype=torch.long))
            response_starts.append(resp_start)
            labels.append(normalize_label(item.get("label", "Safe")))
            gt_starts.append(parse_unsafe_start_index(item.get("unsafe_start_index", -1)))

        padded_inputs = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = (padded_inputs != tokenizer.pad_token_id).long()

        return {
            "input_ids": padded_inputs,
            "attention_mask": attention_mask,
            "response_starts": response_starts,
            "labels": labels,
            "gt_starts": gt_starts
        }
    return collate_fn

def run_eval(args):
    ds = load_guardtest_split(args.dataset, args.split)
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))

    eval_boundary = (args.split == "response_loc")

    tokenizer, backbone, head_r = load_model(args.model, args.head)
    
    dataloader = DataLoader(
        ds, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn_builder(tokenizer, args.max_length)
    )

    y_true, y_pred = [], []
    mae_true, mae_pred = [], []
    boundary_out_of_range = 0
    safe_false_stop_positions = []

    amp_ctx = torch.amp.autocast(DEVICE.type, dtype=DTYPE) if DEVICE.type != "cpu" else nullcontext()

    with torch.inference_mode():
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            
            with amp_ctx:
                outputs = backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                h = outputs.hidden_states[-1]
                logits = head_r(h) # (batch_size, seq_len, 2)
            
            preds = torch.argmax(logits, dim=-1).cpu() # (batch_size, seq_len)

            for i in range(len(batch["labels"])):
                label = batch["labels"][i]
                gt_start = batch["gt_starts"][i]
                resp_start = batch["response_starts"][i]
                
                actual_seq_len = attention_mask[i].sum().item()
                
                start_idx = min(resp_start, actual_seq_len)
                resp_preds = preds[i, start_idx:actual_seq_len]
                
                unsafe_positions = (resp_preds == 1).nonzero(as_tuple=True)[0]
                pred_unsafe_start = unsafe_positions[0].item() if len(unsafe_positions) > 0 else -1
                pred_label = 1 if pred_unsafe_start >= 0 else 0

                y_true.append(label)
                y_pred.append(pred_label)

                if label == 0 and pred_unsafe_start >= 0:
                    safe_false_stop_positions.append(pred_unsafe_start)
                
                elif label == 1 and eval_boundary:
                    if gt_start >= 0:
                        if gt_start < len(resp_preds):
                            mae_true.append(gt_start)
                            mae_pred.append(pred_unsafe_start if pred_unsafe_start >= 0 else len(resp_preds)) 
                        else:
                            boundary_out_of_range += 1

            if (step + 1) % max(1, (len(dataloader) // 10)) == 0:
                print(f"  Progression: {step + 1}/{len(dataloader)} batchs", end="\r")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    acc = (tn + tp) / max(1, len(y_true))
    safe_total = tn + fp
    unsafe_total = fn + tp

    print(f"Accuracy: {acc:.4f}")
    print(f"tn / fp: {tn:>9} | {fp:>11}")
    print(f"fn / tp: {fn:>9} | {tp:>11}")

    if safe_total > 0:
        print(f"\n- Pass through: {tn}/{safe_total} ({tn/safe_total:.1%})")
        print(f"-False positives : {fp}/{safe_total} ({fp/safe_total:.1%})")
    if unsafe_total > 0:
        print(f"- Recall: {tp}/{unsafe_total} ({tp/unsafe_total:.1%})")

    if eval_boundary:
        if mae_true:
            mae = sum(abs(t - p) for t, p in zip(mae_true, mae_pred)) / len(mae_true)
            print(f"MAE on first unsafe token: {mae:.2f} tokens")
            print(f"Valid samples used for MAE: {len(mae_true)}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate SafetyHead on Qwen3GuardTest")
    parser.add_argument("--dataset", default="Qwen/Qwen3GuardTest")
    parser.add_argument("--split", default="thinking", choices=["thinking", "response_loc"])
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--head", default=DEFAULT_HEAD_PATH)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    run_eval(args)

if __name__ == "__main__":
    main()