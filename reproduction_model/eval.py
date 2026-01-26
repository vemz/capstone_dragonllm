import argparse
import json
import torch
from datasets import load_dataset
from tqdm import tqdm
from model import StreamGuardModel
from transformers import AutoTokenizer
import torch.nn.functional as F

def get_label_from_category(category_str):
    if not category_str:
        return 0
    cat_lower = str(category_str).lower()
    if cat_lower in ["safe", "none", "nan"]:
        return 0
    elif "controversial" in cat_lower:
        return 1
    else:
        return 2

def calculate_metrics(results, target_class, name):
    tp = sum(1 for r in results if r["pred_r"] == target_class and r["true_label"] == target_class)
    fp = sum(1 for r in results if r["pred_r"] == target_class and r["true_label"] != target_class)
    fn = sum(1 for r in results if r["pred_r"] != target_class and r["true_label"] == target_class)
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    print(f"Classe {name} -> F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

def run_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head_q_path", type=str, default="head_q.pth")
    parser.add_argument("--head_r_path", type=str, default="head_r.pth")
    parser.add_argument("--dataset_name", type=str, default="Qwen/Qwen3GuardTest") 
    parser.add_argument("--split", type=str, default="thinking")
    parser.add_argument("--output_file", type=str, default="qwen_eval_results.jsonl")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.dataset_name} ({args.split}) on {device}...")
    
    try:
        dataset = load_dataset(args.dataset_name, split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    MODEL_NAME = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = StreamGuardModel(MODEL_NAME, num_classes=3)
    
    model.head_q.load_state_dict(torch.load(args.head_q_path, map_location=device))
    model.head_r.load_state_dict(torch.load(args.head_r_path, map_location=device))
    
    model.to(device).eval()
    model.backbone.to(torch.bfloat16)
    model.head_q.to(torch.bfloat16)
    model.head_r.to(torch.bfloat16)

    results = []
    
    print("Running inference...")
    with open(args.output_file, "w", encoding="utf-8") as f:
        for example in tqdm(dataset):
            messages = example.get("message", [])
            if not messages: continue

            category_raw = example.get("unsafe_type", "")
            true_label = get_label_from_category(category_raw)
            category_display = category_raw if category_raw else "Safe"
            
            user_msgs = [m for m in messages if m['role'] == 'user']
            user_text = tokenizer.apply_chat_template(user_msgs, tokenize=False, add_generation_prompt=False)
            user_tokens = tokenizer(user_text, add_special_tokens=False).input_ids
            sep_index = len(user_tokens) - 1
            
            full_text = tokenizer.apply_chat_template(messages, tokenize=False)
            inputs = tokenizer(full_text, return_tensors="pt").to(device)
            
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits_q, logits_r, _ = model(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask
                    )
            
            if sep_index < logits_q.size(1):
                pred_q = torch.argmax(logits_q[0, sep_index, :]).item()
            else:
                pred_q = 0

            start_gen = sep_index + 1
            final_pred_r = 0
            
            if start_gen < logits_r.size(1):
                r_seq_logits = logits_r[0, start_gen:, :]
                r_probs = torch.softmax(r_seq_logits, dim=-1)
                r_preds = torch.argmax(r_probs, dim=-1).tolist()
                
                for i in range(len(r_preds) - 1):
                    if r_preds[i] > 0 and r_preds[i+1] > 0:
                        final_pred_r = max(r_preds[i], r_preds[i+1])
                        break
            
            res_entry = {
                "unique_id": example.get("unique_id", "N/A"),
                "category": category_display,
                "true_label": true_label,
                "pred_q": pred_q,
                "pred_r": final_pred_r,
                "match": (final_pred_r == true_label)
            }
            results.append(res_entry)
            f.write(json.dumps(res_entry) + "\n")

    print("\n" + "="*30)
    print("EVALUATION RESULTS")
    print("="*30)
    
    calculate_metrics(results, target_class=2, name="Unsafe (Global)")
    
    print("\n--- Breakdown by Category ---")
    categories = set(r["category"] for r in results)
    
    for cat in sorted(categories):
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results: continue
        
        total = len(cat_results)
        detected = sum(1 for r in cat_results if r["pred_r"] >= 1)
        rate = detected / total
        
        if cat == "Safe":
            print(f"[{cat}] {total} samples | False Positive Rate: {rate:.2%} (Lower is better)")
        else:
            print(f"[{cat}] {total} samples | Detection Rate: {rate:.2%} (Higher is better)")

if __name__ == "__main__":
    run_evaluation()