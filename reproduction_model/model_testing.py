import argparse
from contextlib import nullcontext
import os
from typing import Any, Dict, List, Tuple

import torch
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import SafetyHead


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Base"
DEFAULT_HEAD_PATH = os.path.join(SCRIPT_DIR, "head_r2.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
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
    messages_value = item.get("message")
    if not isinstance(messages_value, list) or len(messages_value) == 0:
        messages_value = item.get("messages", [])
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


def build_input_ids_from_messages(
    tokenizer,
    messages: List[Dict[str, Any]],
    max_length: int,
) -> Tuple[List[int], int]:
    prompt_messages, _ = split_prompt_and_assistant(messages)
    try:
        full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        prompt_ids = tokenizer.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True)
    except Exception:
        prompt_text = ""
        assistant_text = ""
        for m in messages:
            role = str(m.get("role", "")).strip().lower()
            if role == "user":
                prompt_text = str(m.get("content", ""))
            elif role == "assistant":
                assistant_text = str(m.get("content", ""))
        raw = f"User: {prompt_text}\nAssistant: {assistant_text}"
        full_ids = tokenizer(raw, add_special_tokens=False).input_ids
        prompt_only = f"User: {prompt_text}\nAssistant:"
        prompt_ids = tokenizer(prompt_only, add_special_tokens=False).input_ids

    if max_length is not None and max_length > 0 and len(full_ids) > max_length:
        full_ids = full_ids[:max_length]

    response_start = min(len(prompt_ids), len(full_ids))
    return full_ids, response_start


def load_guardtest_split(dataset_name: str, split: str) -> Dataset:
    ds = load_dataset(dataset_name)
    if isinstance(ds, DatasetDict):
        if split not in ds:
            available = ", ".join(ds.keys())
            raise ValueError(f"Requested split '{split}' not found. Available splits: {available}")
        return ds[split]
    if isinstance(ds, Dataset):
        return ds
    raise ValueError("Unexpected dataset format")


def load_model(model_name: str, head_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=DTYPE)
    backbone.to(DEVICE)
    backbone.eval()

    hidden_size = backbone.config.hidden_size
    head_r = SafetyHead(hidden_size, num_classes=2)
    head_r.load_state_dict(torch.load(head_path, map_location=DEVICE, weights_only=True))
    head_r.to(DTYPE).to(DEVICE)
    head_r.eval()

    return tokenizer, backbone, head_r


@torch.inference_mode()
def predict_first_toxic_token(
    backbone,
    head_r,
    input_ids_list: List[int],
    response_start: int,
) -> Dict[str, int]:
    if not input_ids_list:
        return {"pred_label": 0, "pred_unsafe_start": -1, "response_token_count": 0}

    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=DEVICE)
    attention_mask = torch.ones_like(input_ids, device=DEVICE)

    amp_ctx = torch.amp.autocast(DEVICE.type, dtype=DTYPE) if DEVICE.type != "cpu" else nullcontext()
    with amp_ctx:
        outputs = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        h = outputs.hidden_states[-1]
        logits = head_r(h)

    preds = torch.argmax(logits, dim=-1)[0].detach().cpu()
    start = max(0, min(int(response_start), int(preds.numel())))
    resp_preds = preds[start:]

    unsafe_positions = (resp_preds == 1).nonzero(as_tuple=True)[0]
    pred_unsafe_start = unsafe_positions[0].item() if len(unsafe_positions) > 0 else -1
    pred_label = 1 if pred_unsafe_start >= 0 else 0

    return {
        "pred_label": pred_label,
        "pred_unsafe_start": pred_unsafe_start,
        "response_token_count": int(resp_preds.numel()),
    }


def run_eval(args):
    ds = load_guardtest_split(args.dataset, args.split)
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))

    print(f"Samples: {len(ds)} | split={args.split}")
    print(f"Model: {args.model} | head: {args.head}")
    print("Toxic start rule: first predicted unsafe token")

    tokenizer, backbone, head_r = load_model(args.model, args.head)

    y_true: List[int] = []
    y_pred: List[int] = []

    mae_true: List[int] = []
    mae_pred: List[int] = []
    boundary_out_of_range = 0

    safe_false_stop_positions: List[int] = []
    safe_total = 0
    unsafe_total = 0
    unsafe_stopped = 0

    used_dataset_input_ids = 0
    used_retokenized = 0

    for idx, item in enumerate(ds):
        label = normalize_label(item.get("label", "Safe"))
        gt_start = parse_unsafe_start_index(item.get("unsafe_start_index", -1))
        messages = get_messages(item)

        input_ids_list = item.get("input_ids")
        if isinstance(input_ids_list, list) and len(input_ids_list) > 0:
            _, response_start = build_input_ids_from_messages(tokenizer, messages, max_length=0)
            used_dataset_input_ids += 1
        else:
            input_ids_list, response_start = build_input_ids_from_messages(tokenizer, messages, max_length=args.max_length)
            used_retokenized += 1

        pred = predict_first_toxic_token(
            backbone,
            head_r,
            input_ids_list,
            response_start,
        )

        y_true.append(label)
        y_pred.append(pred["pred_label"])

        if label == 0:
            safe_total += 1
            if pred["pred_unsafe_start"] >= 0:
                safe_false_stop_positions.append(pred["pred_unsafe_start"])
        else:
            unsafe_total += 1
            if pred["pred_unsafe_start"] >= 0:
                unsafe_stopped += 1

            if gt_start >= 0:
                if gt_start < pred["response_token_count"]:
                    mae_true.append(gt_start)
                    mae_pred.append(pred["pred_unsafe_start"])
                else:
                    boundary_out_of_range += 1

        if (idx + 1) % 50 == 0 or (idx + 1) == len(ds):
            print(f"  {idx + 1}/{len(ds)}", end="\r")
    print()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    acc = (tn + tp) / max(1, len(y_true))

    safe_pass_through = tn
    safe_false_stops = fp
    safe_false_stop_mean = float(sum(safe_false_stop_positions) / len(safe_false_stop_positions)) if safe_false_stop_positions else float("nan")

    unsafe_stop_recall = tp / max(1, unsafe_total)

    if mae_true:
        mae = float(torch.mean(torch.abs(torch.tensor(mae_true, dtype=torch.float32) - torch.tensor(mae_pred, dtype=torch.float32))).item())
    else:
        mae = float("nan")

    print(f"Eval modes: dataset_input_ids={used_dataset_input_ids}, retokenized={used_retokenized}")
    print(f"Accuracy (Safe vs Unsafe): {acc:.4f}")
    print("Confusion matrix (rows=true, cols=pred)")
    print("            Safe    Unsafe")
    print(f"Safe    {tn:>8} {fp:>9}")
    print(f"Unsafe  {fn:>8} {tp:>9}")

    print("\nSafe behavior")
    print(f"Safe total: {safe_total}")
    print(f"Safe passed through to the end (pred Safe): {safe_pass_through}/{safe_total} ({safe_pass_through/max(1, safe_total):.1%})")
    print(f"Unjustified early stop on Safe (pred Unsafe): {safe_false_stops}/{safe_total} ({safe_false_stops/max(1, safe_total):.1%})")
    if safe_false_stops > 0:
        print(f"Average unjustified stop token on Safe: {safe_false_stop_mean:.2f}")

    print("\nUnsafe behavior")
    print(f"Unsafe total: {unsafe_total}")
    print(f"Unsafe correctly stopped (pred Unsafe): {tp}/{unsafe_total} ({unsafe_stop_recall:.1%})")
    print(f"Unsafe missed (pred Safe): {fn}/{unsafe_total} ({fn/max(1, unsafe_total):.1%})")

    print("\nBoundary quality on Unsafe")
    print(f"MAE first toxic token: {mae:.2f} tokens")
    print(f"Boundary samples used for MAE: {len(mae_true)}")
    print(f"Boundary gt out-of-range after tokenization/truncation: {boundary_out_of_range}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SafetyHead on Qwen3GuardTest")
    parser.add_argument("--dataset", default="Qwen/Qwen3GuardTest")
    parser.add_argument("--split", default="thinking")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--head", default=DEFAULT_HEAD_PATH)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_eval(args)


if __name__ == "__main__":
    main()