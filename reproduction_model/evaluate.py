"""
Evaluate head_r on the held-out test set (last 20% of the dataset).

Usage:
  python evaluate.py                        # full test set
  python evaluate.py --limit 500            # first 500 test items
  python evaluate.py --head path/to/head.pth
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix

from model import StreamGuardModel, Collator, LABEL_MAP

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME  = "Qwen/Qwen3-0.6B-Base"
MAX_LENGTH  = 1024
BATCH_SIZE  = 32
TRAIN_RATIO = 0.70   # must match train.py
VAL_RATIO   = 0.10
# test = last 20%
DATA_FILE   = os.path.join(SCRIPT_DIR, "results", "labelling", "rtp_labeled_mixed_25K_cleaned.jsonl")
HEAD_PATH   = os.path.join(SCRIPT_DIR, "head_r.pth")

ID2LABEL = {v: k for k, v in LABEL_MAP.items()}   # 0→Safe, 1→Unsafe
UNSAFE   = LABEL_MAP["Unsafe"]
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model(head_path):
    print(f"Loading backbone: {MODEL_NAME}")
    model = StreamGuardModel(MODEL_NAME)
    model.backbone.to(torch.bfloat16)
    print(f"Loading head: {os.path.basename(head_path)}")
    model.head_r.load_state_dict(torch.load(head_path, map_location="cpu"))
    model.to(DEVICE)
    model.eval()
    return model


# ── Forward pass ───────────────────────────────────────────────────────────────
@torch.inference_mode()
def predict_batch(model, batch):
    input_ids      = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)

    with torch.amp.autocast(DEVICE.type, dtype=torch.bfloat16):
        logits_r, _ = model(input_ids, attention_mask)

    r_preds_cls = logits_r.argmax(dim=-1).cpu()   # (B, L)
    labels_r    = batch["labels_r"]               # (B, L)
    seq_lens    = attention_mask.cpu().sum(dim=-1).tolist()

    tok_preds, tok_trues   = [], []
    boundary_preds         = []   # predicted unsafe_token_index (relative to response start)

    for b in range(input_ids.size(0)):
        slen = int(seq_lens[b])

        # token-level: only active non-ignored positions
        for pos in range(slen):
            true_lbl = labels_r[b, pos].item()
            if true_lbl == -100:
                continue
            tok_trues.append(true_lbl)
            tok_preds.append(r_preds_cls[b, pos].item())

        # boundary: find sep_index by scanning labels_r for first non -100 position
        start_gen = next((p for p in range(slen) if labels_r[b, p].item() != -100), slen)

        unsafe_positions = (r_preds_cls[b, start_gen:slen] == UNSAFE).nonzero(as_tuple=True)[0]
        if len(unsafe_positions) > 0:
            boundary_preds.append(unsafe_positions[0].item())
        else:
            boundary_preds.append(-1)

    return tok_preds, tok_trues, boundary_preds, batch


# ── Main eval ──────────────────────────────────────────────────────────────────
def run_eval(model, tokenizer, data_path, limit=None):
    all_items = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                all_items.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    n        = len(all_items)
    n_train  = int(n * TRAIN_RATIO)
    n_val    = int(n * VAL_RATIO)
    items    = all_items[n_train + n_val:]   # last 20%

    items = [it for it in items if it.get("safety_label") in LABEL_MAP]
    if limit:
        items = items[:limit]

    test_pct = 1 - TRAIN_RATIO - VAL_RATIO
    print(f"Test set: {len(items)} items (last {test_pct:.0%} of {n})")

    collator = Collator(tokenizer, MAX_LENGTH)

    tok_all_preds, tok_all_trues = [], []
    boundary_true, boundary_pred = [], []

    for i in range(0, len(items), BATCH_SIZE):
        batch_items = items[i : i + BATCH_SIZE]
        batch       = collator(batch_items)

        tok_preds, tok_trues, b_preds, _ = predict_batch(model, batch)

        tok_all_preds.extend(tok_preds)
        tok_all_trues.extend(tok_trues)

        for it, pred_idx in zip(batch_items, b_preds):
            gt_idx = it.get("unsafe_token_index", -1)
            if LABEL_MAP.get(it["safety_label"]) == UNSAFE and gt_idx >= 0:
                boundary_true.append(gt_idx)
                boundary_pred.append(pred_idx)

        print(f"  {min(i + BATCH_SIZE, len(items))}/{len(items)}", end="\r")
    print()

    # ── Token-level report ────────────────────────────────────────────────────
    labels_present = sorted(set(tok_all_trues + tok_all_preds))
    target_names   = [ID2LABEL[l] for l in labels_present]

    print("\n" + "=" * 55)
    print("head_r — token-level classification")
    print("=" * 55)
    print(classification_report(tok_all_trues, tok_all_preds,
                                labels=labels_present,
                                target_names=target_names,
                                zero_division=0))
    cm = confusion_matrix(tok_all_trues, tok_all_preds, labels=labels_present)
    print("Confusion matrix (rows=true, cols=pred)")
    print("        " + "  ".join(f"{n:>8}" for n in target_names))
    for name, row in zip(target_names, cm):
        print(f"{name:>8}" + "  ".join(f"{v:>8}" for v in row))

    # ── Boundary metrics ──────────────────────────────────────────────────────
    if boundary_true:
        bt = torch.tensor(boundary_true, dtype=torch.float)
        bp = torch.tensor(boundary_pred, dtype=torch.float)

        detected_mask = bp >= 0
        n_total       = len(bt)
        n_detected    = detected_mask.sum().item()

        mae_all = (bt - bp).abs().mean().item()
        if detected_mask.any():
            mae_det = (bt[detected_mask] - bp[detected_mask]).abs().mean().item()
            within5 = ((bt[detected_mask] - bp[detected_mask]).abs() <= 5).float().mean().item()
        else:
            mae_det = float("nan")
            within5 = 0.0

        print("\n" + "=" * 55)
        print("head_r — boundary detection (unsafe items only)")
        print("=" * 55)
        print(f"  Unsafe items with gt index    : {n_total}")
        print(f"  Detected  (pred_idx >= 0)     : {n_detected}  ({n_detected/n_total:.1%})")
        print(f"  Missed    (pred_idx = -1)      : {n_total - n_detected}  ({(n_total-n_detected)/n_total:.1%})")
        print(f"  MAE (all, missed penalized)   : {mae_all:.2f} tokens")
        print(f"  MAE (detected only)            : {mae_det:.2f} tokens")
        print(f"  Within ±5 tokens (detected)    : {within5:.1%}")
    else:
        print("\nNo unsafe items with a valid ground-truth boundary index.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default=DATA_FILE)
    parser.add_argument("--head",  default=HEAD_PATH, help="Path to head_r.pth")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.head)
    run_eval(model, tokenizer, args.data, args.limit)


if __name__ == "__main__":
    main()
