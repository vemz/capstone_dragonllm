import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
from datasets import load_dataset
from model import StreamGuardModel, Collator
from tqdm import tqdm
import wandb
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = "Qwen/Qwen3-0.6B"
BATCH_SIZE = 64      
LR         = 5e-5
EPOCHS     = 3
MAX_LENGTH = 1024
DTYPE      = torch.bfloat16
DEVICE     = torch.device("cuda") if torch.cuda.is_available() else "cpu"
TRAIN_FILE  = os.path.join(SCRIPT_DIR, "results", "labelling", "rtp_labeled_mixed_25K_cleaned.jsonl")
TRAIN_RATIO = 0.70   
VAL_RATIO   = 0.10



@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        input_ids      = batch['input_ids'].to(DEVICE, non_blocking=True)
        attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
        sep_indices    = batch['sep_indices'].to(DEVICE, non_blocking=True)
        labels_q       = batch['labels_q'].to(DEVICE, non_blocking=True)
        labels_r       = batch['labels_r'].to(DEVICE, non_blocking=True)
        with torch.amp.autocast(DEVICE.type, dtype=DTYPE):
            _, _, loss = model(input_ids, attention_mask, labels_q, labels_r, sep_indices)
        if loss is not None:
            total_loss += loss.item()
            n_batches  += 1
    model.train()
    return total_loss / n_batches if n_batches > 0 else float("inf")


def main():
    wandb.init(project="qwen3guard-repro", config={"lr": LR, "epochs": EPOCHS, "batch_size": BATCH_SIZE})
    torch.set_float32_matmul_precision('high')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = StreamGuardModel(MODEL_NAME)
    model.backbone.to(DTYPE)
    model.to(DEVICE)

    collator = Collator(tokenizer, MAX_LENGTH)
    ds_full  = load_dataset("json", data_files=TRAIN_FILE, split="train")
    n        = len(ds_full)
    n_train  = int(n * TRAIN_RATIO)
    n_val    = int(n * VAL_RATIO)
 

    ds_train = ds_full.select(range(n_train))
    ds_val   = ds_full.select(range(n_train, n_train + n_val))
    print(f"Dataset: {n} total → {len(ds_train)} train / {len(ds_val)} val / {n - n_train - n_val} test (held-out)")

    train_loader = DataLoader(
        ds_train, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collator, num_workers=8,
        persistent_workers=True, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        ds_val, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collator, num_workers=4,
        persistent_workers=True, pin_memory=torch.cuda.is_available(),
    )

    optimizer = AdamW([
        {'params': model.head_q.parameters()},
        {'params': model.head_r.parameters()},
    ], lr=LR)

    best_val_loss  = float("inf")
    patience_count = 0
    patience       = 2
    step           = 0

    model.train()
    for epoch in range(EPOCHS):
        # ── Training ──────────────────────────────────────────────────────────
        epoch_loss, n_batches = 0.0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
        for batch in progress_bar:
            step += 1
            input_ids      = batch['input_ids'].to(DEVICE, non_blocking=True)
            attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
            sep_indices    = batch['sep_indices'].to(DEVICE, non_blocking=True)
            labels_q       = batch['labels_q'].to(DEVICE, non_blocking=True)
            labels_r       = batch['labels_r'].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(DEVICE.type, dtype=DTYPE):
                _, _, loss = model(input_ids, attention_mask, labels_q, labels_r, sep_indices)
            if loss is not None:
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1
                wandb.log({"train/loss": loss.item(), "step": step})
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = epoch_loss / n_batches if n_batches > 0 else float("inf")

        # ── Validation ────────────────────────────────────────────────────────
        val_loss = evaluate(model, val_loader)
        wandb.log({"train/loss_epoch": train_loss, "val/loss": val_loss, "epoch": epoch})
        print(f"Epoch {epoch} | train={train_loss:.4f} | val={val_loss:.4f}")

        # ── Best checkpoint + early stopping ──────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save(model.head_q.state_dict(), os.path.join(SCRIPT_DIR, "head_q_best.pth"))
            torch.save(model.head_r.state_dict(), os.path.join(SCRIPT_DIR, "head_r_best.pth"))
            print(f"  ✓ Best val={best_val_loss:.4f} — saved head_q/r_best.pth")
        else:
            patience_count += 1
            print(f"  No improvement ({patience_count}/{patience})")
            if patience_count >= patience:
                print("Early stopping.")
                break

    # Dernier epoch
    torch.save(model.head_q.state_dict(), os.path.join(SCRIPT_DIR, "head_q.pth"))
    torch.save(model.head_r.state_dict(), os.path.join(SCRIPT_DIR, "head_r.pth"))
    print(f"Done. Best val={best_val_loss:.4f} → head_q/r_best.pth")


if __name__ == "__main__":
    main()
