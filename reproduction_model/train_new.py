import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import AutoTokenizer
from datasets import load_dataset
from model_new import StreamGuardModel, Collator
from tqdm import tqdm
import wandb
import os

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME    = "Qwen/Qwen3-0.6B"
BATCH_SIZE    = 64
LR            = 5e-5
EPOCHS        = 3
MAX_LENGTH    = 1024
DTYPE         = torch.bfloat16
DEVICE        = torch.device("cuda") if torch.cuda.is_available() else "cpu"
TRAIN_FILE    = os.path.join(SCRIPT_DIR, "results", "labelling", "rtp_labeled_mixed_25K_cleaned.jsonl")
GRAD_CLIP     = 1.0
N_LAYERS      = 4
LOSS_R_WEIGHT = 2.0
TRAIN_RATIO   = 0.70   # 70% train / 10% val / 20% test — identique dans evaluate_new.py
VAL_RATIO     = 0.10
# test = derniers 20% → jamais vus pendant le training
PATIENCE      = 2


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        sep_indices    = batch["sep_indices"].to(DEVICE, non_blocking=True)
        labels_q       = batch["labels_q"].to(DEVICE, non_blocking=True)
        labels_r       = batch["labels_r"].to(DEVICE, non_blocking=True)

        with torch.amp.autocast(DEVICE.type, dtype=DTYPE):
            _, _, loss = model(input_ids, attention_mask, labels_q, labels_r, sep_indices)

        if loss is not None:
            total_loss += loss.item()
            n_batches  += 1

    model.train()
    return total_loss / n_batches if n_batches > 0 else float("inf")


def main():
    wandb.init(
        project="qwen3guard-repro",
        name="train_new",
        config={
            "lr": LR, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "grad_clip": GRAD_CLIP, "n_layers": N_LAYERS,
            "loss_r_weight": LOSS_R_WEIGHT, "patience": PATIENCE,
            "train_ratio": TRAIN_RATIO, "val_ratio": VAL_RATIO,
        },
    )
    torch.set_float32_matmul_precision("high")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading backbone (may take a few minutes if not cached)...")
    model = StreamGuardModel(MODEL_NAME, n_layers=N_LAYERS, loss_r_weight=LOSS_R_WEIGHT)
    print(f"Moving model to {DEVICE}...")
    model.to(DEVICE)

    print("Loading dataset...")
    collator = Collator(tokenizer, MAX_LENGTH)
    ds_full  = load_dataset("json", data_files=TRAIN_FILE, split="train")
    n        = len(ds_full)
    n_train  = int(n * TRAIN_RATIO)
    n_val    = int(n * VAL_RATIO)
    # test set = derniers 20% → jamais utilisés ici
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

    optimizer = AdamW(
        [
            {"params": model.head_q.parameters()},
            {"params": model.head_r.parameters()},
        ],
        lr=LR, weight_decay=1e-2,
    )

    total_steps = EPOCHS * len(train_loader)
    scheduler = OneCycleLR(
        optimizer, max_lr=LR, total_steps=total_steps,
        pct_start=0.05, anneal_strategy="cos",
    )

    best_val_loss  = float("inf")
    patience_count = 0
    step           = 0

    model.train()
    for epoch in range(EPOCHS):
        # ── Training ──────────────────────────────────────────────────────────
        epoch_loss, n_batches = 0.0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
        for batch in progress_bar:
            step += 1
            input_ids      = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            sep_indices    = batch["sep_indices"].to(DEVICE, non_blocking=True)
            labels_q       = batch["labels_q"].to(DEVICE, non_blocking=True)
            labels_r       = batch["labels_r"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(DEVICE.type, dtype=DTYPE):
                _, _, loss = model(input_ids, attention_mask, labels_q, labels_r, sep_indices)

            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
                n_batches  += 1
                wandb.log({"train/loss": loss.item(), "lr": scheduler.get_last_lr()[0], "step": step})
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = epoch_loss / n_batches if n_batches > 0 else float("inf")

        # ── Validation ────────────────────────────────────────────────────────
        print(f"Epoch {epoch} — running validation...")
        val_loss = evaluate(model, val_loader)

        wandb.log({"train/loss_epoch": train_loss, "val/loss": val_loss, "epoch": epoch})
        print(f"Epoch {epoch} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        # ── Checkpoint du meilleur modèle ─────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save(model.head_q.state_dict(), os.path.join(SCRIPT_DIR, "head_q_new_best.pth"))
            torch.save(model.head_r.state_dict(), os.path.join(SCRIPT_DIR, "head_r_new_best.pth"))
            print(f"  ✓ New best val_loss={best_val_loss:.4f} — saved best checkpoints")
        else:
            patience_count += 1
            print(f"  No improvement ({patience_count}/{PATIENCE})")
            if patience_count >= PATIENCE:
                print("Early stopping triggered.")
                break

    # Sauvegarde finale (dernier epoch)
    torch.save(model.head_q.state_dict(), os.path.join(SCRIPT_DIR, "head_q_new.pth"))
    torch.save(model.head_r.state_dict(), os.path.join(SCRIPT_DIR, "head_r_new.pth"))
    print(f"Saved last checkpoints. Best val_loss={best_val_loss:.4f} (head_q/r_new_best.pth)")


if __name__ == "__main__":
    main()
