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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME     = "Qwen/Qwen3-0.6B"
BATCH_SIZE     = 64          # L40S 48 GB VRAM
LR             = 5e-5
EPOCHS         = 3
MAX_LENGTH     = 1024
DTYPE          = torch.bfloat16
DEVICE         = torch.device("cuda") if torch.cuda.is_available() else "cpu"
TRAIN_FILE     = os.path.join(SCRIPT_DIR, "results", "labelling", "rtp_labeled_mixed_25K_cleaned.jsonl")
GRAD_CLIP      = 1.0         # FIX 5: gradient clipping
N_LAYERS       = 4           # FIX 4: moyenne des 4 dernières couches backbone
LOSS_R_WEIGHT  = 2.0         # FIX 2: pondération loss_r


def main():
    wandb.init(
        project="qwen3guard-repro",
        name="train_new",
        config={
            "lr": LR, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "grad_clip": GRAD_CLIP, "n_layers": N_LAYERS,
            "loss_r_weight": LOSS_R_WEIGHT,
        },
    )
    torch.set_float32_matmul_precision("high")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading backbone (may take a few minutes if not cached)...")
    model = StreamGuardModel(MODEL_NAME, n_layers=N_LAYERS, loss_r_weight=LOSS_R_WEIGHT)
    print("Moving backbone to bfloat16...")
    model.backbone.to(DTYPE)
    print(f"Moving model to {DEVICE}...")
    model.to(DEVICE)

    print("Loading dataset...")
    ds_train = load_dataset("json", data_files=TRAIN_FILE, split="train")
    print(f"Dataset loaded: {len(ds_train)} items")

    train_loader = DataLoader(
        ds_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=Collator(tokenizer, MAX_LENGTH),
        num_workers=8,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = AdamW(
        [
            {"params": model.head_q.parameters()},
            {"params": model.head_r.parameters()},
        ],
        lr=LR,
        weight_decay=1e-2,
    )

    # OneCycleLR: warmup + cosine decay en un seul scheduler
    total_steps = EPOCHS * len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LR,
        total_steps=total_steps,
        pct_start=0.05,          # 5% des steps = warmup
        anneal_strategy="cos",
    )

    model.train()
    step = 0
    for epoch in range(EPOCHS):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
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
                # FIX 5: gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                wandb.log({
                    "loss": loss.item(),
                    "lr":   scheduler.get_last_lr()[0],
                    "step": step,
                })
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    torch.save(model.head_q.state_dict(), os.path.join(SCRIPT_DIR, "head_q_new.pth"))
    torch.save(model.head_r.state_dict(), os.path.join(SCRIPT_DIR, "head_r_new.pth"))
    print("Saved head_q_new.pth and head_r_new.pth")


if __name__ == "__main__":
    main()
