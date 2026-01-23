import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from model import StreamGuardModel, Collator, adapter_rtp, adapter_qwen3guard
from tqdm import tqdm
import wandb

MODEL_NAME = "Qwen/Qwen3-0.6B"
BATCH_SIZE = 64
LR = 5e-4
EPOCHS = 3
MAX_LENGTH = 1024
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"

def main():
    wandb.init(project="qwen3guard-repro", config={"lr": LR, "epochs": EPOCHS, "batch_size": BATCH_SIZE})

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = StreamGuardModel(MODEL_NAME)
    model.backbone.to(DTYPE)
    model.to(DEVICE)

    ds_qwen_raw = load_dataset("Qwen/Qwen3GuardTest", split="thinking")
    ds_rtp_raw = load_dataset("allenai/real-toxicity-prompts", split="train[:20000]")

    ds_qwen = ds_qwen_raw.map(
        adapter_qwen3guard, 
        remove_columns=ds_qwen_raw.column_names 
    )

    ds_rtp = ds_rtp_raw.map(
        adapter_rtp, 
        remove_columns=ds_rtp_raw.column_names 
    )

    train_loader = DataLoader(
        concatenate_datasets([ds_qwen, ds_rtp]).shuffle(seed=42),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=Collator(tokenizer, MAX_LENGTH),
        num_workers=8,
        pin_memory=True
    )

    optimizer = AdamW([{'params': model.head_q.parameters()}, {'params': model.head_r.parameters()}], lr=LR)
    scaler = torch.amp.GradScaler('cuda')

    model.train()
    for epoch in range(EPOCHS):
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
            attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
            sep_indices = batch['sep_indices'].to(DEVICE, non_blocking=True)
            labels_q = batch['labels_q'].to(DEVICE, non_blocking=True)
            labels_r = batch['labels_r'].to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=DTYPE):
                _, _, loss = model(input_ids, attention_mask, labels_q, labels_r, sep_indices)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            wandb.log({"loss": loss.item()})

    torch.save(model.head_q.state_dict(), "head_q.pth")
    torch.save(model.head_r.state_dict(), "head_r.pth")

if __name__ == "__main__":
    main()