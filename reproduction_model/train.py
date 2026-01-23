import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
from datasets import load_dataset
from model import StreamGuardModel, Collator, adapter_rtp 
from tqdm import tqdm
import wandb

MODEL_NAME = "Qwen/Qwen3-0.6B" 
BATCH_SIZE = 16 
LR = 5e-5 
EPOCHS = 1 
MAX_LENGTH = 1024
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"

def main():
    wandb.init(project="qwen3guard-repro", config={"lr": LR, "epochs": EPOCHS, "batch_size": BATCH_SIZE})

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = StreamGuardModel(MODEL_NAME)
    model.backbone.to(DTYPE)
    model.to(DEVICE)

    ds_rtp = load_dataset("allenai/real-toxicity-prompts", split="train")

    ds_train = ds_rtp.map(
        adapter_rtp, 
        remove_columns=ds_rtp.column_names,
        num_proc=8 
    )

    train_loader = DataLoader(
        ds_train.shuffle(seed=42), 
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=Collator(tokenizer, MAX_LENGTH),
        num_workers=4,
        pin_memory=True
    )

    optimizer = AdamW([
        {'params': model.head_q.parameters()}, 
        {'params': model.head_r.parameters()}
    ], lr=LR)
    
    scaler = torch.amp.GradScaler('cuda')

    model.train()
    
    step = 0
    for epoch in range(EPOCHS):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            step += 1
            input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
            attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
            sep_indices = batch['sep_indices'].to(DEVICE, non_blocking=True)
            labels_q = batch['labels_q'].to(DEVICE, non_blocking=True)
            labels_r = batch['labels_r'].to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=DTYPE):
                _, _, loss = model(input_ids, attention_mask, labels_q, labels_r, sep_indices)
            
            if loss is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                wandb.log({"loss": loss.item()})
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            

    torch.save(model.head_q.state_dict(), "head_q.pth")
    torch.save(model.head_r.state_dict(), "head_r.pth")

if __name__ == "__main__":
    main()