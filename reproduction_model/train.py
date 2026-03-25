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

MODEL_NAME = "Qwen/Qwen3-4B-Base"
BATCH_SIZE = 16      
LR = 1.5e-6
EPOCHS = 2
MAX_LENGTH = 1024
LOGGING_STEPS = 10
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"
TRAIN_FILE = os.path.join(SCRIPT_DIR, "results", "labelling", "rtp_labeled_mixed_25K_cleaned.jsonl")

def main():
    wandb.init(
        project="qwen3guard-repro",
        config={
            "model_name": MODEL_NAME,
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
        },
    )
    torch.set_float32_matmul_precision('high')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")

    model = StreamGuardModel(MODEL_NAME)
    model.backbone.to(DTYPE)
    model.to(DEVICE)

    ds_train = load_dataset("json", data_files=TRAIN_FILE, split="train")

    train_loader = DataLoader(
        ds_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=Collator(tokenizer, MAX_LENGTH),
        num_workers=8,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available()
    )

    optimizer = AdamW([
        {'params': model.head_r.parameters()}
    ], lr=LR)

    model.train()
    
    step = 0
    for epoch in range(EPOCHS):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            step += 1
            input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
            attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
            labels_r = batch['labels_r'].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(DEVICE.type, dtype=DTYPE):
                _, loss = model(input_ids, attention_mask, labels_r)
            
            if loss is not None:
                loss.backward() 
                optimizer.step()

                if step % LOGGING_STEPS == 0:
                    wandb.log({"loss": loss.item()}, step=step)
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    torch.save(model.head_r.state_dict(), os.path.join(SCRIPT_DIR, "head_r2.pth"))

if __name__ == "__main__":
    main()