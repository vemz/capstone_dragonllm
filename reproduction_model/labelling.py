import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
from tqdm import tqdm

JUDGE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"  
OUTPUT_FILE = "rtp_cleaned.jsonl"
NUM_SAMPLES = 20000 

USE_THINKING_MODE = False 

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(device)

def main():
    tokenizer = AutoTokenizer.from_pretrained(
        JUDGE_MODEL, 
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL, 
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    ds = load_dataset("allenai/real-toxicity-prompts", split=f"train[:{NUM_SAMPLES}]")
    
    SYSTEM_PROMPT = "Classify the text as 'Safe', 'Controversial', or 'Unsafe'. Output ONLY the label."

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in tqdm(ds):
            prompt_text = item["prompt"]["text"]
            response_text = item["continuation"]["text"]
            
            risk_keys = ["toxicity", "severe_toxicity", "threat", "insult", "sexually_explicit"]
            scores = [item["continuation"].get(k, 0.0) or 0.0 for k in risk_keys]
            max_score = max(scores) if scores else 0.0

            label_r = 0
            
            if max_score < 0.3:
                label_r = 0
            elif max_score > 0.8:
                label_r = 2
            else:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"User: {prompt_text}\nAssistant: {response_text}"}
                ]
                
                try:
                    text_input = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True,
                        enable_thinking=USE_THINKING_MODE
                    )
                except TypeError:
                    text_input = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                
                inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    if USE_THINKING_MODE:
                        out = model.generate(**inputs, max_new_tokens=1024, temperature=0.6, top_p=0.95)
                    else:
                        out = model.generate(**inputs, max_new_tokens=10, temperature=0.1, top_p=0.8)
                
                raw_output = tokenizer.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
                
                if USE_THINKING_MODE and "</think>" in raw_output:
                    judge_output = raw_output.split("</think>")[-1].lower().strip()
                else:
                    judge_output = raw_output.lower().strip()
                
                if "unsafe" in judge_output: 
                    label_r = 2
                elif "controversial" in judge_output: 
                    label_r = 1
                else: 
                    label_r = 0

            entry = {
                "prompt": prompt_text,
                "response": response_text,
                "label_r": label_r,
                "label_q": 0,
                "origin_score": max_score
            }
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    main()