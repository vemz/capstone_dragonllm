import os
import json
from transformers import AutoTokenizer

GUARD_MODEL = "Qwen/Qwen3Guard-Gen-8B"

INPUT_FILE = os.path.join(os.path.dirname(__file__), "rtp_labeled_mixed.jsonl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results", "labelling")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "rtp_labeled_mixed_reformatted.jsonl")

try:
    tokenizer = AutoTokenizer.from_pretrained(GUARD_MODEL)
except:
    tokenizer = None

total = 0
safe_count = 0
unsafe_count = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    with open(INPUT_FILE, "r", encoding="utf-8") as fin:
        for line in fin:
            item = json.loads(line)
            total += 1

            resp = item.get("assistant_response", "")
            if not isinstance(resp, str):
                resp = " ".join(str(x) for x in resp) if isinstance(resp, list) else str(resp)
                item["assistant_response"] = resp

            is_toxic = item["intended_mode"] == "toxic_attempt"

            item["guard_original_label"] = item.get("safety_label", "Unknown")

            if is_toxic:
                item["safety_label"] = "Unsafe"
                if tokenizer:
                    tokens = tokenizer.encode(resp, add_special_tokens=False)
                    item["unsafe_token_index"] = len(tokens)
                else:
                    item["unsafe_token_index"] = len(item["assistant_response"].split())
                unsafe_count += 1
            else:
                item["safety_label"] = "Safe"
                item["unsafe_token_index"] = -1
                safe_count += 1

            fout.write(json.dumps(item) + "\n")

print(f"Total: {total}")
print(f"Safe: {safe_count}")
print(f"Unsafe: {unsafe_count}")
print(f"Output: {OUTPUT_FILE}")
