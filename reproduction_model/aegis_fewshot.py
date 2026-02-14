import json
import random
from datasets import load_dataset

DATASET_NAME = "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
OUTPUT_FILE = "few_shot_toxic_ready_to_paste.txt"

LEN_SHORT = 20
LEN_LONG = 100

def get_word_count(text):
    if not text: return 0
    return len(text.split())

def format_for_few_shot(prompt, response):
    clean_prompt = prompt.replace("\n", " ").replace('"', "'").strip()
    clean_response = response.replace("\n", " ").replace('"', "'").strip()

    return f'Snippet: "{clean_prompt}"\nJSON: {{{{"user": "{clean_prompt}", "assistant": "{clean_response}"}}}}'

def main():
    try:
        ds = load_dataset(DATASET_NAME, split="train")
    except Exception as e:
        return

    short_bucket = []   
    medium_bucket = []  
    long_bucket = []    

    for item in ds:
        r_label = item.get('response_label') 
        prompt = item.get('prompt') or item.get('instruction_text')
        response = item.get('response') or item.get('response_text')

        if r_label != 'unsafe':
            continue
            
        if prompt and response:
            wc = get_word_count(response)
            entry = {"prompt": prompt, "response": response}

            if wc <= LEN_SHORT:
                short_bucket.append(entry)
            elif wc > LEN_SHORT and wc < LEN_LONG:
                medium_bucket.append(entry)
            elif wc >= LEN_LONG:
                long_bucket.append(entry)

    selection = []
    selection.extend(random.sample(short_bucket, min(len(short_bucket), 30)))
    selection.extend(random.sample(medium_bucket, min(len(medium_bucket), 30)))
    selection.extend(random.sample(long_bucket, min(len(long_bucket), 30)))

    random.shuffle(selection)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in selection:
            block = format_for_few_shot(item['prompt'], item['response'])
            f.write(block + "\n\n")

if __name__ == "__main__":
    main()