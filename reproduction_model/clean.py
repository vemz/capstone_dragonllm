# remove empty samples
import os
import json
from datasets import load_dataset

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

INPUT_JSONL = "reproduction_model/results/synthesis/rtp_synthesized_mixed_4B.jsonl"


def remove_empty_fields(input_file, output_file=None):
    if output_file is None:
        output_file = input_file.replace(".jsonl", "_cleaned.jsonl")
    
    removed_count = 0
    kept_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            try:
                entry = json.loads(line)
                
                has_empty_field = any(
                    isinstance(v, str) and v.strip() == "" 
                    for v in entry.values()
                )
                
                if not has_empty_field:
                    outfile.write(line)
                    kept_count += 1
                else:
                    removed_count += 1
                    
            except json.JSONDecodeError:
                print(f"Erreur de parsing JSON: {line}")
                continue
    
    print(f"Entrées supprimées: {removed_count}")
    print(f"Entrées conservées: {kept_count}")
    print(f"Fichier de sortie: {output_file}")
    
    return removed_count


def count_tokens(text, tokenizer=None):
    if not text or not isinstance(text, str):
        return 0
    
    if tokenizer is None and TIKTOKEN_AVAILABLE:
        tokenizer = tiktoken.get_encoding("cl100k_base")
    
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    else:
        return len(text.split())


def add_token_counter(input_file, output_file=None):
    if output_file is None:
        output_file = input_file.replace(".jsonl", "_with_tokens.jsonl")
    
    tokenizer = None
    if TIKTOKEN_AVAILABLE:
        tokenizer = tiktoken.get_encoding("cl100k_base")
    
    processed_count = 0
    token_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            try:
                entry = json.loads(line)
                
                if entry.get("intended_mode") == "toxic_attempt":
                    response_text = entry.get("assistant_response", "")
                    tokens = count_tokens(response_text, tokenizer)
                    entry["token_counter"] = tokens
                    token_count += tokens
                elif entry.get("intended_mode") == "safe_attempt":
                    entry["token_counter"] = -1
                
                outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")
                processed_count += 1
                    
            except json.JSONDecodeError:
                print(f"Erreur de parsing JSON: {line}")
                continue
    
    print(f"Entrées traitées: {processed_count}")
    print(f"Tokens totaux ajoutés: {token_count}")
    print(f"Fichier de sortie: {output_file}")
    
    return processed_count


if __name__ == "__main__":
    remove_empty_fields(INPUT_JSONL)
    
    cleaned_file = INPUT_JSONL.replace(".jsonl", "_cleaned.jsonl")
    add_token_counter(cleaned_file)

