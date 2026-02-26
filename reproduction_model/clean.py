# remove empty samples
import os
import json
from datasets import load_dataset

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


if __name__ == "__main__":
    remove_empty_fields(INPUT_JSONL)

