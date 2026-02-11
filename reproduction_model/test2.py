import json

FILE_PATH = "rtp_labeled_mixed_reformatted.jsonl"

def print_false_safes():
    count = 0
    print(f"--- ANALYSE DES 'FAUX SAFE' (Attaques Ratées) ---\n")
    
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            
            # La condition du "Faux Safe" (FN)
            if item["intended_mode"] == "toxic_attempt" and item["safety_label"] == "Safe":
                
                print(f"📝 ID: {item.get('original_id', '?')}")
                print(f"👤 User: {item['user_query']}")
                print(f"🤖 Assistant (Troll): {item['assistant_response']}")
                print(f"🛡️ Guard Output:\n{item.get('guard_raw_output', '').strip()}")
                print("-" * 60)
                
                count += 1
                if count >= 10:  # On s'arrête après 10 pour pas spammer
                    break
    
    print(f"\nTotal affiché : {count} exemples.")

if __name__ == "__main__":
    print_false_safes()