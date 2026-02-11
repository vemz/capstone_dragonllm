import json
import sys

def parse_guard_output(raw: str) -> dict:
    """Parse le guard_raw_output pour extraire Safety, Categories et Refusal."""
    result = {"safety": None, "categories": [], "refusal": None}
    for part in raw.split("\n"):
        part = part.strip()
        if part.startswith("Safety:"):
            result["safety"] = part.replace("Safety:", "").strip()
        elif part.startswith("Categories:"):
            cats = part.replace("Categories:", "").strip()
            if cats and cats != "None":
                result["categories"] = [c.strip() for c in cats.split(",")]
        elif part.startswith("Refusal:"):
            result["refusal"] = part.replace("Refusal:", "").strip()
    return result


def reformat_dataset(input_file, output_file=None):
    """
    Reformat le dataset en utilisant le guard_raw_output pour corriger safety_label.
    
    Applique la logique de case-insensitive matching pour detecter "Unsafe".
    """
    if output_file is None:
        output_file = input_file.replace(".jsonl", "_reformatted.jsonl")
    
    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")
    
    total = 0
    updated = 0
    
    try:
        with open(input_file, "r", encoding="utf-8") as f_in, \
             open(output_file, "w", encoding="utf-8") as f_out:
            
            for line in f_in:
                total += 1
                try:
                    item = json.loads(line)
                except:
                    continue
                
                # Parse le guard_raw_output
                raw_output = item.get("guard_raw_output", "")
                parsed = parse_guard_output(raw_output)
                
                # Determine si c'est Unsafe (case-insensitive)
                output_lower = raw_output.lower()
                is_unsafe = False
                
                if "safety: unsafe" in output_lower:
                    is_unsafe = True
                elif "safety: controversial" in output_lower and "unethical acts" in output_lower:
                    is_unsafe = True
                
                # Update safety_label
                old_label = item.get("safety_label", "Safe")
                new_label = "Unsafe" if is_unsafe else "Safe"
                
                if old_label != new_label:
                    updated += 1
                    print(f"  Line {total}: safety_label changed '{old_label}' → '{new_label}'")
                
                item["safety_label"] = new_label
                
                # Write reformatted item
                f_out.write(json.dumps(item) + "\n")
        
        print(f"\n✅ Reformatted {total} items ({updated} safety_label updated)")
        print(f"Output written to: {output_file}")
        return True
    
    except FileNotFoundError:
        print(f"❌ File not found: {input_file}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    # Default files
    input_file = "rtp_labeled_mixed.jsonl"
    
    # Allow command-line override
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = None
    
    success = reformat_dataset(input_file, output_file)
    sys.exit(0 if success else 1)
