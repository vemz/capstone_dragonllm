import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import SafetyHead
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
HEAD_PATH = os.path.join(SCRIPT_DIR, "head_r.pth")
BACKBONE_FT_PATH = os.path.join(SCRIPT_DIR, "backbone_finetuned.pth")
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = {0: "Safe", 1: "Unsafe"}


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE)

    # Load fine-tuned backbone layers if available
    if os.path.exists(BACKBONE_FT_PATH):
        ft_state = torch.load(BACKBONE_FT_PATH, map_location=DEVICE, weights_only=True)
        backbone.load_state_dict(ft_state, strict=False)
        print(f"Loaded {len(ft_state)} fine-tuned backbone tensors")

    backbone.eval()
    backbone.to(DEVICE)

    hidden_size = backbone.config.hidden_size
    head_r = SafetyHead(hidden_size, num_classes=2)
    head_r.load_state_dict(torch.load(HEAD_PATH, map_location=DEVICE, weights_only=True))
    head_r.to(DTYPE).to(DEVICE)
    head_r.eval()

    return tokenizer, backbone, head_r


def stream_generate(tokenizer, backbone, head_r, user_query, max_new_tokens=256):
    """
    The backbone generates the response token by token. At each step,
    head_r classifies the token as Safe/Unsafe. Stops on first unsafe or EOS.
    """
    prompt_text = f"User: {user_query}\nAssistant:"
    input_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)

    results = []
    past_key_values = None

    print(f"\n{'='*60}")
    print(f"User: {user_query}")
    print(f"{'='*60}")
    print("Assistant (streaming with safety guard):\n")

    for i in range(max_new_tokens):
        with torch.no_grad(), torch.amp.autocast(DEVICE.type, dtype=DTYPE):
            outputs = backbone(
                input_ids=input_ids,
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values

            # Get next token via greedy decoding
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Classify with head_r
            h_last = outputs.hidden_states[-1][:, -1, :]
            safety_logits = head_r(h_last)
            probs = torch.softmax(safety_logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, pred].item()

        token_text = tokenizer.decode(next_token_id[0])
        label = LABEL_NAMES[pred]

        color = "\033[92m" if pred == 0 else "\033[91m"
        reset = "\033[0m"
        print(f"{color}{token_text}{reset}", end="", flush=True)

        results.append({
            "token_idx": i,
            "token": token_text,
            "label": label,
            "confidence": confidence,
            "prob_safe": probs[0, 0].item(),
            "prob_unsafe": probs[0, 1].item(),
        })

        # Stop on EOS
        if next_token_id.item() == tokenizer.eos_token_id:
            break

        # Stop on first unsafe token
        if pred == 1:
            print(f"\n\n{color}[UNSAFE DETECTED at token {i}: \"{token_text}\" "
                  f"(confidence: {confidence:.2%})]{reset}")
            break

        # Next iteration: feed only the new token (KV cache handles the rest)
        input_ids = next_token_id

    print("\n")
    return results


def main():
    print("Loading model...")
    tokenizer, backbone, head_r = load_model()
    print("Model loaded. Interactive mode ready.\n")

    while True:
        try:
            user_query = input("\033[94mUser query (or 'quit'):\033[0m ").strip()
            if user_query.lower() in ('quit', 'exit', 'q'):
                break
            if not user_query:
                continue

            results = stream_generate(tokenizer, backbone, head_r, user_query)

            safe_count = sum(1 for r in results if r["label"] == "Safe")
            unsafe_count = sum(1 for r in results if r["label"] == "Unsafe")
            print(f"Summary: {safe_count} safe tokens, {unsafe_count} unsafe tokens out of {len(results)} generated\n")

        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break


if __name__ == "__main__":
    main()
