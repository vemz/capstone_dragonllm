import argparse
import os
from typing import Any, Dict, List, Tuple

import torch
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import SafetyHead


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
HEAD_PATH = os.path.join(SCRIPT_DIR, "head_r.pth")
BACKBONE_FT_PATH = os.path.join(SCRIPT_DIR, "backbone_finetuned.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.bfloat16
ID2LABEL = {0: "Safe", 1: "Unsafe"}


def normalize_label(value: Any) -> int:
	if isinstance(value, str):
		v = value.strip().lower()
		if v == "unsafe":
			return 1
		return 0
	if isinstance(value, bool):
		return int(value)
	if isinstance(value, (int, float)):
		return 1 if int(value) == 1 else 0
	return 0


def parse_unsafe_start_index(value: Any) -> int:
	if value is None:
		return -1
	if isinstance(value, str):
		v = value.strip().lower()
		if v in {"", "none", "null", "nan"}:
			return -1
		try:
			return int(float(v))
		except ValueError:
			return -1
	if isinstance(value, bool):
		return int(value)
	if isinstance(value, (int, float)):
		if isinstance(value, float) and value != value:
			return -1
		return int(value)
	return -1


def extract_user_assistant(messages: Any) -> Tuple[str, str]:
	if not isinstance(messages, list):
		return "", ""

	user_content = ""
	assistant_content = ""
	for msg in messages:
		if not isinstance(msg, dict):
			continue
		role = str(msg.get("role", "")).strip().lower()
		content = str(msg.get("content", ""))
		if role == "user":
			user_content = content
		elif role == "assistant":
			assistant_content = content
	return user_content, assistant_content


def load_guardtest_split(dataset_name: str, split: str) -> Dataset:
	ds = load_dataset(dataset_name)
	if isinstance(ds, DatasetDict):
		if split in ds:
			return ds[split]
		for candidate in ("test", "validation", "train"):
			if candidate in ds:
				print(f"Split '{split}' introuvable, fallback vers '{candidate}'.")
				return ds[candidate]
		first_split = next(iter(ds.keys()))
		print(f"Split '{split}' introuvable, fallback vers '{first_split}'.")
		return ds[first_split]
	if isinstance(ds, Dataset):
		return ds
	raise ValueError("Format de dataset inattendu.")


def load_model(head_path: str, backbone_ft_path: str = ""):
	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	print(f"Loading backbone: {MODEL_NAME}")
	backbone = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE)

	if backbone_ft_path and os.path.exists(backbone_ft_path):
		ft_state = torch.load(backbone_ft_path, map_location=DEVICE, weights_only=True)
		backbone.load_state_dict(ft_state, strict=False)
		print(f"Loaded {len(ft_state)} fine-tuned backbone tensors")
	elif backbone_ft_path:
		print(f"Backbone FT introuvable: {backbone_ft_path} (on continue sans)")

	backbone.to(DEVICE)
	backbone.eval()

	hidden_size = backbone.config.hidden_size
	head_r = SafetyHead(hidden_size, num_classes=2)
	head_r.load_state_dict(torch.load(head_path, map_location=DEVICE, weights_only=True))
	head_r.to(DTYPE).to(DEVICE)
	head_r.eval()

	return tokenizer, backbone, head_r


@torch.inference_mode()
def predict_from_dataset_input_ids(backbone, head_r, input_ids_list: List[int]) -> Dict[str, Any]:
	if not input_ids_list:
		return {
			"pred_label": 0,
			"pred_unsafe_start": -1,
			"response_token_count": 0,
		}

	input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=DEVICE)
	attention_mask = torch.ones_like(input_ids, device=DEVICE)

	with torch.amp.autocast(DEVICE.type, dtype=DTYPE):
		outputs = backbone(
			input_ids=input_ids,
			attention_mask=attention_mask,
			output_hidden_states=True,
		)
		h = outputs.hidden_states[-1]
		logits = head_r(h)

	preds = torch.argmax(logits, dim=-1)[0].detach().cpu()
	unsafe_positions = (preds == 1).nonzero(as_tuple=True)[0]
	pred_unsafe_start = unsafe_positions[0].item() if len(unsafe_positions) > 0 else -1
	pred_label = 1 if pred_unsafe_start >= 0 else 0

	return {
		"pred_label": pred_label,
		"pred_unsafe_start": pred_unsafe_start,
		"response_token_count": int(preds.numel()),
	}


@torch.inference_mode()
def predict_response_tokens(
	tokenizer,
	backbone,
	head_r,
	user_query: str,
	assistant_response: str,
	max_length: int,
) -> Dict[str, Any]:
	prompt = f"User: {user_query}\nAssistant:"
	full_text = f"{prompt} {assistant_response}"

	enc = tokenizer(
		full_text,
		max_length=max_length,
		truncation=True,
		padding=False,
		return_tensors="pt",
	)
	input_ids = enc.input_ids.to(DEVICE)
	attention_mask = enc.attention_mask.to(DEVICE)

	prompt_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
	seq_len = int(attention_mask[0].sum().item())
	start_gen = min(prompt_len, seq_len)

	with torch.amp.autocast(DEVICE.type, dtype=DTYPE):
		outputs = backbone(
			input_ids=input_ids,
			attention_mask=attention_mask,
			output_hidden_states=True,
		)
		h = outputs.hidden_states[-1]
		logits = head_r(h)

	preds = torch.argmax(logits, dim=-1)[0].detach().cpu()
	resp_preds = preds[start_gen:seq_len]

	unsafe_positions = (resp_preds == 1).nonzero(as_tuple=True)[0]
	pred_unsafe_start = unsafe_positions[0].item() if len(unsafe_positions) > 0 else -1
	pred_label = 1 if pred_unsafe_start >= 0 else 0

	return {
		"pred_label": pred_label,
		"pred_unsafe_start": pred_unsafe_start,
		"response_token_count": int(resp_preds.numel()),
	}


def run_eval(args):
	ds = load_guardtest_split(args.dataset, args.split)
	if args.limit is not None:
		ds = ds.select(range(min(args.limit, len(ds))))

	print(f"Samples a evaluer: {len(ds)}")
	tokenizer, backbone, head_r = load_model(args.head, args.backbone_ft)

	y_true: List[int] = []
	y_pred: List[int] = []

	boundary_true: List[int] = []
	boundary_pred: List[int] = []
	boundary_missed = 0
	boundary_truncated = 0
	used_dataset_input_ids = 0
	fallback_text_retokenized = 0
	skipped_missing_input_ids = 0

	examples: List[str] = []

	for idx, item in enumerate(ds):
		label = normalize_label(item.get("label", "Safe"))
		gt_start = parse_unsafe_start_index(item.get("unsafe_start_index", -1))

		input_ids_list = item.get("input_ids")
		if isinstance(input_ids_list, list) and len(input_ids_list) > 0:
			pred = predict_from_dataset_input_ids(backbone, head_r, input_ids_list)
			used_dataset_input_ids += 1
		elif args.require_input_ids:
			skipped_missing_input_ids += 1
			continue
		else:
			messages_value = item.get("messages")
			if not isinstance(messages_value, list) or len(messages_value) == 0:
				messages_value = item.get("message", [])

			user_query, assistant_response = extract_user_assistant(messages_value)
			pred = predict_response_tokens(
				tokenizer,
				backbone,
				head_r,
				user_query,
				assistant_response,
				args.max_length,
			)
			fallback_text_retokenized += 1

		y_true.append(label)
		y_pred.append(pred["pred_label"])

		if label == 1 and gt_start >= 0:
			if gt_start >= pred["response_token_count"]:
				boundary_truncated += 1
			else:
				boundary_true.append(gt_start)
				boundary_pred.append(pred["pred_unsafe_start"])
				if pred["pred_unsafe_start"] < 0:
					boundary_missed += 1

		if args.show_examples > 0 and len(examples) < args.show_examples:
			mismatch = (label != pred["pred_label"])
			boundary_bad = label == 1 and gt_start >= 0 and pred["pred_unsafe_start"] != gt_start
			if mismatch or boundary_bad:
				examples.append(
					f"idx={idx} | gt_label={ID2LABEL[label]} pred_label={ID2LABEL[pred['pred_label']]} | "
					f"gt_start={gt_start} pred_start={pred['pred_unsafe_start']} | "
					f"resp_tokens={pred['response_token_count']}"
				)

		effective_done = len(y_true) + skipped_missing_input_ids
		if (idx + 1) % 50 == 0 or idx + 1 == len(ds):
			print(f"  {effective_done}/{len(ds)}", end="\r")
	print()

	if not y_true:
		print("Aucun sample evalue apres filtrage. Essaie --require-input-ids false ou un autre split.")
		return

	print("\n" + "=" * 60)
	print("Sample-level metrics (Safe vs Unsafe)")
	print("=" * 60)
	print(f"Mode alignement: dataset input_ids={used_dataset_input_ids}, fallback re-tokenization={fallback_text_retokenized}")
	if skipped_missing_input_ids > 0:
		print(f"Samples ignores (input_ids manquants): {skipped_missing_input_ids}")
	print(
		classification_report(
			y_true,
			y_pred,
			labels=[0, 1],
			target_names=["Safe", "Unsafe"],
			zero_division=0,
		)
	)
	cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
	print("Confusion matrix (rows=true, cols=pred)")
	print("            Safe    Unsafe")
	print(f"Safe    {cm[0,0]:>8} {cm[0,1]:>9}")
	print(f"Unsafe  {cm[1,0]:>8} {cm[1,1]:>9}")

	print("\n" + "=" * 60)
	print("Unsafe boundary metrics (unsafe_start_index)")
	print("=" * 60)
	if boundary_true:
		bt = torch.tensor(boundary_true, dtype=torch.float32)
		bp = torch.tensor(boundary_pred, dtype=torch.float32)

		detected = bp >= 0
		n_total = len(boundary_true)
		n_detected = int(detected.sum().item())
		mae_all = torch.abs(bt - bp).mean().item()

		if n_detected > 0:
			mae_detected = torch.abs(bt[detected] - bp[detected]).mean().item()
			within5 = (torch.abs(bt[detected] - bp[detected]) <= 5).float().mean().item()
		else:
			mae_detected = float("nan")
			within5 = 0.0

		print(f"Unsafe avec gt index valide       : {n_total}")
		print(f"Detectes (pred >= 0)             : {n_detected} ({n_detected/n_total:.1%})")
		print(f"Manques (pred = -1)              : {boundary_missed} ({boundary_missed/n_total:.1%})")
		print(f"MAE (all)                        : {mae_all:.2f} tokens")
		print(f"MAE (detected only)              : {mae_detected:.2f} tokens")
		print(f"Within +/-5 tokens (detected)    : {within5:.1%}")
		print(f"Skips pour troncature max_length : {boundary_truncated}")
	else:
		print("Aucun exemple Unsafe avec unsafe_start_index exploitable.")

	if examples:
		print("\nExemples de mismatches:")
		for line in examples:
			print(f"  - {line}")


def main():
	parser = argparse.ArgumentParser(description="Evaluate head_r on Qwen/Qwen3GuardTest")
	parser.add_argument("--dataset", default="Qwen/Qwen3GuardTest")
	parser.add_argument("--split", default="thinking_loc")
	parser.add_argument("--head", default=HEAD_PATH)
	parser.add_argument("--backbone-ft", default=BACKBONE_FT_PATH)
	parser.add_argument("--max-length", type=int, default=1024)
	parser.add_argument("--limit", type=int, default=None)
	parser.add_argument("--show-examples", type=int, default=5)
	parser.add_argument(
		"--require-input-ids",
		action="store_true",
		help="Ignore les samples sans input_ids du dataset (evaluation strictement alignee).",
	)
	args = parser.parse_args()

	run_eval(args)


if __name__ == "__main__":
	main()