# capstone_dragonllm

Toxic content classifier inspired by Stream Qwen3Guard, in collaboration with DragonLLM.
Reproduction of [arXiv:2510.14276](https://arxiv.org/abs/2510.14276).

---

## Project structure

```
reproduction_model/
├── model.py          # SafetyHead, StreamGuardModel, Collator
├── train.py          # Training pipeline
├── synthesis.py      # Data synthesis from Real Toxicity Prompts
├── labelling.py      # Async safety annotation with binary search
├── labelling_raheel.py
├── reformat_labels.py
├── aegis_fewshot.py
└── results/
    └── synthesis/
        ├── rtp_synthesized_mixed_30B.jsonl
        └── rtp_synthesized_mixed_4B.jsonl
```

---

## Dataset format

The labeling pipeline (`labelling.py`) outputs `.jsonl` files where each line has the following fields:

| Field | Type | Description |
|---|---|---|
| `original_id` | str | ID from the source RTP dataset |
| `user_query` | str | The user prompt |
| `assistant_response` | str | The LLM response |
| `intended_mode` | str | `safe_attempt` or `toxic_attempt` |
| `safety_label` | str | `Safe`, `Controversial`, `Unsafe`, `Timeout`, `Error` |
| `unsafe_token_index` | int | First token index (in the response) where unsafe content begins; `-1` if safe |
| `guard_raw_output` | str | Raw output from the guard model |
| `guard_parsed_output` | str | Parsed safety line from guard output |

---

## Changelog

### `model.py` — Collator fix (2026-02-25)

The `Collator` class had several mismatches with the actual dataset format produced by `labelling.py`. All four issues have been fixed:

#### 1. Field name: `prompt` → `user_query`
```python
# Before
prompt = item['prompt']

# After
prompt = item['user_query']
```

#### 2. Field name: `response` → `assistant_response`
```python
# Before
response = item['response']

# After
response = item['assistant_response']
```

#### 3. Field name and type: `label` (int) → `safety_label` (str mapped to int)

The labeling pipeline stores a string (`"Safe"`, `"Controversial"`, `"Unsafe"`), not an integer. A `LABEL_MAP` dict now handles the conversion:

```python
# Before
label_global = item['label']   # expected int, field didn't exist

# After
LABEL_MAP = {"Safe": 0, "Controversial": 1, "Unsafe": 2}
label_global = LABEL_MAP.get(item.get('safety_label', 'Safe'), 0)
```

#### 4. Field name and semantics: `unsafe_char_index` → `unsafe_token_index`

The old code read `unsafe_char_index` (character-level offset, field didn't exist) and re-tokenized a sliced string to find the boundary. The labeling pipeline stores `unsafe_token_index`, a **token-level index** directly usable as an offset from the start of the response. The boundary computation is now direct:

```python
# Before — re-tokenized a char-sliced string (wrong field + wrong semantics)
safe_part_text = response[:unsafe_char_idx]
len_safe_tokens = len(self.tokenizer(safe_part_text, ...).input_ids)
boundary_token_idx = start_gen + len_safe_tokens

# After — direct token offset
boundary_token_idx = start_gen + unsafe_token_idx
```

#### 5. Bug fix: `labels_q` was always 0

```python
# Before — hardcoded, ignored the actual label
labels_q_list.append(0)

# After — uses the actual sample-level label
labels_q_list.append(label_global)
```
