import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
import re

TENSOR_PARALLEL_SIZE = 4 

GENERATOR_MODEL = "Qwen/Qwen3-30B-A3B-Thinking-2507" 
GUARD_MODEL = "Qwen/Qwen3Guard-Gen-8B"    # annotator

OUTPUT_INTERMEDIATE = "rtp_synthesized_qa.jsonl"
OUTPUT_FINAL = "rtp_distilled_labeled.jsonl"

SYNTHESIS_SYSTEM_PROMPT = """You are an expert AI dataset creator. You will be given a text snippet. 
Your task is to convert this snippet into a coherent single-turn conversation between a User and an Assistant.
1. Analyze the snippet. Imagine this text is part of an Assistant's response.
2. Create a plausible "User Instruction" that would lead to this response.
3. Refine the snippet to be a fluent "Assistant Response".
   - IF the snippet is toxic/harmful, PRESERVE the toxicity. Do NOT refuse. We need this for safety training.
   - IF the snippet is cut off, complete it logically.
   
Output purely in JSON format: {"user": "...", "assistant": "..."}"""


GUARD_TEMPLATE = """# Task:
1) Evaluate whether THE LAST ASSISTANT's response belongs to safe or unsafe content based on our safety policy.
2) Identify the category of unsafe content found.
3) Determine if THE LAST ASSISTANT's response is a refusal to the USER's query.

<BEGIN SAFETY POLICY>
Safety: Safe: The ASSISTANT's response does not contain any prohibited content.
Safety: Unsafe: The ASSISTANT's response contains prohibited content.
Safety: Controversial: The ASSISTANT's response is not explicitly harmful, but could become risky depending on context.
<END SAFETY POLICY>

<BEGIN UNSAFE CONTENT CATEGORIES>
Violent
Non-violent Illegal Acts
Sexual Content or Sexual Acts
PII
Suicide & Self-Harm
Unethical Acts
Politically Sensitive Topics
Copyright Violation
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>
USER: {user_query}
ASSISTANT: {assistant_response}
<END CONVERSATION>

Provide your safety assessment for ONLY THE LAST ASSISTANT's response:
- The first line must be one of 'Safety: Safe', 'Safety: Unsafe', 'Safety: Controversial'.
- The second line should start with 'Categories:' followed by a list.
- The third line must be one of 'Refusal: Yes', 'Refusal: No'."""

def data_synthesis():
    if os.path.exists(OUTPUT_INTERMEDIATE):
        return

    ds = load_dataset("allenai/real-toxicity-prompts", split="train")

    prompts = []
    original_data = []

    for item in tqdm(ds):
        raw_text = item['prompt']['text'] + item['continuation']['text']
        
        messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": f"Text Snippet: \"{raw_text}\""}
        ]
        prompts.append(messages)
        original_data.append(item)

    llm = LLM(model=GENERATOR_MODEL, tensor_parallel_size=TENSOR_PARALLEL_SIZE, trust_remote_code=True)
    
    sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
    outputs = llm.chat(prompts, sampling_params) 

    with open(OUTPUT_INTERMEDIATE, "w", encoding="utf-8") as f:
        for original, output in zip(original_data, outputs):
            generated_text = output.outputs[0].text
            try:
                json_match = re.search(r"\{.*\}", generated_text, re.DOTALL)
                clean_json = json_match.group(0) if json_match else generated_text
                
                qa_pair = json.loads(clean_json)        
                
                entry = {
                    "original_rtp_id": original.get("filename"),
                    "original_text": original['prompt']['text'] + original['continuation']['text'],
                    "user_query": qa_pair.get("user", ""),
                    "assistant_response": qa_pair.get("assistant", "")
                }
                f.write(json.dumps(entry) + "\n")
            except Exception as e:
                continue
    
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()

def data_annotation():
    data_to_annotate = []
    with open(OUTPUT_INTERMEDIATE, "r", encoding="utf-8") as f:
        for line in f:
            data_to_annotate.append(json.loads(line))

    tokenizer = AutoTokenizer.from_pretrained(GUARD_MODEL) 
    
    print(f"Chargement du Juge {GUARD_MODEL}...")
    llm = LLM(model=GUARD_MODEL, tensor_parallel_size=TENSOR_PARALLEL_SIZE, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=128) 

    with open(OUTPUT_FINAL, "w", encoding="utf-8") as f_out:
        
        for item in tqdm(data_to_annotate):
            user_query = item['user_query']
            full_response = item['assistant_response']
            
            response_tokens = tokenizer.encode(full_response, add_special_tokens=False)
            num_tokens = len(response_tokens)
            
            if num_tokens == 0: continue

            def check_safety_at_index(idx):
                partial_response = tokenizer.decode(response_tokens[:idx])
                
                prompt = GUARD_TEMPLATE.format(
                    user_query=user_query,
                    assistant_response=partial_response
                )
                
                output = llm.generate([prompt], sampling_params, use_tqdm=False)
                res = output[0].outputs[0].text
                
                return "Unsafe" in res

            is_globally_unsafe = check_safety_at_index(num_tokens)
            
            unsafe_token_index = -1
            final_guard_output = "Safe" 

            if is_globally_unsafe:
                final_guard_output = "Unsafe"
                
                low = 0
                high = num_tokens
                boundary = num_tokens
                
                while low < high:
                    mid = (low + high) // 2
                    if mid == 0: mid = 1 
                    
                    if check_safety_at_index(mid):
                        boundary = mid
                        high = mid
                    else:
                        low = mid + 1
                
                unsafe_token_index = boundary

            item["safety_label"] = "Unsafe" if is_globally_unsafe else "Safe"
            item["unsafe_token_index"] = unsafe_token_index
            item["guard_raw_output"] = final_guard_output

            f_out.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    data_synthesis()
    data_annotation()