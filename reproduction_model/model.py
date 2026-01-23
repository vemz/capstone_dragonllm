from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn

model_name = "Qwen/Qwen3-0.6B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)

# streaming model

class SafetyHead(nn.Module):
    def __init__(self, hidden_size, num_classes=2):
        super().__init__()
        self.W_pre = nn.Linear(hidden_size, hidden_size) 
        self.layer_norm = nn.LayerNorm(hidden_size) 
        self.W_risk = nn.Linear(hidden_size, num_classes) 
        nn.init.normal_(self.W_pre.weight, std=0.02)
        nn.init.zeros_(self.W_pre.bias)
        nn.init.normal_(self.W_risk.weight, std=0.02)
        nn.init.zeros_(self.W_risk.bias)

    def forward(self, h):
        pre_output = self.W_pre(h) 
        x = self.layer_norm(pre_output) 
        y_risk = self.W_risk(x)
        
        return y_risk
    
class StreamGuardModel(nn.Module):
    def __init__(self, model_name, num_classes=2):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(model_name)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        hidden_size = self.backbone.config.hidden_size
        
        self.head_q = SafetyHead(hidden_size, num_classes) # query
        self.head_r = SafetyHead(hidden_size, num_classes) # response
        
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, labels_q=None, labels_r=None, sep_indices=None):
        
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                output_hidden_states=True
            )
        h = outputs.hidden_states[-1] 
        
        logits_q = self.head_q(h) 
        logits_r = self.head_r(h)

        loss = None
        
        if labels_q is not None and labels_r is not None and sep_indices is not None:
            
            batch_size = input_ids.size(0)
            batch_indices = torch.arange(batch_size, device=input_ids.device)
            selected_q_logits = logits_q[batch_indices, sep_indices] 
            
            loss_q = self.loss(selected_q_logits, labels_q)
            
            flat_logits_r = logits_r.view(-1, logits_r.size(-1))
            flat_labels_r = labels_r.view(-1)
            
            loss_r = self.loss(flat_logits_r, flat_labels_r)
            
            loss = loss_q + loss_r

        return logits_q, logits_r, loss
    
# now we prepare the data

def adapter_qwen3guard(example):

    unsafe_type = example.get("unsafe_type", "")
    is_unsafe = 1 if (unsafe_type and unsafe_type != "Safe") else 0

    return {
        "messages": example["message"], 
        "label_q": is_unsafe, 
        "label_r": is_unsafe
    }

def adapter_rtp(example):
    THRESHOLD = 0.5
    
    p_score = example["prompt"].get("toxicity") or 0.0
    r_score = example["continuation"].get("toxicity") or 0.0
    
    msgs = [
        {"role": "user", "content": example["prompt"]["text"]},
        {"role": "assistant", "content": example["continuation"]["text"]}
    ]
    
    return {
        "messages": msgs,
        "label_q": 1 if p_score > THRESHOLD else 0,
        "label_r": 1 if r_score > THRESHOLD else 0
    }

class Collator:
    def __init__(self, tokenizer, max_length=1024): 
        self.tokenizer = tokenizer
        self.max_length = max_length

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch):
        input_ids_list = []
        attention_mask_list = []
        sep_indices_list = []
        labels_q_list = []
        labels_r_list = []

        for item in batch:
            messages = item['messages']
            lbl_q = item['label_q']
            lbl_r = item['label_r']

            full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            
            user_msg = [m for m in messages if m['role'] == 'user']
            prompt_text = self.tokenizer.apply_chat_template(user_msg, tokenize=False, add_generation_prompt=True)
            
            encoding = self.tokenizer(
                full_text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            prompt_tokens = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
            sep_index = len(prompt_tokens) - 1

            if sep_index >= self.max_length: 
                sep_index = self.max_length - 2

            input_ids = encoding.input_ids[0]
            attention_mask = encoding.attention_mask[0]

            labels_r_tensor = torch.full_like(input_ids, -100)
            
            seq_len = int(attention_mask.sum())
            
            if sep_index + 1 < seq_len:
                labels_r_tensor[sep_index+1 : seq_len] = lbl_r

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            sep_indices_list.append(sep_index)
            
            labels_q_list.append(lbl_q)
            labels_r_list.append(labels_r_tensor)

        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "sep_indices": torch.tensor(sep_indices_list, dtype=torch.long),
            "labels_q": torch.tensor(labels_q_list, dtype=torch.long),
            "labels_r": torch.stack(labels_r_list)
        }