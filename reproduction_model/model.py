from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn

class SafetyHead(nn.Module):
    def __init__(self, hidden_size, num_classes=3): 
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
    def __init__(self, model_name, num_classes=3):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(model_name)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        hidden_size = self.backbone.config.hidden_size
        
        self.head_q = SafetyHead(hidden_size, num_classes) 
        self.head_r = SafetyHead(hidden_size, num_classes)
        
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

def adapter_rtp(example):
    
    risk_keys = [
        "toxicity", 
        "severe_toxicity", 
        "threat", 
        "identity_attack", 
        "insult", 
        "sexually_explicit"
    ]
    
    def get_max_score(item_dict):
        if not item_dict: return 0.0
        scores = []
        for key in risk_keys:
            val = item_dict.get(key)

            if val is not None:
                scores.append(val)
            else:
                scores.append(0.0)
        return max(scores) if scores else 0.0

    p_score = get_max_score(example.get("prompt"))
    r_score = get_max_score(example.get("continuation"))
    
    def get_label(score):
        if score < 0.3: 
            return 0 
        elif score < 0.7: 
            return 1 
        else: 
            return 2 
    
    msgs = [
        {"role": "user", "content": example["prompt"]["text"]},
        {"role": "assistant", "content": example["continuation"]["text"]}
    ]
    
    return {
        "messages": msgs,
        "label_q": get_label(p_score), 
        "label_r": get_label(r_score)   
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

            user_msg = [m for m in messages if m['role'] == 'user']
            
            user_text_only = self.tokenizer.apply_chat_template(user_msg, tokenize=False, add_generation_prompt=False)
            user_tokens = self.tokenizer(user_text_only, add_special_tokens=False).input_ids
            
            sep_index = len(user_tokens) - 1

            full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            encoding = self.tokenizer(
                full_text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            input_ids = encoding.input_ids[0]
            attention_mask = encoding.attention_mask[0]

            if sep_index >= self.max_length: 
                sep_index = self.max_length - 1

            labels_r_tensor = torch.full_like(input_ids, -100)
            seq_len = int(attention_mask.sum())
            
            start_gen_index = sep_index + 1 
            
            if start_gen_index < seq_len:
                if lbl_r == 0: 
                    labels_r_tensor[start_gen_index : seq_len] = 0
                else: 
                    labels_r_tensor[start_gen_index : seq_len] = lbl_r

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