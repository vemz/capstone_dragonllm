from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn

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
    def __init__(self, model_name, num_classes=2, backbone_kwargs=None):
        super().__init__()
        if backbone_kwargs is None:
            backbone_kwargs = {}
        self.backbone = AutoModelForCausalLM.from_pretrained(model_name, **backbone_kwargs)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        hidden_size = self.backbone.config.hidden_size
        
        self.head_r = SafetyHead(hidden_size, num_classes)
        
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input_ids, attention_mask, labels_r=None):
        
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                output_hidden_states=True
            )

        h = outputs.hidden_states[-1]
        # try with the sum of the average of all the layers / the last 4 layers
        # h = sum(outputs.hidden_states[-4:]) / 4
        
        logits_r = self.head_r(h)

        loss = None
        
        if labels_r is not None:
            flat_logits_r = logits_r.view(-1, logits_r.size(-1))
            flat_labels_r = labels_r.view(-1)
            flat_attention_mask = attention_mask.view(-1)
            # Compute loss for all tokens, then mask
            loss_per_token = self.loss(flat_logits_r, flat_labels_r)
            # Only apply loss on non-padding tokens (where attention_mask == 1)
            loss = (loss_per_token * flat_attention_mask).sum() / flat_attention_mask.sum()

        return logits_r, loss

LABEL_MAP = {"Safe": 0, "Unsafe": 1}

class Collator:
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch):
        input_ids_list = []
        attention_mask_list = []
        labels_r_list = []

        for item in batch:
            prompt = item['user_query']
            response = item['assistant_response']
            label_global = LABEL_MAP.get(item.get('safety_label', 'Safe'), 0)
            
            unsafe_token_idx = item.get('unsafe_token_index', -1)

            user_part = f"User: {prompt}\nAssistant:"
            full_text = f"{user_part} {response}"

            encoding = self.tokenizer(
                full_text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            input_ids = encoding.input_ids[0]
            mask = encoding.attention_mask[0]

            len_prompt_tokens = len(self.tokenizer(user_part, add_special_tokens=False).input_ids)
            sep_index = len_prompt_tokens - 1

            labels_r_tensor = torch.full_like(input_ids, 0)  # Default to Safe (0)
            seq_len = int(mask.sum())
            start_gen = sep_index + 1

            # Labelize all response tokens
            if label_global == 0:
                # All response tokens are Safe
                labels_r_tensor[start_gen : seq_len] = 0
            else:
                # Some tokens are Unsafe
                if unsafe_token_idx <= 0:
                    # boundary not found or first token is unsafe: full response is unsafe
                    labels_r_tensor[start_gen : seq_len] = 1
                else:
                    # Tokens before boundary are safe, after boundary are unsafe
                    boundary_token_idx = start_gen + unsafe_token_idx
                    if boundary_token_idx < seq_len:
                        labels_r_tensor[start_gen : boundary_token_idx] = 0
                        labels_r_tensor[boundary_token_idx : seq_len] = 1
                    else:
                        # Boundary is beyond sequence: all response tokens are safe
                        labels_r_tensor[start_gen : seq_len] = 0
            
            # Ignore prompt tokens by setting them to padding label (0 is fine since attention_mask will ignore)
            # Actually keep all tokens, attention_mask will handle the masking

            input_ids_list.append(input_ids)
            attention_mask_list.append(mask)
            labels_r_list.append(labels_r_tensor)

        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels_r": torch.stack(labels_r_list)
        }