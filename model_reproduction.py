from transformers import AutoModelForCausalLM, AutoTokenizer

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

# dataset import

from datasets import load_dataset
ds = load_dataset("Qwen/Qwen3GuardTest")

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
        print(logits_r)
        
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