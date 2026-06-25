from transformers import AutoConfig
model_name="mistralai/Mixtral-8x22B-v0.1"
cfg = AutoConfig.from_pretrained(
    model_name,
    # "meta-llama/Meta-Llama-3.1-70B",
    trust_remote_code=True
)

print("hidden_size:", cfg.hidden_size)
print("num_attention_heads:", cfg.num_attention_heads)
print("num_key_value_heads:", getattr(cfg, "num_key_value_heads", None))

print("derived head_dim:", cfg.hidden_size // cfg.num_attention_heads)
print("torch_dtype in config:", cfg.torch_dtype)
hidden_size = cfg.hidden_size
q_heads = cfg.num_attention_heads
kv_heads = cfg.num_key_value_heads
layers = cfg.num_hidden_layers
head_dim = hidden_size // q_heads
tokens = 16
dtype_bytes = 2  # fp16 / bf16

values_per_token = 2 * kv_heads * head_dim * layers
bytes_total = values_per_token * tokens * dtype_bytes

print(f"{model_name} KV size for 16 tokens:", bytes_total / 1024, "KB")