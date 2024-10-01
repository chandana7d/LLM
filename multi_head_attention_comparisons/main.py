# main.py
import torch

# Import the MHA classes
from models.causal_attention import Ch03_MHA_Wrapper
from models.multi_head_attention import Ch03_MHA

# Set random seed for reproducibility
torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define parameters
batch_size = 8
context_len = 1024
embed_dim = 768
embeddings = torch.randn((batch_size, context_len, embed_dim), device=device)

# Initialize MHA wrapper from chapter 3
mha_ch03_wrapper = Ch03_MHA_Wrapper(
    d_in=embed_dim,
    d_out=embed_dim // 12,
    context_length=context_len,
    dropout=0.0,
    num_heads=12,
    qkv_bias=False
).to(device)

# Forward pass through the MHA wrapper
out_wrapper = mha_ch03_wrapper(embeddings)
print(f"MHA Wrapper Output Shape: {out_wrapper.shape}")  # Should be (batch_size, context_len, d_out * num_heads)

# Initialize the Ch03_MHA
mha_ch03 = Ch03_MHA(
    d_in=embed_dim,
    d_out=embed_dim,
    context_length=context_len,
    dropout=0.0,
    num_heads=12,
    qkv_bias=False
).to(device)

# Forward pass through the Ch03_MHA
out_mha = mha_ch03(embeddings)
print(f"Ch03_MHA Output Shape: {out_mha.shape}")  # Should be (batch_size, context_len, d_out)
