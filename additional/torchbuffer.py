import torch
import torch.nn as nn

# Causal Attention Without Buffers
class CausalAttentionWithoutBuffers(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec


# Example initialization and run of the module without buffers
torch.manual_seed(123)

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your     (x^1)
     [0.55, 0.87, 0.66],  # journey  (x^2)
     [0.57, 0.85, 0.64],  # starts   (x^3)
     [0.22, 0.58, 0.33],  # with     (x^4)
     [0.77, 0.25, 0.10],  # one      (x^5)
     [0.05, 0.80, 0.55]]  # step     (x^6)
)

batch = torch.stack((inputs, inputs), dim=0)
context_length = batch.shape[1]
d_in = inputs.shape[1]
d_out = 2

ca_without_buffer = CausalAttentionWithoutBuffers(d_in, d_out, context_length, 0.0)

# Run on CPU
with torch.no_grad():
    context_vecs = ca_without_buffer(batch)

print("Context Vectors Without Buffers (CPU):")
print(context_vecs)

# Transfer to GPU and run again
print("Machine has GPU:", torch.cuda.is_available())

batch = batch.to("cuda")
ca_without_buffer.to("cuda")

# Now let's check the device locations of some tensors
print("W_query.device:", ca_without_buffer.W_query.weight.device)
print("mask.device:", ca_without_buffer.mask.device)

# Move the mask to GPU
ca_without_buffer.mask = ca_without_buffer.mask.to("cuda")

# Run on GPU
with torch.no_grad():
    context_vecs_cuda = ca_without_buffer(batch)

print("\nContext Vectors Without Buffers (GPU):")
print(context_vecs_cuda)

# Causal Attention With Buffers
class CausalAttentionWithBuffer(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec


# Initialize and run the module with buffers
ca_with_buffer = CausalAttentionWithBuffer(d_in, d_out, context_length, 0.0)
ca_with_buffer.to("cuda")

print("\nW_query.device:", ca_with_buffer.W_query.weight.device)
print("mask.device:", ca_with_buffer.mask.device)

# Run on GPU with buffers
with torch.no_grad():
    context_vecs_buffer = ca_with_buffer(batch)

print("\nContext Vectors With Buffers (GPU):")
print(context_vecs_buffer)

# Comparison of outputs
print("\nComparison of Outputs:")
print("Without Buffers (GPU):")
print(context_vecs_cuda)
print("\nWith Buffers (GPU):")
print(context_vecs_buffer)

# Check state_dicts
print("\nState Dict Without Buffers:")
print(ca_without_buffer.state_dict())
print("\nState Dict With Buffers:")
print(ca_with_buffer.state_dict())
