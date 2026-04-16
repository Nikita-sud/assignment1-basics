import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        
        std = math.sqrt(2.0 / (in_features + out_features))
        
        nn.init.trunc_normal_(
            self.W, 
            mean=0.0, 
            std=std, 
            a=-3.0 * std, 
            b=3.0 * std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.W)
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.Embeddings = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        
        nn.init.trunc_normal_(
            self.Embeddings, 
            mean=0.0,
            std=1,
            a=-3.0,
            b=3.0
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.Embeddings[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x_fp32**2, dim=-1, keepdim=True) + self.eps)
        x_norm = x_fp32 / rms
        res = x_norm * self.g
        return res.to(in_dtype)
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = int(8/3 * d_model)
            d_ff = (d_ff // 64) * 64

        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.W1(x)
        x3 = self.W3(x)
        gated = F.silu(x1) * x3
        return self.W2(gated)
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_k: int, max_seq_len: int, theta: float = 10000.0, device=None):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))

        t = torch.arange(max_seq_len, device=device)
        freqs = torch.outer(t, inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:

        cos = self.cos[token_positions] 
        sin = self.sin[token_positions]

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        x_rotated = self._rotate_half(x)
        return (x * cos) + (x_rotated * sin)
    
def safe_softmax(x, dim):
    max_x = torch.max(x, dim=dim, keepdim=True).values
    x_safe = x - max_x

    e_x = torch.exp(x_safe)
    return e_x / torch.sum(e_x, dim=dim, keepdim=True)
    
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]

    scores = torch.einsum("...qd, ...kd -> ...qk", Q, K) / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))

    probs = safe_softmax(scores, dim=-1)

    return torch.einsum("...qk, ...kv -> ...qv", probs, V)


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RotaryPositionalEmbedding, device=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rope = rope

        self.W_q = Linear(d_model, d_model, device=device)
        self.W_k = Linear(d_model, d_model, device=device)
        self.W_v = Linear(d_model, d_model, device=device)
        self.W_o = Linear(d_model, d_model, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        b, s, _ = x.shape

        q = self.W_q(x).view(b, s, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(b, s, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(b, s, self.num_heads, self.d_k).transpose(1, 2)

        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = attn_out.transpose(1, 2).contiguous().view(b, s, -1)

        return self.W_o(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RotaryPositionalEmbedding, device=None):
        super().__init__()
        self.attention_rms_norm = RMSNorm(d_model=d_model, device=device)
        self.attention = CausalMultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, rope=rope, device=device)
        self.ffn_rms_norm = RMSNorm(d_model=d_model, device=device)
        self.ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):

        x_norm = self.attention_rms_norm(x)
        attn_out = self.attention(x_norm, token_positions)
        x = x + attn_out

        x_norm = self.ffn_rms_norm(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, theta: float = 10000.0, device=None):
        super().__init__()
        
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device)
        
        self.rope = RotaryPositionalEmbedding(d_k=d_model // num_heads, max_seq_len=context_length, theta=theta, device=device)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, 
                             rope=self.rope, device=device)
            for _ in range(num_layers)
        ])
        
        self.rms_norm = RMSNorm(d_model=d_model, device=device)

        self.output_linear = Linear(in_features=d_model, out_features=vocab_size, device=device)

    def forward(self, token_ids: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)
        
        for block in self.blocks:
            x = block(x, token_positions)
            
        x = self.rms_norm(x)
        logits = self.output_linear(x)
        
        return logits