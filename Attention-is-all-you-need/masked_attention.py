# Implement masked self-attention, a variation of the attention mechanism used in sequence modeling tasks such as text generation. Your task is to compute masked self-attention using query (Q), key (K), value (V) matrices and an attention mask.

import torch
import torch.nn.functional as F

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return torch.matmul(X, W_q), torch.matmul(X, W_k), torch.matmul(X, W_v)

def masked_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute masked self-attention.
    """
    # Your code here
    d_k = K.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)
    scores = scores + mask
    att = F.softmax(scores, dim=-1)
    return att @ V
