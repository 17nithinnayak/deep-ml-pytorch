# Implement the multi-head attention mechanism, a critical component of transformer models. You need to implement three functions:

# compute_qkv(X, W_q, W_k, W_v): Compute Query, Key, and Value matrices by multiplying input X with weight matrices. Returns a tuple (Q, K, V) where each has the same shape as X.

# self_attention(Q, K, V): Compute scaled dot-product attention for a single head. Returns the attention output with the same shape as V.

# multi_head_attention(Q, K, V, n_heads): Split Q, K, V into multiple heads along the feature dimension, compute self-attention for each head independently, and concatenate results. Returns output with the same shape as Q.

import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    return Q, K, V

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    d_k = K.shape[1]

    scores = (Q @ K.T) / np.sqrt(d_k)

    max_score = np.max(scores, axis=1, keepdims=True)
    exp_score = np.exp(scores - max_score)
    att = exp_score / np.sum(exp_score, axis=1, keepdims=True)

    output = att @ V

    return output


def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    Q_heads = np.split(Q, n_heads, axis=1)
    K_heads = np.split(K, n_heads, axis=1)
    V_heads = np.split(V, n_heads, axis=1)
    
    head_outputs = []
    
    # 2. Compute self-attention for each head independently
    for i in range(n_heads):
        # Pass the sliced Q, K, V chunks into our self_attention engine
        head_out = self_attention(Q_heads[i], K_heads[i], V_heads[i])
        head_outputs.append(head_out)
        
    # 3. Stitch the outputs back together horizontally
    final_output = np.concatenate(head_outputs, axis=1)
    
    return final_output
