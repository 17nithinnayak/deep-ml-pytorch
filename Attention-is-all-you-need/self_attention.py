/*
Implement the self-attention mechanism, a fundamental component of transformer models used in NLP and computer vision.

Your task is to implement the self_attention function that computes attention output given Query (Q), Key (K), and Value (V) matrices.

The self-attention formula is: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

where d_k is the dimensionality of the key vectors (number of columns in K).
*/
import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(Q, K, V):
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
    
    Returns:
        Attention output of shape (seq_len, d_v)
    """
    d_k = K.shape[1]

    scores = (Q @ K.T) / np.sqrt(d_k)
    max_score = np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores - max_score)

    attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return attention_weights @ V
    
