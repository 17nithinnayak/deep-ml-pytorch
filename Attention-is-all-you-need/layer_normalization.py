# Implement a function to perform Layer Normalization on an input tensor. Given a 3D array representing batch_size, sequence length, and feature dimensions, normalize the data across the feature dimension for each sequence, then apply scaling and shifting parameters.


import torch

def layer_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Perform Layer Normalization across the feature dimension.
    """
    # 1. Calculate the mean along the feature dimension (d_model)
    # keepdim=True ensures shape stays (batch, seq_len, 1) instead of (batch, seq_len)
    mean = X.mean(dim=-1, keepdim=True)
    
    # 2. Calculate the variance (MUST use unbiased=False for neural network math)
    var = X.var(dim=-1, keepdim=True, unbiased=False)
    
    # 3. Normalize the tensor to mean=0 and variance=1
    X_normalized = (X - mean) / torch.sqrt(var + epsilon)
    
    # 4. Apply the learned scale (gamma) and shift (beta)
    # Broadcasting automatically applies these (1, 1, d_model) tensors across the batch/seq
    output = gamma * X_normalized + beta
    
    return output
    
