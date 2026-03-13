# Write a Python function that computes the transpose of a given 2D matrix. The transpose of a matrix is formed by turning its rows into columns and columns into rows. For an mÃn matrix, the transpose will be an nÃm matrix.
import torch

def transpose_matrix(a) -> torch.Tensor:
    """
    Transpose a 2D matrix using PyTorch.
    
    Args:
        a: A 2D matrix (can be list, numpy array, or torch.Tensor)
    
    Returns:
        A transposed torch.Tensor
    """
    a_t = torch.as_tensor(a)
    # Your code here
    return a_t.t()

# Explaination: The function `transpose_matrix` takes a 2D matrix `a` as input, which can be a Python list, a NumPy array, or a PyTorch tensor. It first converts the input into a PyTorch tensor using `torch.as_tensor()`. Then, it uses the `.t()` method to compute the transpose of the matrix and returns the transposed tensor.