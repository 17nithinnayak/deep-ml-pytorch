# Write a Python function that computes the dot product of a matrix and a vector. The function should return a list representing the resulting vector if the operation is valid, or -1 if the matrix and vector dimensions are incompatible. A matrix (a list of lists) can be dotted with a vector (a list) only if the number of columns in the matrix equals the length of the vector. For example, an n x m matrix requires a vector of length m.

# Example:
# Input:
# a = [[1, 2], [2, 4]], b = [1, 2]
# Output:
# [5, 10]


import torch

def matrix_dot_vector(a, b) -> torch.Tensor:
    """
    Compute the product of matrix `a` and vector `b` using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 1-D tensor of length m, or tensor(-1) if dimensions mismatch.
    """
    a_t = torch.as_tensor(a, dtype=torch.float)
    b_t = torch.as_tensor(b, dtype=torch.float)
    # Dimension mismatch check
    if a_t.size(1) != b_t.size(0):
        return torch.tensor(-1)
    # Your implementation here
    result = torch.matmul(a_t, b_t)
    return result
    

# Explaination: The function `matrix_dot_vector` takes two inputs, `a` and `b`, which can be Python lists, NumPy arrays, or PyTorch tensors. It first converts these inputs into PyTorch tensors of type float. Then, it checks if the number of columns in the matrix `a` matches the length of the vector `b`. If they do not match, it returns a tensor with the value -1 to indicate an error. If they are compatible, it uses the `torch.matmul` function to compute the dot product of the matrix and vector, and returns the resulting tensor.