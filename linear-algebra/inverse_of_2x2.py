# Write a Python function that calculates the inverse of a 2x2 matrix. The inverse of a matrix A is another matrix A_inv such that A * A_inv = I (the identity matrix).\n\nFor a 2x2 matrix [[a, b], [c, d]], the inverse exists only if the determinant (ad - bc) is non-zero.\n\nReturn None if the matrix is not invertible (i.e., when the determinant equals zero).

import torch

def inverse_2x2(matrix) -> torch.Tensor | None:
    """
    Compute the inverse of a 2x2 matrix using PyTorch.
    
    Args:
        matrix: A 2x2 matrix (can be list, numpy array, or torch.Tensor)
    
    Returns:
        A 2x2 tensor containing the inverse, or None if the matrix is singular
    """
    m = torch.as_tensor(matrix, dtype=torch.float)
    # Your code here
    a, b = m[0, 0], m[0, 1]
    c, d = m[1, 0], m[1, 1]

    det = a*d - b*c

    if det == 0:
        return None

    adjugate = torch.tensor([[d, -b], [-c, a]], dtype = torch.float)

    inverse = (1/det) * adjugate

    return inverse

# Explanation: The function `inverse_2x2` takes a 2x2 matrix as input and first converts it into a PyTorch tensor of type float. It then extracts the elements of the matrix and calculates the determinant. If the determinant is zero, it returns None, indicating that the matrix is not invertible. If the determinant is non-zero, it computes the adjugate of the matrix and divides it by the determinant to obtain the inverse, which is returned as a 2x2 tensor.