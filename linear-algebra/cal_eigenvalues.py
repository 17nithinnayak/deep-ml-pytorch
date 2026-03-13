# Write a Python function that calculates the eigenvalues of a 2x2 matrix. The function should return a list containing the eigenvalues, sort values from highest to lowest.

import torch

def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvalues of a 2x2 matrix using PyTorch.
    Input: 2x2 tensor; Output: 1-D tensor with the two eigenvalues in descending order (highest to lowest).
    """
    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]

    trace = a + d
    det = a*d - b*c

    descr = trace**2 - 4 * det
    sqrt_dest = torch.sqrt(descr)

    lambda1 = (trace + sqrt_dest) / 2
    lambda2 = (trace - sqrt_dest) / 2

    eigenvalues = torch.stack([lambda1, lambda2])
    return eigenvalues[torch.argsort(eigenvalues.real, descending=True)]

# Explanation: The function `calculate_eigenvalues` takes a 2x2 matrix as input and computes its eigenvalues using the characteristic polynomial. It calculates the trace and determinant of the matrix, then uses these to find the discriminant. The eigenvalues are computed using the quadratic formula, and finally, they are sorted in descending order before being returned as a 1-D tensor.