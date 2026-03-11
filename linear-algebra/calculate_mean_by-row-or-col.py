# Write a Python function that calculates the mean of a matrix either by row or by column, based on a given mode. The function should take a matrix (list of lists) and a mode ('row' or 'column') as input and return a list of means according to the specified mode.

import torch

def calculate_matrix_mean(matrix, mode: str) -> torch.Tensor:
    """
    Calculate mean of a 2D matrix per row or per column using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 1-D tensor of means or raises ValueError on invalid mode.
    """
    a_t = torch.as_tensor(matrix, dtype=torch.float)
    # Your implementation here
    if mode == 'row':
        return torch.mean(a_t, dim=1)
    elif mode == 'column':
        return torch.mean(a_t, dim=0)
    else:
        raise ValueError("No proper mode described")

# Explanation: The function `calculate_matrix_mean` takes a 2D matrix and a mode as input. It first converts the input matrix into a PyTorch tensor of type float. Then, it checks the mode: if the mode is 'row', it calculates the mean across rows (dim=1) and returns it; if the mode is 'column', it calculates the mean across columns (dim=0) and returns it. If an invalid mode is provided, it raises a ValueError with an appropriate message.