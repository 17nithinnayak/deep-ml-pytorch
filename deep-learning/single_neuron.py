import torch
import torch.nn.functional as F
from typing import List, Tuple

def single_neuron_model(
    features: List[List[float]],
    labels: List[float],
    weights: List[float],
    bias: float
) -> Tuple[List[float], float]:
    """
    Compute output probabilities and MSE for a single neuron.
    Uses built-in sigmoid and MSE loss.
    """
    
    X = torch.tensor(features, dtype=torch.float32)
    y_true = torch.tensor(labels, dtype=torch.float32)
    W = torch.tensor(weights, dtype=torch.float32)
    B = torch.tensor(bias, dtype=torch.float32)

    Z = X @ W + B

    probs = torch.sigmoid(Z)

    loss = F.mse_loss(probs, y_true)

    probs_round = [round(p.item(), 4) for p in probs]
    loss_round = round(loss.item(), 4)

    return probs_round, loss_round
    
# Explaination:
# 1. We convert the input lists to PyTorch tensors for efficient computation.
# 2. We compute the linear combination Z = X @ W + B.
# 3. We apply the sigmoid function to Z to get the output probabilities.
# 4. We compute the MSE loss between the predicted probabilities and the true labels.
# 5. Finally, we round the probabilities and loss to 4 decimal places