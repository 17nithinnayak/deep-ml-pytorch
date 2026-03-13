import torch

def sigmoid(z: float) -> float:
	#Your code here
	Z = torch.tensor(z, dtype=torch.float32)
	result = 1 / (1 + torch.exp(-Z))
	return round(result.item(), 4)

# Explanation:
# 1. We convert the input z to a PyTorch tensor for efficient computation.
# 2. We compute the sigmoid function using the formula: sigmoid(z) = 1 / (1 + exp(-z)).
# 3. Finally, we round the result to 4 decimal places and return it as a float.