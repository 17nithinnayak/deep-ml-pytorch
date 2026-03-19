/*
Implement the PReLU (Parametric ReLU) activation function, a variant of the ReLU activation function that introduces a learnable parameter for negative inputs. Your task is to compute the PReLU activation value for a given input.
*/
def prelu(x: float, alpha: float = 0.25) -> float:
	"""
	Implements the PReLU (Parametric ReLU) activation function.

	Args:
		x: Input value
		alpha: Slope parameter for negative values (default: 0.25)

	Returns:
		float: PReLU activation value
	"""
	# Your code here
	if(x>0):
		return x
	else:
		return alpha * x
