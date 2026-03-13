# Write a Python function to calculate the dot product of two vectors. The function should take two 1D NumPy arrays as input and return the dot product as a single number.
import numpy as np

def calculate_dot_product(vec1, vec2):
	"""
	Calculate the dot product of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	Returns:
		The dot product of the two vectors.
	"""
	# Your code here
	return vec1 @ vec2

# Explanation: The function `calculate_dot_product` takes two 1D NumPy arrays as input, representing the vectors. It uses the `@` operator, which is a shorthand for the dot product in NumPy, to compute and return the dot product of the two vectors.