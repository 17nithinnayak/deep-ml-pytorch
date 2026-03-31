import math

def poisson_probability(k, lam):
	"""
	Calculate the probability of observing exactly k events in a fixed interval,
	given the mean rate of events lam, using the Poisson distribution formula.
	:param k: Number of events (non-negative integer)
	:param lam: The average rate (mean) of occurrences in a fixed interval
	"""
	
	val = (math.pow(lam, k) * math.exp(-lam)) / (math.factorial(k))
	return round(val,5)

# Explainantion: The function calculates the probability for a given number of events occurring in a fixed interval, based on the mean rate of occurrences.
# Applications
# The Poisson distribution is widely used in:

# Modeling the number of arrivals at a queue (e.g., calls at a call center)
# Counting occurrences over time (e.g., number of emails received per hour)
# Biology (e.g., distribution of mutations in a DNA strand)
# Traffic flow analysis (e.g., number of cars passing through an intersection)
# This distribution is essential for understanding and predicting rare events in real-world scenarios.
