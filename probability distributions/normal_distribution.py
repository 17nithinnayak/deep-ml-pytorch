import math

def normal_pdf(x, mean, std_dev):
    """
    Calculate the probability density function (PDF) of the normal distribution.
    """
    # All these lines must start at the same indentation level
    variance = std_dev ** 2 
    coefficient = 1.0 / math.sqrt(2 * math.pi * variance)
    exponent = math.exp(-((x - mean) ** 2) / (2 * variance))
    
    pdf = coefficient * exponent
    return round(pdf, 5)

# Explaination: The Normal Distribution, also known as the Gaussian Distribution, is a continuous probability distribution that is symmetrical and bell-shaped, representing the distribution of data around the mean.

# Key Characteristics
# Symmetry: The distribution is symmetric around the mean, which means the left and right halves of the graph are mirror images.
# Mean, Median, and Mode: In a perfectly normal distribution, the mean, median, and mode are all equal.

# formula at deep-ml.com
