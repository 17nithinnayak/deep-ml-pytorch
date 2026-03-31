import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    # Convert the list of lists to a NumPy array
    matrix = np.array(vectors)

    # Calculate the covariance matrix using NumPy.
    # By default, np.cov assumes rows are variables and columns are observations.
    # If your input 'vectors' is structured as (observations, variables),
    # you must set 'rowvar=False' to treat columns as variables instead.
    # Example: cov_matrix = np.cov(matrix, rowvar=False)
    cov_matrix = np.cov(matrix)

    # Round everything to 4 decimal places and convert back to a standard Python list
    return np.round(cov_matrix, 4).tolist()
# Explainantion: Covariance: Measures the directional relationship between two random variables. A positive covariance indicates that the variables increase together, while a negative covariance indicates that one variable increases as the other decreases.
