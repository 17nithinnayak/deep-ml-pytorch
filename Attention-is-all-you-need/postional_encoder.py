#Write a Python function to implement the Positional Encoding layer for Transformers. The function should calculate positional encodings for a sequence length (position) and model dimensionality (d_model) using sine and cosine functions as specified in the Transformer architecture. The function should return -1 if position is 0, or if d_model is less than or equal to 0. The output should be a numpy array of type float16.

import numpy as np

def pos_encoding(position: int, d_model: int):
    """
    Generate Interleaved Positional Encoding matrix.
    """
    if position == 0 or d_model <= 0:
        return -1
        
    # 1. Initialize an empty 2D matrix (seq_len, d_model)
    pe = np.zeros((position, d_model))
    
    # 2. Calculate the frequencies (the 'Gears')
    half_d = d_model // 2
    exponents = (2 * np.arange(half_d)) / d_model
    denominator = np.power(10000, exponents)
    
    # 3. Create the position column (the 'Time')
    pos_matrix = np.arange(position)[:, np.newaxis]
    
    # 4. Calculate the angles
    args = pos_matrix / denominator
    
    # 5. The Interleaving Magic
    # 0::2 means start at index 0, and jump by 2 (columns 0, 2, 4...)
    pe[:, 0::2] = np.sin(args)
    
    # 1::2 means start at index 1, and jump by 2 (columns 1, 3, 5...)
    pe[:, 1::2] = np.cos(args)
    
    # 6. Cast to float16 as strictly requested
    pe = np.float16(pe)
    
    return pe
