# Write a Python function that implements a simple Recurrent Neural Network (RNN) cell. The function should process a sequence of input vectors and produce the final hidden state. Use the tanh activation function for the hidden state updates. The function should take as inputs the sequence of input vectors, the initial hidden state, the weight matrices for input-to-hidden and hidden-to-hidden connections, and the bias vector. The function should return the final hidden state after processing the entire sequence, rounded to four decimal places.
# ht = tanh(W 
x
​
 x 
t
​
 +W 
h
​
 h 
t−1
​
 +b)
import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
    """
    Perform a forward pass through a simple RNN cell.
    """
    # 1. Convert everything to NumPy arrays for easy matrix math
    h = np.array(initial_hidden_state)
    Wx = np.array(Wx)
    Wh = np.array(Wh)
    b = np.array(b)
    
    # 2. Loop through the sequence one time-step (word) at a time
    for x_val in input_sequence:
        x = np.array(x_val)
        
        # 3. The Core RNN Equation
        # Combine the input and the memory, add bias, and squash with tanh
        h = np.tanh(np.dot(x, Wx) + np.dot(h, Wh) + b)
        
    # 4. Round the final memory state to 4 decimal places and return as a list
    final_hidden_state = np.round(h, 4).tolist()
    
    return final_hidden_state
