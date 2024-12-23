"""
Activation functions for neural networks.
"""
import math

def relu(x):
    """ReLU activation function."""
    return max(0, x)

def softmax(x):
    """Softmax activation function."""
    exp_x = [math.exp(i) for i in x]
    sum_exp = sum(exp_x)
    return [i/sum_exp for i in exp_x]