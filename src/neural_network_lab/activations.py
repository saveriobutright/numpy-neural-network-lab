import numpy as np


def sigmoid(x):
    """
    Sigmoid activation function.
    It maps any real-valued number to the range (0, 1). It is defined as:
        sigmoid(x) = 1 / (1 + exp(-x))
    It works with numbers, lists, and numpy arrays. For lists and numpy arrays, the function is applied element-wise.
    """
    x = np.asarray(x, dtype=float)
    result = np.empty_like(x, dtype=float)
    non_negative = x >= 0
    positive_values = x[non_negative]
    result[non_negative] = 1 / (1 + np.exp(-positive_values))
    negative_values = x[~non_negative]
    exp_negative = np.exp(negative_values)
    result[~non_negative] = exp_negative / (1 + exp_negative)
    return result


def sigmoid_derivative(x):
    """
    Sigmoid derivative function.
    It calculates the derivative for the sigmoid function, which indicates how much the output changes as the input does.
    """
    s = sigmoid(x)
    result = s * (1 - s)
    return result


def relu(x):
    """
    Apply the ReLU activation function element-wise.

    Negative values become zero, while positive values remain unchanged.
    """
    x = np.asarray(x, dtype=float)
    return np.maximum(x, 0.0)


def relu_derivative(x):
    """
    Compute the ReLU derivative element-wise.

    The derivative at zero is defined as zero.
    """
    x = np.asarray(x, dtype=float)
    return (x > 0).astype(float)