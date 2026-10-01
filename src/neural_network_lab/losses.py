import numpy as np


def mean_squared_error(y_true, y_pred):
    """
    Compute the Mean Squared Error (MSE) loss.

    Parameters:
    y_true (numpy.ndarray): True labels.
    y_pred (numpy.ndarray): Predicted labels.

    Returns:
    float: The MSE loss value.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must be the same.")
    mse = np.mean((y_pred - y_true) ** 2)
    return float(mse)


def mean_squared_error_derivative(y_true, y_pred):
    """
    Compute the derivative of the Mean Squared Error (MSE) loss with respect to predictions.

    Parameters:
    y_true (numpy.ndarray): True labels.
    y_pred (numpy.ndarray): Predicted labels.

    Returns:
    numpy.ndarray: The derivative of the MSE loss.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must be the same.")
    mse_derivative = 2 * (y_pred - y_true) / y_true.size
    return mse_derivative
