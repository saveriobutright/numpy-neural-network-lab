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


def binary_cross_entropy(y_true, y_pred):
    """
    Compute the Binary Cross-Entropy (BCE) loss.

    Parameters:
    y_true (numpy.ndarray): True binary labels (0 or 1).
    y_pred (numpy.ndarray): Predicted probabilities (between 0 and 1).

    Returns:
    float: The BCE loss value.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must be the same.")
    invalid_probabilities = (
        ~np.isfinite(y_pred)
        | (y_pred < 0)
        | (y_pred > 1)
    )
    if np.any(invalid_probabilities):
        raise ValueError(
            "Predicted probabilities must be finite and in the range [0, 1]."
        )
    if np.any((y_true != 0) & (y_true != 1)):
        raise ValueError("True labels must be binary (0 or 1).")
    epsilon = 1e-12
    clipped_y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    bce_values = -(
        y_true * np.log(clipped_y_pred)
        + (1 - y_true) * np.log(1 - clipped_y_pred)
    )
    return float(np.mean(bce_values))


def binary_cross_entropy_derivative(y_true, y_pred):
    """
    Compute the BCE derivative with respect to predictions.

    Parameters:
    y_true (numpy.ndarray): True binary labels (0 or 1).
    y_pred (numpy.ndarray): Predicted probabilities (between 0 and 1).

    Returns:
    numpy.ndarray: The derivative of the BCE loss.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must be the same.")
    invalid_probabilities = (
        ~np.isfinite(y_pred)
        | (y_pred < 0)
        | (y_pred > 1)
    )
    if np.any(invalid_probabilities):
        raise ValueError(
            "Predicted probabilities must be finite and in the range [0, 1]."
        )
    if np.any((y_true != 0) & (y_true != 1)):
        raise ValueError("True labels must be binary (0 or 1).")
    epsilon = 1e-12
    clipped_y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    bce_derivative = (
        -(y_true / clipped_y_pred)
        + (1 - y_true) / (1 - clipped_y_pred)
    )
    return bce_derivative / y_true.size
