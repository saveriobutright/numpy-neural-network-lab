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


def _validate_categorical_inputs(y_true, y_pred):
    """
    Validate the inputs for categorical cross-entropy functions.

    Parameters:
    y_true (numpy.ndarray): True one-hot encoded labels.
    y_pred (numpy.ndarray): Predicted probabilities for each class.

    Raises:
    ValueError: If the inputs are invalid.

    Returns:
    tuple: The validated y_true and y_pred arrays.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must be the same.")

    if y_true.ndim not in (1, 2):
        raise ValueError("y_true and y_pred must be 1D or 2D arrays.")

    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    invalid_probabilities = (
        ~np.isfinite(y_pred)
        | (y_pred < 0)
        | (y_pred > 1)
    )

    if np.any(invalid_probabilities):
        raise ValueError(
            "Predicted probabilities must be finite and in the range [0, 1]."
        )

    prediction_sums = np.sum(y_pred, axis=-1)

    if not np.allclose(prediction_sums, 1.0):
        raise ValueError(
            "y_pred must contain probability distributions that sum to 1."
        )

    if np.any((y_true != 0) & (y_true != 1)):
        raise ValueError("y_true must contain only 0 and 1.")

    target_sums = np.sum(y_true, axis=-1)

    if not np.allclose(target_sums, 1.0):
        raise ValueError(
            "y_true must contain one-hot vectors that sum to 1."
        )

    return y_true, y_pred


def categorical_cross_entropy(y_true, y_pred):
    """
    Compute the Categorical Cross-Entropy (CCE) loss.

    Parameters:
    y_true (numpy.ndarray): True one-hot encoded labels.
    y_pred (numpy.ndarray): Predicted probabilities for each class.
    It accepts only 1D or 2D non-empty arrays.
    It requires one-hot targets with sum = 1 on the last axis.
    It requires finite predictions in the range [0, 1] with sum = 1 on the last axis.

    Returns:
    float: The CCE loss value.
    """
    y_true, y_pred = _validate_categorical_inputs(y_true, y_pred)

    epsilon = 1e-12
    clipped_y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    sample_losses = -np.sum(
        y_true * np.log(clipped_y_pred),
        axis=-1,
    )

    return float(np.mean(sample_losses))


def categorical_cross_entropy_derivative(y_true, y_pred):
    """
    Compute the CCE derivative with respect to predictions.

    Parameters:
    y_true (numpy.ndarray): True one-hot encoded labels.
    y_pred (numpy.ndarray): Predicted probabilities for each class.

    Returns:
    numpy.ndarray: The derivative of the CCE loss.
    """
    y_true, y_pred = _validate_categorical_inputs(y_true, y_pred)

    epsilon = 1e-12
    clipped_y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    cce_derivative = -y_true / clipped_y_pred
    sample_count = 1 if y_true.ndim == 1 else y_true.shape[0]

    return cce_derivative / sample_count


def softmax_categorical_cross_entropy_derivative(y_true, y_pred):
    """
    Compute the Softmax-CCE gradient with respect to logits.

    Parameters:
    y_true (numpy.ndarray): True one-hot encoded labels.
    y_pred (numpy.ndarray): Predicted probabilities for each class.

    Returns:
    numpy.ndarray: The derivative of the CCE loss with respect to logits.
    """
    y_true, y_pred = _validate_categorical_inputs(y_true, y_pred)

    sample_count = 1 if y_true.ndim == 1 else y_true.shape[0]

    return (y_pred - y_true) / sample_count
