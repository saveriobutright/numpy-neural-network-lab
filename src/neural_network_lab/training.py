import numpy as np


def iterate_minibatches(
    inputs,
    targets,
    batch_size,
    shuffle=True,
    seed=None,
):
    """Yield mini-batches of inputs and corresponding targets."""
    inputs = np.asarray(inputs)
    targets = np.asarray(targets)

    if inputs.ndim == 0 or targets.ndim == 0:
        raise ValueError("Inputs and targets must have at least one dimension.")

    if inputs.shape[0] != targets.shape[0]:
        raise ValueError(
            "Inputs and targets must contain the same number of samples."
        )

    if inputs.shape[0] == 0:
        raise ValueError("Dataset must not be empty.")

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, (int, np.integer))
        or batch_size <= 0
    ):
        raise ValueError("Batch size must be a positive integer.")

    if not isinstance(shuffle, (bool, np.bool_)):
        raise ValueError("Shuffle must be a boolean.")

    indices = np.arange(inputs.shape[0])

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, inputs.shape[0], batch_size):
        end = start + batch_size
        batch_indices = indices[start:end]

        yield inputs[batch_indices], targets[batch_indices]


def train_epoch(
    layer,
    inputs,
    targets,
    loss_function,
    loss_derivative,
    optimizer,
    batch_size,
    shuffle=True,
    seed=None,
):
    """Train a single layer for one epoch."""
    total_loss = 0.0
    sample_count = 0

    for batch_inputs, batch_targets in iterate_minibatches(
        inputs,
        targets,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    ):
        predictions = layer.forward(batch_inputs)

        batch_loss = loss_function(
            batch_targets,
            predictions,
        )
        output_gradients = loss_derivative(
            batch_targets,
            predictions,
        )

        layer.backward(output_gradients)
        optimizer.step(layer)

        current_batch_size = batch_inputs.shape[0]
        total_loss += batch_loss * current_batch_size
        sample_count += current_batch_size

    return float(total_loss / sample_count)
