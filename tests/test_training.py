import numpy as np
import pytest

from neural_network_lab.layers import Dense
from neural_network_lab.losses import (
    mean_squared_error,
    mean_squared_error_derivative,
)
from neural_network_lab.optimizers import SGD
from neural_network_lab.training import (
    iterate_minibatches,
    train_epoch,
)


def test_iterate_minibatches_preserves_order_and_final_partial_batch():
    inputs = np.arange(20).reshape(10, 2)
    targets = np.arange(10)
    batch_size = 4
    shuffle = False

    batches = list(iterate_minibatches(inputs, targets, batch_size, shuffle))

    assert len(batches) == 3

    batch_sizes = [
        batch_inputs.shape[0]
        for batch_inputs, _ in batches
    ]
    assert batch_sizes == [4, 4, 2]

    reconstructed_inputs = np.concatenate([
        batch_inputs
        for batch_inputs, _ in batches
    ])
    reconstructed_targets = np.concatenate([
        batch_targets
        for _, batch_targets in batches
    ])

    np.testing.assert_array_equal(
        reconstructed_inputs,
        inputs,
    )
    np.testing.assert_array_equal(
        reconstructed_targets,
        targets,
    )


def test_iterate_minibatches_shuffles_reproducibly_and_preserves_pairs():
    inputs = np.arange(20).reshape(10, 2)
    targets = np.arange(10)

    first_batches = list(
        iterate_minibatches(
            inputs,
            targets,
            batch_size=4,
            shuffle=True,
            seed=42,
        )
    )
    second_batches = list(
        iterate_minibatches(
            inputs,
            targets,
            batch_size=4,
            shuffle=True,
            seed=42,
        )
    )

    first_inputs = np.concatenate([
        batch_inputs
        for batch_inputs, _ in first_batches
    ])
    first_targets = np.concatenate([
        batch_targets
        for _, batch_targets in first_batches
    ])

    second_inputs = np.concatenate([
        batch_inputs
        for batch_inputs, _ in second_batches
    ])
    second_targets = np.concatenate([
        batch_targets
        for _, batch_targets in second_batches
    ])

    np.testing.assert_array_equal(
        first_inputs,
        second_inputs,
    )
    np.testing.assert_array_equal(
        first_targets,
        second_targets,
    )
    np.testing.assert_array_equal(
        first_inputs,
        inputs[first_targets],
    )


@pytest.mark.parametrize(
    "invalid_batch_size",
    [
        0,
        -1,
        1.5,
        True,
    ],
)
def test_iterate_minibatches_rejects_invalid_batch_sizes(
    invalid_batch_size,
):
    inputs = np.arange(12).reshape(6, 2)
    targets = np.arange(6)

    with pytest.raises(ValueError):
        list(
            iterate_minibatches(
                inputs,
                targets,
                batch_size=invalid_batch_size,
            )
        )


@pytest.mark.parametrize(
    ("inputs", "targets"),
    [
        (1.0, [0]),
        (np.ones((3, 2)), np.arange(2)),
        (np.empty((0, 2)), np.array([])),
    ],
)
def test_iterate_minibatches_rejects_invalid_datasets(
    inputs,
    targets,
):
    with pytest.raises(ValueError):
        list(
            iterate_minibatches(
                inputs,
                targets,
                batch_size=2,
            )
        )


@pytest.mark.parametrize(
    "invalid_shuffle",
    [
        0,
        1,
        "yes",
        None,
    ],
)
def test_iterate_minibatches_rejects_non_boolean_shuffle(
    invalid_shuffle,
):
    inputs = np.arange(12).reshape(6, 2)
    targets = np.arange(6)

    with pytest.raises(ValueError):
        list(
            iterate_minibatches(
                inputs,
                targets,
                batch_size=2,
                shuffle=invalid_shuffle,
            )
        )


def test_train_epoch_updates_parameters_and_returns_loss():
    inputs = np.array([
        [-1.0],
        [0.0],
        [1.0],
    ])
    targets = np.array([
        [-1.0],
        [1.0],
        [3.0],
    ])

    layer = Dense(1, 1, seed=42)
    layer.weights = np.zeros((1, 1))
    layer.biases = np.zeros(1)

    optimizer = SGD(learning_rate=0.1)

    loss = train_epoch(
        layer,
        inputs,
        targets,
        mean_squared_error,
        mean_squared_error_derivative,
        optimizer,
        batch_size=3,
        shuffle=False,
    )

    np.testing.assert_allclose(loss, 11 / 3, rtol=1e-7)
    np.testing.assert_allclose(
        layer.weights,
        [[4 / 15]],
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        layer.biases,
        [0.2],
        rtol=1e-7,
    )


def test_train_epoch_learns_linear_relationship():
    inputs = np.array([
        [-2.0],
        [-1.0],
        [0.0],
        [1.0],
        [2.0],
    ])
    targets = 2 * inputs + 1

    layer = Dense(1, 1, seed=42)
    optimizer = SGD(learning_rate=0.05)

    initial_loss = mean_squared_error(
        targets,
        layer.forward(inputs),
    )

    for epoch in range(100):
        train_epoch(
            layer,
            inputs,
            targets,
            mean_squared_error,
            mean_squared_error_derivative,
            optimizer,
            batch_size=2,
            shuffle=True,
            seed=epoch,
        )

    final_loss = mean_squared_error(
        targets,
        layer.forward(inputs),
    )

    assert final_loss < initial_loss
    np.testing.assert_allclose(
        layer.weights,
        [[2.0]],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        layer.biases,
        [1.0],
        atol=1e-7,
    )
