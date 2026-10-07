import numpy as np
import pytest

from neural_network_lab.layers import Dense
from neural_network_lab.optimizers import SGD


def test_sgd_updates_weights_and_biases():
    layer = Dense(2, 2, seed=42)
    layer.weights = np.array([
        [0.5, 1.0],
        [-0.5, 0.25],
    ])
    layer.biases = np.array([0.1, -0.2])

    layer.forward([2.0, -1.0])
    layer.backward([0.3, -0.2])

    optimizer = SGD(learning_rate=0.1)
    optimizer.step(layer)

    expected_weights = np.array([
        [0.44, 1.04],
        [-0.47, 0.23],
    ])
    expected_biases = np.array([0.07, -0.18])

    np.testing.assert_allclose(
        layer.weights,
        expected_weights,
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        layer.biases,
        expected_biases,
        rtol=1e-7,
    )


def test_sgd_requires_gradients_before_updating_parameters():
    layer = Dense(2, 2, seed=42)
    optimizer = SGD(learning_rate=0.1)

    with pytest.raises(RuntimeError):
        optimizer.step(layer)


@pytest.mark.parametrize(
    "invalid_learning_rate",
    [
        0,
        -0.1,
        True,
        np.nan,
        np.inf,
        -np.inf,
        "0.1",
        None,
    ],
)
def test_sgd_rejects_invalid_learning_rates(
    invalid_learning_rate,
):
    with pytest.raises(ValueError):
        SGD(learning_rate=invalid_learning_rate)