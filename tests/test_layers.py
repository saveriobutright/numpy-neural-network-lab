import numpy as np
import pytest

from neural_network_lab.layers import Dense


def test_dense_initializes_parameters_with_expected_shapes():
    input_size = 3
    output_size = 4

    layer = Dense(input_size, output_size, seed=42)

    assert layer.weights.shape == (input_size, output_size)
    assert layer.biases.shape == (output_size,)

    np.testing.assert_allclose(layer.biases, np.zeros(output_size), rtol=1e-7)


def test_dense_seed_reproduces_weights():
    input_size = 3
    output_size = 4
    seed = 42

    layer1 = Dense(input_size, output_size, seed=seed)
    layer2 = Dense(input_size, output_size, seed=seed)

    np.testing.assert_allclose(layer1.weights, layer2.weights, rtol=1e-7)


def test_dense_weights_stay_within_xavier_bounds():
    input_size = 3
    output_size = 4
    seed = 42

    layer = Dense(input_size, output_size, seed=seed)

    xavier_limit = np.sqrt(6 / (input_size + output_size))
    assert np.all(layer.weights >= -xavier_limit)
    assert np.all(layer.weights <= xavier_limit)


def test_dense_forward_returns_expected_output_for_single_input():
    input_size = 2
    output_size = 2
    seed = 42

    layer = Dense(input_size, output_size, seed=seed)
    layer.weights = np.array([
        [0.5, 1.0],
        [-0.5, 0.25],
    ])
    layer.biases = np.array([0.1, -0.2])

    inputs = np.array([2.0, -1.0])
    expected = np.array([1.6, 1.55])

    output = layer.forward(inputs)

    np.testing.assert_allclose(output, expected, rtol=1e-7)


def test_dense_forward_returns_expected_output_for_batch():
    input_size = 2
    output_size = 2
    seed = 42

    layer = Dense(input_size, output_size, seed=seed)
    layer.weights = np.array([
        [0.5, 1.0],
        [-0.5, 0.25],
    ])
    layer.biases = np.array([0.1, -0.2])

    inputs = np.array([
        [2.0, -1.0],
        [0.0, 2.0],
    ])
    expected = np.array([
        [1.6, 1.55],
        [-0.9, 0.3],
    ])

    output = layer.forward(inputs)

    np.testing.assert_allclose(output, expected, rtol=1e-7)


@pytest.mark.parametrize(
    ("input_size", "output_size"),
    [
        (0, 2),
        (-1, 2),
        (2, 0),
        (2, -1),
        (2.5, 2),
        (2, 2.5),
    ],
)
def test_dense_rejects_invalid_layer_sizes(input_size, output_size):
    with pytest.raises(ValueError):
        Dense(input_size, output_size)


@pytest.mark.parametrize(
    "invalid_inputs",
    [
        1.0,                   # Scalar
        np.ones((1, 1, 2)),    # Three-dimensional input
        [],                    # Empty input
        [1.0, 2.0, 3.0],      # Wrong number of features
    ],
)
def test_dense_forward_rejects_invalid_inputs(invalid_inputs):
    input_size = 2
    output_size = 2
    seed = 42

    layer = Dense(input_size, output_size, seed=seed)

    with pytest.raises(ValueError):
        layer.forward(invalid_inputs)
