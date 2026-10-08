import numpy as np
import pytest

from neural_network_lab.layers import Dense
from neural_network_lab.optimizers import Adam, Momentum, SGD


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


def test_momentum_preserves_velocity_between_steps():
    layer = Dense(1, 1, seed=42)
    layer.weights.fill(1.0)
    layer.biases.fill(0.0)

    optimizer = Momentum(
        learning_rate=0.1,
        momentum=0.9,
    )

    layer.forward([1.0])
    layer.backward([0.5])
    optimizer.step(layer)

    np.testing.assert_allclose(layer.weights, [[0.95]])
    np.testing.assert_allclose(layer.biases, [-0.05])

    layer.forward([1.0])
    layer.backward([0.5])
    optimizer.step(layer)

    np.testing.assert_allclose(layer.weights, [[0.855]])
    np.testing.assert_allclose(layer.biases, [-0.145])


def test_momentum_keeps_separate_velocities_for_each_layer():
    first_layer = Dense(1, 1, seed=42)
    second_layer = Dense(1, 1, seed=42)

    for layer in (first_layer, second_layer):
        layer.weights.fill(1.0)
        layer.biases.fill(0.0)

    optimizer = Momentum(
        learning_rate=0.1,
        momentum=0.9,
    )

    first_layer.forward([1.0])
    first_layer.backward([0.5])
    optimizer.step(first_layer)

    second_layer.forward([1.0])
    second_layer.backward([1.0])
    optimizer.step(second_layer)

    np.testing.assert_allclose(first_layer.weights, [[0.95]])
    np.testing.assert_allclose(first_layer.biases, [-0.05])
    np.testing.assert_allclose(second_layer.weights, [[0.9]])
    np.testing.assert_allclose(second_layer.biases, [-0.1])

    first_layer.forward([1.0])
    first_layer.backward([0.5])
    optimizer.step(first_layer)

    np.testing.assert_allclose(first_layer.weights, [[0.855]])
    np.testing.assert_allclose(first_layer.biases, [-0.145])
    np.testing.assert_allclose(second_layer.weights, [[0.9]])
    np.testing.assert_allclose(second_layer.biases, [-0.1])


def test_momentum_zero_matches_sgd():
    sgd_layer = Dense(1, 1, seed=42)
    momentum_layer = Dense(1, 1, seed=42)

    sgd = SGD(learning_rate=0.1)
    momentum = Momentum(
        learning_rate=0.1,
        momentum=0.0,
    )

    for gradient in [0.5, -0.2, 0.1]:
        sgd_layer.forward([1.0])
        sgd_layer.backward([gradient])
        sgd.step(sgd_layer)

        momentum_layer.forward([1.0])
        momentum_layer.backward([gradient])
        momentum.step(momentum_layer)

        np.testing.assert_allclose(
            momentum_layer.weights,
            sgd_layer.weights,
        )
        np.testing.assert_allclose(
            momentum_layer.biases,
            sgd_layer.biases,
        )


@pytest.mark.parametrize(
    "invalid_momentum",
    [
        -0.1,
        1.0,
        1.1,
        True,
        np.nan,
        np.inf,
        -np.inf,
        "0.9",
        None,
    ],
)
def test_momentum_rejects_invalid_momentum(invalid_momentum):
    with pytest.raises(ValueError):
        Momentum(momentum=invalid_momentum)


def test_momentum_requires_gradients_before_updating_parameters():
    layer = Dense(2, 2, seed=42)
    optimizer = Momentum(learning_rate=0.1, momentum=0.9)

    with pytest.raises(RuntimeError):
        optimizer.step(layer)


def test_adam_corrects_initial_bias_and_preserves_state():
    layer = Dense(1, 1, seed=42)
    layer.weights.fill(1.0)
    layer.biases.fill(0.0)

    epsilon = 1e-8
    optimizer = Adam(
        learning_rate=0.1,
        epsilon=epsilon,
    )

    expected_update = 0.1 * 0.5 / (0.5 + epsilon)

    for step in (1, 2):
        layer.forward([1.0])
        layer.backward([0.5])
        optimizer.step(layer)

        np.testing.assert_allclose(
            layer.weights,
            [[1.0 - step * expected_update]],
            rtol=1e-7,
        )
        np.testing.assert_allclose(
            layer.biases,
            [-step * expected_update],
            rtol=1e-7,
        )
        assert optimizer.states[layer]["t"] == step


def test_adam_updates_using_changing_gradients():
    layer = Dense(1, 1, seed=42)
    layer.weights.fill(1.0)
    layer.biases.fill(0.0)

    epsilon = 1e-8
    optimizer = Adam(
        learning_rate=0.1,
        beta1=0.5,
        beta2=0.5,
        epsilon=epsilon,
    )

    layer.forward([1.0])
    layer.backward([0.5])
    optimizer.step(layer)

    layer.forward([1.0])
    layer.backward([1.0])
    optimizer.step(layer)

    first_update = 0.1 * 0.5 / (0.5 + epsilon)
    second_update = (
        0.1 * (5 / 6) / (np.sqrt(3 / 4) + epsilon)
    )

    np.testing.assert_allclose(
        layer.weights,
        [[1.0 - first_update - second_update]],
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        layer.biases,
        [-first_update - second_update],
        rtol=1e-7,
    )


def test_adam_keeps_independent_state_for_each_layer():
    first_layer = Dense(1, 1, seed=42)
    second_layer = Dense(1, 1, seed=42)

    for layer in (first_layer, second_layer):
        layer.weights.fill(1.0)
        layer.biases.fill(0.0)

    epsilon = 1e-8
    optimizer = Adam(learning_rate=0.1, epsilon=epsilon)

    first_layer.forward([1.0])
    first_layer.backward([0.5])
    optimizer.step(first_layer)

    second_layer.forward([1.0])
    second_layer.backward([-0.5])
    optimizer.step(second_layer)

    first_layer.forward([1.0])
    first_layer.backward([0.5])


def test_adam_keeps_separate_states_for_each_layer():
    first_layer = Dense(1, 1, seed=42)
    second_layer = Dense(1, 1, seed=42)

    for layer in (first_layer, second_layer):
        layer.weights.fill(1.0)
        layer.biases.fill(0.0)

    optimizer = Adam(learning_rate=0.1)
    epsilon = optimizer.epsilon

    first_layer.forward([1.0])
    first_layer.backward([0.5])
    optimizer.step(first_layer)

    second_layer.forward([1.0])
    second_layer.backward([-1.0])
    optimizer.step(second_layer)

    first_layer.forward([1.0])
    first_layer.backward([0.5])
    optimizer.step(first_layer)

    first_update = 0.1 * 0.5 / (0.5 + epsilon)
    second_update = 0.1 / (1.0 + epsilon)

    np.testing.assert_allclose(
        first_layer.weights,
        [[1.0 - 2 * first_update]],
    )
    np.testing.assert_allclose(
        first_layer.biases,
        [-2 * first_update],
    )
    np.testing.assert_allclose(
        second_layer.weights,
        [[1.0 + second_update]],
    )
    np.testing.assert_allclose(
        second_layer.biases,
        [second_update],
    )

    assert optimizer.states[first_layer]["t"] == 2
    assert optimizer.states[second_layer]["t"] == 1


def test_adam_handles_zero_gradients_without_changing_parameters():
    layer = Dense(1, 1, seed=42)
    layer.weights.fill(1.0)
    layer.biases.fill(0.0)

    layer.forward([1.0])
    layer.backward([0.0])

    optimizer = Adam()

    with np.errstate(divide="raise", invalid="raise"):
        optimizer.step(layer)

    np.testing.assert_array_equal(layer.weights, [[1.0]])
    np.testing.assert_array_equal(layer.biases, [0.0])


def test_adam_requires_gradients_before_updating_parameters():
    layer = Dense(2, 2, seed=42)
    optimizer = Adam()

    with pytest.raises(RuntimeError):
        optimizer.step(layer)


@pytest.mark.parametrize("parameter_name", ["beta1", "beta2"])
@pytest.mark.parametrize(
    "invalid_value",
    [
        -0.1,
        1.0,
        1.1,
        True,
        np.nan,
        np.inf,
        -np.inf,
        "0.9",
        None,
    ],
)
def test_adam_rejects_invalid_betas(parameter_name, invalid_value):
    with pytest.raises(ValueError):
        Adam(**{parameter_name: invalid_value})


@pytest.mark.parametrize(
    "invalid_epsilon",
    [
        0,
        -1e-8,
        True,
        np.nan,
        np.inf,
        -np.inf,
        "1e-8",
        None,
    ],
)
def test_adam_rejects_invalid_epsilon(invalid_epsilon):
    with pytest.raises(ValueError):
        Adam(epsilon=invalid_epsilon)
