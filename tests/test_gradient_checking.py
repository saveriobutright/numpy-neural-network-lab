import numpy as np
import pytest

from neural_network_lab.layers import Dense
from neural_network_lab.losses import (
    mean_squared_error,
    mean_squared_error_derivative,
)

@pytest.mark.parametrize(
    ("inputs", "targets"),
    [
        (
            np.array([0.5, -1.0]),
            np.array([0.2, 0.8]),
        ),
        (
            np.array([
                [0.5, -1.0],
                [1.5, 2.0],
            ]),
            np.array([
                [0.2, 0.8],
                [1.0, -0.5],
            ]),
        ),
    ],
)
@pytest.mark.parametrize(
    ("parameter_name", "gradient_name"),
    [
        ("weights", "weight_gradients"),
        ("biases", "bias_gradients"),
    ],
)
def test_dense_gradients_match_numerical_gradients(
    inputs,
    targets,
    parameter_name,
    gradient_name,
):
    layer = Dense(2, 2, seed=42)

    predictions = layer.forward(inputs)
    output_gradients = mean_squared_error_derivative(
        targets,
        predictions,
    )
    layer.backward(output_gradients)

    parameters = getattr(layer, parameter_name)
    analytical_gradients = getattr(layer, gradient_name).copy()
    numerical_gradients = np.zeros_like(parameters)
    epsilon = 1e-5

    for index in np.ndindex(parameters.shape):
        original_value = parameters[index]

        try:
            parameters[index] = original_value + epsilon
            loss_plus = mean_squared_error(
                targets,
                layer.forward(inputs),
            )

            parameters[index] = original_value - epsilon
            loss_minus = mean_squared_error(
                targets,
                layer.forward(inputs),
            )
        finally:
            parameters[index] = original_value

        numerical_gradients[index] = (
            (loss_plus - loss_minus) / (2 * epsilon)
        )

    np.testing.assert_allclose(
        analytical_gradients,
        numerical_gradients,
        rtol=1e-5,
        atol=1e-7,
    )
