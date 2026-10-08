import numpy as np


class SGD:
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, learning_rate=0.01):
        if (
            isinstance(learning_rate, bool)
            or not isinstance(
                learning_rate,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError(
                "Learning rate must be a positive finite number."
            )

        self.learning_rate = float(learning_rate)

    def step(self, layer):
        """Update a layer's parameters using its gradients."""
        if (
            layer.weight_gradients is None
            or layer.bias_gradients is None
        ):
            raise RuntimeError(
                "Backward pass must be called before updating parameters."
            )

        layer.weights -= (
            self.learning_rate * layer.weight_gradients
        )
        layer.biases -= (
            self.learning_rate * layer.bias_gradients
        )


class Momentum(SGD):
    """Gradient descent with momentum."""

    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)

        if (
            isinstance(momentum, bool)
            or not isinstance(
                momentum,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(momentum)
            or not 0 <= momentum < 1
        ):
            raise ValueError(
                "Momentum must be a finite number in the range [0, 1)."
            )

        self.momentum = float(momentum)
        self.velocities = {}

    def step(self, layer):
        """Update parameters using persistent velocities."""
        if (
            layer.weight_gradients is None
            or layer.bias_gradients is None
        ):
            raise RuntimeError(
                "Backward pass must be called before updating parameters."
            )

        if layer not in self.velocities:
            self.velocities[layer] = (
                np.zeros_like(layer.weights),
                np.zeros_like(layer.biases),
            )

        weight_velocity, bias_velocity = self.velocities[layer]

        weight_velocity *= self.momentum
        weight_velocity -= (
            self.learning_rate * layer.weight_gradients
        )

        bias_velocity *= self.momentum
        bias_velocity -= (
            self.learning_rate * layer.bias_gradients
        )

        layer.weights += weight_velocity
        layer.biases += bias_velocity
