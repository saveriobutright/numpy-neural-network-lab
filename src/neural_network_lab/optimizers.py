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