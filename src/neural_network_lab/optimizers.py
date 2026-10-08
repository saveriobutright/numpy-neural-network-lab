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


class Adam(SGD):
    """Adaptive Moment Estimation optimizer."""

    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ):
        super().__init__(learning_rate)

        for name, value in (
            ("beta1", beta1),
            ("beta2", beta2),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(
                    value,
                    (int, float, np.integer, np.floating),
                )
                or not np.isfinite(value)
                or not 0 <= value < 1
            ):
                raise ValueError(
                    f"{name} must be a finite number in the range [0, 1)."
                )

        if (
            isinstance(epsilon, bool)
            or not isinstance(
                epsilon,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(epsilon)
            or epsilon <= 0
        ):
            raise ValueError(
                "Epsilon must be a positive finite number."
            )

        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.states = {}

    def step(self, layer):
        """Update parameters using bias-corrected moment estimates."""
        if (
            layer.weight_gradients is None
            or layer.bias_gradients is None
        ):
            raise RuntimeError(
                "Backward pass must be called before updating parameters."
            )

        if layer not in self.states:
            self.states[layer] = {
                "t": 0,
                "weight_m": np.zeros_like(layer.weights),
                "weight_v": np.zeros_like(layer.weights),
                "bias_m": np.zeros_like(layer.biases),
                "bias_v": np.zeros_like(layer.biases),
            }

        state = self.states[layer]
        state["t"] += 1
        t = state["t"]

        state["weight_m"] *= self.beta1
        state["weight_m"] += (
            (1 - self.beta1) * layer.weight_gradients
        )
        state["weight_v"] *= self.beta2
        state["weight_v"] += (
            (1 - self.beta2) * layer.weight_gradients ** 2
        )

        state["bias_m"] *= self.beta1
        state["bias_m"] += (
            (1 - self.beta1) * layer.bias_gradients
        )
        state["bias_v"] *= self.beta2
        state["bias_v"] += (
            (1 - self.beta2) * layer.bias_gradients ** 2
        )

        weight_m_hat = state["weight_m"] / (1 - self.beta1 ** t)
        weight_v_hat = state["weight_v"] / (1 - self.beta2 ** t)
        bias_m_hat = state["bias_m"] / (1 - self.beta1 ** t)
        bias_v_hat = state["bias_v"] / (1 - self.beta2 ** t)

        layer.weights -= (
            self.learning_rate
            * weight_m_hat
            / (np.sqrt(weight_v_hat) + self.epsilon)
        )
        layer.biases -= (
            self.learning_rate
            * bias_m_hat
            / (np.sqrt(bias_v_hat) + self.epsilon)
        )