import numpy as np


class Dense:
    """A fully connected neural network layer."""

    def __init__(self, input_size, output_size, seed=None):
        """Initialize the layer using Xavier uniform initialization."""
        if (
            not isinstance(input_size, int)
            or not isinstance(output_size, int)
            or input_size <= 0
            or output_size <= 0
        ):
            raise ValueError("Input and output sizes must be positive integers.")
        self.input_size = input_size
        self.output_size = output_size
        rng = np.random.default_rng(seed)
        xavier_limit = np.sqrt(6 / (input_size + output_size))
        self.weights = rng.uniform(
            -xavier_limit,
            xavier_limit,
            size=(input_size, output_size),
        )
        self.biases = np.zeros((output_size,), dtype=float)

        self.inputs = None
        self.weight_gradients = None
        self.bias_gradients = None

    def forward(self, inputs):
        """Perform the forward pass through the layer."""
        inputs = np.asarray(inputs, dtype=float)

        if inputs.ndim not in (1, 2):
            raise ValueError("Inputs must be a 1D or 2D array.")

        if inputs.size == 0:
            raise ValueError("Inputs must not be empty.")

        if inputs.shape[-1] != self.input_size:
            raise ValueError("Input size mismatch.")

        self.inputs = inputs
        return inputs @ self.weights + self.biases

    def backward(self, output_gradients):
        """Perform the backward pass through the layer."""
        if self.inputs is None:
            raise RuntimeError("Forward pass must be called before backward.")

        output_gradients = np.asarray(output_gradients, dtype=float)

        if self.inputs.ndim == 1:
            expected_shape = (self.output_size,)
        else:
            expected_shape = (self.inputs.shape[0], self.output_size)

        if output_gradients.shape != expected_shape:
            raise ValueError(
                f"Output gradients must have shape {expected_shape}."
            )

        if self.inputs.ndim == 1:
            self.weight_gradients = np.outer(
                self.inputs,
                output_gradients,
            )
            self.bias_gradients = output_gradients.copy()
        else:
            self.weight_gradients = self.inputs.T @ output_gradients
            self.bias_gradients = np.sum(output_gradients, axis=0)

        return output_gradients @ self.weights.T