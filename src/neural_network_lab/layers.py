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

    def forward(self, inputs):
        """Perform the forward pass through the layer."""
        inputs = np.asarray(inputs, dtype=float)

        if inputs.ndim not in (1, 2):
            raise ValueError("Inputs must be a 1D or 2D array.")

        if inputs.size == 0:
            raise ValueError("Inputs must not be empty.")

        if inputs.shape[-1] != self.input_size:
            raise ValueError("Input size mismatch.")

        return inputs @ self.weights + self.biases
