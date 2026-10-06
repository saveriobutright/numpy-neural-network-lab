# Neural Network Lab

A neural network library implemented from scratch with Python and NumPy.

The project focuses on the mathematical and software foundations behind neural
networks. Its components are developed incrementally, tested independently, and
documented as they are introduced. High-level machine learning frameworks are
reserved for later validation and benchmarking rather than for the core
implementation.

## Goals

- Understand forward propagation and backpropagation by implementing them.
- Build reusable, vectorized neural network components with NumPy.
- Verify mathematical properties and numerical behavior with automated tests.
- Compare the completed implementation with a professional deep learning
  framework.
- Produce reproducible experiments on synthetic and real datasets.

## Current Features

- Numerically stable sigmoid activation.
- Sigmoid derivative for backpropagation.
- ReLU activation with an explicit derivative convention at zero.
- Numerically stable Softmax activation for individual vectors and batches.
- Mean Squared Error loss and derivative with input-shape validation.
- Numerically stable Binary Cross-Entropy loss and derivative.
- Categorical Cross-Entropy for one-hot targets and batches.
- Simplified Softmax–CCE gradient with respect to logits.
- Vectorized operations for scalar, list, and NumPy array inputs.
- Automated tests for expected values, numerical stability, mathematical
  properties, probability normalization, and input validation.
- Dense layers with Xavier uniform initialization.
- Dense forward propagation for individual inputs and batches.
- Vectorized Dense backward propagation for input, weight, and bias gradients.

## Project Structure

```text
NeuralNetworkLab/
|-- src/
|   `-- neural_network_lab/
|       |-- __init__.py
|       |-- activations.py
|       |-- layers.py
|       `-- losses.py
|-- tests/
|   |-- test_activations.py
|   |-- test_layers.py
|   `-- test_losses.py
|-- pyproject.toml
`-- README.md
```

## Development Setup

Python 3.10 or later is required.

Create and activate a virtual environment on Windows:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project and its development dependencies:

```bash
python -m pip install --editable ".[dev]"
```

Run the test suite:

```bash
python -m pytest
```

## Example

```python
import numpy as np

from neural_network_lab import (
    Dense,
    binary_cross_entropy,
    binary_cross_entropy_derivative,
    categorical_cross_entropy,
    categorical_cross_entropy_derivative,
    mean_squared_error,
    mean_squared_error_derivative,
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    softmax,
    softmax_categorical_cross_entropy_derivative,
)

values = np.array([-2.0, 0.0, 2.0])

print(sigmoid(values))
print(sigmoid_derivative(values))
print(relu(values))
print(relu_derivative(values))
print(softmax(values))

targets = np.array([1.0, 0.0, 1.0])
predictions = np.array([0.7, 0.2, 0.9])

print(mean_squared_error(targets, predictions))
print(mean_squared_error_derivative(targets, predictions))
print(binary_cross_entropy(targets, predictions))
print(binary_cross_entropy_derivative(targets, predictions))

class_targets = np.array([
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
])
class_predictions = np.array([
    [0.1, 0.7, 0.2],
    [0.8, 0.1, 0.1],
])

print(categorical_cross_entropy(class_targets, class_predictions))
print(categorical_cross_entropy_derivative(class_targets, class_predictions))
print(
    softmax_categorical_cross_entropy_derivative(
        class_targets,
        class_predictions,
    )
)

dense = Dense(3, 2, seed=42)
dense_inputs = np.array([
    [1.0, 2.0, 3.0],
    [0.5, -1.0, 2.0],
])

dense_outputs = dense.forward(dense_inputs)
dense_output_gradients = np.ones_like(dense_outputs)
dense_input_gradients = dense.backward(dense_output_gradients)

print(dense_outputs)
print(dense_input_gradients)
print(dense.weight_gradients)
print(dense.bias_gradients)
```

## Roadmap

- [x] Stable sigmoid activation and derivative
- [x] ReLU activation and derivative
- [x] Numerically stable Softmax activation with batch support
- [x] Loss functions
  - [x] Mean Squared Error and derivative
  - [x] Binary Cross-Entropy and derivative
  - [x] Categorical Cross-Entropy and Softmax integration
- [x] Dense layers and parameter initialization
- [x] Vectorized backpropagation
- [ ] Mini-batch training
- [ ] SGD, Momentum, and Adam optimizers
- [ ] Gradient checking
- [ ] Experiments on synthetic datasets and Fashion MNIST
- [ ] Comparison with PyTorch
- [ ] Interactive training visualizations

## Project Status

Neural Network Lab is under active development. Features are documented as
they become implemented and covered by tests.
