import numpy as np

from neural_network_lab.activations import sigmoid, sigmoid_derivative, relu, relu_derivative


def test_sigmoid_of_zero_is_half():
    sigmoid_zero = sigmoid(0)
    np.testing.assert_allclose(sigmoid_zero, 0.5, rtol=1e-7)


def test_sigmoid_is_applied_element_wise():
    input_array = np.array([-1.0, 0.0, 1.0])
    result = sigmoid(input_array)
    np.testing.assert_allclose(result, [0.26894142, 0.5, 0.73105858], rtol=1e-7)


def test_sigmoid_handles_extreme_values_without_overflow():
    input_array = np.array([-1000.0, 1000.0])
    with np.errstate(over="raise"):
        result = sigmoid(input_array)
    np.testing.assert_allclose(result, [0.0, 1.0], rtol=1e-7)


def test_sigmoid_derivative_at_zero_is_one_quarter():
    s = sigmoid_derivative(0.0)
    np.testing.assert_allclose(s, 0.25, rtol=1e-7)


def test_sigmoid_derivative_is_symmetric():
    input_array = np.array([0.5, 1.0, 2.0, 10.0])
    s = sigmoid_derivative(input_array)
    t = sigmoid_derivative(-input_array)
    np.testing.assert_allclose(s, t, rtol=1e-7)


def test_relu_is_applied_element_wise():
    input_array = np.array([-3.0, -0.5, 0.0, 2.0, 5.0])
    result = relu(input_array)
    np.testing.assert_allclose(result, [0.0, 0.0, 0.0, 2.0, 5.0])


def test_relu_derivative_uses_zero_at_origin():
    input_array = np.array([-3.0, -0.5, 0.0, 2.0, 5.0])
    result = relu_derivative(input_array)
    np.testing.assert_allclose(result, [0.0, 0.0, 0.0, 1.0, 1.0])