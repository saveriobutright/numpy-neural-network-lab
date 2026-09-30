import numpy as np

from neural_network_lab.activations import sigmoid, sigmoid_derivative, relu, relu_derivative, softmax


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


def test_softmax_returns_probability_distribution():
    input_array = np.array([2.0, 1.0, 0.0])
    result = softmax(input_array)

    np.testing.assert_allclose(result, [0.66524096, 0.24472847, 0.09003057])
    np.testing.assert_allclose(np.sum(result), 1.0)


def test_softmax_is_stable_for_large_logits():
    input_array_1 = np.array([2.0, 1.0, 0.0])
    result_1 = softmax(input_array_1)
    input_array_2 = np.array([1002.0, 1001.0, 1000.0])

    with np.errstate(over="raise"):
        result_2 = softmax(input_array_2)

    np.testing.assert_allclose(result_1, result_2)


def test_softmax_normalizes_each_batch_item():
    input_matrix = np.array([
        [2.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ])
    expected = np.array([
        [0.66524096, 0.24472847, 0.09003057],
        [1 / 3, 1 / 3, 1 / 3],
    ])

    result = softmax(input_matrix)

    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(np.sum(result, axis=1), [1.0, 1.0])
