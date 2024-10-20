import numpy as np
from typing import Literal, Dict, Tuple, Callable

"""
activations.py

This module defines various activation functions and their derivatives

Supported functions: "relu", "tanh", "sigmoid", "softmax", "linear".
"""

def _set_activation(
    layer: object,
    activation: Literal["relu", "tanh", "sigmoid", "softmax", "linear"]
) -> None:
    """
    Sets the activation function and its derivative for a given layer.

    Args:
        layer: The neural network layer to set the activation function for.
        activation: The name of the activation function to use.

    Raises:
        ValueError: If the specified activation function is not supported.
    """
    if activation in activation_functions:
        layer.activation, layer.activation_derivative = activation_functions[activation]
    else:
        raise ValueError(f"Unsupported activation function '{activation}'. Supported activation functions are {list(activation_functions.keys())}.")


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def _relu_derivative(z: np.ndarray) -> np.ndarray:
    return np.where(z > 0, 1, 0)


def _tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def _tanh_derivative(z: np.ndarray) -> np.ndarray:
    return 1 - np.square(np.tanh(z))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def _sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    s = _sigmoid(z)
    return s * (1 - s)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -100, 100)  # Clip z to avoid overflow
    e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return e_z / (e_z.sum(axis=0, keepdims=True) + 1e-10)


def _linear(z: np.ndarray) -> np.ndarray:
    return z


def _linear_derivative(z: np.ndarray) -> np.ndarray:
    return np.ones_like(z)

activation_functions: Dict[str, Tuple[Callable, Callable]] = {
    "relu": (_relu, _relu_derivative),
    "tanh": (_tanh, _tanh_derivative),
    "sigmoid": (_sigmoid, _sigmoid_derivative),
    "softmax": (_softmax, _linear_derivative),
    "linear": (_linear, _linear_derivative),
}
