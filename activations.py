import numpy as np
from typing import Literal, Dict, Tuple, Callable, Optional
from functools import partial

"""
activations.py

This module defines various activation functions and their derivatives

Supported functions: "linear", "sigmoid", "tanh", "relu", "prelu", "elu", "swish", "softmax"
"""

def _set_activation(
    layer: object,
    activation: Literal["linear", "sigmoid", "tanh", "relu", "prelu", "elu", "swish", "softmax"],
    alpha: Optional[float],
    beta: Optional[float]
) -> None:
    """
    Sets the activation function and its derivative for a given layer.

    Args:
        layer: The neural network layer to set the activation function for.
        activation: The name of the activation function to use.
        alpha: Used only for "prelu" (default: 0.01) and "elu" (default: 1.0).
        beta: Used only for "swish" (default: 1.0).

    Raises:
        ValueError: If the specified activation function is not supported.
    """
    
    if activation not in SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"Unsupported activation function '{activation}'. "
            f"Supported activations: {SUPPORTED_ACTIVATIONS}."
        )      
              
    layer._activation_name = activation
    
    if activation == "prelu":
        layer.alpha = 0.01 if alpha is None else alpha
        layer.activation = partial(_prelu, alpha=layer.alpha)
        layer.activation_derivative = partial(_prelu_derivative, alpha=layer.alpha)
    elif activation == "elu":
        layer.alpha = 1.0 if alpha is None else alpha
        layer.activation = partial(_elu, alpha=layer.alpha)
        layer.activation_derivative = partial(_elu_derivative, alpha=layer.alpha)
    elif activation == "swish":
        layer.beta = 1.0 if beta is None else beta
        layer.activation = partial(_swish, beta=layer.beta)
        layer.activation_derivative = partial(_swish_derivative, beta=layer.beta)
    else:
        layer.activation, layer.activation_derivative = non_param_activations[activation]        

def _linear(z: np.ndarray) -> np.ndarray:
    return z


def _linear_derivative(z: np.ndarray) -> np.ndarray:
    return np.ones_like(z)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def _sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    s = _sigmoid(z)
    return s * (1 - s)


def _tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def _tanh_derivative(z: np.ndarray) -> np.ndarray:
    return 1 - np.square(np.tanh(z))


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def _relu_derivative(z: np.ndarray) -> np.ndarray:
    return np.where(z > 0, 1, 0)


def _prelu(z: np.ndarray, alpha: float) -> np.ndarray:
    return np.where(z >= 0, z, alpha * z)


def _prelu_derivative(z: np.ndarray, alpha: float) -> np.ndarray:
    return np.where(z >= 0, 1, alpha)


def _elu(z: np.ndarray, alpha: float) -> np.ndarray:
    return np.where(z >= 0, z, alpha * (np.exp(z) - 1))


def _elu_derivative(z: np.ndarray, alpha: float) -> np.ndarray:
    return np.where(z >= 0, 1, alpha * np.exp(z))


def _swish(z: np.ndarray, beta: float) -> np.ndarray:
    return z * _sigmoid(beta * z)


def _swish_derivative(z: np.ndarray, beta: float) -> np.ndarray:
    s = _sigmoid(beta * z)
    return s + beta * z * s * (1 - s)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -100, 100)  # Clip z to avoid overflow
    e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return e_z / (e_z.sum(axis=0, keepdims=True) + 1e-10)

non_param_activations: Dict[str, Tuple[Callable, Callable]] = {
    "linear": (_linear, _linear_derivative),
    "sigmoid": (_sigmoid, _sigmoid_derivative),
    "tanh": (_tanh, _tanh_derivative),
    "relu": (_relu, _relu_derivative),
    "softmax": (_softmax, _linear_derivative),
}

SUPPORTED_ACTIVATIONS = set(non_param_activations.keys()) | {"prelu", "elu", "swish"}