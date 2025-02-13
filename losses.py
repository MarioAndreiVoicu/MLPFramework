import numpy as np
from typing import Literal

"""
losses.py

This module provides implementations of various loss functions and their
gradients for use in the Multilayer Perceptron.

Loss Functions Supported:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Huber Loss (HL)
- Binary Crossentropy (BCE)
- Categorical Crossentropy (CCE)
"""

def _set_loss(
    model: object,
    loss_function: Literal[
        "mean_squared_error", 
        "mean_absolute_error", 
        "huber_loss",
        "binary_crossentropy", 
        "categorical_crossentropy"
    ]
) -> None:
    """
    Sets the loss function and its gradient for the model.

    Args:
        model: The model to set the loss function for.
        loss_function: The name of the loss function. Supported loss functions:
            "mean_squared_error", "mean_absolute_error", "huber_loss",
            "binary_crossentropy", "categorical_crossentropy".
    
    Raises:
        ValueError: If the loss function is not supported.
    """
    if loss_function in loss_functions:
        model.loss_function, model.loss_gradient = loss_functions[loss_function]
    else:
         raise ValueError(f"Unsupported loss function: {loss_function}. Supported loss functions are: {list(loss_functions.keys())}")

def _mean_squared_error(Y: np.ndarray, Yhat: np.ndarray) -> float:
    m: int = Y.shape[1]
    cost: float = np.sum(np.square((Y - Yhat))) / (2 * m)
    return cost

def _MSE_gradient(Y: np.ndarray, Yhat: np.ndarray) -> np.ndarray:
    m: int = Y.shape[1]
    gradient: np.ndarray = -2 * (Y - Yhat) / m
    return gradient

def _mean_absolute_error(Y: np.ndarray, Yhat: np.ndarray) -> float:
    m: int = Y.shape[1]
    cost: float = np.sum(np.abs(Yhat - Y)) / m
    return cost

def _MAE_gradient(Y: np.ndarray, Yhat: np.ndarray) -> np.ndarray:
    m: int = Y.shape[1]
    gradient: np.ndarray = np.sign(Yhat - Y) / m
    return gradient

def _huber_loss(Y: np.ndarray, Yhat: np.ndarray) -> float:
    """
    A default delta value of 1.0 is used.
    """
    delta = 1.0
    m: int = Y.shape[1]
    error = Yhat - Y
    abs_error = np.abs(error)
    loss = np.where(abs_error <= delta, 0.5 * error**2, delta * (abs_error - 0.5*delta))
    return np.sum(loss) / m

def _huber_gradient(Y: np.ndarray, Yhat: np.ndarray) -> np.ndarray:
    delta = 1.0
    m: int = Y.shape[1]
    error = Yhat - Y
    abs_error = np.abs(error)
    grad = np.where(abs_error <= delta, error, delta * np.sign(error))
    return grad / m

def _binary_crossentropy(Y: np.ndarray, Yhat: np.ndarray) -> float:
    m: int = Y.shape[1]
    cost: float = -np.sum(Y * np.log(Yhat + 1e-10) + (1 - Y) * np.log(1 - Yhat + 1e-10)) / m
    return cost

def _BCE_gradient(Y: np.ndarray, Yhat: np.ndarray) -> np.ndarray:
    m: int = Y.shape[1]
    gradient: np.ndarray = (Yhat - Y) / (Yhat * (1 - Yhat) + 1e-10)
    return gradient

def _categorical_crossentropy(Y: np.ndarray, Yhat: np.ndarray) -> float:
    """
    Computes the categorical cross-entropy loss
    
    Clipping is used to avoid log(0) which is undefined.
    It clips the values of Yhat to be between 1e-10 and 1.0.
    """
    m: int = Y.shape[1]
    Yhat: np.ndarray = np.clip(Yhat, 1e-10, 1.0)  # Clipping to avoid log(0)
    return -np.sum(Y * np.log(Yhat)) / m

def _CCE_gradient(Y: np.ndarray, Yhat: np.ndarray) -> np.ndarray:
    """
    Computes the gradient of the categorical cross-entropy loss
    
    The gradient is computed as Yhat - Y.
    """
    return Yhat - Y

loss_functions = {
    "mean_squared_error" : (_mean_squared_error, _MSE_gradient),
    "mean_absolute_error": (_mean_absolute_error, _MAE_gradient),
    "huber_loss" : (_huber_loss, _huber_gradient),
    "binary_crossentropy": (_binary_crossentropy, _BCE_gradient),
    "categorical_crossentropy": (_categorical_crossentropy, _CCE_gradient)
}