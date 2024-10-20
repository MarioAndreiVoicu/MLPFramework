import numpy as np
from typing import Literal

"""
regularization.py

This module contains functions for implementing regularization in neural networks.
The supported regularization methods are L1, L2, L1_L2 (Elastic Net).

The module contains the following functions:
- set_regularization: Sets the regularization parameters for the model.
- _compute_regularization_term: Calculates the regularization penalty for the model's weights.
- _compute_gradient_regularization: Computes the gradient regularization term for the weights.


"""

def _set_regularization(model: object ,lambda_: float,
                               regularization_type: Literal["L1", "L2", "L1_L2"],
                               l1_l2_ratio: float) -> None:
    """
    Sets the regularization parameters for the model.
    
    Args:
        model: The neural network model.
        lambda_: The regularization parameter.
        regularization_type: The type of regularization to use.
        l1_l2_ratio: The ratio of L1 regularization to L2 regularization.
    
    Raises:
        ValueError: If an unsupported regularization type is provided.
    """
    if regularization_type not in ["L1", "L2", "L1_L2"]:
      raise ValueError(f"Unsupported regularization type: {regularization_type}. Supported regularization types are: L1, L2, L1_L2")
    
    model.regularization_type = regularization_type
    model.lambda_ = lambda_
    model.l1_l2_ratio = l1_l2_ratio
  
def _compute_regularization_term(model: object, m: int) -> float:
    """
    Calculates the regularization penalty for the model's weights

    This function iterates through each layer of the model and computes
    the regularization term (L1, L2, or a combination) for the model's weights.
    It adjusts the penalty according to the number of training examples.
      
    Args:
        model: The neural network model.
        m: The number of training examples.
        
    Returns:
        float: The regularization term for the model.
    """
    
    regularization_term = 0
    for layer in model.layers:
        if hasattr(layer, "W"):
            if model.regularization_type == "L1":
                regularization_term += np.sum(np.abs(layer.W)) * (model.lambda_ / m) 
            elif model.regularization_type == "L2":
                regularization_term += np.sum(np.square(layer.W)) * (model.lambda_ / (2 * m))
            elif model.regularization_type == "L1_L2":
                lambda_l1 = model.lambda_ * model.l1_l2_ratio
                lambda_l2 = model.lambda_ * (1 - model.l1_l2_ratio)
                regularization_term += (np.sum(np.abs(layer.W)) * (lambda_l1 / m) +
                                        np.sum(np.square(layer.W)) * (lambda_l2 / (2 * m)))
    return regularization_term

def _compute_gradient_regularization(layer: object, m: int,
                                     regularization_type: Literal["L1", "L2", "L1_L2"],
                                     lambda_: float, l1_l2_ratio: float) -> np.ndarray:
    """
    Compute the gradient regularization term for the weights.

    Parameters:
        layer: The layer instance.
        m: The number of training examples.
        regularization_type: The type of regularization to apply.
        lambda_: The regularization parameter.
        l1_l2_ratio: The ratio between L1 and L2 regularization when using "L1_L2" regularization.

    Returns:
        numpy.ndarray: The computed regularization term for the gradient.
    """
    regularization_term = 0
    if regularization_type == "L1":
        regularization_term += (lambda_ / m) * np.sign(layer.W)
    elif regularization_type == "L2":
        regularization_term += (lambda_ / m) * layer.W
    elif regularization_type == "L1_L2":
        lambda_l1 = lambda_ * l1_l2_ratio
        lambda_l2 = lambda_ * (1 - l1_l2_ratio)
        regularization_term += (lambda_l1 / m) * np.sign(layer.W) + (lambda_l2 / m) * layer.W
    return regularization_term