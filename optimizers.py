import numpy as np
from typing import Tuple, Literal

"""
optimizers.py

This module defines various optimization algorithms used for updating the parameters 
of the multilayer perceptron layers during training.

Supported optimizers include:
- Stochastic Gradient Descent (SGD)
- Adam
- RMSprop
- Momentum

The module also includes functionality for gradient clipping and setting optimizers 
for the layers in a model.
"""

class Optimizer:
    """
    Base class for all optimizers.
    
    Attributes:
        learning_rate (float): The learning rate.
        clip_gradients (bool): Whether to apply gradient clipping.
        max_norm (float): The maximum norm to use for gradient clipping.
    """
    def __init__(self, learning_rate: float = 0.001, clip_gradients: bool = False,
                 max_norm: float = 1.0) -> None:
        """
        Initializes the optimizer with the specified parameters.
        
        Args:
            learning_rate: The learning rate.
            clip_gradients: Whether to apply gradient clipping.
            max_norm: The maximum norm to use for gradient clipping
        """
        self.learning_rate = learning_rate
        self.clip_gradients = clip_gradients
        self.max_norm = max_norm

    def _apply_gradient_clipping(self, dW: np.ndarray, db: np.ndarray
                                 )-> Tuple[np.ndarray, np.ndarray]:
        """
        Applies gradient clipping to the gradients of the weights and biases.
        
        The gradients are clipped only if their norm exceeds the maximum norm.
        
        Args:
            dW: The gradient of the weights.
            db: The gradient of the biases.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: The clipped gradients for the weights and biases.
        """
        epsilon = 1e-7
        norm_dW = np.linalg.norm(dW)
        norm_db = np.linalg.norm(db)

        if norm_dW > self.max_norm:
            scaling_factor_dW = self.max_norm / (norm_dW + epsilon)
            dW *= scaling_factor_dW

        if norm_db > self.max_norm:
            scaling_factor_db = self.max_norm / (norm_db + epsilon)
            db *= scaling_factor_db

        return dW, db

    def _update(self, layer: object, dW: np.ndarray, db: np.ndarray) -> None:
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("This method should be overridden by subclasses.")

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, learning_rate: float = 0.001, clip_gradients: bool = False, max_norm: float = 1.0) -> None:
        super().__init__(learning_rate, clip_gradients, max_norm)
        """
        Initializes the SGD optimizer with the specified parameters.
        
        Args:
            learning_rate: The learning rate.
            clip_gradients: Whether to apply gradient clipping.
            max_norm: The maximum norm to use for gradient clipping.
        """

    def _update(self, layer: object, dW: np.ndarray, db: np.ndarray) -> None:
        """Updates the weights and biases using SGD.
        
        Clipping is applied if enabled.
        
        Args:
            layer: The layer to update.
            dW: The gradient of the weights.
            db: The gradient of the biases.
        """
        if self.clip_gradients:
            dW, db = self._apply_gradient_clipping(dW, db)
        
        layer.W -= self.learning_rate * dW
        layer.b -= self.learning_rate * db

class Adam(Optimizer):
    """
    Adam optimizer.
    
    Adam is designed to compute adaptive learning rates for each parameter based 
    on the first and second moments of the gradients.

    Attributes:
        t (int): The time step.
        beta1 (float): The exponential decay rate for the first moment estimates.
        beta2 (float): The exponential decay rate for the second moment estimates.
        epsilon (float): A small value to prevent division by zero.
        m_W (numpy.ndarray): The first moment estimate for the weights.
        m_b (numpy.ndarray): The first moment estimate for the biases.
        v_W (numpy.ndarray): The second moment estimate for the weights.
        v_b (numpy.ndarray): The second moment estimate for the biases.
    """
    def __init__(self, layer: object, learning_rate: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8,
                 clip_gradients: bool = False, max_norm: float = 1.0) -> None:
        super().__init__(learning_rate, clip_gradients, max_norm)
        """
        Initializes the Adam optimizer with the specified parameters.
        
        Args:
            layer: The layer to optimize.
            learning_rate: The learning rate.
            beta1: The exponential decay rate for the first moment estimates.
            beta2: The exponential decay rate for the second moment estimates.
            epsilon: A small value to prevent division by zero.
            clip_gradients: Whether to apply gradient clipping.
            max_norm: The maximum norm to use for gradient clipping.
        """
        self.t = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_W = np.zeros_like(layer.W)
        self.m_b = np.zeros_like(layer.b)
        self.v_W = np.zeros_like(layer.W)
        self.v_b = np.zeros_like(layer.b)

    def _update(self, layer: object, dW: np.ndarray, db: np.ndarray) -> None:
        """
        Updates the weights and biases using the Adam optimizer.
        
        Clipping is applied if enabled.
        
        Args:
            layer: The layer to update.
            dW: The gradient of the weights.
            db: The gradient of the biases.
        """
        if self.clip_gradients:
            dW, db = self._apply_gradient_clipping(dW, db)
        
        self.t += 1
        
        self.m_W = self.beta1 * self.m_W + (1 - self.beta1) * dW
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db
        self.v_W = self.beta2 * self.v_W + (1 - self.beta2) * dW**2
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * db**2

        m_W_hat = self.m_W / (1 - self.beta1**self.t)
        m_b_hat = self.m_b / (1 - self.beta1**self.t)
        v_W_hat = self.v_W / (1 - self.beta2**self.t)
        v_b_hat = self.v_b / (1 - self.beta2**self.t)

        layer.W -= self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
        layer.b -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

class RMSprop(Optimizer):
    """
    RMSprop optimizer.
    
    RMSprop is an adaptive learning rate optimization algorithm that divides
    the learning rate by an exponentially decaying average of squared gradients.
    
    Attributes:
        beta (float): The exponential decay rate.
        epsilon (float): A small value to prevent division by zero.
        cache_W (numpy.ndarray): The cache for the squared gradients of the weights.
        cache_b (numpy.ndarray): The cache for the squared gradients of the biases.
        """
    def __init__(self, layer: object, learning_rate: float = 0.001,
                 beta: float = 0.9, epsilon: float = 1e-8,
                 clip_gradients: bool = False, max_norm: float = 1.0) -> None:
        super().__init__(learning_rate, clip_gradients, max_norm)
        """
        Initializes the RMSprop optimizer with the specified parameters.
        
        Args:
            layer: The layer to optimize.
            learning_rate: The learning rate.
            beta: The exponential decay rate.
            epsilon: A small value to prevent division by zero.
            clip_gradients: Whether to apply gradient clipping.
            max_norm: The maximum norm to use for gradient clipping.
        """
        self.beta = beta
        self.epsilon = epsilon
        self.cache_W = np.zeros_like(layer.W)
        self.cache_b = np.zeros_like(layer.b)

    def _update(self, layer: object, dW: np.ndarray, db: np.ndarray) -> None:
        """
        Updates the weights and biases using the RMSprop optimizer.
        
        Clipping is applied if enabled.
        
        Args:
            layer: The layer to update.
            dW: The gradient of the weights.
            db: The gradient of the biases.
        """
        if self.clip_gradients:
            dW, db = self._apply_gradient_clipping(dW, db)
        
        self.cache_W = self.beta * self.cache_W + (1 - self.beta) * dW**2
        self.cache_b = self.beta * self.cache_b + (1 - self.beta) * db**2

        layer.W -= self.learning_rate * dW / (np.sqrt(self.cache_W) + self.epsilon)
        layer.b -= self.learning_rate * db / (np.sqrt(self.cache_b) + self.epsilon)

class Momentum(Optimizer):
    """
    Momentum optimizer.
    
    Momentum is a method that helps accelerate SGD in the relevant direction
    and dampens oscillations.
    
    Attributes:
        beta (float): The momentum parameter.
        v_W (numpy.ndarray): The velocity for the weights.
        v_b (numpy.ndarray): The velocity for the biases.
        """   
    def __init__(self, layer: object, learning_rate: float = 0.001,
                 beta: float = 0.9, clip_gradients: bool = False,
                 max_norm: float = 1.0) -> None:
        super().__init__(learning_rate, clip_gradients, max_norm)
        """
        Initializes the Momentum optimizer with the specified parameters.
        
        Args:
            layer: The layer to optimize.
            learning_rate: The learning rate.
            beta: The momentum parameter.
            clip_gradients: Whether to apply gradient clipping.
            max_norm: The maximum norm to use for gradient clipping.
        """
        self.beta = beta
        self.v_W = np.zeros_like(layer.W)
        self.v_b = np.zeros_like(layer.b)

    def _update(self, layer:object, dW: np.ndarray, db: np.ndarray) -> None:
        """
        Updates the weights and biases using the Momentum optimizer.
        
        Clipping is applied if enabled.
        
        Args:
            layer: The layer to update.
            dW: The gradient of the weights.
            db: The gradient of the biases.
        """
        if self.clip_gradients:
            dW, db = self._apply_gradient_clipping(dW, db)
        
        self.v_W = self.beta * self.v_W + (1 - self.beta) * dW
        self.v_b = self.beta * self.v_b + (1 - self.beta) * db

        layer.W -= self.learning_rate * self.v_W
        layer.b -= self.learning_rate * self.v_b

def _set_optimizers(model: object,
                            optimizer_name: Literal["sgd", "adam", "momentum", "rmsprop"],
                            learning_rate: float = 0.001, beta1: float = 0.9,
                            beta2: float = 0.999, epsilon: float = 1e-8,
                            clip_gradients: bool = False, max_norm: float = 1.0
                            ) -> None:
    """Sets the optimizer for each layer in the model."""
    for layer in model.layers:
        if not hasattr(layer, "W") or not hasattr(layer, "b"):
            continue
        
        if optimizer_name == "sgd":
            layer.optimizer = SGD(learning_rate, clip_gradients, max_norm)
        elif optimizer_name == "adam":
            layer.optimizer = Adam(layer, learning_rate, beta1, beta2, epsilon, clip_gradients, max_norm)
        elif optimizer_name == "momentum":
            layer.optimizer = Momentum(layer, learning_rate, beta1, clip_gradients, max_norm)
        elif optimizer_name == "rmsprop":
            layer.optimizer = RMSprop(layer, learning_rate, beta1, clip_gradients, max_norm)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}. Supported optimizers are: sgd, adam, momentum, rmsprop")
