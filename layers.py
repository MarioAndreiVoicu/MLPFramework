import numpy as np
import activations
import regularization
from typing import Literal, Tuple, Optional

"""
layers.py

This module defines various layers for a Multilayer Perceptron, including:
- DenseLayer: A fully connected layer.
- DropoutLayer: A layer that applies dropout for regularization.

Each layer supports various activation functions and weight initialization methods.
"""

class DenseLayer:
    """
    A dense (fully connected) layer for a Multilayer Perceptron.

    Attributes:
        input_size (int): Number of input features.
        output_size (int): Number of output neurons.
        b (np.ndarray): Bias for the layer.
        W (np.ndarray): Weights for the layer.
        activation (Callable): Activation function for the layer.
        activation_derivative (Callable): Derivative of the activation function.
        _activation_name (str): The name of the activation function
        alpha (float, optional): Parameter for PReLU (default 0.01) or ELU (default 1.0). Ignored for other activations.
        beta (float, optional): Parameter for Swish (default 1.0). Ignored for other activations.
        optimizer (object): The optimizer used to update weights and biases.
    """
    
    def __init__(self, input_size: int,
                 output_size: int,
                 activation: Literal["relu", "tanh", "sigmoid", "softmax", "linear", "prelu", "elu", 'swish'] = "linear",
                 alpha: Optional[float] = None,
                 beta: Optional[float] = None,
                 init_method: Literal["auto", "he", "xavier", "random"] = "auto"
                 ) -> None:
        """
        Initializes the dense layer with weights, biases, and activation.

        Args:
            input_size: Number of input features.
            output_size: Number of output neurons.
            activation: The activation function to use. Defaults to "relu".
                The parameter is a string and the class attribute is a function.
            alpha: Parameter for PReLU (default 0.01) or ELU (default 1.0). Ignored for other activations.
            beta: Parameter for Swish (default 1.0). Ignored for other activations.
            init_method: The method for weight initialization. Defaults to "auto".
                On "auto", the method is chosen based on the activation function.
                For "relu", "he" initialization is used. For "tanh", "sigmoid"
                and "softmax", "xavier" initialization is used. For other
                activation functions, "random" initialization is used.
                Random initialization uses small random values for weights.
        """
        self.input_size = input_size
        self.output_size = output_size
        activations._set_activation(self, activation, alpha, beta)
        _initialize_parameters(self, input_size, output_size,layer_type="dense", init_method=init_method)
      
    def _forward_propagation(self, A_prev: np.ndarray) -> np.ndarray:
        """
        Performs forward propagation through the dense layer.

        Args:
            A_prev: Activations from the previous layer.

        Returns:
            np.ndarray: Output activations after applying the weights, biases,
                and activation function.
        """
        self.A_prev: np.ndarray = A_prev
        self.Z: np.ndarray = np.dot(self.W, A_prev) + self.b
        A: np.ndarray = self.activation(self.Z)
        return A
 
    def _backward_propagation(self, dA: np.ndarray, model: object) -> np.ndarray:
        """
        Performs backpropagation through the dense layer.

        Args:
            dA: Gradient of the loss with respect to the output.
            model: The Multilayer Perceptron containing the layer.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input.
        """
        m: int = dA.shape[1]
    
        dZ: np.ndarray = dA * self.activation_derivative(self.Z)
        dW: np.ndarray = np.dot(dZ, self.A_prev.T) / m
        db: np.ndarray = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev: np.ndarray = np.dot(self.W.T, dZ)
    
        if model.lambda_ > 0:
            regularization_term = regularization._compute_gradient_regularization(self, m, model.regularization_type, model.lambda_, model.l1_l2_ratio)
            dW += regularization_term
        
        self.optimizer._update(self, dW, db)
    
        return dA_prev
      
    def _summary(self) -> Tuple[str, Tuple[str, int], int]:
        """
        Generates a summary of the dense layer to be used in the model summary.
        
        Returns:
            A tuple containing the layer type, output size, and number of parameters.
        """ 
        layer_type = "Dense"
        output_size = ("batch size", self.output_size)
        num_params = (self.input_size + 1) * self.output_size
        return layer_type, output_size, num_params
      
class DropoutLayer:
    """
    A dropout layer for a Multilayer Perceptron.
    
    Attributes:
        dropout_rate: Probability of dropping a neuron. Defaults to 0.5.
            Values should be between 0 and 1.A value of 0 means no neurons are dropped.
            A value of 1 means all neurons are dropped.
        mask: Binary mask used during training.
        inference: Flag to indicate inference mode (no dropout applied).
            In training mode, inference is set to False because dropout is
            applied. In inference mode, dropout is disabled because the model
            should use all neurons for prediction.
    """
    def __init__(self, dropout_rate: float = 0.5) -> None:
        """
        Initializes the Dropout layer with a given dropout rate.

        Args:
            dropout_rate: Probability of dropping neurons.Defaults to 0.5.
                Values should be between 0 and 1.
                
        Raises:
            ValueError: If the dropout rate is outside the range [0, 1).
        """
        self._set_dropout_rate(dropout_rate)
        self.mask: np.ndarray = None
        self.inference: bool = False
        
    def _set_dropout_rate(self, dropout_rate: float) -> None:
        """
        Sets the dropout rate for the layer.

        Args:
            dropout_rate: Probability of dropping neurons.

        Raises:
            ValueError: If the dropout rate is outside the range [0, 1).
        """
        if dropout_rate < 0 or dropout_rate >= 1:
            raise ValueError("Dropout rate must be in the range [0, 1).")
        self.dropout_rate: float = dropout_rate
    
    def _forward_propagation(self, A_prev: np.ndarray) -> np.ndarray:
        """
        Performs forward propagation through the dropout layer.
        
        Applies the dropout mask to the activations during training.

        Args:
            A_prev: Activations from the previous layer.

        Returns:
            np.ndarray: The modified activations after applying dropout.
        """
        
        if self.inference:
            return A_prev
        self.mask = (np.random.rand(*A_prev.shape) >= self.dropout_rate) / (1.0 - self.dropout_rate)
        return A_prev * self.mask
  
    def _backward_propagation(self, dA: np.ndarray) -> np.ndarray:
        """
        Performs backpropagation through the dropout layer.
        
        Applies the dropout mask to the gradient during training.

        Args:
            dA: Gradient of the loss with respect to the output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input.

        Raises:
            ValueError: If the forward propagation hasn't been called, and
                the mask is not available.
        """
        
        if self.mask is None:
            raise ValueError("Mask not found. Please ensure forward propagation is called before backpropagation")
        return dA * self.mask
    
    def _summary(self) -> Tuple[str, int, int]:
        """
        Generates a summary of the dense layer to be used in the model summary.
        
        Returns:
            A tuple containing the layer type, output size, and number of parameters.
        """ 
        layer_type = "Dropout"
        output_size = self.output_size
        num_params = 0
        return layer_type, output_size, num_params
  
    def _set_inference_mode(self, inference: bool) -> None:
        """
        Sets the inference mode for the dropout layer.

        Args:
            inference (bool): If True, disables dropout for inference.
        """
        self.inference = inference
    
def _initialize_parameters(layer: object, input_size: int, output_size: int,
                           layer_type: Literal["dense", "conv"],
                           init_method: Literal["auto", "he", "xavier", "random"]
                           ) -> None:
    """
    Initializes parameters (weights and biases) for a layer.

    Args:
        layer: The layer to initialize parameters for.
        input_size: The size of the input to the layer.
        output_size: The size of the output from the layer.
        layer_type: The type of layer (e.g., "dense", "conv").
        init_method: The method for weight initialization. Defaults to "auto".

    Raises:
        ValueError: If the provided initialization method is not supported.
    """
    
    if init_method == "auto":
        if layer._activation_name in ("relu", "prelu", "elu", "swish"):
            init_method = "he"
        elif layer._activation_name in ("linear", "tanh", "sigmoid", "softmax"):
            init_method = "xavier"
        else:
            init_method = "random"

    if layer_type == "dense":
        layer.b = np.zeros((output_size, 1))
        if init_method == "he":
            layer.W = np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)
        elif init_method == "xavier":
            layer.W = np.random.randn(output_size, input_size) * np.sqrt(1 / input_size)
        elif init_method == "random":
            layer.W = np.random.randn(output_size, input_size) * 0.01
        else:
            raise ValueError(f"Unsupported initialization method '{init_method}'. Supported methods are 'auto', 'he', 'xavier', and 'random'.")
    elif layer_type == "conv":
        pass