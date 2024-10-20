import numpy as np
import layers
import losses
import optimizers
import regularization
import callbacks
import metrics
import json
from typing import List, Optional, Literal, Generator, Tuple

"""
neural_network.py

This module defines a neural network model class that can be used to build, 
compile, train, and evaluate custom neural networks.

Key Features:
- Build custom neural networks using a list of layers.
- Compile models with various loss functions, optimizers, and metrics.
- Train the model using backpropagation and gradient descent.
- Evaluate model performance with a range of metrics and early stopping.
- Make predictions based on the trained model.
- Save and load models using JSON format.

Dependencies:
- numpy: For numerical operations and handling data in arrays.
- json: Used for saving and loading model configurations and weights.
"""

class NeuralNetwork:
    """
    Class representing a neural network model.
    
    Attributes:
        layers (list): A list of the layers in the model.
        compiled (bool): Whether the model has been compiled.
        inference_mode (bool): Whether the model is in inference mode.
        verbose (bool): Whether to print progress updates.
        metrics (Metrics): An instance of the Metrics class.
        main_metric (str): The metric displayed in the progress bar.
            This can be one of "train_cost", "val_cost", or "train_val_cost".
        loss_function (function): The loss function used by the model.
            This can be one of "binary_crossentropy", "categorical_crossentropy",
            "mean_squared_error", "mean_absolute_error"
        loss_gradient (function): The gradient of the loss function.
        regularization_type (str): The type of regularization used by the model.
            This can be "L1", "L2", or "L1_L2".
        lambda_ (float): The regularization parameter.
        l1_l2_ratio (float): The ratio of L1 regularization to L2 regularization.
            This is only used if regularization_type is "L1_L2".
    """
    
    def __init__(self, layers: object, verbose:bool = True):
        """
        Initialize the neural network model.
        
        Args:
            layers (list): A list of the layers in the model.
            verbose (bool): Whether to print progress updates.
        """
        self.layers = layers
        self.compiled = False
        self.inference_mode = False
        self.verbose = verbose

    def compile(
        self,
        loss_function: str,
        optimizer: str,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        lambda_: float = 0.0,
        regularization_type: str = "L1_L2",
        l1_l2_ratio: float = 0.5,
        metrics_list: Optional[List[str]] = None,
        main_metric: str = "train_cost",
        max_metric_computation_size: int = 5000,
        metrics_average_type: str = "macro",
        use_colors_metrics: bool = False,
        clip_gradients: bool = False,
        max_norm: float = 1.0
    ) -> None:
        """
        Compile the neural network model.
        
        Args:
            loss_function (str): The loss function used by the model.
                This can be one of "binary_crossentropy", "categorical_crossentropy",
                "mean_squared_error", "mean_absolute_error".
            optimizer (str): The optimizer used to train the model.
                This can be one of "adam", "rmsprop", "momentum", "sgd".
                An instance of the optimizer is created and stored in each layer separately.
            learning_rate (float): The learning rate used by the optimizer.
            beta1 (float): The exponential decay rate for the first moment estimates.
                Used by Adam, RMSprop, and Momentum optimizers.
            beta2 (float): The exponential decay rate for the second moment estimates.
                Used by Adam optimizer.
            epsilon (float): A small constant for numerical stability.
                Used by Adam and RMSprop optimizers.
            lambda_ (float): The regularization parameter.
            regularization_type (str): The type of regularization used by the model.
                This can be "L1", "L2", or "L1_L2".
            l1_l2_ratio (float): The ratio of L1 regularization to L2 regularization.
                This is only used if regularization_type is "L1_L2".
            metrics_list (list): A list of metrics to evaluate the model on.
                This can contain any of "accuracy", "precision", "recall",
                "f1_score", "mae", "mse".
            main_metric (str): The metric displayed in the progress bar.
                This can be one of "train_cost", "val_cost", or "train_val_cost".
            max_metric_computation_size: The maximum number of examples to use
                for metric computation. This is used to speed up the computation
                of metrics for large datasets.
            metrics_average_type (str): The type of averaging used for the metrics.
                This can be "macro" or "micro".
            use_colors_metrics (bool): Whether to use colored text for the metrics.
                If metrics are displayed, setting this to True makes each metric
                have a different color, making it easier to distinguish them.
            clip_gradients (bool): Whether to clip the gradients during training.
            max_norm (float): The maximum norm value for gradient clipping.
        """
        losses._set_loss(self, loss_function=loss_function)
        optimizers._set_optimizers(self, optimizer_name=optimizer, learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon, clip_gradients=clip_gradients, max_norm=max_norm)
        regularization._set_regularization(self, lambda_=lambda_, regularization_type=regularization_type, l1_l2_ratio=l1_l2_ratio)
        self.metrics = metrics.Metrics(model=self, metrics_list=metrics_list, main_metric=main_metric, max_dataset_size=max_metric_computation_size, average=metrics_average_type, use_colors=use_colors_metrics)
        self.compiled = True
                
    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
        use_early_stopping: bool = False,
        early_stopping_metric: Literal["cost", "accuracy", "precision", "recall",
                                       "f1_score", "mae", "mse"] = "cost",
        patience: int = 5,
    ) -> None:
        """
        Train the neural network model.
        
        Args:
            X_train: The training data with shape (num_samples, num_features).
                - num_samples is the number of training examples in the data.
                - num_features is the number of features in the input data.
            Y_train: The training labels with shape (num_samples, output_size).
                - output_size is the number of classes for classification tasks
                or the number of output units for regression tasks.
            epochs: The number of epochs to train the model for.
            batch_size: The size of each batch.
            X_val: The validation data.
            Y_val: The validation labels.
            use_early_stopping: Whether to use early stopping.
            early_stopping_metric: The metric to use for early stopping.
                Default is "cost", but can be any of "accuracy", "precision",
                "recall", "f1_score", "mae", "mse".If this is set to "cost",
                the validation cost will be used for early stopping if validation
                data is provided, otherwise the training cost will be used.
            patience: The number of epochs to wait before stopping training.
        
        Raises:
            ValueError: If the model has not been compiled.
        """
        if not self.compiled:
            raise ValueError("Model must be compiled before training.")

        has_validation_data = X_val is not None and Y_val is not None
        
        # Model works internally with data in shape (num_features, num_samples)
        X_train, Y_train = self._transpose_data(X_train, Y_train)
        if has_validation_data:
           X_val, Y_val =  self._transpose_data(X_val, Y_val)

        if use_early_stopping:
            early_stopping = callbacks.EarlyStopping(self, metric=early_stopping_metric, patience=patience,has_validation_data=has_validation_data, verbose=self.verbose)

        batch_size, num_batches = self._get_batch_info(batch_size, X_train, verbose=self.verbose)

        progress_bar = callbacks.ProgressBar(num_batches, epochs) if self.verbose else None
  
        for epoch in range(1, epochs + 1):
            current_batch_num = 0
            for X_batch, Y_batch in self._generate_batches(X_train, Y_train, batch_size):
                AL = self._forward_propagation(X_batch)
                cost = self.metrics._compute_cost(Y_batch, AL)
                self._backward_propagation(Y_batch, AL)
                
                self.metrics.costs["train"].append(cost)
                current_batch_num += 1  
                progress_bar._update(epoch, current_batch_num, with_metric=False) if self.verbose else None

            if has_validation_data: 
                self.metrics._evaluate_validation_cost(self, X_val, Y_val)
            if self.verbose:
                metric_value = self.metrics._get_main_metric(self, has_validation_data, X_train, Y_train)
                progress_bar._update(epoch, current_batch_num, with_metric=True, metric_name=self.main_metric, metric_value=metric_value)
                progress_bar._finish()            
            
            if self.metrics.has_non_cost_metrics:
                self.metrics._evaluate_model_metrics(self, X_train, Y_train, has_validation_data, X_val, Y_val)
                            
            if use_early_stopping and early_stopping._should_stop(
                    X_val if has_validation_data else X_train, 
                    Y_val if has_validation_data else Y_train):
                break
            
    def predict(self, X: np.ndarray, threshold: float = 0.5,
                transpose_data: bool = True) -> np.ndarray:
        """
        Generate predictions for the input data.
        
        Automatically determines the appropriate output format based on the
        activation function of the output layer.
        For sigmoid activation, the output is binary (0 or 1).
        For softmax activation, the output is the class with the highest probability.
        For linear activation, the output is the raw prediction.
        For tanh activation, the output is the tanh of the raw prediction.
        For relu activation, the output is the max of 0 and the raw prediction.
        
        Args:
            X: The input data. Expected dimensions are (num_samples, num_features).
            threshold: The threshold for binary classification.
            transpose_data: Whether to transpose the input data.If the data is
                already in the shape (num_features, num_samples), set this to False.
                The model works internally with data in the shape
                (num_features, num_samples) so the default value is True because
                the input data is usually in the shape (num_samples, num_features).
            
        Returns:
            np.ndarray: The model's predictions.
        """
        
        if transpose_data:
            X = X.T.copy() # Model works internally with data in shape (num_features, num_samples)
        
        self.inference_mode = True
        AL = self._forward_propagation(X)
        activation = self.layers[-1].activation.__name__

        if activation == "_sigmoid":
            return (AL > threshold).astype(int)
        elif activation == "_softmax":
            return np.argmax(AL, axis=0)
        elif activation == "_linear":
            return AL
        elif activation == "_tanh":
            return np.tanh(AL)
        elif activation == "_relu":
            return np.maximum(0, AL)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
    def summary(self, detailed: bool = False) -> None:
        """
        Print a summary of the model.
        
        Args:
            detailed (bool): Whether to include detailed layer information.
                If False, the summary will include the layer type, output shape,
                and number of parameters.
                If True, the summary will include additional details:
                loss function, optimizer, learning_rate, regularization_type,
                metrics information, max_metric_computation_size,
                gradient_clipping, max_norm.
                
        Raises:
            ValueError: If the model has not been compiled.
        """
        if not self.compiled:
            raise ValueError("Model must be compiled before printing the summary.")
        
        total_parameters = 0
        print("\nModel Summary")
        print("--------------------------------------------------------------------------------")
        print(f"{'Layer Type':<30} {'Output Shape':<30} {'Parameter Count':>15}")
        print("================================================================================")
        for layer in self.layers:
            if layer.__class__.__name__ == "DropoutLayer":
                layer_type = "Dropout"
                param_count = 0
                # output_shape will be the same as the previous layer
            else: 
                layer_type, output_shape, param_count = layer._summary()
            total_parameters += param_count
            print(f"{layer_type:<30} {str(output_shape):<30} {param_count:>15}")
        print(f"Total Parameters: {total_parameters}")
        print("================================================================================")
        if detailed:
            for layer in self.layers:
                if isinstance(layer, layers.DenseLayer):
                    layer_instance = layer
                    break
            print("Loss Function:", self.loss_function.__name__.lstrip("_"))
            print("Optimizer:", layer_instance.optimizer.__class__.__name__)
            print("Learning Rate:", layer_instance.optimizer.learning_rate)
            print("Regularization Type:", self.regularization_type)
            print("Lambda:", self.lambda_)
            if self.regularization_type == "L1_L2":
                print("L1_L2 Ratio:", self.l1_l2_ratio)
            if self.metrics.has_non_cost_metrics:
                print(f"Metrics: {', '.join(self.metrics.metrics_values.keys())}")
                print("Metrics Average Type:", self.metrics.average)
                print("Use Colors for Metrics:", self.metrics.use_colors)
            else:
                print("Non Cost Metrics: None")
            print("Main Metric:", self.main_metric)
            print(f"Max Metric Dataset Size: {self.metrics.max_dataset_size}")
            print(f"Gradient Clipping: {layer_instance.optimizer.clip_gradients}")
            if layer_instance.optimizer.clip_gradients:
                print(f"Max Norm: {layer_instance.optimizer.max_norm}")
        
    def get_metrics(self):
        """
        Get the metrics for the model.
        
        Returns:
            dict: A dictionary containing the model's metrics.
        """
        if self.metrics.has_non_cost_metrics is False:
            raise ValueError("No metrics have been defined for this model.")
        return self.metrics._get_metrics()
    
    def get_costs(self):
        """
        Get the costs for the model.
        
        Returns:
            dict: A dictionary containing the model's costs with the following keys:
                - "train": Training costs for each batch (always computed).
                - "validation": Validation costs after each epoch.
                    (if validation data is provided).
                - "epoch_train": Training costs after each epoch.
                    (if applicable based on main_metric).
        """
        return self.metrics.costs
    
    def save_model(self, filename: str) -> None:
        """
        Save the model to a JSON file.
        
        Args:
            filename (str): The name of the file to save the model to.
            
        Raises:
            ValueError: If the model has not been compiled.
        """
        if not self.compiled:
            raise ValueError("Model must be compiled before saving.")
        
        for layer in self.layers:
            if isinstance(layer, layers.DenseLayer):
                layer_instance = layer
                break
            
        model_data = {
            'layers': [],
            'verbose': self.verbose,
            'loss_function': self.loss_function.__name__.lstrip('_'),
            'optimizer_name': layer_instance.optimizer.__class__.__name__.lower(),
            'learning_rate': layer_instance.optimizer.learning_rate,
            'beta1': layer_instance.optimizer.beta1,
            'beta2': layer_instance.optimizer.beta2,
            'epsilon': layer_instance.optimizer.epsilon,
            'clip_gradients': layer_instance.optimizer.clip_gradients,
            'max_norm': layer_instance.optimizer.max_norm,
            'lambda_': self.lambda_,
            'regularization_type': self.regularization_type,
            'l1_l2_ratio': self.l1_l2_ratio,
            'metrics_list': list(self.metrics.metrics_values.keys()) if self.metrics.has_non_cost_metrics else None,
            'main_metric': self.main_metric,
            'max_metric_computation_size': self.metrics.max_dataset_size,
            'metrics_average_type': self.metrics.average,
            'use_colors_metrics': self.metrics.use_colors,
        }
        
        for layer in self.layers:
            layer_data = {
                'type': layer.__class__.__name__,
                'input_size': layer.input_size if hasattr(layer, 'input_size') else None,
                'output_size': layer.output_size if hasattr(layer, 'output_size') else None,
                'activation': layer.activation.__name__.lstrip('_') if hasattr(layer, 'activation') else None,
                'weights': layer.W.tolist() if hasattr(layer, 'W') else None,
                'biases': layer.b.tolist() if hasattr(layer, 'b') else None,
                'dropout_rate': layer.dropout_rate if hasattr(layer, 'dropout_rate') else None,
            }
            model_data['layers'].append(layer_data)
        
        with open(filename, 'w') as json_file:
            json.dump(model_data, json_file)

    
    @staticmethod
    def load_model(filename: str) -> 'NeuralNetwork':
        """
        Load a model from a JSON file.
        
        Args:
            filename (str): The name of the file to load the model from.
            
        Returns:
            NeuralNetwork: The loaded model.
        """
        with open(filename, 'r') as json_file:
            model_data = json.load(json_file)

        layers_list = []
        for layer_data in model_data['layers']:
            layer_type = layer_data['type']
            input_size = layer_data['input_size']
            output_size = layer_data['output_size']
            activation = layer_data['activation']
            weights = layer_data['weights']
            biases = layer_data['biases']
            dropout_rate = layer_data['dropout_rate']
            if layer_type == 'DenseLayer':
                layer = layers.DenseLayer(input_size, output_size, activation)
                layer.W = np.array(weights)
                layer.b = np.array(biases)
            elif layer_type == 'DropoutLayer':
                layer = layers.DropoutLayer(dropout_rate)
            layers_list.append(layer)

        model = NeuralNetwork(layers_list, verbose=model_data['verbose'])
        model.compile(
            loss_function=model_data['loss_function'],
            optimizer=model_data['optimizer_name'],
            learning_rate=model_data['learning_rate'],
            beta1=model_data['beta1'],
            beta2=model_data['beta2'],
            epsilon=model_data['epsilon'],
            lambda_=model_data['lambda_'],
            regularization_type=model_data['regularization_type'],
            l1_l2_ratio=model_data['l1_l2_ratio'],
            metrics_list=model_data['metrics_list'],
            main_metric=model_data['main_metric'],
            max_metric_computation_size=model_data['max_metric_computation_size'],
            metrics_average_type=model_data['metrics_average_type'],
            use_colors_metrics=model_data['use_colors_metrics'],
            clip_gradients=model_data['clip_gradients'],
            max_norm=model_data['max_norm']
        )
        
        return model


    def _generate_batches(self, X: np.ndarray, Y: np.ndarray, batch_size: int
                          ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate batches of data for training.
        
        Args:
            X: The input data.
            Y: The target labels.
            batch_size: The size of each batch.
            
        Yields:
            tuple: A tuple containing the input data and target labels for each batch.
        """
        num_samples = X.shape[1]
        permutation = np.random.permutation(num_samples)
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[:, permutation]
        for i in range(0, num_samples, batch_size):
            yield X[:, i : i + batch_size], Y[:, i : i + batch_size]
    
    def _get_batch_info(self, batch_size: int, X_train: np.ndarray, verbose: bool = True) -> Tuple[int, int]:
        """
        Validate the batch size and calculate the number of batches.

        Args:
            batch_size (int): The size of each batch.
            X_train (np.ndarray): The training data.
            verbose (bool): Whether to print warnings.

        Returns:
            tuple: A tuple containing the validated batch size and the number of batches.
        """
        num_samples = X_train.shape[1]
        
        if batch_size <= 0:
            raise ValueError(f"Batch size must be a positive integer. Got {batch_size}.")
        if batch_size > num_samples:
            if verbose:
                print(f"Warning: Batch size ({batch_size}) is larger than the number of samples ({num_samples}).Batch size automatically set to {num_samples}.")
            batch_size = num_samples
        
        num_batches = int(np.ceil(num_samples / batch_size))
        
        return batch_size, num_batches  
    
    def _transpose_data(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Makes a copy and transposes the input data and target labels.
        
        The model works internally with the input data in the shape
        (num_features, num_samples) and the target labels in the shape
        (output_size, num_samples). But it takes input data in the shape
        (num_samples, num_features) and target labels in the shape
        (num_samples, output_size). This method transposes the data
        to the required shape.
        
        Args:
            X: The input data.
            Y: The target labels.
        
        Returns:
            tuple: A tuple containing a copy of the transposed input data and target labels.
        """
        return X.T.copy(), Y.T.copy()
    
    def _forward_propagation(self, X: np.ndarray) -> np.ndarray:
        """
        Perform forward propagation through the model.
        
        Args:
            X: The input data.
        
        Returns:
            np.ndarray: The model's result after forward propagation.
        """
        A = X
        for layer in self.layers:
            if isinstance(layer, layers.DropoutLayer):
                layer._set_inference_mode(self.inference_mode)
            A = layer._forward_propagation(A)
        return A

    def _backward_propagation(self, Y: np.ndarray, AL: np.ndarray) ->None:
        """
        Perform backward propagation through the model.
        
        Args:
            Y: The target labels.
            AL: The model's output after forward propagation.
        """
        dA = self.loss_gradient(Y, AL)
        for layer in reversed(self.layers):
            if isinstance(layer, layers.DenseLayer):
                dA = layer._backward_propagation(dA=dA, model=self)
            elif isinstance(layer, layers.DropoutLayer):
                dA = layer._backward_propagation(dA)