import numpy as np
import regularization

from typing import List, Literal, Union, Tuple, Dict, Callable

"""
metrics.py

This module contains the Metrics class which is used to compute and display
metrics for a neural network model. The class supports various metrics such as
accuracy, precision, recall, F1 score, mean absolute error, and mean squared error,
costs, and the main metric for the model. The metrics are computed during training
and displayed in the progress bar. The class also supports colored output for
displaying metrics.

Attributes:
    - model: A neural network model instance from the neural_network module.
    - costs: Dictionary storing training and validation costs, updated during training.
    - metrics_functions: Dictionary mapping metric names to their computation functions.
    - metrics_values: Dictionary storing computed metrics, updated during training.
    - has_non_cost_metrics: Boolean indicating whether non-cost metrics are computed.
    - average: The averaging method for multiclass metrics (macro or micro).
    - use_colors: Whether to use colored output for metrics display.


Classes:
    Metrics: A class to compute, store, and display performance metrics for a 
    neural network model during the training process.
"""

class Metrics:
    """
    A class to compute and display metrics for a neural network model.
    
    Attributes:
        model (NeuralNetwork): The neural network model.
        costs (dict): Dictionary to store the training and validation costs.
            It has the following keys: "train", "validation", "epoch_train".
            "train" stores the training costs for each batch, which are always
            computed during training.
            "validation" stores the validation costs after each epoch, only if
            validation data is provided.
            "epoch_train" stores the training costs after each epoch only if the
            main_metric is either "train_cost" or "train_val_cost".
        metrics_functions (dict): Dictionary of metric functions to compute
            various metrics. The supported metrics are: "accuracy", "precision",
            "recall", "f1_score", "mae", "mse".
        metrics_values (dict): Dictionary to store the computed metrics. The keys
            are the metric names and the values are lists of metric values.
            These values are computed during training after each epoch on the
            validation data if provided, or on the training data otherwise.
        max_dataset_size: The maximum number of examples to use for metric
            computation. This is used to speed up the computation of metrics
            for large datasets.
        has_non_cost_metrics (bool): Whether there are non-cost metrics to compute.
        average (str): The type of averaging to perform for the metrics.
        use_colors (bool): Whether to use colored output for displaying metrics.
            If True, each metric is displayed in a different color, otherwise
            the metrics are displayed in the default color.
        """
    def __init__(self, model: object,
                 metrics_list: List[Literal["accuracy", "precision", "recall",
                                            "f1_score", "mae", "mse"]] = None,
                 main_metric: Literal["train_cost", "val_cost",
                                      "train_val_cost"] = "train_cost",
                 max_dataset_size: int = 5000,
                 average: Literal["macro", "micro"] = "macro",
                 use_colors: bool = False) -> None:
        """
        Initializes the metrics for the neural network model.

        Args:
            model: The neural network model.
            metrics_list: List of strings specifying the metrics to compute.
                Supported metrics are: "accuracy", "precision", "recall",
                "f1_score", "mae", "mse".
            main_metric: The name of the main metric to use for the model.
                This metric is computed during training and it is displayed
                in the progress bar. Supported metrics are: "train_cost",
                "val_cost", "train_val_cost". Defaults to "train_cost".
            max_dataset_size: The maximum number of examples to use for metric
                computation. This is used to speed up the computation of
                metrics for large datasets.
            average: The type of averaging to perform for the metrics.
                Supported values are: "macro", "micro". Defaults to "macro".
                Macro averaging computes the metric independently for each class
                and then takes the average.
                Micro averaging computes the metric globally by counting the
                total true positives, false negatives, and false positives.
            use_colors: Whether to use colored output for displaying metrics.
                Defaults to False.
        """
        self.model: object = model
        self._set_main_metric(self.model, main_metric)
        self.costs: Dict[str, List[float]] = {"train": [], "validation": [], "epoch_train": []}
        self.metrics_functions: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
            "accuracy": self._accuracy,
            "precision": self._precision,
            "recall": self._recall,
            "f1_score": self._f1_score,
            "mae": self._mean_absolute_error,
            "mse": self._mean_squared_error,
        }
        self._validate_metrics(metrics_list)
        self.has_non_cost_metrics: bool = metrics_list is not None
        self.metrics_values: Dict[str, List[float]] = {}
        if self.has_non_cost_metrics:
            self._initialize_metrics(metrics_list)
        self._set_max_dataset_size(max_dataset_size)
        self._validate_average(average)
        self.average: Literal["macro", "micro"] = average
        self.use_colors: bool = use_colors
        
    def _set_main_metric(self, model: object,
                                main_metric: Literal["train_cost", "val_cost",
                                      "train_val_cost"]) -> None:
        """
        Set the main metric for the model.
        
        This method sets the main metric for the model based on the specified
        metric name. The main metric can be "train_cost", "val_cost", or "train_val_cost".
        
        Args:
            model: The neural network model to set the main metric for.
            main_metric: The name of the main metric to use.
            
        Raises:
            ValueError: If an unsupported metric name is provided.
        """
        if main_metric not in ["train_cost", "val_cost", "train_val_cost"]:
            raise ValueError(f"Unsupported main metric: {main_metric}. Supported metrics are: ['train_cost', 'val_cost', 'train_val_cost']")
        
        model.main_metric = main_metric    

    def _validate_metrics(self, metrics_list: List[str]) -> None:
        """
        Validate the metrics list
        
        Args:
            metrics_list: The list of metrics to compute.
            
        Raises:
            ValueError: If the metrics list is not provided or is empty.
            ValueError: If the metric is not supported.
        """
        if metrics_list is not None:
            for metric in metrics_list:
                if metric not in self.metrics_functions:
                    raise ValueError(f"Unsupported metric: {metric}. Supported metrics are: {list(self.metrics_functions.keys())}")
        
    def _set_max_dataset_size(self, max_dataset_size: int) -> None:
        """
        Set the maximum dataset size for metric computation.
        
        Args:
            max_dataset_size: The maximum number of examples to use for metric computation.
            
        Raises:
            ValueError: If the maximum dataset size is not a positive integer.
        """
        if max_dataset_size <= 0:
            raise ValueError("max_dataset_size must be a positive integer")
        
        self.max_dataset_size: int = max_dataset_size    
    
    def _validate_average(self, average: str) -> None:
        if average not in ["macro", "micro"]:
            raise ValueError("average must be either 'macro' or 'micro'")
        
    def _initialize_metrics(self, metrics_list: List[str]) -> None:
        """Initialize the metrics dictionary with empty lists."""
        for metric in metrics_list:
            self.metrics_values[metric] = []

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Compute the specified metrics for the given true and predicted values.
        
        Computes all specified metrics apart from the costs
        The metrics computed are stored in the metrics_values dictionary
        Converts one-hot encoded labels to class indices if necessary

        Parameters:
            y_true: The ground truth labels.
            y_pred: The predicted labels.
            
        Raises:
            ValueError: If there are no metrics to compute
        """
        if not self.has_non_cost_metrics:
            raise ValueError("There are no metrics to compute")
        
        if len(y_pred.shape) > 1:  # If y_pred is one-hot encoded
            y_pred = np.argmax(y_pred, axis=0)
        if len(y_true.shape) > 1:  # If y_true is one-hot encoded
            y_true = np.argmax(y_true, axis=0)
        
        for metric in self.metrics_values:
            self.metrics_values[metric].append(self.metrics_functions[metric](y_true, y_pred))
            
    def _compute_single_metric(self, metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute a single metric for the given true and predicted values.
        
        This method computes a single metric for the given true labels (`y_true`)
        and predicted labels (`y_pred`), converting them from one-hot encoded format
        if necessary.
        
        Args:
            metric_name: The name of the metric to compute.
            y_true: The ground truth labels.
            y_pred: The predicted labels.
            
        Returns:
            float: The computed metric value.
        """
        if len(y_pred.shape) > 1:  # If y_pred is one-hot encoded
            y_pred = np.argmax(y_pred, axis=0)
        if len(y_true.shape) > 1:  # If y_true is one-hot encoded
            y_true = np.argmax(y_true, axis=0)
            
        if metric_name not in self.metrics_functions:
            raise ValueError(f"Unsupported metric: {metric_name}. Supported metrics are: {list(self.metrics_functions.keys())}")    
            
        return self.metrics_functions[metric_name](y_true, y_pred)
    
    def _evaluate_model_metrics(self, model: object,
                                X_train: np.ndarray, Y_train: np.ndarray,
                                has_validation_data: bool,
                                X_val: np.ndarray = None, Y_val: np.ndarray = None
                                 ) -> None:
        """
        Evaluate the model metrics on the training or validation data.
        
        If validation data is provided, the metrics are computed on the validation data.
        Otherwise, the metrics are computed on the training data.
        
        Parameters:
            model: The neural network model.
            X_train: The training input data.
            Y_train: The training output data.
            has_validation_data: Whether validation data is provided.
            X_val: The validation input data.
            Y_val: The validation output data.
        """
        if self.has_non_cost_metrics:
            model.inference_mode = True
            
            if has_validation_data:
                val_predictions = model.predict(X_val, transpose_data=False)
                self._compute_metrics(Y_val, val_predictions)
            else:
                train_predictions = model.predict(X_train, transpose_data=False)
                self._compute_metrics(Y_train, train_predictions)

            model.inference_mode = False  
                            
            if model.verbose and self.has_non_cost_metrics:
                self._display_last_metrics()
                
    def _evaluate_validation_cost(self, model: object, X_val: np.ndarray, Y_val: np.ndarray) -> None:
        """
        Evaluate the cost on validation data for the neural network model.
        
        The cost is stored in the costs dictionary under the "validation" key.
        
        Args:
            model (NeuralNetwork): The neural network model.
            X_val (np.ndarray): The validation input data.
            Y_val (np.ndarray): The validation output data.
        """
        model.inference_mode = True
        val_predictions = model._forward_propagation(X_val)
        val_cost = self._compute_cost(Y_val, val_predictions)
        self.costs["validation"].append(val_cost)
        model.inference_mode = False
            
    def _get_main_metric(self, model: object,
                         has_validation_data: bool,
                         X_train: np.ndarray = None, Y_train: np.ndarray = None
                         ) -> Union[float, Tuple[float, float]]:
        """
        Computes and returns the specified metric for the neural network.
        
        Parameters:
            has_validation_data: Whether validation data is provided.
            X_train: The training input data.
            Y_train: The training output data.
            
        Raises:
            ValueError: If validation data is not provided and the metric name
                is "val_cost" or "train_val_cost".
            ValueError: If an unsupported metric name is provided.
            
        Returns:
            float or Tuple[float, float]: The computed metric value.
                float: If the metric name is "train_cost".
                float: If the metric name is "val_cost".
                Tuple[float, float]: If the metric name is "train_val_cost".
        """                   
        if not has_validation_data and (model.main_metric == "val_cost"
                                        or model.main_metric == "train_val_cost"):
            raise ValueError("Validation data must be provided to compute the validation cost.")
        
        if X_train.shape[1] > self.max_dataset_size: # cap dataset for efficiency
            indices = np.random.choice(X_train.shape[1], self.max_dataset_size, replace=False)
            X: np.ndarray = X_train[:, indices]
            Y: np.ndarray = Y_train[:, indices]
        else:
            X: np.ndarray = X_train
            Y: np.ndarray = Y_train
            
        if model.main_metric == "train_cost" or model.main_metric == "train_val_cost":
            model.inference_mode = True
            predictions: np.ndarray = model._forward_propagation(X)
            model.inference_mode = False
        
        if model.main_metric == "train_cost":
            epoch_train_cost: float = self._compute_cost(Y, predictions)
            self.costs["epoch_train"].append(epoch_train_cost)
            return epoch_train_cost
        elif model.main_metric == "val_cost":
            return self.costs["validation"][-1]
        elif model.main_metric == "train_val_cost":
            epoch_train_cost: float = self._compute_cost(Y, predictions)
            self.costs["epoch_train"].append(epoch_train_cost)
            return epoch_train_cost, self.costs["validation"][-1]
        else:
            raise ValueError(f"Unsupported metric name: {model.main_metric}")

    def _display_last_metrics(self) -> None:
        """
        Display the metrics computed in the last epoch
        
        This method constructs a string representation of various metrics
        If `self.use_colors` is True, the metrics will be displayed with colors
        """
        colors = {
            "accuracy": "\033[31m",      # Red
            "precision": "\033[35m",     # Magenta
            "recall": "\033[33m",        # Yellow
            "f1_score": "\033[36m",      # Cyan
            "mae": "\033[33m",           # Yellow
            "mse": "\033[31m",           # Red
        }
        reset_color = "\033[0m"

        output: str = "Metrics | "
        for metric in self.metrics_values:
            value = self.metrics_values[metric][-1]
            if self.use_colors:
                output += f"{colors[metric]}{metric}: {value:.4f}{reset_color} | "
            else:
                output += f"{metric}: {value:.4f} | "
        print(output.strip(" | "))
        
    def _get_metrics(self) -> Dict[str, List[float]]:
        """Return the dictionary of metrics and their values."""
        return self.metrics_values
    
    def _get_costs(self) -> Dict[str, List[float]]:
        """Return the dictionary of costs for training and validation.
        
        The dictionary has the keys: "train", "validation", "epoch_train".
        It stores lists for the values computed for each key.
        "train" stores the training costs for each batch, which are always
        computed during training.
        "validation" stores the validation costs after each epoch, only if
        validation data is provided.If no validation data is provided the list is empty
        "epoch_train" stores the training costs after each epoch only if the
        main_metric is either "train_cost" or "train_val_cost".   
        """
        return self.costs
    
    def _compute_cost(self, Y: np.ndarray, AL: np.ndarray) -> float:
        """
        Compute the cost for the given true and predicted values.
        
        The cost is computed using the loss function of the model.
        If the model has regularization enabled, the regularization term is
        added to the cost.
        
        Args:
            Y (np.ndarray): The ground truth labels.
            AL (np.ndarray): The predicted labels.
        
        Returns:
            float: The computed cost.
        """
        cost: float = self.model.loss_function(Y, AL)

        if self.model.lambda_ > 0:
            m: int = Y.shape[1]
            regularization_term: float = regularization._compute_regularization_term(self.model, m)
            cost += regularization_term

        return cost        

    def _accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the accuracy score for the given true and predicted values.
        
        The accuracy score is the ratio of correct predictions to the total number
        of predictions made by the classifier.
        
        Args:
            y_true (np.ndarray): The ground truth labels.
            y_pred (np.ndarray): The predicted labels.
            
        Returns:
            float: The accuracy score.
        """
        correct_predictions: int = np.sum(y_true == y_pred)
        total_predictions: int = len(y_true)
        return correct_predictions / total_predictions if total_predictions > 0 else 0

    def _precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the precision score for the given true and predicted values.
        
        The precision score is the ratio of true positive predictions to the total
        number of positive predictions made by the classifier.
        
        Args:
            y_true (np.ndarray): The ground truth labels.
            y_pred (np.ndarray): The predicted labels.
            
        Returns:
            float: The precision score.
        """
        classes: np.ndarray = np.unique(y_true)
        
        if self.average == "macro":
            precisions: List[float] = []
            for cls in classes:
                true_positive: int = np.sum((y_pred == cls) & (y_true == cls))
                false_positive: int = np.sum((y_pred == cls) & (y_true != cls))
                precision: float = (
                    true_positive / (true_positive + false_positive)
                    if (true_positive + false_positive) != 0
                    else 0
                )
                precisions.append(precision)
            return np.mean(precisions)
        elif self.average == "micro":
            true_positive: int = 0
            false_positive: int = 0
            for cls in classes:
                true_positive += np.sum((y_pred == cls) & (y_true == cls))
                false_positive += np.sum((y_pred == cls) & (y_true != cls))
            return true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
        
    def _recall(self, y_true: np.ndarray, y_pred:np.ndarray) -> float:
        """
        Compute the recall score for the given true and predicted values.
        
        The recall score is the ratio of true positive predictions to the total
        number of actual positive instances in the data.
        
        Args:
            y_true (np.ndarray): The ground truth labels.
            y_pred (np.ndarray): The predicted labels.
            
        Returns:
            float: The recall score.
        """
        
        classes: np.ndarray = np.unique(y_true)

        if self.average == "macro":
            recalls: List[float] = []
            for cls in classes:
                true_positive: int = np.sum((y_pred == cls) & (y_true == cls))
                false_negative: int = np.sum((y_pred != cls) & (y_true == cls))
                recall_value: float = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
                recalls.append(recall_value)
            return np.mean(recalls)
        elif self.average == "micro":
            true_positive: int = 0
            false_negative: int = 0
            for cls in classes:
                true_positive += np.sum((y_pred == cls) & (y_true == cls))
                false_negative += np.sum((y_pred != cls) & (y_true == cls))
            return true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
              
    def _f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the F1 score for the given true and predicted values.
        
        The F1 score is the harmonic mean of precision and recall, providing a
        balance between the two metrics.
        
        Args:
            y_true (np.ndarray): The ground truth labels.
            y_pred (np.ndarray): The predicted labels.
            
        Returns:
            float: The F1 score.
        """
        precision_value: float = self._precision(y_true, y_pred)
        recall_value: float = self._recall(y_true, y_pred)
        f1_value: float = 2 * (precision_value * recall_value) / (precision_value + recall_value) if (precision_value + recall_value) != 0 else 0
        return f1_value
    
    def _mean_absolute_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the mean absolute error between the true and predicted values.
        
        The mean absolute error is the average of the absolute differences between
        the true and predicted values.
        
        Args:
            y_true (np.ndarray): The ground truth labels.
            y_pred (np.ndarray): The predicted labels.
            
        Returns:
            float: The mean absolute error.
        """
        return np.mean(np.abs(y_true - y_pred))

    def _mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the mean squared error between the true and predicted values.
        
        The mean squared error is the average of the squared differences between
        the true and predicted values.
        
        Args:
            y_true (np.ndarray): The ground truth labels.
            y_pred (np.ndarray): The predicted labels.
            
        Returns:
            float: The mean squared error.
        """
        return np.mean((y_true - y_pred) ** 2)
    
    def _get_supported_metrics(self) -> List[str]:
        """Return the list of supported metrics.
        
        Supported metrics are: "accuracy", "precision", "recall", "f1_score",
        "mae", "mse".
        Costs are not included in the list but are computed during training.
        """
        return list(self.metrics_functions.keys())