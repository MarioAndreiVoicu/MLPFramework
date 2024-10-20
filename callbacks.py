import numpy as np
import sys
from losses import loss_functions
from typing import Literal, List

"""
callbacks.py

This module defines various callbacks for neural network training.

Supported callbacks: "ProgressBar", "EarlyStopping".
"""

class ProgressBar:
    """
    A class to implement a progress bar for neural network training.
    
    The progress bar shows the progress of the current epoch, the current batch
    number and a customizable metric at the end of the progress bar.
    """
    
    def __init__(self, total_batches: int, total_epochs: int, bar_length: int = 25) ->None:
        """
        Initializes the progress bar with the total number of batches and epochs.

        Args:
            total_batches: Total number of batches.
            total_epochs: Total number of epochs.
            bar_length: Length of the progress bar.
        """
        self.total_batches: int = total_batches
        self.total_epochs: int = total_epochs
        self.bar_length: int = bar_length

    def _update(self, current_epoch: int, current_batch: int, with_metric: bool = False,
                metric_name: Literal["train_cost", "val_cost", "train_val_cost"] = None,
                metric_value: float = None):
        """
        Updates the progress bar and prints it to the console.
        
        Displays the current epoch, the progress bar, the current batch number,
        and an optional custom metric at the end of the progress bar that can be
        either the training cost, validation cost, or both.
        
        The progress bar color changes based on the progress:
            - Red (< 33%)
            - Yellow (33% - 66%)
            - Green (>= 66%)
        
        Args:
            current_epoch: The current epoch number.
            current_batch: The current batch number within the epoch.
            with_metric: Whether to display a metric at the end of the progress bar.
            metric_name: The name of the metric to display.
                One of {"train_cost", "val_cost", "train_val_cost"}.
            metric_value (float): The value of the metric to display.
        """
        
        progress: float = current_batch / self.total_batches
        arrow: str = "=" * int(round(progress * self.bar_length) - 1) + ">"
        spaces: str = " " * (self.bar_length - len(arrow))
        
        bar_color: str = "\033[32m" if progress > 0.66 else "\033[33m" if progress > 0.33 else "\033[31m"  # Green, Yellow, Red
        epoch_color: str = "\033[36m"  # Cyan
        metric_color: str = "\033[34m"   # Blue
        
        message: str = (f"\r{epoch_color}Epoch {current_epoch}/{self.total_epochs} | "
                f"{bar_color}[{arrow}{spaces}] {int(progress * 100)}% | "
                f"Batch {current_batch}/{self.total_batches} | ")
            
        if with_metric:
            if metric_value is None or metric_name is None:
                raise ValueError("metric_name and metric_value must be provided")
            if metric_name == "train_cost":
                message += f"{metric_color}Train Cost: {metric_value:.4f}"
            elif metric_name == "val_cost":
                message += f"{metric_color}Validation Cost: {metric_value:.4f}"
            elif metric_name == "train_val_cost":
                message += f"{metric_color}Train Cost: {metric_value[0]:.4f} | Val Cost: {metric_value[1]:.4f}"
        
        message += "\033[0m"  # Reset color at the end

        sys.stdout.write(message)
        sys.stdout.flush()

    def _finish(self) -> None:
        """Finishes the progress bar by printing a newline character."""
        sys.stdout.write("\n")
        sys.stdout.flush()

class EarlyStopping:
    """
    A class to implement early stopping during neural network training.

    This class monitors the model's performance on the specified metric and
    stops training if the performance does not improve for a given number of
    epochs (patience).
    
    Attributes:
        model (NeuralNetwork): The neural network model to monitor for early stopping.
        patience (int): Number of epochs with no improvement after which training
            will be stopped.
        metric (str): The performance metric to evaluate. Default is "cost".
            If cost is used, validation cost is used if validation data is provided,
            otherwise training cost is used.
            Other supported metrics are "accuracy", "precision", "recall", "f1_score",
            "mae", "mse".
        has_validation_data (bool): Whether validation data is provided.
        verbose (bool): Whether to print a message when early stopping is triggered.
        best_score (float): The best score achieved on the specified metric.
        epochs_without_improvement (int): Number of epochs with no improvement.
        stop_training (bool): Whether to stop training based on the early stopping criteria.
        mode (str): The mode of operation, either "minimize" or "maximize".
            If the metric is "cost" or one of ["mae", "mse"], the mode is "minimize".
            Otherwise, the mode is "maximize".
    """
    
    def __init__(self, model: object, patience: int = 5,
                 metric: Literal["accuracy", "precision", "recall", "f1_score",
                                 "cost", "mae", "mse"] = "cost",
                 has_validation_data: bool = False, verbose: bool = False) -> None:
        """
        Initializes EarlyStopping with given parameters.

        Args:
            model: The neural network model to monitor for early stopping.
            patience: Number of epochs with no improvement after which training
                will be stopped.
            metric: The performance metric to evaluate. Default is "cost".
                If cost is used, validation cost is used if validation data is
                provided, otherwise training cost is used.
                Other supported metrics are "accuracy", "precision", "recall",
                "f1_score", "mae", "mse".
            has_validation_data: Whether validation data is provided.
            verbose: Whether to print a message when early
                stopping is triggered.
        """
        self.model: object = model
        self._set_metric(metric, has_validation_data)
        self.patience: int = patience
        self.verbose: bool = verbose
        self.best_score: float = -float("inf")
        self.epochs_without_improvement: int = 0
        self.stop_training: bool = False
        self._set_mode()
        
    def _set_metric(self, metric: str, has_validation_data: bool) -> None:
        """
        Validates and sets the metric attribute.
        
        Args:
            metric: The metric to evaluate.
            has_validation_data: Whether validation data is provided.
            
        Raises:
            ValueError: If the metric is invalid.
        """
        valid_metrics: List[str] = self.model.metrics._get_supported_metrics()
        if metric != "cost" and metric not in valid_metrics:
            raise ValueError(f"metric must be 'cost' or {valid_metrics}. Got {metric}")
        
        if metric == "cost":
            self.metric: str = "val_cost" if has_validation_data else "train_cost"
        else:
            self.metric: str = metric
        
    def _set_mode(self):
        """
        Sets the mode of operation based on the specified metric.

        This method determines whether the mode should be "minimize" or "maximize"
        based on the value of the `metric` attribute. If the metric is "cost" or 
        one of ["mae", "mse"], the mode is set to "minimize". Otherwise, the mode 
        is set to "maximize".
        """
        if self.metric == "cost" or self.metric in ["val_cost", "train_cost"] or self.metric in ["mae", "mse"]:
            self.mode: str = "minimize"
        else:
            self.mode: str = "maximize"
       
    def _get_current_score(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Returns the current score based on the chosen metric.

        Returns:
            float: The current score based on the metric.
        """
        if self.metric == "val_cost":
            return self.model.metrics.costs["validation"][-1]
        elif self.metric == "train_cost":
            return self.model.metrics.costs["epoch_train"][-1]
        elif self.metric in self.model.metrics._get_supported_metrics():
            self.model.inference_mode = True
            predictions = self.model.predict(X, transpose_data=False)
            self.model.inference_mode = False
            return self.model.metrics._compute_single_metric(self.metric, Y, predictions)
       
    def _should_stop(self, X: np.ndarray, Y: np.ndarray) -> bool:
        """
        Checks if training should be stopped based on the early stopping criteria.

        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        current_score = self._get_current_score(X, Y)
         
        # if the metric is cost or error, we want to minimize it so we negate the score
        if self.mode == "minimize":
            current_score = -current_score
            
        if current_score < self.best_score:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print(f"Early stopping due to no improvement for {self.patience} epochs.")
        else:
            self.best_score = current_score
            self.epochs_without_improvement = 0
    
        return self.stop_training
