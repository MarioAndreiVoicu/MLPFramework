## Multilayered Perceptron Framework

A NumPy-based library for building and training Multilayer Perceptrons.

## Key Features

- **Custom Architectures**: Build Multilayer Perceptrons with custom architectures.
- **Layer Types**: Dense, Dropout.
- **Activation Functions**: Linear, Sigmoid, Tanh, Softmax, ReLU, PReLU, ELU, Swish.
- **Loss Functions**: Binary Crossentropy, Categorical Crossentropy, Mean Squared Error, Mean Absolute Error, Huber Loss
- **Optimizers**: Adam, RMSprop, Momentum, SGD.
- **Metrics**: Accuracy, Precision, Recall, F1 Score, MAE, MSE.
- **Callbacks**: Early Stopping, Progress Bar.
- **Model Handling**: Save and Load Models.
- **Regularization**: L1, L2, L1_L2.

## Additional Features

- Weight Initialization (He, Xavier, Random).
- Model Summary.
- Mini-batch Gradient Descent.
- Custom Early Stopping Metrics.
- Colored Metrics Display.
- Verbosity Control.
- Gradient Clipping.

## Dependencies

- `numpy`: Core numerical operations.
- `json`: Used for saving and loading models.
- `typing`: Used for type hints.

> **Note**: Example files utilize `sklearn` and `matplotlib` for dataset import, preprocessing, and visualization, but these are not required for the core library.

## Example Files

- **`example_load_model.py`**: Demonstrates loading a pre-trained model and using it for predictions on new data.
- **`example_train_model.py`**: Shows how to create, compile, train, and evaluate a model on a dataset.

> **Data format requirements**: Input data should have the shape `(num_samples, num_features)`, and target labels should be in the shape `(num_samples, output_size)`.

## Example Usage

Here is a minimal example that demonstrates how to create, compile, and train a simple Multilayer Perceptron:

```python
from multilayer_perceptron import layers, MultilayerPerceptron

# Define the architecture of the model
layers = [
    layers.DenseLayer(input_size, 128, activation="swish"),
    layers.DropoutLayer(0.3),
    layers.DenseLayer(128, 64, activation="swish"),
    layers.DropoutLayer(0.3),
    layers.DenseLayer(64, output_size, activation="softmax")
]

# Initialize the model with the defined layers
model = MultilayerPerceptron(layers)

# Compile the model with a loss function, optimizer, and (optional) evaluation metrics
model.compile(
    loss_function="categorical_crossentropy",
    optimizer="adam",
    metrics_list=["accuracy", "f1_score"]
)

# Train the model with training data, validation data, and early stopping
model.fit(
    X_train, Y_train,
    batch_size=128,
    X_val=X_val, Y_val=Y_val,
    epochs=10,
    use_early_stopping=True,
    patience=2
)
```
