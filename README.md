## Neural Network Library

A NumPy-based library for building and training neural networks.

## Key Features
- **Custom Architectures**: Build neural networks with custom architectures.
- **Layer Types**: Dense, Dropout.
- **Activation Functions**: Sigmoid, Tanh, ReLU, Linear, Softmax.
- **Loss Functions**: Binary Crossentropy, Categorical Crossentropy, Mean Squared Error, Mean Absolute Error.
- **Optimizers**: Adam, RMSprop, Momentum, SGD.
- **Metrics**: Accuracy, Precision, Recall, F1 Score, MAE, MSE.
- **Callbacks**: Early Stopping, Progress Bar.
- **Model Handling**: Save and Load Models.
- **Regularization**: L1, L2, L1_L2.

## Additional Features
- Model Summary.
- Weight Initialization (He, Xavier, Random).
- Mini-batch Gradient Descent.
- Gradient Clipping.
- Custom Early Stopping Metrics.
- Colored Metrics Display.
- Verbosity Control.

## Dependencies

- `numpy`: Core numerical operations.
- `json`: Only used for saving and loading models.
- `typing`: Only used for type hints to improve code readability.
  
> **Note**: Example files utilize `sklearn` and `matplotlib` for dataset import, preprocessing, and visualization, but these are not required for the core library.

## Example Files

- **`example_load_model.py`**: Demonstrates loading a pre-trained model and using it for predictions on new data.
- **`example_train_model.py`**: Shows how to create, compile, train, and evaluate a model on a dataset.

> **Data format requirements**: Input data should have the shape `(num_samples, num_features)`, and target labels should be in the shape `(num_samples, output_size)`.

## Example Usage

Here’s a minimal example that demonstrates how to create, compile, and train a simple neural network:

```python
from neural_network import layers, NeuralNetwork

# Define the architecture of the model
layers = [
    layers.DenseLayer(input_size, 128, activation="relu"),
    layers.DropoutLayer(0.3),
    layers.DenseLayer(128, 64, activation="relu"),
    layers.DropoutLayer(0.3),
    layers.DenseLayer(64, output_size, activation="softmax")
]

# Initialize the model with the defined layers
model = NeuralNetwork(layers)

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
