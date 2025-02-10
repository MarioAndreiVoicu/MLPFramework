"""
MNIST Neural Network Training Example

This is an example of training a neural network on the MNIST dataset.
Input format: 28x28 grayscale images of handwritten digits (0-9).
Output format: One-hot encoded labels for each digit (10 classes).

Architecture:
- Hidden layer with 128 neurons and ReLU activation function
- Dropout layer with a dropout rate of 0.3
- Hidden layer with 64 neurons and ReLU activation function
- Dropout layer with a dropout rate of 0.3
- Output layer with 10 neurons (one for each digit) and softmax activation function

Training:
- Optimizer: Adam
- Loss Function: Categorical Cross-Entropy

Data Preprocessing:
- Data is split into training, validation, and test sets.
- Features are rescaled to the range [0, 1].
- Target variable is one-hot encoded.
"""


import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# import neural_network and layers
from neural_network import NeuralNetwork, layers

# Load MNIST data
mnist = fetch_openml("mnist_784", version=1)
X = mnist.data.to_numpy()
y = mnist.target.astype(int)

# Rescale features
X = X / 255.0

# One-hot encode the target variable
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.values.reshape(-1, 1))

# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_val, X_test, Y_val, Y_test = train_test_split(
    X_test, Y_test, test_size=0.5, random_state=42
)

input_size = X_train.shape[1]
output_size = Y_train.shape[1]

# Initialize the layers
layers = [
    layers.DenseLayer(input_size, 128, activation="swish"),
    layers.DropoutLayer(0.3),
    layers.DenseLayer(128, 64, activation="swish"),
    layers.DropoutLayer(0.3),
    layers.DenseLayer(64, output_size, activation="softmax"),
]

# Initialize and train the neural network
model = NeuralNetwork(layers)
model.compile(
    loss_function="categorical_crossentropy",
    optimizer="adam",
    learning_rate=0.001,
    lambda_=0.001,
    main_metric="train_val_cost",
    # metrics_list=["accuracy", "precision", "recall", "f1_score"], # Uncomment this line to get the metrics
    # use_colors_metrics=True # Uncomment this line to get the metrics in colors in training
)

model.fit(
    X_train,
    Y_train,
    batch_size=128,
    X_val=X_val,
    Y_val=Y_val,
    epochs=10,
    use_early_stopping=True,
    patience=2,
)

# Save the model
NeuralNetwork.save_model(model, "mnist_model.json")

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Evaluate the model
Y_train_indices = np.argmax(Y_train, axis=1)
Y_test_indices = np.argmax(Y_test, axis=1)
train_accuracy = accuracy_score(Y_train_indices, train_predictions)
print(f"Train Accuracy: {train_accuracy:.4f}")
test_accuracy = accuracy_score(Y_test_indices, test_predictions)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Get the costs and plot them
costs = model.get_costs()
train_costs = costs["epoch_train"]
val_costs = costs["validation"]
plt.plot(train_costs, label="Training Cost")
plt.plot(val_costs, label="Validation Cost")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Training and Validation Costs")
plt.legend()
plt.show()

"""
Uncomment the metrics list in the compile method to get the metrics.
Then you can plot them by uncommenting the following code.
"""
# metrics = model.get_metrics()
# plt.plot(metrics["accuracy"], label="Accuracy")
# plt.plot(metrics["precision"], label="Precision")
# plt.plot(metrics["recall"], label="Recall")
# plt.plot(metrics["f1_score"], label="F1 Score")
# plt.xlabel("Epoch")
# plt.ylabel("Metric Value")
# plt.title("Metrics")
# plt.legend()
# plt.show()