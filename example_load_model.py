"""
MNIST Multilayer Perceptron Example (Load Model)

This is an example of loading a pretrained MLP model.
Input format: 28x28 grayscale images of handwritten digits (0-9).
Output format: One-hot encoded labels for each digit (10 classes).

The saved model is the one trained in example_train_model.py.

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

from multilayer_perceptron import MultilayerPerceptron

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

# Load the model
model = MultilayerPerceptron.load_model("mnist_model.json")

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