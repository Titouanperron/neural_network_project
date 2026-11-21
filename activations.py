"""Vectorized activation functions and their derivatives."""

import numpy as np


def relu(z):
    """Return the positive part of each input."""
    return np.maximum(0, z)


def relu_derivative(z):
    """Use zero as the derivative at the ReLU kink."""
    return (np.asarray(z) > 0).astype(float)


def tanh(z):
    """Return the hyperbolic tangent."""
    return np.tanh(z)


def tanh_derivative(z):
    """Differentiate tanh with respect to its input."""
    return 1 - np.tanh(z) ** 2


def logistic(z):
    """Compute the sigmoid without exponentiating large positive values."""
    return np.exp(-np.logaddexp(0, -np.asarray(z, dtype=float)))


def logistic_derivative(z):
    """Differentiate the sigmoid with respect to its input."""
    z = np.asarray(z, dtype=float)
    return logistic(z) * logistic(-z)


def softmax(z):
    """Normalize logits into probabilities along the last axis."""
    shifted = z - np.max(z, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)
