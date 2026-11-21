"""Shared forward pass, gradient computation and training loop."""

from abc import ABC, abstractmethod
from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state

from activations import logistic, logistic_derivative, relu, relu_derivative, tanh, tanh_derivative


class BaseSLPEstimator(BaseEstimator, ABC):
    """One hidden layer trained with mini-batch SGD and optional momentum.

    batch_size=None uses the full training set. With early_stopping=True,
    validation_fraction is held out and the lowest-validation-loss weights
    are restored. Patience counts epochs without a tol-sized improvement.
    """

    def __init__(
        self,
        hidden_layer_size=100,
        activation="logistic",
        learning_rate=0.001,
        max_iter=200,
        random_state=None,
        *,
        batch_size=16,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=15,
        tol=1e-4,
        momentum=0.9,
    ):
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.momentum = momentum

    def _get_activation_function(self):
        if self.activation == "relu":
            return relu, relu_derivative
        if self.activation == "tanh":
            return tanh, tanh_derivative
        if self.activation in ("logistic", "sigmoid"):
            return logistic, logistic_derivative
        if self.activation == "identity":
            return np.asarray, np.ones_like
        raise ValueError(f"Unknown activation: {self.activation}")

    def _validate_parameters(self):
        integers = {
            "hidden_layer_size": self.hidden_layer_size,
            "max_iter": self.max_iter,
            "n_iter_no_change": self.n_iter_no_change,
        }
        if self.batch_size is not None:
            integers["batch_size"] = self.batch_size
        for name, value in integers.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "tol", "validation_fraction", "momentum"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.learning_rate <= 0 or self.tol < 0:
            raise ValueError("learning_rate must be positive and tol must be nonnegative")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be at least 0 and below 1")
        if not isinstance(self.early_stopping, (bool, np.bool_)):
            raise ValueError("early_stopping must be a boolean")
        self._get_activation_function()

    def _initialize_weights(self):
        """Use Glorot uniform weights and center the initial output logits."""
        hidden = self.hidden_layer_size
        limit1 = np.sqrt(6 / (self.n_features_in_ + hidden))
        limit2 = np.sqrt(6 / (hidden + self.n_outputs_))
        self.W1_ = self._rng.uniform(-limit1, limit1, (self.n_features_in_, hidden))
        self.b1_ = np.zeros(hidden)
        self.W2_ = self._rng.uniform(-limit2, limit2, (hidden, self.n_outputs_))
        self.b2_ = np.zeros(self.n_outputs_)
        if self.activation in ("logistic", "sigmoid"):
            # A zero hidden pre-activation gives sigmoid(0) = 0.5.
            self.b2_ = -0.5 * self.W2_.sum(axis=0)

    def _forward_propagation(self, X):
        activation, _ = self._get_activation_function()
        z1 = X @ self.W1_ + self.b1_
        a1 = activation(z1)
        z2 = a1 @ self.W2_ + self.b2_
        return z1, a1, z2, self._output_activation(z2)

    @abstractmethod
    def _output_activation(self, z2):
        """Convert output logits to predictions."""

    @abstractmethod
    def _output_error(self, y, y_pred):
        """Differentiate per-sample loss with respect to output logits."""

    def _backward_propagation(self, X, y, z1, a1, z2, y_pred):
        _, derivative = self._get_activation_function()
        output_delta = self._output_error(y, y_pred) / X.shape[0]
        hidden_delta = (output_delta @ self.W2_.T) * derivative(z1)
        return (
            X.T @ hidden_delta,
            hidden_delta.sum(axis=0),
            a1.T @ output_delta,
            output_delta.sum(axis=0),
        )

    @abstractmethod
    def _compute_loss(self, y_true, y_pred):
        """Return the mean loss."""

    def _evaluate_loss(self, X, y):
        return self._compute_loss(y, self._forward_propagation(X)[-1])

    def _prepare_targets(self, y):
        """Fit and apply any target transformation on training data."""
        return y

    def _transform_targets(self, y):
        return y

    def _train(self, X, y):
        self._rng = check_random_state(self.random_state)
        if self.early_stopping:
            n_validation = int(np.ceil(len(X) * self.validation_fraction))
            if n_validation >= len(X):
                raise ValueError("early_stopping requires separate training and validation samples")
            indices = self._rng.permutation(len(X))
            validation = indices[:n_validation]
            train = indices[n_validation:]
            X_val, y_val = X[validation], y[validation]
            X, y = X[train], y[train]

        y = self._prepare_targets(y)
        if self.early_stopping:
            y_val = self._transform_targets(y_val)

        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = y.shape[1]
        self._initialize_weights()
        parameters = (self.W1_, self.b1_, self.W2_, self.b2_)
        velocities = [np.zeros_like(parameter) for parameter in parameters]
        self.loss_curve_ = []
        self.validation_loss_curve_ = []
        self.best_iteration_ = None
        best_loss = patience_reference = np.inf
        stale_epochs = 0
        batch_size = self.batch_size or len(X)

        for epoch in range(1, self.max_iter + 1):
            indices = self._rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                batch = indices[start:start + batch_size]
                X_batch, y_batch = X[batch], y[batch]
                gradients = self._backward_propagation(
                    X_batch, y_batch, *self._forward_propagation(X_batch)
                )
                for parameter, velocity, gradient in zip(parameters, velocities, gradients):
                    velocity *= self.momentum
                    velocity -= self.learning_rate * gradient
                    parameter += velocity

            training_loss = self._evaluate_loss(X, y)
            if not np.isfinite(training_loss):
                raise ValueError("Training diverged. Try scaling the data or reducing learning_rate.")
            self.loss_curve_.append(training_loss)
            if self.early_stopping:
                validation_loss = self._evaluate_loss(X_val, y_val)
                if not np.isfinite(validation_loss):
                    raise ValueError("Validation loss became non-finite.")
                self.validation_loss_curve_.append(validation_loss)
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    best_parameters = [parameter.copy() for parameter in parameters]
                    self.best_iteration_ = epoch
                if patience_reference - validation_loss > self.tol:
                    patience_reference = validation_loss
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                if stale_epochs >= self.n_iter_no_change:
                    break

        self.n_iter_ = len(self.loss_curve_)
        self.loss_ = self.loss_curve_[-1]
        if self.early_stopping:
            for parameter, best in zip(parameters, best_parameters):
                parameter[:] = best
            self.loss_ = self.loss_curve_[self.best_iteration_ - 1]
        self.coefs_ = [self.W1_, self.W2_]
        self.intercepts_ = [self.b1_, self.b2_]

    def _predict_output(self, X):
        check_is_fitted(self, "coefs_")
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        return self._forward_propagation(X)[-1]
