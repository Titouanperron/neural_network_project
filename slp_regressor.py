"""Continuous single- or multi-output regression with one hidden layer."""

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y

from slp_base import BaseSLPEstimator


class SimpleSLPRegressor(RegressorMixin, BaseSLPEstimator):
    """Use an identity output and minimize mean squared error."""

    def _output_activation(self, z2):
        return z2

    def _output_error(self, y, y_pred):
        return 2 * (y_pred - y) / y.shape[1]

    def _compute_loss(self, y_true, y_pred):
        return float(np.mean((y_pred - y_true) ** 2))

    def _prepare_targets(self, y):
        self._target_scaler = StandardScaler().fit(y)
        return self._transform_targets(y)

    def _transform_targets(self, y):
        return self._target_scaler.transform(y)

    def fit(self, X, y):
        """Train from feature rows and one or more continuous targets."""
        self._validate_parameters()
        X, y = check_X_y(X, y, dtype=float, multi_output=True, y_numeric=True)
        y = np.asarray(y, dtype=float).reshape(len(X), -1)
        self._train(X, y)
        return self

    def predict(self, X):
        """Return a vector for one output and a matrix for multiple outputs."""
        output = self._predict_output(X)
        predictions = self._target_scaler.inverse_transform(output)
        return predictions.ravel() if self.n_outputs_ == 1 else predictions
