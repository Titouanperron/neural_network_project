"""Binary and multiclass classification with one hidden layer."""

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y

from activations import logistic, softmax
from slp_base import BaseSLPEstimator


class SimpleSLPClassifier(ClassifierMixin, BaseSLPEstimator):
    """Use sigmoid for two classes and softmax for more than two."""

    def _output_activation(self, z2):
        if len(self.classes_) == 1:
            return np.ones_like(z2)
        return logistic(z2) if self.n_outputs_ == 1 else softmax(z2)

    def _output_error(self, y, y_pred):
        # Sigmoid/softmax combined with cross-entropy simplifies to p - y.
        return y_pred - y

    def _compute_loss(self, y_true, y_pred):
        probabilities = np.clip(y_pred, np.finfo(float).eps, 1 - np.finfo(float).eps)
        if probabilities.shape[1] == 1:
            terms = y_true * np.log(probabilities) + (1 - y_true) * np.log1p(-probabilities)
        else:
            terms = y_true * np.log(probabilities)
        return float(-terms.sum(axis=1).mean())

    def _evaluate_loss(self, X, y):
        """Evaluate cross-entropy directly from logits for numerical stability."""
        if len(self.classes_) == 1:
            return 0.0
        _, _, logits, _ = self._forward_propagation(X)
        if self.n_outputs_ == 1:
            signed_logits = (1 - 2 * y) * logits
            return float(np.logaddexp(0, signed_logits).mean())
        shifted = logits - logits.max(axis=1, keepdims=True)
        return float((np.log(np.exp(shifted).sum(axis=1)) - (y * shifted).sum(axis=1)).mean())

    def fit(self, X, y):
        """Train from feature rows and discrete class labels."""
        self._validate_parameters()
        X, y = check_X_y(X, y, dtype=float)
        check_classification_targets(y)
        self.classes_, indices = np.unique(y, return_inverse=True)
        if len(self.classes_) == 1:
            target = np.ones((len(y), 1))
        elif len(self.classes_) == 2:
            target = indices[:, None].astype(float)
        else:
            target = (indices[:, None] == np.arange(len(self.classes_))).astype(float)
        self._train(X, target)
        return self

    def predict_proba(self, X):
        """Return one probability column per label in classes_ order."""
        probabilities = self._predict_output(X)
        if len(self.classes_) == 2:
            return np.column_stack((1 - probabilities, probabilities))
        return probabilities

    def predict(self, X):
        """Return the label with the highest probability for each row."""
        indices = self.predict_proba(X).argmax(axis=1)
        return self.classes_[indices]
