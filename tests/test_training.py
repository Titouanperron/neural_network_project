"""Check gradients and restoration of the best validation checkpoint."""

import numpy as np
import pytest

from slp_classifier import SimpleSLPClassifier
from slp_regressor import SimpleSLPRegressor


@pytest.mark.parametrize(
    "estimator, targets",
    [
        (SimpleSLPClassifier, [0, 1, 0]),
        (SimpleSLPClassifier, [0, 1, 2]),
        (SimpleSLPRegressor, [[1, 3], [2, 1], [4, 2]]),
    ],
)
def test_gradients_match_finite_differences(estimator, targets):
    X = np.array([[-1., 2.], [0.5, 1.], [1., -0.5]])
    y = np.asarray(targets)
    model = estimator(hidden_layer_size=3, activation="tanh", max_iter=1, random_state=42)
    model.fit(X, y)
    if estimator is SimpleSLPRegressor:
        y = model._transform_targets(y)
    elif len(model.classes_) == 2:
        y = y[:, None]
    else:
        y = np.eye(3)[y]

    gradients = model._backward_propagation(X, y, *model._forward_propagation(X))
    step = 1e-5
    for parameter, gradient in zip((model.W1_, model.b1_, model.W2_, model.b2_), gradients):
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + step
            upper = model._evaluate_loss(X, y)
            parameter[index] = original - step
            lower = model._evaluate_loss(X, y)
            parameter[index] = original
            assert gradient[index] == pytest.approx((upper - lower) / (2 * step), abs=1e-7)


def test_early_stopping_restores_best_validation_loss():
    X = np.linspace(-1, 1, 40)[:, None]
    y = (3 * X[:, 0]) + 1
    model = SimpleSLPRegressor(
        hidden_layer_size=4, random_state=42, early_stopping=True,
        validation_fraction=0.25, n_iter_no_change=3, tol=10, max_iter=30,
    ).fit(X, y)

    validation = np.random.RandomState(42).permutation(len(X))[:10]
    assert model.n_iter_ == 4
    assert model.best_iteration_ == np.argmin(model.validation_loss_curve_) + 1
    assert model._evaluate_loss(
        X[validation], model._transform_targets(y[validation, None])
    ) == pytest.approx(min(model.validation_loss_curve_))
