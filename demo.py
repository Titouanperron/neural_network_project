"""Compare both networks on held-out data and save the loss curves."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, make_regression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from slp_classifier import SimpleSLPClassifier
from slp_regressor import SimpleSLPRegressor


SEED = 42
HIDDEN_SIZE = 24
BATCH_SIZE = 16
EPOCHS = 200


def compare(X, y, classification, learning_rate):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED,
        stratify=y if classification else None,
    )
    features = StandardScaler()
    X_train = features.fit_transform(X_train)
    X_test = features.transform(X_test)

    custom_type, reference_type = (
        (SimpleSLPClassifier, MLPClassifier) if classification
        else (SimpleSLPRegressor, MLPRegressor)
    )
    common = dict(
        activation="tanh", batch_size=BATCH_SIZE, max_iter=EPOCHS,
        momentum=0.0, random_state=SEED, tol=0.0,
    )
    custom = custom_type(
        hidden_layer_size=HIDDEN_SIZE, learning_rate=learning_rate, **common
    )
    # sklearn minimizes half the MSE, so double its step for regression.
    reference = reference_type(
        hidden_layer_sizes=(HIDDEN_SIZE,), solver="sgd", alpha=0.0,
        learning_rate_init=learning_rate if classification else 2 * learning_rate,
        nesterovs_momentum=False, n_iter_no_change=EPOCHS, **common
    )
    custom.fit(X_train, y_train)
    if classification:
        reference.fit(X_train, y_train)
        reference_score = reference.score(X_test, y_test)
        reference_loss = np.asarray(reference.loss_curve_)
    else:
        target = StandardScaler().fit(y_train[:, None])
        reference.fit(X_train, target.transform(y_train[:, None]).ravel())
        reference_score = reference.score(X_test, target.transform(y_test[:, None]).ravel())
        reference_loss = 2 * np.asarray(reference.loss_curve_)

    return custom.score(X_test, y_test), reference_score, custom.loss_curve_, reference_loss


def main():
    datasets = [
        ("Diabetes", *load_diabetes(return_X_y=True), False, 0.003),
        ("Breast Cancer", *load_breast_cancer(return_X_y=True), True, 0.025),
        ("Iris", *load_iris(return_X_y=True), True, 0.045),
        ("Synthetic regression", *make_regression(
            n_samples=500, n_features=8, n_informative=5, noise=15, random_state=SEED,
        ), False, 0.005),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, (name, X, y, classification, learning_rate) in zip(axes.flat, datasets):
        custom_score, reference_score, custom_loss, reference_loss = compare(
            X, y, classification, learning_rate
        )
        axis.plot(range(1, len(custom_loss) + 1), custom_loss, label="SimpleSLP")
        axis.plot(range(1, len(reference_loss) + 1), reference_loss,
                  label="scikit-learn", linestyle="--")
        axis.set(title=name, xlabel="Epoch", ylabel="Cross-entropy" if classification else "MSE")
        axis.grid(alpha=0.2)
        axis.legend()
        metric = "Accuracy" if classification else "R²"
        print(f"{name:22} {metric:8} SimpleSLP={custom_score:.4f}  sklearn={reference_score:.4f}")

    figure.savefig(Path(__file__).with_name("loss_curves.png"), dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
