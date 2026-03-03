"""
Tests for ml.py - Linear Regression machine learning function.
"""

import numpy as np
import pytest

from ml import LinearRegression, mean_squared_error


def test_fit_and_predict_simple():
    """Model learns a simple linear relationship y = 2x + 1."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 10, size=(100, 1))
    y = 2 * X[:, 0] + 1

    model = LinearRegression(learning_rate=0.05, n_iterations=2000)
    model.fit(X, y)
    y_pred = model.predict(X)

    assert mean_squared_error(y, y_pred) < 0.01


def test_r2_score_near_one():
    """R² should be close to 1 for a noiseless linear dataset."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-5, 5, size=(200, 1))
    y = 3 * X[:, 0] - 2

    model = LinearRegression(learning_rate=0.05, n_iterations=2000)
    model.fit(X, y)

    assert model.score(X, y) > 0.99


def test_loss_decreases():
    """Training loss should decrease monotonically."""
    rng = np.random.default_rng(7)
    X = rng.uniform(0, 5, size=(50, 1))
    y = X[:, 0] * 4 + rng.normal(0, 0.1, size=50)

    model = LinearRegression(learning_rate=0.05, n_iterations=500)
    model.fit(X, y)

    losses = model.loss_history
    assert losses[0] > losses[-1], "Loss should decrease over training"


def test_predict_before_fit_raises():
    """predict() before fit() should raise RuntimeError."""
    model = LinearRegression()
    with pytest.raises(RuntimeError):
        model.predict(np.array([[1.0]]))


def test_multi_feature():
    """Model works with multiple input features."""
    rng = np.random.default_rng(99)
    X = rng.uniform(0, 10, size=(200, 3))
    w_true = np.array([1.5, -2.0, 0.5])
    y = X.dot(w_true) + 3.0

    model = LinearRegression(learning_rate=0.01, n_iterations=3000)
    model.fit(X, y)

    assert model.score(X, y) > 0.99


def test_mean_squared_error():
    """MSE should be 0 for identical arrays and positive otherwise."""
    y = np.array([1.0, 2.0, 3.0])
    assert mean_squared_error(y, y) == 0.0

    y_pred = np.array([2.0, 3.0, 4.0])
    assert mean_squared_error(y, y_pred) == pytest.approx(1.0)
