"""
Machine Learning - Regressão Linear (Linear Regression)

Implementação de regressão linear usando gradiente descendente.
Linear regression implementation using gradient descent.
"""

import numpy as np


class LinearRegression:
    """
    Regressão Linear usando gradiente descendente.
    Linear Regression using gradient descent.

    Parameters
    ----------
    learning_rate : float
        Taxa de aprendizado (step size for gradient descent).
    n_iterations : int
        Número de iterações do treinamento.
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """
        Treina o modelo com os dados fornecidos.
        Train the model with the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Dados de entrada (input features).
        y : array-like of shape (n_samples,)
            Valores alvo (target values).

        Returns
        -------
        self
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            y_pred = self._predict_raw(X)
            error = y_pred - y

            dw = (1 / n_samples) * X.T.dot(error)
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = (1 / (2 * n_samples)) * np.sum(error ** 2)
            self.loss_history.append(loss)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Faz previsões para os dados de entrada.
        Make predictions for the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Dados de entrada (input features).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Valores previstos (predicted values).
        """
        if self.weights is None:
            raise RuntimeError("O modelo precisa ser treinado antes de fazer previsões. "
                               "Call fit() before predict().")
        return self._predict_raw(np.array(X, dtype=float))

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        return X.dot(self.weights) + self.bias

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula o coeficiente de determinação R².
        Compute the R² (coefficient of determination) score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Dados de entrada (input features).
        y : array-like of shape (n_samples,)
            Valores reais (true target values).

        Returns
        -------
        r2 : float
            Coeficiente R² (R² score). Valor 1.0 indica ajuste perfeito.
        """
        y = np.array(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return 1.0
        return float(1 - ss_res / ss_tot)


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula o Erro Quadrático Médio (Mean Squared Error).

    Parameters
    ----------
    y_true : array-like
        Valores reais (true values).
    y_pred : array-like
        Valores previstos (predicted values).

    Returns
    -------
    mse : float
        Erro quadrático médio (MSE).
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))
