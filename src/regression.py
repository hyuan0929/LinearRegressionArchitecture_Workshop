from __future__ import annotations
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def predict(theta0: float, theta1: float, X: np.ndarray) -> np.ndarray:
    return theta0 + theta1 * X[:, 0]

def gradient_descent(X: np.ndarray, y: np.ndarray, lr: float, iters: int):
    m = len(y)
    theta0, theta1 = 0.0, 0.0
    history = []
    for _ in range(iters):
        y_pred = predict(theta0, theta1, X)
        err = y_pred - y
        d0 = (1/m) * np.sum(err)
        d1 = (1/m) * np.sum(err * X[:, 0])
        theta0 -= lr * d0
        theta1 -= lr * d1
        history.append(float(np.mean((err) ** 2)))
    return theta0, theta1, history

def train_sklearn(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    m = LinearRegression()
    m.fit(X_train, y_train)
    return m

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }