#Model Implementation: Linear Regression from Scratch using Gradient Descent

# From-scratch Implementation (Gradient Descent)
import numpy as np

def train_linear_regression_gd(
    X_train_scaled,
    y_train,
    learning_rate=0.05,
    epochs=3000
):
    """
    Train a univariate linear regression model using gradient descent.

    Hypothesis:
        h(x) = theta0 + theta1 * x
    """
    x = X_train_scaled[:, 0]
    n = len(y_train)

    theta0 = 0.0  # intercept
    theta1 = 0.0  # slope
    loss_history = []

    for _ in range(epochs):
        # Predictions
        y_pred = theta0 + theta1 * x

        # Errors
        error = y_pred - y_train

        # Gradients (MSE)
        d_theta0 = (2 / n) * np.sum(error)
        d_theta1 = (2 / n) * np.sum(error * x)

        # Parameter update
        theta0 -= learning_rate * d_theta0
        theta1 -= learning_rate * d_theta1

        # Store training loss
        loss_history.append(np.mean(error ** 2))

    return theta0, theta1, loss_history


def predict_linear_regression(theta0, theta1, X_scaled):
    """
    Make predictions using a trained univariate linear regression model.
    """
    x = X_scaled[:, 0]
    return theta0 + theta1 * x


# Train the model
theta0, theta1, loss_history = train_linear_regression_gd(
    X_train_scaled, y_train
)

# Predict on test set
y_pred_gd = predict_linear_regression(
    theta0, theta1, X_test_scaled
)

print("From-scratch model parameters:")
print("Intercept (theta0):", theta0)
print("Slope (theta1):", theta1)
print("Final training MSE:", loss_history[-1])


# Model Implementation: scikit-learn Baseline Implementation
from sklearn.linear_model import LinearRegression

# Train scikit-learn model
sk_model = LinearRegression()
sk_model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred_sk = sk_model.predict(X_test_scaled)

print("scikit-learn model parameters:")
print("Intercept:", sk_model.intercept_)
print("Coefficient:", sk_model.coef_[0])