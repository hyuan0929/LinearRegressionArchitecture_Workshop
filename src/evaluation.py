# Model Evaluation Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate_model(model_name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"--- {model_name} ---")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE : {mae:.2f}")
    print(f"R^2 : {r2:.4f}\n")

# Evaluate both models
evaluate_model("From-scratch (Gradient Descent)", y_test, y_pred_gd)
evaluate_model("scikit-learn Baseline", y_test, y_pred_sk)

# Visualization of Results
import matplotlib.pyplot as plt
import numpy as np

# Order by X values for smooth regression lines
order = np.argsort(X_test_scaled[:, 0])
x_sorted = X_test_scaled[order, 0]

y_line_gd = y_pred_gd[order]
y_line_sk = y_pred_sk[order]

plt.figure(figsize=(8, 6))

# Scatter plot of test data
plt.scatter(
    X_test_scaled[:, 0],
    y_test,
    s=10,
    alpha=0.5,
    label="Data points"
)

# Regression lines
plt.plot(x_sorted, y_line_gd, label="From-scratch Regression Line")
plt.plot(x_sorted, y_line_sk, label="scikit-learn Regression Line")

# Axis limits
plt.xlim(0, 10)
plt.ylim(0, 6000000)

# Labels and title
plt.xlabel(f"{independent_variable} (standardized)")
plt.ylabel(dependent_variable)
plt.title("Regression Line vs Data Points")
plt.legend()

plt.show()