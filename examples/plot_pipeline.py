"""
==================================================
Gaussian Process Regression with Pipeline
==================================================

This example demonstrates GPs with pipelines and kernel optimization in sklearn
"""

# Authors: Your Name
# License: BSD-3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern

##############################################################################
# Data Generation
# ---------------
# Create synthetic data with non-linear relationships

X, y = make_regression(n_samples=200, n_features=1, noise=20, random_state=42)
y = y**2 + 50 * np.sin(X.squeeze())  # Add non-linearity

##############################################################################
# Train-Test Split
# ----------------
# Split data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

##############################################################################
# Model Pipeline
# --------------
# Create pipeline with scaling and GP regression
# Using a Matern kernel
# For more kernel examples see: 
# https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_prior_posterior.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-prior-posterior-py

kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)

gp_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('gpr', GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.1,  # Reduced alpha since we have WhiteKernel
        n_restarts_optimizer=10,  # More restarts for better optimization
        random_state=42
    ))
])

# Fit the pipeline
gp_pipeline.fit(X_train, y_train)

##############################################################################
# Prediction and Plotting
# ----------------------
# Make predictions and visualize results

# Create figure explicitly
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)  # Get explicit Axes reference

# Get scaled values for plotting (accessed through the pipeline)
X_train_scaled = gp_pipeline.named_steps['scaler'].transform(X_train)
X_test_scaled = gp_pipeline.named_steps['scaler'].transform(X_test)

# Plot training and test data
ax.scatter(X_train_scaled, y_train, c='blue', label='Training data')
ax.scatter(X_test_scaled, y_test, c='green', label='Test data')

# Create prediction line
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_plot_scaled = gp_pipeline.named_steps['scaler'].transform(X_plot)
y_pred, y_std = gp_pipeline.named_steps['gpr'].predict(X_plot_scaled, return_std=True)

# Plot prediction and uncertainty
ax.plot(X_plot_scaled, y_pred, 'r-', label='GP prediction')
ax.fill_between(
    X_plot_scaled.squeeze(),
    y_pred - 1.96 * y_std,
    y_pred + 1.96 * y_std,
    color='pink',
    alpha=0.3,
    label='95% confidence interval'
)

ax.set_title("Gaussian Process Regression with Pipeline")
ax.set_xlabel("Normalized feature values")
ax.set_ylabel("Target values")
ax.legend()

# Show final kernel parameters
final_kernel = gp_pipeline.named_steps['gpr'].kernel_
ax.text(0.02, 0.95, f"Optimized kernel:\n{final_kernel}", 
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Final adjustments
plt.tight_layout()

plt.show()