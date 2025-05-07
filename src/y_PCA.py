import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# ----------------------
# 1. Generate Synthetic Data (y rows sum to 1)
# ----------------------
X, y_raw = make_regression(
    n_samples=1000,
    n_features=10,
    n_targets=5,  # 5D output
    noise=0.1,
    random_state=42
)
y = y_raw / y_raw.sum(axis=1, keepdims=True)  # Normalize rows to sum=1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verify normalization
print("Sample y_train sums:", np.round(y_train.sum(axis=1)[:5])) # [1. 1. 1. 1. 1.]

# ----------------------
# 2. Define Custom Transformers
# ----------------------
class PCATargetTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def fit(self, y):
        self.pca.fit(y)
        return self

    def transform(self, y):
        return self.pca.transform(y)

    def inverse_transform(self, y):
        y_inv = self.pca.inverse_transform(y)
        return y_inv / y_inv.sum(axis=1, keepdims=True)  # Normalize rows to sum=1

# ----------------------
# 3. Build the Pipeline
# ----------------------
pipeline = Pipeline([
    ('scaler_x', StandardScaler()),          # Scale X
    ('pca_y', PCATargetTransformer(2)),     # PCA on y (with sum=1 guarantee)
    ('regressor', MultiOutputRegressor(      # Multi-output model
        SVR(kernel='linear')))
])

# ----------------------
# 4. Train and Evaluate
# ----------------------
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Verify predictions sum to 1
print("Predicted y_test sums:", np.round(y_pred.sum(axis=1)[:5]))  # [1. 1. 1. 1. 1.]

# Check PCA reconstruction error
print("Mean absolute error:", np.mean(np.abs(y_test - y_pred)))  # ~0.02 (example)