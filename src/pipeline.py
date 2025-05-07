import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.model_selection import train_test_split

# =============================================================================
# 1. Synthetic Data Generation
# =============================================================================
n_samples = 1000
n_weights = 10        # Input dimension (weight vectors)
height, width = 16, 16  # Output map dimensions

# Generate random weights and normalized 2D maps
X = np.random.rand(n_samples, n_weights)
y = np.random.rand(n_samples, height, width)
y = y / y.sum(axis=(1, 2), keepdims=True)  # Force maps to sum to 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================================================================
# 2. Custom Transformers
# =============================================================================
class MapFlattener(BaseEstimator, TransformerMixin):
    """Flattens 2D maps into 1D vectors for PCA."""
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if y is None:
            return X
        return y.reshape(-1, self.height * self.width)

class MapReconstructor(BaseEstimator, TransformerMixin):
    """Reconstructs 2D maps from PCA components with normalization."""
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if y is None:
            return X
        # Reshape and enforce constraints
        y_maps = y.reshape(-1, self.height, self.width)
        y_maps = np.maximum(y_maps, 0)  # Non-negativity
        y_maps = y_maps / y_maps.sum(axis=(1, 2), keepdims=True)  # Normalize
        return y_maps

class PCARegressor(BaseEstimator, RegressorMixin):
    """Combines PCA and regression into a single step."""
    def __init__(self, n_components, height, width, regressor=None):
        self.n_components = n_components
        self.height = height
        self.width = width
        self.regressor = regressor if regressor else LinearRegression()
        self.pca = PCA(n_components=n_components)

    def fit(self, X, y):
        y_flat = y.reshape(-1, self.height * self.width)
        self.pca.fit(y_flat)
        y_pca = self.pca.transform(y_flat)
        self.regressor.fit(X, y_pca)
        return self

    def predict(self, X):
        y_pca = self.regressor.predict(X)
        y_flat = self.pca.inverse_transform(y_pca)
        y_maps = y_flat.reshape(-1, self.height, self.width)
        y_maps = np.maximum(y_maps, 0)
        y_maps = y_maps / y_maps.sum(axis=(1, 2), keepdims=True)
        return y_maps

# =============================================================================
# 3. Build and Run Pipeline
# =============================================================================
pipeline = Pipeline([
    ('flatten', MapFlattener(height, width)),  # Flatten maps for PCA
    ('pca_regress', PCARegressor(
        n_components=20,
        height=height,
        width=width,
        regressor=LinearRegression()
    ))
])

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# =============================================================================
# 4. Evaluation
# =============================================================================
# Check map properties
print("Input map sums (sample):", np.round(y_test[0].sum(), 4))  # Should be 1.0
print("Predicted map sums (sample):", np.round(y_pred[0].sum(), 4))  # Should be 1.0

# Reconstruction error
mae = np.mean(np.abs(y_test - y_pred))
print(f"Mean Absolute Error: {mae:.4f}")