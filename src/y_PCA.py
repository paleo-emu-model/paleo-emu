import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import BaseEstimator, TransformerMixin

# ----------------------
# 1. Generate Synthetic Data (y rows sum to 1)
# ----------------------
X, y_raw = make_regression(
    n_samples=1000,
    n_features=10,
    n_targets=5,
    noise=0.1,
    random_state=42
)
y = y_raw / y_raw.sum(axis=1, keepdims=True)  # Normalize rows to sum=1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------
# 2. Custom PCA Transformer for y (with n_components tuning)
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
        return self.pca.inverse_transform(y)  # No renormalization

# ----------------------
# 3. Build Pipeline with TransformedTargetRegressor
# ----------------------
inner_pipeline = Pipeline([
    ('scaler_x', StandardScaler()),
    ('regressor', MultiOutputRegressor(SVR()))
])

pipeline = Pipeline([
    ('model', TransformedTargetRegressor(
        regressor=inner_pipeline,
        transformer=PCATargetTransformer()  # n_components will be tuned
    ))
])

# ----------------------
# 4. Define Hyperparameter Grid
# ----------------------
param_grid = {
    'model__transformer__n_components': [2, 3, 4],  # PCA components for y
    'model__regressor__regressor__estimator__C': [0.1, 1, 10],  # SVR hyperparameters
    'model__regressor__regressor__estimator__kernel': ['linear', 'rbf']
}

# ----------------------
# 5. Run GridSearchCV
# ----------------------
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',  # Or any regression metric
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# ----------------------
# 6. Evaluate Best Model
# ----------------------
print("Best parameters:", grid_search.best_params_)
y_pred = grid_search.predict(X_test)
print("Test MAE:", np.mean(np.abs(y_test - y_pred)))