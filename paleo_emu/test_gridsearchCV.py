# main.py
from sklearn.datasets import make_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from paleo_emu.load_config import load_config, make_kernel

# ---- 1. Load & validate config ----
cfg = load_config("./paleo_emu/config.yaml")

# ---- 2. Build param_grid from config ----
param_grid = {
    # Note: we use "gpr__kernel" because the GPR is in a Pipeline step named "gpr"
    "gpr__kernel": [
        make_kernel(name, cfg.regressor_config)
        for name in cfg.regressor_config.kernels
    ],
}

# ---- 3. Fake data ----
X, y = make_regression(
    n_samples=200,
    n_features=5,
    noise=0.2,
    random_state=cfg.random_state,
)

# ---- 4. Base estimator (inside a Pipeline) ----
gpr = GaussianProcessRegressor(
    normalize_y=True,
    n_restarts_optimizer=cfg.regressor_config.n_restarts_optimizer,
    random_state=cfg.random_state,
)

# Pipeline: normalize X, then fit GPR
pipe = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("gpr", gpr),
    ]
)

# ---- 5. GridSearchCV over kernels ----
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=cfg.cv.folds,
    n_jobs=cfg.cv.n_jobs,
    scoring=cfg.cv.scoring,
)

grid.fit(X, y)

print("Best params:", grid.best_params_)
print("Best score:", grid.best_score_)
