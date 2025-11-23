# _load.py (or config_loader.py)
from typing import List, Optional, Union
import yaml
from pydantic import BaseModel, field_validator, ValidationError, model_validator, ConfigDict
import os 
import xarray as xr
import pandas as pd
from pathlib import Path

try:
    from typing import Literal
except ImportError:  # pragma: no cover
    from typing_extensions import Literal

from typing import List, Optional, Union, Tuple

# Registry of valid kernels 
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel

_BASE_KERNEL_REGISTRY = {
    "RBF": RBF(length_scale=1.0),
    "Matern_nu_15": Matern(length_scale=1.0, nu=1.5),
    "Matern_nu_25": Matern(length_scale=1.0, nu=2.5),
}

# Regressor config
class _RegressorConfig(BaseModel):
    kernels: List[str]
    n_restarts_optimizer: int = 0
    whitekernel_noise_level: float = 1e-2
    whitekernel_noise_level_bounds: Tuple[float, float] = (1e-5, 1e1)

    @field_validator("kernels")
    @classmethod
    def validate_kernels(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("kernels list must not be empty")
        unknown = [k for k in v if k not in _BASE_KERNEL_REGISTRY]
        if unknown:
            raise ValueError(f"Unknown kernel names: {unknown}")
        return v

    @field_validator("whitekernel_noise_level")
    @classmethod
    def validate_noise_level(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("whitekernel_noise_level must be > 0")
        return v

    @field_validator("whitekernel_noise_level_bounds")
    @classmethod
    def validate_noise_bounds(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        if len(v) != 2:
            raise ValueError("whitekernel_noise_level_bounds must have length 2")
        low, high = v
        if low <= 0 or high <= 0:
            raise ValueError("whitekernel_noise_level_bounds must be positive")
        if low >= high:
            raise ValueError(
                "whitekernel_noise_level_bounds[0] must be < whitekernel_noise_level_bounds[1]"
            )
        return v

def make_kernel(name: str, cfg: _RegressorConfig):
    """
    Construct a kernel from its base name and the WhiteKernel
    parameters specified in the regressor_config.
    """
    base_kernel = _BASE_KERNEL_REGISTRY[name]
    return base_kernel + WhiteKernel(
        noise_level=cfg.whitekernel_noise_level,
        noise_level_bounds=cfg.whitekernel_noise_level_bounds,
    )



# Encoder configs
class _PCAEncoderConfig(BaseModel):
    encoder_type: Literal["PCA"]
    n_components: Optional[int] = None
    pca_variance_ratio: Optional[float] = None

    # Forbid unexpected keys (e.g. latent_dim in PCA config)
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def check_pca_fields(self) -> "PCAEncoderConfig":
        n_components = self.n_components
        var_ratio = self.pca_variance_ratio

        # Require at least one of the two
        if n_components is None and var_ratio is None:
            raise ValueError(
                "For encoder_type='PCA', you must set either "
                "'n_components' or 'pca_variance_ratio'."
            )

        if n_components is not None and n_components <= 0:
            raise ValueError("n_components must be > 0")

        if var_ratio is not None and not (0.0 < var_ratio <= 1.0):
            raise ValueError("pca_variance_ratio must be in (0, 1].")

        return self


class _LearnedEncoderConfig(BaseModel):
    # e.g. VAE / AE-style encoder; no encoder_type needed in YAML
    latent_dim: int
    epochs: int
    learning_rate: float
    batch_size: int
    kl_weight: float

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def check_learned_fields(self) -> "LearnedEncoderConfig":
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be > 0")
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.kl_weight < 0:
            raise ValueError("kl_weight must be >= 0")
        return self


# Cross validation config
class _CVConfig(BaseModel):
    folds: int
    n_jobs: int = 1
    scoring: str = "neg_mean_squared_error"

    @field_validator("folds")
    @classmethod
    def validate_folds(cls, v: int) -> int:
        if v < 2:
            raise ValueError("folds must be >= 2")
        return v

# Top-level config

class PaleoEmuConfig(BaseModel):
    regressor_config: _RegressorConfig
    cv: _CVConfig
    random_state: int
    model_run_name: str
    # encoder_config can be either PCA-style or learned-encoder-style
    encoder_config: Union[_PCAEncoderConfig, _LearnedEncoderConfig]

    training_file_path: Path
    X_input_file_name: str
    Y_input_file_name: str
    X_column_names: List[str]

    @field_validator("X_column_names")
    @classmethod
    def validate_x_column_names(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("X_column_names must not be empty")
        return v


def load_config(path: str) -> PaleoEmuConfig:
    """
    Load and validate YAML config. Raises ValidationError or yaml.YAMLError
    if something is wrong.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # Pydantic v2 does all structural + semantic checks here
    return PaleoEmuConfig(**raw)