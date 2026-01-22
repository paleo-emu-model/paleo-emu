"""
Paleo-emu package initialization.
Suppress ConvergenceWarnings from sklearn's Gaussian Process kernels.
"""
import warnings

# Suppress ConvergenceWarnings about kernel bounds - these are expected for PCA-encoded data
# with different characteristic scales and don't affect model performance
warnings.filterwarnings(
    "ignore",
    category=Warning,
    module="sklearn.gaussian_process.kernels"
)
# Suppress ConvergenceWarnings about noise_level bounds
warnings.filterwarnings(
    "ignore",
    message=".*optimal value found for dimension.*parameter k2__noise_level.*",
    category=Warning
)