"""
Paleo-emu package initialization.
Suppress ConvergenceWarnings from sklearn's Gaussian Process kernels.
"""
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress all ConvergenceWarnings from sklearn (covers child worker processes too)
warnings.filterwarnings("ignore", category=ConvergenceWarning)