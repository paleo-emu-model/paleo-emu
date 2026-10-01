"""
Paleo-emu package initialization.
Suppress ConvergenceWarnings from sklearn's Gaussian Process kernels.
"""
import warnings
from importlib.metadata import version, PackageNotFoundError
from sklearn.exceptions import ConvergenceWarning

try:
    __version__ = version("paleo-emu")
except PackageNotFoundError:
    __version__ = "unknown"

# Suppress all ConvergenceWarnings from sklearn (covers child worker processes too)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from paleo_emu.runner import PaleoEmuRunner    # noqa: E402
from paleo_emu.plotter import PaleoEmuPlotter  # noqa: E402

plotter = PaleoEmuPlotter()  # module-level singleton: from paleo_emu import plotter