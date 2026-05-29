"""
High-level runner for training and prediction.
"""

import inspect
import itertools
from pathlib import Path

import joblib
import numpy as np

from paleo_emu.config import NumberedSweep, load_config
from paleo_emu.export import save_prediction
from paleo_emu.load import load_forcing_data, load_training_data
from paleo_emu.training import TrainingGenerator


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------

def _format_sweep_filter_values(v) -> list[str]:
    """Parse runtime sweep filters into strings for comparison."""
    if isinstance(v, NumberedSweep):
        return v.values()
    if isinstance(v, dict):
        return NumberedSweep(**v).values()
    if isinstance(v, list):
        return [str(x) for x in v]
    return [str(v)]


def _norm(s: str) -> str:
    """Strip leading zeros from pure integers for comparison."""
    return str(int(s)) if s.isdigit() else s


def _matches_filter(sweep_vals: dict, sweep_overrides: dict) -> bool:
    """Return True if sweep_vals satisfies all runtime sweep filters."""
    for k, v in sweep_overrides.items():
        allowed = {_norm(x) for x in _format_sweep_filter_values(v)}
        if _norm(str(sweep_vals.get(k, ""))) not in allowed:
            return False
    return True


def _expand_pattern_config(config):
    """Yield (sweep_dict, forcing_file) pairs for a pattern scenario."""
    sweep_values = config.expanded_sweep_values()
    keys = list(sweep_values)
    value_lists = [sweep_values[key] for key in keys]

    for combo in itertools.product(*value_lists):
        values = dict(zip(keys, combo))
        forcing_file = config.forcing_input_pattern.format(**values)
        yield values, forcing_file


def _expand_scenario(scenario_cfg):
    """Yield (sweep_dict, forcing_file) pairs for a validated scenario config."""
    if scenario_cfg.kind == "single":
        yield {}, scenario_cfg.forcing_input
        return

    yield from _expand_pattern_config(scenario_cfg)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class PaleoEmuRunner:
    """
    High-level interface for training and prediction.

    Parameters
    ----------
    cfg_path : str or Path
        Path to a YAML configuration file.  If the path does not exist as-is,
        it is resolved relative to the caller's script location.

    Examples
    --------
    >>> runner = PaleoEmuRunner("example_PCA_GP.yml")
    >>> runner.train()
    >>> runner.predict()                                        # all scenarios
    >>> runner.predict("SSP585")                                # single file
    >>> runner.predict("past800ka_ens")                         # all members
    >>> runner.predict("past800ka_ens", member={"start": 1, "end": 10})  # members 1–10
    >>> runner.predict("past800ka_var")                         # all vars
    >>> runner.predict("past800ka_var", var=["sst", "precip"])  # subset of vars
    """

    def __init__(self, cfg_path):
        p = Path(cfg_path)
        if not p.is_absolute() and not p.exists():
            caller_dir = Path(inspect.stack()[1].filename).parent
            p = caller_dir / p
        self.cfg_path = p
        self.cfg = load_config(str(self.cfg_path))

    # ------------------------------------------------------------------
    def train(self) -> Path:
        """Load training data, fit the model, and save the artifact."""
        X, Y, var_name, _, lat_array, lon_array, _, var_attrs = load_training_data(self.cfg)
        training      = TrainingGenerator(self.cfg, X, Y, lat_array, lon_array,
                                          var_name=var_name, var_attrs=var_attrs)
        artifact_path = Path(training.run_training())
        print(f"[TRAIN] artifact saved → {artifact_path}")
        return artifact_path

    # ------------------------------------------------------------------
    def predict(self, scenario=None, output_dir=None, **sweep_overrides) -> None:
        """
        Predict forcing scenarios and save results as NetCDF.

        Parameters
        ----------
        scenario : str, optional
            Scenario key from the YAML ``forcing_data`` section.
            Defaults to all scenarios in the config.
        output_dir : str or Path, optional
            Output directory.  Defaults to ``<cfg_dir>/outputs/``.
        **sweep_overrides
            Runtime filter for pattern-based scenarios.  The keyword name must
            match a sweep dimension defined in the YAML.  Accepts the same
            formats as the YAML value (string, list, or a dict with start/end/width).
            Examples: ``member={"start": 1, "end": 10}``, ``var=["sst", "precip"]``.
        """
        out_dir = Path(output_dir) if output_dir else self.cfg_path.parent / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)

        scenario_list = (
            list(self.cfg.forcing_data.keys()) if scenario is None else [scenario]
        )

        artifact  = joblib.load(self._artifact_path())
        model     = artifact["model"]
        lat_array = artifact["lat_array"]
        lon_array = artifact["lon_array"]
        var_name  = artifact.get("var_name")
        var_attrs = artifact.get("var_attrs", {})
        n_lat, n_lon = len(lat_array), len(lon_array)

        for scen in scenario_list:
            scenario_cfg = self.cfg.forcing_data.get(scen)
            if scenario_cfg is None:
                raise KeyError(
                    f"Scenario '{scen}' not found. "
                    f"Available: {list(self.cfg.forcing_data.keys())}"
                )

            expansions = list(_expand_scenario(scenario_cfg))
            if sweep_overrides:
                expansions = [(sv, ff) for sv, ff in expansions
                              if _matches_filter(sv, sweep_overrides)]
                if not expansions:
                    print(f"[PREDICT] {scen}: no files match {sweep_overrides}, skipping.")
                    continue

            for sweep_vals, forcing_file in expansions:
                X_forcing     = load_forcing_data(self.cfg, forcing_file=forcing_file)
                Y_pred, Y_std = model.predict_with_variance(X_forcing)
                Y_pred_3d     = Y_pred.reshape(-1, n_lat, n_lon)
                Y_var_3d      = (
                    (Y_std ** 2).reshape(-1, n_lat, n_lon)
                    if Y_std is not None else np.zeros_like(Y_pred_3d)
                )

                sweep_suffix = "_".join(str(v) for v in sweep_vals.values())
                fname = f"{self.cfg.model_run_name}_{scen}"
                if sweep_suffix:
                    fname += f"_{sweep_suffix}"
                fname += "_prediction"

                save_prediction(Y_pred_3d, Y_var_3d, lat_array, lon_array,
                                output_dir=str(out_dir), file_name=fname,
                                var_name=var_name, var_attrs=var_attrs)
                print(f"[PREDICT] {scen}"
                      + (f" {sweep_vals}" if sweep_vals else "")
                      + f" → {out_dir / fname}.nc")

    # ------------------------------------------------------------------
    def _artifact_path(self) -> Path:
        artifact_name = (
            self.cfg.artifact_name
            or f"{self.cfg.model_run_name}_fitted_pipeline.joblib"
        )
        if self.cfg.output_dir is not None:
            return Path(self.cfg.output_dir) / artifact_name
        return self.cfg_path.parent / "pretrained" / artifact_name
