"""
High-level plotting interface for paleo-emu predictions.
Reads prediction NetCDF files produced by PaleoEmuRunner.predict().
"""

import inspect
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.util as cutil
import xarray as xr

from paleo_emu.config import load_config


class PaleoEmuPlotter:
    """
    High-level interface for plotting paleo-emu prediction outputs.

    Reads prediction NetCDF files produced by ``PaleoEmuRunner.predict()``.

    Two usage styles:

    **Style 1 — upfront initialisation** (efficient for multiple calls)::

        p = PaleoEmuPlotter("example_PCA_GP.yml")
        p.map("SSP585")
        p.timeseries("SSP585")

    **Style 2 — per-call config** (uses the module-level singleton)::

        from paleo_emu import plotter
        plotter.map("SSP585", cfg="example_PCA_GP.yml")

    For the full parameter reference, call ``help()`` on any method::

        help(PaleoEmuPlotter.map)
        help(PaleoEmuPlotter.delta_map)

    Methods
    -------
    map
        Spatial map (averaged or single step).
    timeseries
        Area-weighted mean timeseries.
    uncertainty_map
        Spatial map of prediction uncertainty (1 σ).
    zonal_mean
        Latitude-vs-value profile.
    delta_map
        Difference map between two scenarios.
    mapgif
        Animated GIF over time steps (requires pillow).
    """

    def __init__(self, cfg_path=None):
        self.cfg      = None
        self.cfg_path = None
        if cfg_path is not None:
            self.cfg_path, self.cfg = _load_cfg(
                cfg_path, Path(inspect.stack()[1].filename).parent
            )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def map(self, scenario, var=None, time=None,
            region=None, zonal_mean=False,
            title=None, label=None, save_name=None, fmt="png", save_dir=None,
            filepath=None, cfg=None):
        """
        Spatial map of the prediction field.

        Parameters
        ----------
        scenario : str
            Key in the YAML ``forcing_data`` block (e.g. ``"SSP585"``).
        var : str, optional
            Must match a ``var`` sweep dimension for this scenario in the
            YAML config.  Used to locate the NC file via glob
            ``*{var}*_{scenario}_prediction.nc`` and as the plot label.
            Omit for single-file scenarios.
        time : None, int, list, range, or slice, default None
            Time selection — see Notes.
        region : list of float, optional
            ``[lon_min, lon_max, lat_min, lat_max]``.  Defaults to global.
        zonal_mean : bool, default False
            Add a latitude-vs-value side panel next to the map.
        title : str, optional
            Override the auto-generated plot title.
        save_name : str, optional
            Output filename stem (no extension).  Default: ``map_<t_label>``.
        fmt : {"png", "pdf"}, default "png"
            Output image format.
        save_dir : str or Path, optional
            Output directory.  Default: ``<cfg_dir>/figures/``.
        filepath : str or Path, optional
            Load this NC directly, bypassing auto-derived path.
            If it conflicts with the derived path you are prompted to choose.
        cfg : str or Path, optional
            YAML config for this call.  Overrides the instance config.

        Notes
        -----
        **time= semantics**

        - ``time=None``            → mean of all steps (default)
        - ``time=[0, 1, 2, 3]``   → mean of those steps
        - ``time=range(a, b, c)`` → mean of that range
        - ``time=slice(a, b, c)`` → mean of that slice
        - ``time=5`` (int)        → only step 5, no averaging

        Examples
        --------
        >>> p = PaleoEmuPlotter("example_PCA_GP.yml")
        >>> p.map("SSP585")
        >>> p.map("SSP585", time=[0, 1, 2, 3])
        >>> p.map("SSP585", time=5, zonal_mean=True)
        >>> p.map("SSP585", region=[220, 305, 30, 90],
        ...       time=range(0, 20, 2), fmt="pdf",
        ...       save_name="ssp585_map", save_dir="figures/")
        >>> p.map("SSP585", filepath="outputs/my_pred.nc")
        >>> p.map("past800ka_var", var="sst", time=[0, 1, 2, 3])
        """
        cfg_obj, cfg_path = self._resolve_cfg(cfg)
        _validate_var(scenario, var, cfg_obj)
        ds        = self._load(scenario, var, filepath, cfg_obj, cfg_path)
        pred_var  = _get_pred_var(ds)
        frame     = _sel_time_avg(ds[pred_var].values, time)
        lat       = ds["latitude"].values
        lon       = ds["longitude"].values
        long_name = ds[pred_var].attrs.get("long_name", var or pred_var)
        units     = ds[pred_var].attrs.get("units", "")

        if region is not None:
            frame, lat, lon = _crop(frame, lat, lon, region)

        t_label = _time_label(time)
        fig, axes = _map_fig(zonal_mean)
        im, bar_label = self._draw_map(
            axes[0], frame, lat, lon, long_name, units,
            title=title or f"{long_name} — {scenario}  {t_label}",
            label=label)
        if zonal_mean:
            _draw_zonal_panel(axes[1], frame, lat)
            cbar = fig.colorbar(im, cax=axes[2], orientation="horizontal")
        else:
            cbar = fig.colorbar(im, ax=axes[0], orientation="horizontal",
                                pad=0.05, fraction=0.046)
        cbar.set_label(bar_label, fontsize=12)
        out = self._out(save_name or f"map_{t_label}", scenario, var,
                        fmt, save_dir, cfg_obj, cfg_path)
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        print(f"[PLOT] map → {out}")

    def timeseries(self, scenario, var=None, time=None,
                   region=None,
                   title=None, label=None, save_name=None, fmt="png", save_dir=None,
                   filepath=None, cfg=None):
        """
        Area-weighted mean timeseries (global or regional).

        Parameters
        ----------
        scenario : str
        var : str, optional
            See ``map()`` for details.
        time : None, int, list, range, or slice, default None
            Steps to include.  ``None`` → all steps.  Unlike ``map()``,
            no averaging is applied — each step becomes a point on the plot.
        region : list of float, optional
            ``[lon_min, lon_max, lat_min, lat_max]``.  Defaults to global.
        title : str, optional
        save_name : str, optional
            Default: ``timeseries``.
        fmt : {"png", "pdf"}, default "png"
        save_dir : str or Path, optional
            Default: ``<cfg_dir>/figures/``.
        filepath : str or Path, optional
        cfg : str or Path, optional

        Examples
        --------
        >>> p = PaleoEmuPlotter("example_PCA_GP.yml")
        >>> p.timeseries("SSP585")
        >>> p.timeseries("SSP585", time=range(50),
        ...              region=[220, 305, 30, 90],
        ...              save_name="north_atl_ts", fmt="pdf")
        """
        cfg_obj, cfg_path = self._resolve_cfg(cfg)
        _validate_var(scenario, var, cfg_obj)
        ds        = self._load(scenario, var, filepath, cfg_obj, cfg_path)
        pred_var  = _get_pred_var(ds)
        pred      = ds[pred_var].values
        lat       = ds["latitude"].values
        lon       = ds["longitude"].values
        long_name = ds[pred_var].attrs.get("long_name", var or pred_var)
        units     = ds[pred_var].attrs.get("units", "")

        if region is not None:
            pred, lat, lon = _crop(pred, lat, lon, region)

        if isinstance(time, int):
            t_idx = [time]
        elif isinstance(time, slice):
            t_idx = list(range(*time.indices(pred.shape[0])))
        elif time is not None:
            t_idx = list(time)
        else:
            t_idx = list(range(pred.shape[0]))
        pred   = pred[t_idx]
        w      = _weights(lat)
        series = np.nansum(pred * w[np.newaxis, :, np.newaxis], axis=(1, 2))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t_idx, series)
        y_label = label if label is not None else (
            f"{long_name} [{units}]" if units else long_name)
        ax.set_xlabel("Time index", fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        region_label = "regional" if region else "global"
        ax.set_title(title or f"{long_name} — {scenario}  {region_label} mean",
                     fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        out = self._out(save_name or "timeseries", scenario, var, fmt,
                        save_dir, cfg_obj, cfg_path)
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        print(f"[PLOT] timeseries → {out}")

    def uncertainty_map(self, scenario, var=None, time=None,
                        region=None, zonal_mean=False,
                        title=None, label=None, save_name=None, fmt="png", save_dir=None,
                        filepath=None, cfg=None):
        """
        Spatial map of prediction uncertainty (1 σ = sqrt of variance).

        Parameters
        ----------
        scenario : str
        var : str, optional
            See ``map()`` for details.
        time : None, int, list, range, or slice, default None
            Same averaging semantics as ``map()``.
        region : list of float, optional
        zonal_mean : bool, default False
        title : str, optional
        save_name : str, optional
            Default: ``uncertainty_<t_label>``.
        fmt : {"png", "pdf"}, default "png"
        save_dir : str or Path, optional
        filepath : str or Path, optional
        cfg : str or Path, optional

        Examples
        --------
        >>> p = PaleoEmuPlotter("example_PCA_GP.yml")
        >>> p.uncertainty_map("SSP585")
        >>> p.uncertainty_map("SSP585", time=[0, 1, 2, 3],
        ...                   zonal_mean=True,
        ...                   save_dir="figures/uncertainty/")
        """
        cfg_obj, cfg_path = self._resolve_cfg(cfg)
        _validate_var(scenario, var, cfg_obj)
        ds        = self._load(scenario, var, filepath, cfg_obj, cfg_path)
        pred_var  = _get_pred_var(ds)
        frame     = np.sqrt(_sel_time_avg(ds["variance"].values, time))
        lat       = ds["latitude"].values
        lon       = ds["longitude"].values
        long_name = ds[pred_var].attrs.get("long_name", var or pred_var)
        units     = ds[pred_var].attrs.get("units", "")

        if region is not None:
            frame, lat, lon = _crop(frame, lat, lon, region)

        t_label = _time_label(time)
        fig, axes = _map_fig(zonal_mean)
        im, bar_label = self._draw_map(
            axes[0], frame, lat, lon,
            f"Uncertainty (1σ) — {long_name}", units,
            cmap="YlOrRd",
            title=title or f"Uncertainty — {scenario}  {t_label}",
            label=label)
        if zonal_mean:
            _draw_zonal_panel(axes[1], frame, lat)
            cbar = fig.colorbar(im, cax=axes[2], orientation="horizontal")
        else:
            cbar = fig.colorbar(im, ax=axes[0], orientation="horizontal",
                                pad=0.05, fraction=0.046)
        cbar.set_label(bar_label, fontsize=12)
        out = self._out(save_name or f"uncertainty_{t_label}", scenario, var,
                        fmt, save_dir, cfg_obj, cfg_path)
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        print(f"[PLOT] uncertainty map → {out}")

    def zonal_mean(self, scenario, var=None, time=None,
                   title=None, label=None, save_name=None, fmt="png", save_dir=None,
                   filepath=None, cfg=None):
        """
        Standalone latitude-vs-value profile (longitude-averaged).

        Parameters
        ----------
        scenario : str
        var : str, optional
            See ``map()`` for details.
        time : None, int, list, range, or slice, default None
            Same averaging semantics as ``map()``.
        title : str, optional
        save_name : str, optional
            Default: ``zonal_mean_<t_label>``.
        fmt : {"png", "pdf"}, default "png"
        save_dir : str or Path, optional
        filepath : str or Path, optional
        cfg : str or Path, optional

        Examples
        --------
        >>> p = PaleoEmuPlotter("example_PCA_GP.yml")
        >>> p.zonal_mean("SSP585")
        >>> p.zonal_mean("SSP585", time=range(0, 10),
        ...              title="Early SSP585 — zonal mean", fmt="pdf")
        """
        cfg_obj, cfg_path = self._resolve_cfg(cfg)
        _validate_var(scenario, var, cfg_obj)
        ds        = self._load(scenario, var, filepath, cfg_obj, cfg_path)
        pred_var  = _get_pred_var(ds)
        frame     = _sel_time_avg(ds[pred_var].values, time)
        lat       = ds["latitude"].values
        long_name = ds[pred_var].attrs.get("long_name", var or pred_var)
        units     = ds[pred_var].attrs.get("units", "")

        t_label = _time_label(time)
        zonal = np.nanmean(frame, axis=1)
        fig, ax = plt.subplots(figsize=(5, 7))
        ax.plot(zonal, lat)
        x_label = label if label is not None else (
            f"{long_name} [{units}]" if units else long_name)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel("Latitude (°)", fontsize=12)
        ax.set_title(title or f"Zonal mean — {scenario}  {t_label}", fontsize=12)
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = self._out(save_name or f"zonal_mean_{t_label}", scenario, var,
                        fmt, save_dir, cfg_obj, cfg_path)
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        print(f"[PLOT] zonal mean → {out}")

    def delta_map(self, scenario, ref_scenario, var=None, time=None,
                  region=None, zonal_mean=False,
                  title=None, label=None, save_name=None, fmt="png", save_dir=None,
                  filepath=None, ref_filepath=None, cfg=None):
        """
        Difference map: ``scenario`` − ``ref_scenario``.

        Parameters
        ----------
        scenario : str
            Scenario to subtract *from* (the signal).
        ref_scenario : str
            Reference scenario (subtracted).
        var : str, optional
            See ``map()`` for details.
        time : None, int, list, range, or slice, default None
            Applied to **both** datasets before differencing.
            Same averaging semantics as ``map()``.
        region : list of float, optional
        zonal_mean : bool, default False
        title : str, optional
        save_name : str, optional
            Default: ``delta_{scenario}_minus_{ref_scenario}_<t_label>``.
        fmt : {"png", "pdf"}, default "png"
        save_dir : str or Path, optional
        filepath : str or Path, optional
            Explicit NC file for ``scenario``.
        ref_filepath : str or Path, optional
            Explicit NC file for ``ref_scenario``.
        cfg : str or Path, optional

        Examples
        --------
        >>> p = PaleoEmuPlotter("example_PCA_GP.yml")
        >>> p.delta_map("SSP585", ref_scenario="SSP126")
        >>> p.delta_map("SSP585", ref_scenario="SSP126",
        ...             time=[0, 1, 2, 3], zonal_mean=True,
        ...             ref_filepath="outputs/ssp126_pred.nc",
        ...             save_name="delta_ssp585_ssp126", fmt="pdf")
        """
        cfg_obj, cfg_path = self._resolve_cfg(cfg)
        _validate_var(scenario,     var, cfg_obj)
        _validate_var(ref_scenario, var, cfg_obj)
        ds_s   = self._load(scenario,     var, filepath,     cfg_obj, cfg_path)
        ds_r   = self._load(ref_scenario, var, ref_filepath, cfg_obj, cfg_path)
        pred_var  = _get_pred_var(ds_s)
        s_fr   = _sel_time_avg(ds_s[pred_var].values, time)
        r_fr   = _sel_time_avg(ds_r[pred_var].values, time)
        lat    = ds_s["latitude"].values
        lon    = ds_s["longitude"].values
        long_name = ds_s[pred_var].attrs.get("long_name", var or pred_var)
        units     = ds_s[pred_var].attrs.get("units", "")

        diff = s_fr - r_fr

        if region is not None:
            diff, lat, lon = _crop(diff, lat, lon, region)

        t_label = _time_label(time)
        fig, axes = _map_fig(zonal_mean)
        lim = np.nanpercentile(np.abs(diff), 95)
        im, bar_label = self._draw_map(
            axes[0], diff, lat, lon,
            f"Δ{long_name}", units,
            cmap="RdBu_r", vmin=-lim, vmax=lim,
            title=title or
            f"Δ{long_name}: {scenario} − {ref_scenario}  {t_label}",
            label=label)
        if zonal_mean:
            _draw_zonal_panel(axes[1], diff, lat)
            cbar = fig.colorbar(im, cax=axes[2], orientation="horizontal")
        else:
            cbar = fig.colorbar(im, ax=axes[0], orientation="horizontal",
                                pad=0.05, fraction=0.046)
        cbar.set_label(bar_label, fontsize=12)
        out = self._out(
            save_name or f"delta_{scenario}_minus_{ref_scenario}_{t_label}",
            scenario, var, fmt, save_dir, cfg_obj, cfg_path)
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        print(f"[PLOT] delta map → {out}")

    def mapgif(self, scenario, var=None, time=None,
               region=None, zonal_mean=False,
               save_name=None, save_dir=None,
               filepath=None, cfg=None):
        """
        Animated GIF: one map frame per time step with a timeseries panel.

        Each frame is an individual time step (no averaging).
        Requires pillow: ``pip install pillow``

        Parameters
        ----------
        scenario : str
        var : str, optional
            See ``map()`` for details.
        time : None, int, list, range, or slice, default None
            Steps to animate.  ``None`` → all steps.  No averaging —
            each step becomes one frame.
        region : list of float, optional
        zonal_mean : bool, default False
            Add a latitude-vs-value side panel beside each frame.
        save_name : str, optional
            Default: ``animation``.
        save_dir : str or Path, optional
            Default: ``<cfg_dir>/figures/``.
        filepath : str or Path, optional
        cfg : str or Path, optional

        Examples
        --------
        >>> p = PaleoEmuPlotter("example_PCA_GP.yml")
        >>> p.mapgif("SSP585")
        >>> p.mapgif("SSP585", time=range(50), zonal_mean=True,
        ...          save_name="ssp585_anim", save_dir="figures/animations/")
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "pillow is required for mapgif.  Install with: pip install pillow"
            )

        cfg_obj, cfg_path = self._resolve_cfg(cfg)
        _validate_var(scenario, var, cfg_obj)
        ds        = self._load(scenario, var, filepath, cfg_obj, cfg_path)
        pred_var  = _get_pred_var(ds)
        pred      = ds[pred_var].values
        lat       = ds["latitude"].values
        lon       = ds["longitude"].values
        long_name = ds[pred_var].attrs.get("long_name", var or pred_var)
        units     = ds[pred_var].attrs.get("units", "")

        if isinstance(time, int):
            t_idx = [time]
        elif isinstance(time, slice):
            t_idx = list(range(*time.indices(pred.shape[0])))
        elif time is not None:
            t_idx = list(time)
        else:
            t_idx = list(range(pred.shape[0]))
        pred  = pred[t_idx]
        if region is not None:
            pred, lat, lon = _crop(pred, lat, lon, region)

        w      = _weights(lat)
        series = np.nansum(pred * w[np.newaxis, :, np.newaxis], axis=(1, 2))
        vmin   = np.nanpercentile(pred, 2)
        vmax   = np.nanpercentile(pred, 98)

        frames_dir = self._save_dir(save_dir, cfg_path) / "_gif_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = []

        for i, t in enumerate(t_idx):
            if zonal_mean:
                fig = plt.figure(figsize=(14, 8))
                gs  = fig.add_gridspec(2, 2, height_ratios=[3, 1],
                                       width_ratios=[4, 1], hspace=0.3)
                ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
                ax_zm  = fig.add_subplot(gs[0, 1])
                ax_ts  = fig.add_subplot(gs[1, :])
            else:
                fig = plt.figure(figsize=(10, 7))
                gs  = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.35)
                ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
                ax_ts  = fig.add_subplot(gs[1])

            im, bar_label = self._draw_map(
                ax_map, pred[i], lat, lon, long_name, units,
                vmin=vmin, vmax=vmax,
                title=f"{long_name} — {scenario}  t={t}")
            if zonal_mean:
                _draw_zonal_panel(ax_zm, pred[i], lat)
                cbar = fig.colorbar(im, ax=[ax_map, ax_zm],
                                    orientation="horizontal", pad=0.05, fraction=0.046)
            else:
                cbar = fig.colorbar(im, ax=ax_map,
                                    orientation="horizontal", pad=0.05, fraction=0.046)
            cbar.set_label(bar_label, fontsize=10)
            ax_ts.plot(t_idx, series, color="steelblue")
            ax_ts.axvline(t, color="red", lw=1.2)
            ax_ts.set_xlabel("Time index", fontsize=10)
            ax_ts.set_ylabel(f"{long_name} [{units}]" if units else long_name,
                             fontsize=10)
            ax_ts.grid(True, alpha=0.3)

            fp = frames_dir / f"frame_{i:04d}.png"
            fig.savefig(fp, dpi=120, bbox_inches="tight", pad_inches=0.2)
            plt.close(fig)
            frame_paths.append(fp)

        out = self._out(save_name or "animation", scenario, var, "gif",
                        save_dir, cfg_obj, cfg_path)
        images = [Image.open(fp) for fp in frame_paths]
        images[0].save(out, save_all=True, append_images=images[1:],
                       loop=0, duration=200)
        for fp in frame_paths:
            fp.unlink()
        frames_dir.rmdir()
        print(f"[PLOT] GIF → {out}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_cfg(self, cfg_arg):
        if cfg_arg is not None:
            caller_dir = Path(inspect.stack()[2].filename).parent
            cfg_path, cfg_obj = _load_cfg(cfg_arg, caller_dir)
            return cfg_obj, cfg_path
        return self.cfg, self.cfg_path

    def _load(self, scenario, var, filepath, cfg_obj, cfg_path):
        derived = self._derived_path(scenario, var, cfg_obj, cfg_path)

        if filepath is not None:
            filepath = Path(filepath)
            if derived is not None and \
               filepath.resolve() != derived.resolve() and derived.exists():
                print(f"[PLOT] Warning: filepath={filepath} conflicts with "
                      f"derived path={derived}.")
                try:
                    resp = input("Use the explicit filepath? [y/N]: ")
                except EOFError:
                    resp = "y"
                if resp.strip().lower() != "y":
                    filepath = derived
            if not filepath.exists():
                raise FileNotFoundError(
                    f"Prediction file not found: {filepath}\n"
                    "Check the filepath argument."
                )
            return xr.open_dataset(filepath, engine="h5netcdf")

        if derived is None:
            raise ValueError(
                "cfg is required to locate the prediction file. "
                "Pass cfg= or initialise PaleoEmuPlotter('config.yml')."
            )
        if not derived.exists():
            raise FileNotFoundError(
                f"Prediction file not found: {derived}\n"
                "Run runner.predict() first, or pass filepath= directly."
            )
        return xr.open_dataset(derived, engine="h5netcdf")

    def _derived_path(self, scenario, var, cfg_obj, cfg_path):
        if cfg_path is None:
            return None
        out_dir = cfg_path.parent / "outputs"
        if var is not None:
            matches = sorted(out_dir.glob(f"*{var}*_{scenario}_prediction.nc"))
            if matches:
                return matches[0]
        if cfg_obj is None:
            return None
        return out_dir / f"{cfg_obj.model_run_name}_{scenario}_prediction.nc"

    def _save_dir(self, save_dir, cfg_path=None):
        if save_dir:
            d = Path(save_dir)
        elif cfg_path is not None:
            d = cfg_path.parent / "figures"
        else:
            d = Path("figures")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _out(self, suffix, scenario, var, fmt, save_dir, cfg_obj, cfg_path):
        model_name = cfg_obj.model_run_name if cfg_obj else "prediction"
        parts = [model_name, scenario]
        if var:
            parts.append(var)
        parts.append(suffix)
        return self._save_dir(save_dir, cfg_path) / f"{'_'.join(parts)}.{fmt}"

    def _draw_map(self, ax, data_2d, lat, lon, long_name, units,
                  cmap="coolwarm", vmin=None, vmax=None, title="", label=None):
        """Draw the map mesh and return (im, bar_label) — caller adds colorbar."""
        if vmin is None:
            vmin = np.nanpercentile(data_2d, 2)
        if vmax is None:
            vmax = np.nanpercentile(data_2d, 98)
        # Drop duplicate 360°=0° column if present, then add a proper cyclic
        # point so pcolormesh has no seam at the 0/360 meridian.
        if lon.size > 1 and np.isclose(lon[-1], 360.0, atol=1.0) and np.isclose(lon[0], 0.0, atol=1.0):
            data_2d = data_2d[..., :-1]
            lon = lon[:-1]
        data_cyc, lon_cyc = cutil.add_cyclic_point(data_2d, coord=lon)
        Lon, Lat = np.meshgrid(lon_cyc, lat)
        im = ax.pcolormesh(Lon, Lat, data_cyc, cmap=cmap,
                           vmin=vmin, vmax=vmax, shading="auto",
                           transform=ccrs.PlateCarree())
        ax.set_global()
        ax.coastlines(linewidth=0.6)
        ax.set_title(title, fontsize=12)
        bar_label = label if label is not None else (
            f"{long_name} [{units}]" if units else long_name)
        return im, bar_label


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _get_pred_var(ds):
    """Return the prediction variable name — first data var that is not 'variance'."""
    candidates = [v for v in ds.data_vars if v != "variance"]
    if not candidates:
        raise KeyError("No prediction variable found in dataset.")
    return candidates[0]


def _load_cfg(cfg_path, caller_dir):
    p = Path(cfg_path)
    if not p.is_absolute() and not p.exists():
        p = caller_dir / p
    return p, load_config(str(p))


def _validate_var(scenario, var, cfg_obj):
    """Raise if var is given but the scenario has no 'var' sweep dimension."""
    if var is None or cfg_obj is None:
        return
    scenario_cfg = cfg_obj.forcing_data.get(scenario, {})
    if "var" not in scenario_cfg:
        sweep_dims = [k for k in scenario_cfg
                      if k not in ("forcing_input", "forcing_input_pattern")]
        raise ValueError(
            f"'var' is not a sweep dimension of scenario '{scenario}'. "
            + (f"Available sweep dimensions: {sweep_dims}"
               if sweep_dims else "This scenario has no sweep dimensions.")
        )
    allowed = scenario_cfg["var"]
    if isinstance(allowed, list) and var not in [str(v) for v in allowed]:
        raise ValueError(
            f"var='{var}' is not in the allowed values for scenario '{scenario}': "
            f"{allowed}"
        )


def _sel_time_avg(arr, time):
    """
    Apply time selection, always returning a single 2-D (lat, lon) frame.
    - time=None              → mean of all steps
    - time=[0, 1, 2, 3]     → mean of those steps
    - time=range(a, b, c)   → mean of that range
    - time=slice(a, b, c)   → mean of that slice
    - time=x (int)          → only step x, no averaging
    """
    if isinstance(time, int):
        return arr[time]
    if isinstance(time, slice):
        arr = arr[time]
    elif time is not None:
        arr = arr[list(time)]
    return np.nanmean(arr, axis=0)


def _time_label(time):
    if isinstance(time, int):
        return f"t{time}"
    return "mean"


def _crop(data, lat, lon, region):
    lon_min, lon_max, lat_min, lat_max = region
    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    lon_mask = (lon >= lon_min) & (lon <= lon_max)
    return (data[..., lat_mask, :][..., lon_mask],
            lat[lat_mask], lon[lon_mask])


def _weights(lat):
    w = np.cos(np.radians(lat))
    return w / w.sum()


def _map_fig(with_zonal):
    """
    Build figure and axes for map (and optional zonal-mean panel).

    Returns
    -------
    fig : Figure
    axes : list
        [ax_map]           — no zonal mean
        [ax_map, ax_zm, cax] — with zonal mean; cax is the dedicated colorbar axes
    """
    if with_zonal:
        fig = plt.figure(figsize=(13, 5))
        # Two-row layout: row 0 = map + zonal panel, row 1 = colorbar under map only.
        # Using explicit gridspec (no constrained layout) so wspace is not overridden.
        gs = fig.add_gridspec(
            2, 2,
            width_ratios=[4, 1],
            height_ratios=[13, 1],
            wspace=0.02,   # fraction of mean axis width — honoured without constrained layout
            hspace=0.08,
            left=0.02, right=0.97,
            top=0.90, bottom=0.08,
        )
        ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        ax_zm  = fig.add_subplot(gs[0, 1])
        cax    = fig.add_subplot(gs[1, 0])   # colorbar axes, below map only
        return fig, [ax_map, ax_zm, cax]
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained",
                           subplot_kw={"projection": ccrs.PlateCarree()})
    return fig, [ax]


def _draw_zonal_panel(ax, data_2d, lat):
    zonal = np.nanmean(data_2d, axis=1)
    ax.plot(zonal, lat)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylim(lat.min(), lat.max())
    ax.set_xlabel("Zonal mean", fontsize=10)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Latitude (°)", fontsize=10)
    ax.grid(True, alpha=0.3)
