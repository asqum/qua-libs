"""2D flux-landscape fitting.

Qubit column: interaction-band activity on state / contrast maps.
Coupler cut: |control − target|, Savitzky–Golay, sliding-window FFT,
decouple = min in flat idle band, gate = first fringe dip toward oscillation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks, savgol_filter

_FRINGE_FREQ_LOW = 0.05
_FRINGE_FREQ_HIGH = 0.45
_MIN_FLAT_POINTS = 4
_GUARD_FRACTION = 0.05
_REFINE_HALF_WIDTH_DECOUPLE = 20
_REFINE_HALF_WIDTH_GATE = 15
_SAVGOL_POLY = 3

ANALYSIS_PRESETS: dict[str, dict] = {
    "default": {
        "flat_power_thresh": 0.03,
        "osc_power_thresh": 0.05,
        "gate_feature_prominence": 0.05,
        "savgol_window_coarse": 21,
        "savgol_window_fine": 7,
        "fft_window": 30,
        "coupler_core_trim_fraction": 0.12,
        "coupler_qubit_marginal_low_fraction": 0.25,
    },
    "noisy": {
        "flat_power_thresh": 0.05,
        "osc_power_thresh": 0.08,
        "gate_feature_prominence": 0.02,
        "savgol_window_coarse": 41,
        "savgol_window_fine": 15,
        "fft_window": 30,
        "coupler_core_trim_fraction": 0.12,
        "coupler_qubit_marginal_low_fraction": 0.25,
    },
    "coarse": {
        "flat_power_thresh": 0.03,
        "osc_power_thresh": 0.04,
        "gate_feature_prominence": 0.08,
        "savgol_window_coarse": 11,
        "savgol_window_fine": 5,
        "fft_window": 15,
        "coupler_core_trim_fraction": 0.12,
        "coupler_qubit_marginal_low_fraction": 0.25,
    },
}


def get_analysis_fit_config(preset: str) -> dict:
    if preset not in ANALYSIS_PRESETS:
        valid = ", ".join(sorted(ANALYSIS_PRESETS))
        raise ValueError(f"analysis_fit_preset must be one of {{{valid}}}, got {preset!r}")
    return ANALYSIS_PRESETS[preset]


def _select_flux_coord(
    ds: xr.Dataset,
    coord_name: str,
    qp_name: str,
    sample_var: str = "state_control",
    *,
    slice_da: xr.DataArray | None = None,
) -> xr.DataArray:
    """Select flux coordinates for one pair (handles shared 1D vs per-qubit 2D coords)."""
    if slice_da is not None and coord_name in slice_da.coords:
        da = slice_da.coords[coord_name]
        if isinstance(da, xr.DataArray) and "qubit" in da.dims:
            return da.sel(qubit=qp_name)
        return da

    if sample_var in ds:
        sample = ds[sample_var].sel(qubit=qp_name)
        if coord_name in sample.coords:
            da = sample.coords[coord_name]
            if isinstance(da, xr.DataArray) and "qubit" in da.dims:
                return da.sel(qubit=qp_name)
            return da

    if coord_name in ds.coords:
        da = ds.coords[coord_name]
    elif coord_name in ds:
        da = ds[coord_name]
    else:
        raise KeyError(f"Coordinate {coord_name!r} not found in dataset for {qp_name!r}")

    if isinstance(da, xr.DataArray) and "qubit" in da.dims:
        return da.sel(qubit=qp_name)
    return da


@dataclass
class FluxLandscapeFit:
    success: bool
    optimal_qubit_flux: float
    optimal_decouple_offset: float = np.nan
    optimal_decouple_coupler_flux_rel: float = np.nan
    optimal_gate_coupler_flux_rel: float = np.nan
    optimal_gate_coupler_flux_total: float = np.nan
    contrast_coupler_rel: Optional[np.ndarray] = None
    contrast_coupler_full: Optional[np.ndarray] = None
    contrast_raw: Optional[np.ndarray] = None
    contrast_smoothed: Optional[np.ndarray] = None
    ac_power_norm: Optional[np.ndarray] = None
    osc_mask: Optional[np.ndarray] = None
    flat_mask: Optional[np.ndarray] = None

    def to_results_dict(self) -> dict:
        return asdict(self)

    def to_save_dict(self) -> dict:
        """Scalar summary safe for JSON export (no ndarray fields)."""
        def _f(v: float) -> float | None:
            return None if not np.isfinite(v) else float(v)

        return {
            "success": bool(self.success),
            "optimal_qubit_flux": _f(self.optimal_qubit_flux),
            "optimal_decouple_offset": _f(self.optimal_decouple_offset),
            "optimal_decouple_coupler_flux_rel": _f(self.optimal_decouple_coupler_flux_rel),
            "optimal_gate_coupler_flux_rel": _f(self.optimal_gate_coupler_flux_rel),
            "optimal_gate_coupler_flux_total": _f(self.optimal_gate_coupler_flux_total),
        }


def _sliding_fringe_power(y: np.ndarray, cfg: dict) -> np.ndarray:
    half_w = cfg["fft_window"] // 2
    n = y.size
    freqs = np.fft.rfftfreq(cfg["fft_window"])
    band = (freqs >= _FRINGE_FREQ_LOW) & (freqs <= _FRINGE_FREQ_HIGH)
    ac_power = np.zeros(n)
    for i in range(n):
        i0 = max(0, i - half_w)
        i1 = min(n, i + half_w)
        seg = y[i0:i1]
        seg_pad = np.pad(seg, (0, cfg["fft_window"] - len(seg)), mode="edge")
        spectrum = np.abs(np.fft.rfft(seg_pad - seg_pad.mean())) ** 2
        ac_power[i] = spectrum[band].mean()
    return ac_power / (ac_power.max() + 1e-12)


def _savgol_smooth(y: np.ndarray, window: int) -> np.ndarray:
    sw = min(window | 1, y.size)
    if sw % 2 == 0:
        sw -= 1
    sw = max(sw, _SAVGOL_POLY + 2)
    return savgol_filter(y, window_length=sw, polyorder=min(_SAVGOL_POLY, sw - 1))


def _region_masks(y_heavy: np.ndarray, cfg: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ac_power_norm = _sliding_fringe_power(y_heavy, cfg)
    osc_mask = ac_power_norm > cfg["osc_power_thresh"]
    flat_mask = ac_power_norm < cfg["flat_power_thresh"]
    return ac_power_norm, osc_mask, flat_mask


def _flat_segments(flat_indices: np.ndarray) -> list[np.ndarray]:
    """Split sorted flat-mask indices into contiguous runs."""
    flat_indices = np.asarray(flat_indices, dtype=int)
    if flat_indices.size == 0:
        return []
    breaks = np.where(np.diff(flat_indices) > 1)[0] + 1
    chunks = np.split(flat_indices, breaks)
    return [c for c in chunks if c.size > 0]


def _decouple_index_in_flat_region(
    smoothed: np.ndarray,
    flat_mask: np.ndarray,
    coupler_rel: np.ndarray | None = None,
    osc_mask: np.ndarray | None = None,
    *,
    mode: Literal["argmin", "argmax"] = "argmin",
) -> int | None:
    """Decouple = extremum in the FFT-flat idle band beside the oscillation fringe."""
    flat_indices = np.where(flat_mask)[0]
    if flat_indices.size < _MIN_FLAT_POINTS:
        return None

    segments = _flat_segments(flat_indices)
    n = smoothed.size
    g = max(1, int(_GUARD_FRACTION * n))

    if osc_mask is not None and np.any(osc_mask):
        osc_idx = np.where(osc_mask)[0]
        osc_lo, osc_hi = int(osc_idx.min()), int(osc_idx.max())
        if coupler_rel is None or float(coupler_rel[-1]) > float(coupler_rel[0]):
            adjacent = [
                seg
                for seg in segments
                if int(seg.min()) > osc_hi and int(seg.max()) < n - g
            ]
        else:
            adjacent = [
                seg
                for seg in segments
                if int(seg.max()) < osc_lo and int(seg.min()) > g
            ]
        if adjacent:
            segments = adjacent
        else:
            segments = [
                seg for seg in segments if int(seg.min()) > g and int(seg.max()) < n - g
            ] or segments

    best_idx: int | None = None
    best_score = float("inf") if mode == "argmin" else float("-inf")
    for seg in segments:
        seg_arr = np.asarray(seg, dtype=int)
        if seg_arr.size < 2:
            continue
        if mode == "argmax":
            idx = int(seg_arr[np.nanargmax(smoothed[seg_arr])])
            score = float(smoothed[idx])
            if score > best_score:
                best_score = score
                best_idx = idx
        else:
            idx = int(seg_arr[np.nanargmin(smoothed[seg_arr])])
            score = float(smoothed[idx])
            if score < best_score:
                best_score = score
                best_idx = idx
    if best_idx is not None:
        return best_idx

    flat_arr = np.asarray(flat_indices, dtype=int)
    if mode == "argmax":
        return int(flat_arr[np.nanargmax(smoothed[flat_arr])])
    return int(flat_arr[np.nanargmin(smoothed[flat_arr])])


def _refine_decouple_index(
    y_fine: np.ndarray,
    flat_mask: np.ndarray,
    coarse_idx: int,
    *,
    mode: Literal["argmin", "argmax"] = "argmin",
) -> int:
    flat_indices = np.where(flat_mask)[0]
    if flat_indices.size < _MIN_FLAT_POINTS:
        return coarse_idx
    in_flat = flat_indices[
        (flat_indices >= max(flat_indices[0], coarse_idx - _REFINE_HALF_WIDTH_DECOUPLE))
        & (flat_indices <= min(flat_indices[-1], coarse_idx + _REFINE_HALF_WIDTH_DECOUPLE))
    ]
    if in_flat.size == 0:
        in_flat = flat_indices
    if mode == "argmax":
        return int(in_flat[np.nanargmax(y_fine[in_flat])])
    return int(in_flat[np.nanargmin(y_fine[in_flat])])


def _refine_gate_index(
    y_fine: np.ndarray,
    coarse_gate: int | None,
    cfg: dict,
    *,
    pick_peak: bool,
) -> int | None:
    if coarse_gate is None:
        return None
    lo = max(0, coarse_gate - _REFINE_HALF_WIDTH_GATE)
    hi = min(len(y_fine) - 1, coarse_gate + _REFINE_HALF_WIDTH_GATE)
    seg = y_fine[lo : hi + 1]
    if seg.size < 3:
        return coarse_gate
    sig_range = float(seg.max() - seg.min())
    prom = max(cfg["gate_feature_prominence"], 0.05 * sig_range)
    if pick_peak:
        features, _ = find_peaks(seg, prominence=prom)
    else:
        features, _ = find_peaks(-seg, prominence=prom)
    if len(features) == 0:
        return coarse_gate
    global_features = lo + features.astype(int)
    return int(global_features[np.argmin(np.abs(global_features - coarse_gate))])


def _first_feature_beyond_decouple(
    seg_smoothed: np.ndarray,
    from_decouple_end: str,
    prominence: float,
    guard_pts: int,
    *,
    pick_peak: bool,
) -> int | None:
    n = len(seg_smoothed)
    if from_decouple_end == "right":
        s0, s1 = guard_pts, n
    else:
        s0, s1 = 0, max(guard_pts + 1, n - guard_pts)
    if s1 <= s0:
        return None
    search_region = seg_smoothed[s0:s1]
    sig_range = search_region.max() - search_region.min()
    prom = max(prominence, 0.05 * sig_range)
    if pick_peak:
        features, _ = find_peaks(search_region, prominence=prom)
    else:
        features, _ = find_peaks(-search_region, prominence=prom)
    if len(features) == 0:
        return None
    local = int(features[0] if from_decouple_end == "right" else features[-1])
    return s0 + local


def _gate_coupler_index_from_cut(
    smoothed: np.ndarray,
    decouple_idx: int,
    osc_mask: np.ndarray,
    coupler_rel: np.ndarray,
    cfg: dict,
    *,
    pick_peak: bool,
) -> int | None:
    osc_indices = np.where(osc_mask)[0]
    if osc_indices.size == 0:
        return None
    guard_pts = max(3, int(_GUARD_FRACTION * len(coupler_rel)))
    osc_center = float(coupler_rel[osc_indices].mean())
    decouple_v = float(coupler_rel[decouple_idx])
    prom = cfg["gate_feature_prominence"]
    dip_l = _first_feature_beyond_decouple(
        smoothed[: decouple_idx + 1], "left", prom, guard_pts, pick_peak=pick_peak
    )
    dip_r = _first_feature_beyond_decouple(smoothed[decouple_idx:], "right", prom, guard_pts, pick_peak=pick_peak)
    gate_l = dip_l
    gate_r = (decouple_idx + dip_r) if dip_r is not None else None
    if osc_center < decouple_v:
        return gate_l
    return gate_r


def _gate_fringe_from_decouple(
    y: np.ndarray,
    decouple_idx: int,
    coupler_rel: np.ndarray,
    cfg: dict,
    *,
    pick_peak: bool,
) -> int | None:
    """First fringe extremum leaving decouple toward the interaction side."""
    n = y.size
    g = max(3, int(_GUARD_FRACTION * n))
    if float(coupler_rel[-1]) > float(coupler_rel[0]):
        order = np.arange(decouple_idx - 1, g - 1, -1, dtype=int)
    else:
        order = np.arange(decouple_idx + 1, n - g, dtype=int)
    if order.size < 3:
        return None
    seg = y[order]
    sig_range = float(np.ptp(seg))
    prom = max(cfg["gate_feature_prominence"] * sig_range, 0.05 * sig_range)
    if pick_peak:
        features, _ = find_peaks(seg, prominence=prom)
    else:
        features, _ = find_peaks(-seg, prominence=prom)
    if features.size == 0:
        return None
    return int(order[int(features[0])])


def _coarse_coupler_indices(
    y_heavy: np.ndarray,
    flat_mask: np.ndarray,
    osc_mask: np.ndarray,
    coupler_rel: np.ndarray,
    cfg: dict,
    *,
    pick_peak: bool,
) -> Tuple[int | None, int | None]:
    decouple = _decouple_index_in_flat_region(
        y_heavy, flat_mask, coupler_rel, osc_mask
    )
    if decouple is None:
        return None, None
    gate = _gate_coupler_index_from_cut(y_heavy, decouple, osc_mask, coupler_rel, cfg, pick_peak=pick_peak)
    if gate is None:
        gate = _gate_fringe_from_decouple(y_heavy, decouple, coupler_rel, cfg, pick_peak=pick_peak)
    return decouple, gate


def _refine_coupler_indices(
    y_fine: np.ndarray,
    flat_mask: np.ndarray,
    decouple_coarse: int | None,
    gate_coarse: int | None,
    cfg: dict,
    *,
    pick_peak: bool,
) -> Tuple[int | None, int | None]:
    if decouple_coarse is None:
        return None, None
    decouple = _refine_decouple_index(y_fine, flat_mask, decouple_coarse)
    gate = _refine_gate_index(y_fine, gate_coarse, cfg, pick_peak=pick_peak)
    return decouple, gate


def _fit_coupler_cut(
    y_raw: np.ndarray,
    coupler_rel: np.ndarray,
    cfg: dict,
    *,
    gate_pick_peak: bool = False,
    normalize: bool = True,
    marginal: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int | None, int | None]:
    """1D coupler cut: SG smooth, FFT flat/osc masks, decouple + gate indices."""
    y = np.asarray(y_raw, dtype=float).copy()
    if normalize:
        y_range = y.max() - y.min()
        if y_range > 1e-12:
            y = y / y_range

    y_heavy = _savgol_smooth(y, cfg["savgol_window_coarse"])
    y_fine = _savgol_smooth(y, cfg["savgol_window_fine"])
    ac_power_norm, osc_mask, flat_mask = _region_masks(y_heavy, cfg)

    decouple_coarse, gate_coarse = _coarse_coupler_indices(
        y_heavy, flat_mask, osc_mask, coupler_rel, cfg, pick_peak=gate_pick_peak
    )
    if decouple_coarse is None and marginal is not None:
        decouple_coarse = _argext_interior(np.asarray(marginal, dtype=float), "argmin")
        gate_coarse = _gate_coupler_index_from_cut(
            y_heavy, decouple_coarse, osc_mask, coupler_rel, cfg, pick_peak=gate_pick_peak
        )
        if gate_coarse is None:
            gate_coarse = _gate_fringe_from_decouple(
                y_heavy, decouple_coarse, coupler_rel, cfg, pick_peak=gate_pick_peak
            )

    decouple_idx, gate_idx = _refine_coupler_indices(
        y_fine, flat_mask, decouple_coarse, gate_coarse, cfg, pick_peak=gate_pick_peak
    )
    if gate_idx is None and decouple_idx is not None:
        gate_idx = _gate_fringe_from_decouple(
            y_fine, decouple_idx, coupler_rel, cfg, pick_peak=gate_pick_peak
        )
    return y_raw, y_fine, ac_power_norm, osc_mask, flat_mask, decouple_idx, gate_idx


def _interior_slice(n: int, guard_fraction: float = _GUARD_FRACTION) -> slice:
    """Slice excluding top/bottom guard_fraction of a 1D sweep."""
    g = max(1, int(guard_fraction * n))
    if n <= 2 * g:
        return slice(None)
    return slice(g, n - g)


def _on_interaction_side(gate_idx: int, decouple_idx: int, coupler_rel: np.ndarray) -> bool:
    """True if gate_idx lies on the strongly-coupled side of decouple_idx."""
    if float(coupler_rel[-1]) > float(coupler_rel[0]):
        return gate_idx < decouple_idx
    return gate_idx > decouple_idx


def _argext_interior(values: np.ndarray, mode: Literal["argmin", "argmax"], guard_fraction: float = _GUARD_FRACTION) -> int:
    """argmin/argmax ignoring sweep boundary points (avoids fringe/noise at edges)."""
    sl = _interior_slice(values.size, guard_fraction)
    seg = np.asarray(values)[sl]
    base = sl.start or 0
    if seg.size == 0:
        return int(np.nanargmin(values) if mode == "argmin" else np.nanargmax(values))
    if mode == "argmin":
        return base + int(np.nanargmin(seg))
    return base + int(np.nanargmax(seg))


def _index_on_sweep_boundary(idx: int, size: int, guard_fraction: float = _GUARD_FRACTION) -> bool:
    if size <= 1 or idx == 0 or idx == size - 1:
        return True
    g = max(1, int(guard_fraction * size))
    return idx < g or idx >= size - g


def _nanptp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Peak-to-peak along axis, ignoring NaNs (``np.nanptp`` is NumPy >= 2.0)."""
    return np.nanmax(a, axis=axis) - np.nanmin(a, axis=axis)


def _coupler_rows_for_qubit_cut(nc: int, cfg: dict) -> slice:
    """Coupler row band for qubit-cut marginals (drops bottom fringe + top idle)."""
    lo = max(0, int(cfg["coupler_qubit_marginal_low_fraction"] * nc))
    hi = nc - max(1, int(cfg["coupler_core_trim_fraction"] * nc))
    if hi - lo < max(3, nc // 10):
        g = max(1, int(_GUARD_FRACTION * nc))
        return slice(g, nc - g)
    return slice(lo, hi)


def _qubit_coupler_slab(
    data: np.ndarray,
    qubit_axis: int,
    coupler_axis: int,
    cfg: dict,
) -> np.ndarray:
    nc = int(data.shape[coupler_axis])
    rows = _coupler_rows_for_qubit_cut(nc, cfg)
    if qubit_axis == 0 and coupler_axis == 1:
        return np.asarray(data[:, rows], dtype=float)
    if qubit_axis == 1 and coupler_axis == 0:
        return np.asarray(data[rows, :], dtype=float)
    slab = np.take(data, np.arange(rows.start or 0, rows.stop or nc), axis=coupler_axis)
    q_other = 0 if coupler_axis != 0 else (1 if data.ndim > 1 else 0)
    if coupler_axis == q_other:
        return np.asarray(slab, dtype=float)
    return np.moveaxis(slab, coupler_axis, q_other)


def _marginal_on_coupler_core(
    data: np.ndarray,
    qubit_axis: int,
    coupler_axis: int,
    func,
    cfg: dict,
) -> np.ndarray:
    """1D qubit profile using the interaction coupler band (excludes bottom fringe)."""
    slab = _qubit_coupler_slab(data, qubit_axis, coupler_axis, cfg)
    if qubit_axis == 0 and coupler_axis == 1:
        return np.asarray(func(slab, axis=1), dtype=float).ravel()
    if qubit_axis == 1 and coupler_axis == 0:
        return np.asarray(func(slab, axis=0), dtype=float).ravel()
    out_axis = 0 if qubit_axis == 0 else 1
    return np.asarray(func(slab, axis=out_axis), dtype=float).ravel()


def _from_smoothed_marginal(values: np.ndarray, cfg: dict, mode: Literal["argmax", "argmin"]) -> int:
    if values.size < 3:
        return int(np.nanargmax(values) if mode == "argmax" else np.nanargmin(values))
    return _argext_interior(_savgol_smooth(values, cfg["savgol_window_coarse"]), mode)


def _qubit_activity_profile(
    data: np.ndarray,
    qubit_axis: int,
    coupler_axis: int,
    cfg: dict,
) -> np.ndarray:
    """Per-qubit chevron strength: coupler ptp + std in the interaction band only."""
    slab = _qubit_coupler_slab(data, qubit_axis, coupler_axis, cfg)
    c_axis = 1 if qubit_axis == 0 else 0
    ptp = _nanptp(slab, axis=c_axis)
    std = np.nanstd(slab, axis=c_axis)
    return np.asarray(ptp, dtype=float) + np.asarray(std, dtype=float)


def _qubit_idx_from_2d_deviation(
    data: np.ndarray,
    qubit_axis: int,
    coupler_axis: int,
    cfg: dict,
) -> int:
    """Qubit index at the strongest 2D blob (peak or dip vs global median)."""
    nc = int(data.shape[coupler_axis])
    rows = _coupler_rows_for_qubit_cut(nc, cfg)
    masked = np.array(data, dtype=float, copy=True)
    if coupler_axis == 1 and masked.ndim == 2:
        masked[:, : rows.start] = np.nan
        masked[:, rows.stop :] = np.nan
    elif coupler_axis == 0 and masked.ndim == 2:
        masked[: rows.start, :] = np.nan
        masked[rows.stop :, :] = np.nan

    sig = max(1.0, cfg["savgol_window_coarse"] / 7.0)
    sigma = [sig if ax == qubit_axis else sig for ax in range(masked.ndim)]
    smoothed = gaussian_filter(np.nan_to_num(masked, nan=float(np.nanmedian(masked))), sigma=sigma)
    score = np.abs(smoothed - float(np.nanmedian(smoothed)))

    gq = max(1, int(_GUARD_FRACTION * score.shape[qubit_axis]))
    if score.shape[qubit_axis] <= 2 * gq:
        flat = int(np.nanargmax(score))
        return int(np.unravel_index(flat, score.shape)[qubit_axis])

    sl_q = slice(gq, score.shape[qubit_axis] - gq)
    if qubit_axis == 0:
        band_score = np.nanmax(score[:, rows], axis=1)[sl_q]
    else:
        band_score = np.nanmax(score[rows, :], axis=0)[sl_q]
    flat = int(np.nanargmax(band_score))
    return gq + flat


def _select_qubit_flux_cut(
    contrast: xr.DataArray,
    cfg: dict,
    *,
    control: xr.DataArray | None = None,
    target: xr.DataArray | None = None,
) -> int:
    """Pick the qubit column through the chevron centre.

    Uses summed coupler-direction activity (ptp + std) in the interaction band,
    combining target / control state maps and |contrast|.  Bottom coupler fringes
    and sweep-edge blobs are excluded before scoring.
    """
    q_dim = "flux_qubit" if "flux_qubit" in contrast.dims else contrast.dims[0]
    c_dim = "flux_coupler" if "flux_coupler" in contrast.dims else contrast.dims[1]
    q_ax = int(contrast.dims.index(q_dim))
    c_ax = int(contrast.dims.index(c_dim))
    n_qubit = int(contrast.sizes[q_dim])

    contrast_arr = np.asarray(contrast.values, dtype=float)
    maps: list[np.ndarray] = []
    if target is not None:
        maps.append(np.asarray(target.values, dtype=float))
    if control is not None:
        maps.append(np.asarray(control.values, dtype=float))
    maps.append(np.abs(contrast_arr))

    combined = np.zeros(n_qubit, dtype=float)
    for arr in maps:
        prof = _qubit_activity_profile(arr, q_ax, c_ax, cfg)
        if prof.size == n_qubit:
            combined += prof

    idx_activity = _from_smoothed_marginal(combined, cfg, "argmax")
    idx_deviation = _qubit_idx_from_2d_deviation(contrast_arr, q_ax, c_ax, cfg)
    if target is not None:
        idx_deviation = int(
            np.median(
                [
                    idx_deviation,
                    _qubit_idx_from_2d_deviation(np.asarray(target.values, dtype=float), q_ax, c_ax, cfg),
                ]
            )
        )

    tol = max(3, int(0.08 * n_qubit))
    if abs(idx_activity - idx_deviation) <= tol:
        return idx_activity
    if combined.size and np.isfinite(combined[idx_activity]):
        return idx_activity
    return idx_deviation


def _decouple_index_from_signed_marginal(contrast: xr.DataArray) -> int:
    """Node-61 decouple estimate: argmin of the qubit-averaged SIGNED contrast.

    Averaging signed contrast over qubit flux cancels the oscillatory chevron
    (it swings symmetrically about zero), leaving the systematic coupler-flux
    dependence.  Uses the full 2D map, so it is far more robust than a single-cut
    FFT flat-band search.
    """
    marginal = contrast.mean(dim="flux_qubit")
    return int(marginal.argmin(dim="flux_coupler").values)


def _valley_half_width(y: np.ndarray, idx: int) -> float:
    """Valley width at half-depth; larger = broader / smoother dip."""
    n = y.size
    if idx <= 0 or idx >= n - 1:
        return 0.0
    y_min = float(y[idx])
    rim = float(max(y[max(0, idx - 30) : min(n, idx + 31)]))
    if rim <= y_min:
        return 0.0
    threshold = y_min + 0.5 * (rim - y_min)
    lo = idx
    while lo > 0 and float(y[lo]) <= threshold:
        lo -= 1
    hi = idx
    while hi < n - 1 and float(y[hi]) <= threshold:
        hi += 1
    return float(hi - lo)


def _decouple_smoothest_dip_index(
    y_raw: np.ndarray,
    flat_mask: np.ndarray,
    coupler_rel: np.ndarray,
    cfg: dict,
    fallback: int,
    *,
    gate_idx: int | None = None,
) -> int:
    """Broadest idle-side local minimum inside the FFT flat band."""
    y = _savgol_smooth(np.asarray(y_raw, dtype=float), cfg["savgol_window_coarse"])
    flat = np.asarray(flat_mask, dtype=bool)
    n = y.size
    g = max(3, int(_GUARD_FRACTION * n))
    idx_all = np.arange(n)

    idle_side = np.ones(n, dtype=bool)
    if gate_idx is not None:
        sweep_up = float(coupler_rel[-1]) > float(coupler_rel[0])
        if sweep_up:
            idle_side = idx_all > int(gate_idx) + g
        else:
            idle_side = idx_all < int(gate_idx) - g

    is_local_min = np.zeros(n, dtype=bool)
    is_local_min[1:-1] = (y[1:-1] <= y[:-2]) & (y[1:-1] <= y[2:])
    candidates = np.where(is_local_min & flat & idle_side)[0]
    if candidates.size:
        order = sorted(
            candidates,
            key=lambda i: (_valley_half_width(y, int(i)), -float(y[int(i)])),
            reverse=True,
        )
        return int(order[0])

    flat_idle = np.where(flat & idle_side)[0]
    if flat_idle.size:
        return int(flat_idle[np.nanargmin(y[flat_idle])])

    flat_idx = np.where(flat)[0]
    if flat_idx.size:
        return int(flat_idx[np.nanargmin(y[flat_idx])])
    return fallback


def fit_coupler_zeropoint_pair(
    ds: xr.Dataset,
    qp_name: str,
    *,
    use_state_discrimination: bool,
    cz_or_iswap: Literal["cz", "iswap"],
    preset: str = "default",
) -> FluxLandscapeFit:
    """Fit coupler zeropoint from |control − target| on a 2D flux landscape."""
    cfg = get_analysis_fit_config(preset)
    if use_state_discrimination:
        control = ds.state_control.sel(qubit=qp_name)
        target = ds.state_target.sel(qubit=qp_name)
        sample_var = "state_control"
    else:
        control = ds.I_control.sel(qubit=qp_name)
        target = ds.I_target.sel(qubit=qp_name)
        sample_var = "I_control"

    contrast = control - target
    coupler_rel = np.asarray(
        _select_flux_coord(ds, "flux_coupler", qp_name, sample_var, slice_da=contrast).values
    ).astype(float)
    flux_qubit_full = _select_flux_coord(ds, "flux_qubit_full", qp_name, sample_var, slice_da=contrast)
    flux_coupler_full = _select_flux_coord(ds, "flux_coupler_full", qp_name, sample_var, slice_da=contrast)

    qubit_map_control = control
    qubit_map_target = target
    if not use_state_discrimination and "state_control" in ds and "state_target" in ds:
        qubit_map_control = ds.state_control.sel(qubit=qp_name)
        qubit_map_target = ds.state_target.sel(qubit=qp_name)

    qubit_idx = _select_qubit_flux_cut(
        contrast,
        cfg,
        control=qubit_map_control,
        target=qubit_map_target,
    )

    optimal_qubit_flux = float(flux_qubit_full.isel(flux_qubit=qubit_idx).values)
    y_raw = np.abs(contrast.isel(flux_qubit=qubit_idx).values.ravel().astype(float))
    n_coupler = y_raw.size
    n_qubit = contrast.sizes["flux_qubit"]
    if n_coupler < 3:
        return FluxLandscapeFit(success=False, optimal_qubit_flux=optimal_qubit_flux)

    y_raw, y_fine, ac_power_norm, osc_mask, flat_mask, decouple_idx, gate_idx = _fit_coupler_cut(
        y_raw,
        coupler_rel,
        cfg,
        gate_pick_peak=False,
        normalize=True,
        marginal=np.abs(contrast.mean(dim="flux_qubit").values),
    )

    # Decouple offset: pick the smoothest dip = the lowest interior valley inside
    # the flat (low-oscillation) band, i.e. the broad idle minimum rather than a
    # sharp fringe valley near the gate.  The gate index from the FFT cut is kept;
    # only when it is missing or ends up on the wrong side of the new decouple do
    # we re-locate the first fringe leaving the decouple point.
    if cfg.get("decouple_smoothest_dip", True):
        marginal_fallback = _decouple_index_from_signed_marginal(contrast)
        decouple_idx = _decouple_smoothest_dip_index(
            y_raw, flat_mask, coupler_rel, cfg, marginal_fallback, gate_idx=gate_idx
        )
        if gate_idx is None or not _on_interaction_side(gate_idx, decouple_idx, coupler_rel):
            gate_fallback = _gate_fringe_from_decouple(
                y_fine, decouple_idx, coupler_rel, cfg, pick_peak=False
            )
            if gate_fallback is not None:
                gate_idx = gate_fallback

    # Validity mirrors node 61: the decouple offset may legitimately sit at the
    # idle sweep edge, so it is not boundary-constrained; only gate / qubit are.
    gate_valid = (
        gate_idx is not None
        and gate_idx != decouple_idx
        and not _index_on_sweep_boundary(gate_idx, n_coupler)
        and _on_interaction_side(gate_idx, decouple_idx, coupler_rel)
    )
    success = gate_valid and not _index_on_sweep_boundary(qubit_idx, n_qubit)

    return FluxLandscapeFit(
        success=success,
        optimal_qubit_flux=optimal_qubit_flux,
        optimal_decouple_offset=(
            float(flux_coupler_full.isel(flux_coupler=decouple_idx).values)
            if decouple_idx is not None
            else np.nan
        ),
        optimal_decouple_coupler_flux_rel=(
            float(coupler_rel[decouple_idx]) if decouple_idx is not None else np.nan
        ),
        optimal_gate_coupler_flux_rel=float(coupler_rel[gate_idx]) if gate_idx is not None else np.nan,
        optimal_gate_coupler_flux_total=(
            float(flux_coupler_full.isel(flux_coupler=gate_idx).values) if gate_idx is not None else np.nan
        ),
        contrast_coupler_rel=coupler_rel.copy(),
        contrast_coupler_full=np.asarray(flux_coupler_full).ravel().astype(float).copy(),
        contrast_raw=y_raw.copy(),
        contrast_smoothed=y_fine.copy(),
        ac_power_norm=ac_power_norm.copy(),
        osc_mask=osc_mask.copy(),
        flat_mask=flat_mask.copy(),
    )


def fit_coupler_zeropoint_to_legacy_results(
    fit: FluxLandscapeFit,
    *,
    decouple_offset: float,
    coupler_center: float | None,
) -> dict:
    """Map FluxLandscapeFit to the legacy coupler-zeropoint result dict."""
    gate_full = fit.optimal_gate_coupler_flux_total
    if np.isfinite(gate_full):
        flux_coupler_max = float(gate_full) - float(decouple_offset)
    else:
        flux_coupler_max = np.nan
    return {
        "flux_coupler_min": float(fit.optimal_decouple_coupler_flux_rel),
        "flux_coupler_min_full": float(fit.optimal_decouple_offset),
        "flux_qubit_max": float(fit.optimal_qubit_flux),
        "flux_coupler_max": flux_coupler_max,
        "flux_coupler_max_relative": float(fit.optimal_gate_coupler_flux_rel),
        "flux_coupler_center": float(coupler_center) if coupler_center is not None else np.nan,
        "flux_coupler_max_full": float(fit.optimal_gate_coupler_flux_total),
        "fit_success": fit.success,
    }


