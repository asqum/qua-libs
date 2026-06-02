from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import xarray as xr
from scipy.signal import find_peaks

__all__ = [
    "BranchDecoupleResult",
    "PairDecoupleResult",
    "DecoupleOffsetAnalysis",
    "analyze_branch_decouple_offset",
    "analyze_pair_decouple_offsets",
    "analyze_decouple_offsets",
    "detrend_linear_per_flux",
    "format_decouple_offset_summary",
    "initial_decouple_offset_from_detrended",
    "select_offset_closer_to_zero",
    "dataset_for_pair_analysis",
    "match_loaded_qubit_pair_dataset",
    "prepare_fetched_qubit_pair_dataset",
    "qubit_names_for_pair",
    "qubits_for_pair",
    "restore_pair_axis",
]

QUBIT_PAIR_ROLES = ["control", "target"]


def qubits_for_pair(qp):
    return [qp.qubit_control, qp.qubit_target]


def qubit_names_for_pair(qp):
    return [qubit.name for qubit in qubits_for_pair(qp)]


def prepare_fetched_qubit_pair_dataset(ds: xr.Dataset, qubit_pairs, dfs, dcs) -> xr.Dataset:
    """Format OPX streams saved as control/target variables on a qubit-pair axis."""
    ds = ds.rename({"qubit": "qubit_pair"})
    pair_names = [qp.name for qp in qubit_pairs]
    ds = ds.assign_coords(qubit_pair=pair_names)
    ds = ds.assign_coords(flux_coupler_full=(["qubit_pair", "flux"], np.array([dcs for _ in qubit_pairs])))
    ds = ds.assign_coords(flux_mV=ds.flux_coupler_full * 1e3)

    for role in QUBIT_PAIR_ROLES:
        qubits = [getattr(qp, f"qubit_{role}") for qp in qubit_pairs]
        readout_lengths = xr.DataArray(
            [qubit.resonator.operations["readout"].length for qubit in qubits],
            dims=["qubit_pair"],
            coords={"qubit_pair": pair_names},
        )
        ds[f"I_{role}"] = ds[f"I_{role}"] * 2**12 / readout_lengths
        ds[f"Q_{role}"] = ds[f"Q_{role}"] * 2**12 / readout_lengths
        ds[f"IQ_abs_{role}"] = np.sqrt(ds[f"I_{role}"] ** 2 + ds[f"Q_{role}"] ** 2)
        ds = ds.assign_coords(
            {
                f"freq_full_{role}": (
                    ["qubit_pair", "freq"],
                    np.array([dfs + qubit.resonator.RF_frequency for qubit in qubits]),
                )
            }
        )
        ds = ds.assign_coords({f"freq_GHz_{role}": ds[f"freq_full_{role}"] / 1e9})

    return ds


def match_loaded_qubit_pair_dataset(ds: xr.Dataset, qubit_pairs):
    """Add or filter the qubit_pair axis for loaded or multi-pair datasets."""
    if "qubit_pair" not in ds.dims:
        loaded_qubits = [str(qubit) for qubit in ds.qubit.values]
        matching_pairs = [qp for qp in qubit_pairs if qubit_names_for_pair(qp) == loaded_qubits]
        if len(qubit_pairs) == 1:
            ds = ds.expand_dims(qubit_pair=[qubit_pairs[0].name])
        elif matching_pairs:
            ds = ds.expand_dims(qubit_pair=[matching_pairs[0].name])
        else:
            raise ValueError(
                "Loaded dataset has no qubit_pair dimension and none of the requested qubit_pairs match "
                f"the dataset qubits {loaded_qubits}."
            )

    available_pairs = {str(qp_name) for qp_name in ds.qubit_pair.values}
    matched_pairs = [qp for qp in qubit_pairs if qp.name in available_pairs]
    if not matched_pairs:
        raise ValueError(f"No requested qubit_pairs are present in the dataset: {sorted(available_pairs)}")
    return ds, matched_pairs


def dataset_for_pair_analysis(ds: xr.Dataset, qp) -> xr.Dataset:
    """Return a two-qubit dataset for one qubit pair using real qubit names."""
    if "I_control" in ds and "I_target" in ds:
        pair_ds = ds.sel(qubit_pair=qp.name)
        return xr.Dataset(
            {
                var: (
                    ["qubit", "flux", "freq"],
                    np.stack([pair_ds[f"{var}_control"].values, pair_ds[f"{var}_target"].values]),
                )
                for var in ["I", "Q", "IQ_abs"]
            },
            coords={
                "qubit": qubit_names_for_pair(qp),
                "flux": ds.flux.values,
                "freq": ds.freq.values,
                "flux_coupler_full": ("flux", pair_ds.flux_coupler_full.values),
                "flux_mV": ("flux", pair_ds.flux_mV.values),
                "freq_full": (
                    ["qubit", "freq"],
                    np.stack([pair_ds.freq_full_control.values, pair_ds.freq_full_target.values]),
                ),
                "freq_GHz": (
                    ["qubit", "freq"],
                    np.stack([pair_ds.freq_GHz_control.values, pair_ds.freq_GHz_target.values]),
                ),
            },
        )

    pair_ds = ds.sel(qubit_pair=qp.name)
    if list(pair_ds.qubit.values) == QUBIT_PAIR_ROLES:
        pair_ds = pair_ds.assign_coords(qubit=qubit_names_for_pair(qp))
    if "freq_GHz" not in pair_ds.coords:
        pair_ds = pair_ds.assign_coords(freq_GHz=pair_ds.freq_full / 1e9)
    return pair_ds


def restore_pair_axis(ds_pair: xr.Dataset, qp) -> xr.Dataset:
    """Store one analyzed pair dataset back with role labels on the qubit axis."""
    ds_pair = ds_pair.assign_coords(qubit=QUBIT_PAIR_ROLES)
    ds_pair = ds_pair.assign_coords(qubit_name=(["qubit"], qubit_names_for_pair(qp)))
    return ds_pair.expand_dims(qubit_pair=[qp.name])


@dataclass
class BranchDecoupleResult:
    qubit: str
    detrended: np.ndarray
    freq_GHz: np.ndarray
    offset_mV: float
    offset_V: float
    wider_label: str
    minima_flux_mV: np.ndarray
    d_m1_m2_mV: float
    d_m2_m3_mV: float


@dataclass
class PairDecoupleResult:
    pair_name: str
    qubits: list[str]
    flux_mV: np.ndarray
    branches: dict[str, BranchDecoupleResult]
    selected_qubit: str
    selected_branch: str
    decouple_offset_mV: float
    decouple_offset_V: float


@dataclass
class DecoupleOffsetAnalysis:
    pairs: dict[str, PairDecoupleResult]
    pair_order: list[str]

    def decouple_offsets_V(self) -> dict[str, float]:
        return {name: self.pairs[name].decouple_offset_V for name in self.pair_order}


def detrend_linear_per_flux(image, fit_min_percentile: float = 35) -> np.ndarray:
    """Linear detrend along frequency, independently for each coupler-flux row."""
    image = np.asarray(image, dtype=float)
    freq_axis = np.linspace(-1.0, 1.0, image.shape[1])
    detrended = np.full_like(image, np.nan)

    for row_index, trace in enumerate(image):
        finite = np.isfinite(trace)
        if finite.sum() < 2:
            detrended[row_index] = trace
            continue

        threshold = np.nanpercentile(trace[finite], fit_min_percentile)
        fit_mask = finite & (trace >= threshold)
        if fit_mask.sum() < 2:
            fit_mask = finite

        coeff = np.polyfit(freq_axis[fit_mask], trace[fit_mask], deg=1)
        background = np.polyval(coeff, freq_axis)
        detrended[row_index] = trace - background + np.nanmedian(background)

    return detrended


def initial_decouple_offset_from_detrended(
    flux_mV: np.ndarray,
    detrended: np.ndarray,
    prominence_fraction: float = 0.05,
    distance: int = 5,
) -> dict:
    """Estimate branch offset from the three deepest row-contrast minima."""
    row_span = np.nanmax(detrended, axis=1) - np.nanmin(detrended, axis=1)
    dip_idx, dip_props = find_peaks(
        -row_span,
        prominence=prominence_fraction * np.nanmax(row_span),
        distance=distance,
    )
    if len(dip_idx) < 3:
        raise ValueError(f"find_peaks found only {len(dip_idx)} minima; lower prominence or distance.")

    prominence = dip_props["prominences"]
    top3_local = np.argsort(prominence)[-3:]
    top3_flux_mV = flux_mV[np.sort(dip_idx[top3_local])]

    d_m1_m2, d_m2_m3 = np.diff(top3_flux_mV)
    if d_m1_m2 >= d_m2_m3:
        wider_label = "M1-M2"
        wider_flux_mV = top3_flux_mV[0], top3_flux_mV[1]
    else:
        wider_label = "M2-M3"
        wider_flux_mV = top3_flux_mV[1], top3_flux_mV[2]

    offset_mV = 0.5 * (wider_flux_mV[0] + wider_flux_mV[1])
    return {
        "offset_mV": float(offset_mV),
        "offset_V": float(offset_mV * 1e-3),
        "wider_label": wider_label,
        "minima_flux_mV": top3_flux_mV,
        "d_m1_m2_mV": float(d_m1_m2),
        "d_m2_m3_mV": float(d_m2_m3),
    }


def select_offset_closer_to_zero(branch_offsets_mV: dict[str, float], qubit_names: Sequence[str]) -> tuple[str, str]:
    """Pick the branch whose |offset| is smaller (closer to 0 V)."""
    top_q, bottom_q = qubit_names[0], qubit_names[1]
    if abs(branch_offsets_mV[top_q]) <= abs(branch_offsets_mV[bottom_q]):
        return top_q, "top"
    return bottom_q, "bottom"


def analyze_branch_decouple_offset(
    pair_ds: xr.Dataset,
    qubit: str,
    flux_mV: np.ndarray,
    fit_min_percentile: float = 35,
    prominence_fraction: float = 0.05,
    distance: int = 5,
) -> BranchDecoupleResult:
    trace = pair_ds.IQ_abs.sel(qubit=qubit).transpose("flux", "freq")
    freq_GHz = (
        trace.freq_GHz.values
        if "freq_GHz" in trace.coords
        else pair_ds.freq_GHz.sel(qubit=qubit).values
    )
    detrended = detrend_linear_per_flux(np.asarray(trace.values, dtype=float), fit_min_percentile)
    offset = initial_decouple_offset_from_detrended(
        flux_mV,
        detrended,
        prominence_fraction=prominence_fraction,
        distance=distance,
    )
    return BranchDecoupleResult(
        qubit=qubit,
        detrended=detrended,
        freq_GHz=freq_GHz,
        **offset,
    )


def analyze_pair_decouple_offsets(
    ds: xr.Dataset,
    qp,
    fit_min_percentile: float = 35,
    prominence_fraction: float = 0.05,
    distance: int = 5,
) -> PairDecoupleResult:
    pair_ds = dataset_for_pair_analysis(ds, qp)
    flux_mV = pair_ds.flux.values * 1e3
    qubits = qubit_names_for_pair(qp)

    branches = {
        qubit: analyze_branch_decouple_offset(
            pair_ds,
            qubit,
            flux_mV,
            fit_min_percentile=fit_min_percentile,
            prominence_fraction=prominence_fraction,
            distance=distance,
        )
        for qubit in qubits
    }
    branch_offsets_mV = {qubit: branches[qubit].offset_mV for qubit in qubits}
    selected_qubit, selected_branch = select_offset_closer_to_zero(branch_offsets_mV, qubits)
    selected = branches[selected_qubit]

    return PairDecoupleResult(
        pair_name=qp.name,
        qubits=qubits,
        flux_mV=flux_mV,
        branches=branches,
        selected_qubit=selected_qubit,
        selected_branch=selected_branch,
        decouple_offset_mV=selected.offset_mV,
        decouple_offset_V=selected.offset_V,
    )


def analyze_decouple_offsets(
    ds: xr.Dataset,
    qubit_pairs,
    fit_min_percentile: float = 35,
    prominence_fraction: float = 0.05,
    distance: int = 5,
) -> DecoupleOffsetAnalysis:
    """Analyze all qubit pairs and pick |offset|-closest-to-zero decouple point per pair."""
    pair_order = [qp.name for qp in qubit_pairs]
    pairs = {
        qp.name: analyze_pair_decouple_offsets(
            ds,
            qp,
            fit_min_percentile=fit_min_percentile,
            prominence_fraction=prominence_fraction,
            distance=distance,
        )
        for qp in qubit_pairs
    }
    return DecoupleOffsetAnalysis(pairs=pairs, pair_order=pair_order)


def format_decouple_offset_summary(analysis: DecoupleOffsetAnalysis) -> str:
    lines = ["Best decouple_offset per pair (|offset| closer to 0):"]
    for pair_name in analysis.pair_order:
        pair = analysis.pairs[pair_name]
        top_q, bottom_q = pair.qubits
        top_mV = pair.branches[top_q].offset_mV
        bottom_mV = pair.branches[bottom_q].offset_mV
        lines.append(
            f"  {pair_name}: top {top_q} = {top_mV:.2f} mV | "
            f"bottom {bottom_q} = {bottom_mV:.2f} mV -> "
            f"decouple_offset = {pair.decouple_offset_mV:.2f} mV "
            f"({pair.decouple_offset_V:.6f} V) from {pair.selected_branch} ({pair.selected_qubit})"
        )
    return "\n".join(lines)

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import least_squares
from scipy.signal import find_peaks, savgol_filter

__all__ = [
    "BranchDecoupleResult",
    "PairDecoupleResult",
    "DecoupleOffsetAnalysis",
    "analyze_branch_decouple_offset",
    "analyze_pair_decouple_offsets",
    "analyze_decouple_offsets",
    "detrend_linear_per_flux",
    "format_decouple_offset_summary",
    "initial_decouple_offset_from_detrended",
    "select_offset_closer_to_zero",
    "fit_resonator_spectroscopy_vs_coupler_flux",
    "dataset_for_pair_analysis",
    "match_loaded_qubit_pair_dataset",
    "prepare_fetched_qubit_pair_dataset",
    "qubit_names_for_pair",
    "qubits_for_pair",
    "restore_pair_axis",
    "select_best_fit_by_r_squared",
]

QUBIT_PAIR_ROLES = ["control", "target"]


def qubits_for_pair(qp):
    return [qp.qubit_control, qp.qubit_target]


def qubit_names_for_pair(qp):
    return [qubit.name for qubit in qubits_for_pair(qp)]


def prepare_fetched_qubit_pair_dataset(ds: xr.Dataset, qubit_pairs, dfs, dcs) -> xr.Dataset:
    """Format OPX streams saved as control/target variables on a qubit-pair axis."""
    ds = ds.rename({"qubit": "qubit_pair"})
    pair_names = [qp.name for qp in qubit_pairs]
    ds = ds.assign_coords(qubit_pair=pair_names)
    ds = ds.assign_coords(flux_coupler_full=(["qubit_pair", "flux"], np.array([dcs for _ in qubit_pairs])))
    ds = ds.assign_coords(flux_mV=ds.flux_coupler_full * 1e3)

    for role in QUBIT_PAIR_ROLES:
        qubits = [getattr(qp, f"qubit_{role}") for qp in qubit_pairs]
        readout_lengths = xr.DataArray(
            [qubit.resonator.operations["readout"].length for qubit in qubits],
            dims=["qubit_pair"],
            coords={"qubit_pair": pair_names},
        )
        ds[f"I_{role}"] = ds[f"I_{role}"] * 2**12 / readout_lengths
        ds[f"Q_{role}"] = ds[f"Q_{role}"] * 2**12 / readout_lengths
        ds[f"IQ_abs_{role}"] = np.sqrt(ds[f"I_{role}"] ** 2 + ds[f"Q_{role}"] ** 2)
        ds = ds.assign_coords(
            {
                f"freq_full_{role}": (
                    ["qubit_pair", "freq"],
                    np.array([dfs + qubit.resonator.RF_frequency for qubit in qubits]),
                )
            }
        )
        ds = ds.assign_coords({f"freq_GHz_{role}": ds[f"freq_full_{role}"] / 1e9})

    return ds


def match_loaded_qubit_pair_dataset(ds: xr.Dataset, qubit_pairs):
    """Add or filter the qubit_pair axis for loaded or multi-pair datasets."""
    if "qubit_pair" not in ds.dims:
        loaded_qubits = [str(qubit) for qubit in ds.qubit.values]
        matching_pairs = [qp for qp in qubit_pairs if qubit_names_for_pair(qp) == loaded_qubits]
        if len(qubit_pairs) == 1:
            ds = ds.expand_dims(qubit_pair=[qubit_pairs[0].name])
        elif matching_pairs:
            ds = ds.expand_dims(qubit_pair=[matching_pairs[0].name])
        else:
            raise ValueError(
                "Loaded dataset has no qubit_pair dimension and none of the requested qubit_pairs match "
                f"the dataset qubits {loaded_qubits}."
            )

    available_pairs = {str(qp_name) for qp_name in ds.qubit_pair.values}
    matched_pairs = [qp for qp in qubit_pairs if qp.name in available_pairs]
    if not matched_pairs:
        raise ValueError(f"No requested qubit_pairs are present in the dataset: {sorted(available_pairs)}")
    return ds, matched_pairs


def dataset_for_pair_analysis(ds: xr.Dataset, qp) -> xr.Dataset:
    """Return a two-qubit dataset for one qubit pair using real qubit names."""
    if "I_control" in ds and "I_target" in ds:
        pair_ds = ds.sel(qubit_pair=qp.name)
        return xr.Dataset(
            {
                var: (
                    ["qubit", "flux", "freq"],
                    np.stack([pair_ds[f"{var}_control"].values, pair_ds[f"{var}_target"].values]),
                )
                for var in ["I", "Q", "IQ_abs"]
            },
            coords={
                "qubit": qubit_names_for_pair(qp),
                "flux": ds.flux.values,
                "freq": ds.freq.values,
                "flux_coupler_full": ("flux", pair_ds.flux_coupler_full.values),
                "flux_mV": ("flux", pair_ds.flux_mV.values),
                "freq_full": (
                    ["qubit", "freq"],
                    np.stack([pair_ds.freq_full_control.values, pair_ds.freq_full_target.values]),
                ),
                "freq_GHz": (
                    ["qubit", "freq"],
                    np.stack([pair_ds.freq_GHz_control.values, pair_ds.freq_GHz_target.values]),
                ),
            },
        )

    pair_ds = ds.sel(qubit_pair=qp.name)
    if list(pair_ds.qubit.values) == QUBIT_PAIR_ROLES:
        pair_ds = pair_ds.assign_coords(qubit=qubit_names_for_pair(qp))
    if "freq_GHz" not in pair_ds.coords:
        pair_ds = pair_ds.assign_coords(freq_GHz=pair_ds.freq_full / 1e9)
    return pair_ds


def restore_pair_axis(ds_pair: xr.Dataset, qp) -> xr.Dataset:
    """Store one analyzed pair dataset back with role labels on the qubit axis."""
    ds_pair = ds_pair.assign_coords(qubit=QUBIT_PAIR_ROLES)
    ds_pair = ds_pair.assign_coords(qubit_name=(["qubit"], qubit_names_for_pair(qp)))
    return ds_pair.expand_dims(qubit_pair=[qp.name])


def fit_resonator_spectroscopy_vs_coupler_flux(
    ds: xr.Dataset,
    qubits=None,
    data_var: str = "IQ_abs",
    qubit_dim: str = "qubit",
    flux_dim: str = "flux",
    freq_dim: str = "freq",
    freq_ref_coord: str = "freq_GHz",
    edge_points: int = 8,
    smooth_window: int = 11,
    prominence_fraction: float = 0.10,
    min_distance: int = 10,
    max_dips_per_flux: Optional[int] = 1,
    weighted: bool = True,
    weight_column: str = "prominence",
    weight_power: float = 1.5,
    normalize_weights: bool = True,
    loss: str = "soft_l1",
    f_scale: float = 0.30,
    num_fit_points: int = 2000,
    enforce_fc_max_above_fr: bool = True,
    min_fc_max_above_fr_GHz: float = 1e-6,
    fit_selection_min_valid_dips: int = 5,
    fit_kwargs: Optional[dict] = None,
) -> Tuple[xr.Dataset, dict]:
    """Detect resonator dips, fit them, and store the analysis back into ``ds``."""
    if data_var not in ds:
        raise KeyError(f"Dataset does not contain data variable {data_var!r}.")

    ds = _ensure_analysis_coordinates(ds, freq_ref_coord, flux_dim, freq_dim)
    qubit_names = _normalize_qubit_names(ds, qubits, qubit_dim)
    flux_values = np.asarray(ds[flux_dim].values, dtype=float)

    fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)
    fit_param_names = ["f_r", "fc_max", "g", "flux_period", "flux_offset"]

    dip_freq = np.full((len(qubit_names), flux_values.size), np.nan)
    dip_freq_abs_GHz = np.full_like(dip_freq, np.nan)
    dip_iq_abs = np.full_like(dip_freq, np.nan)
    dip_prominence = np.full_like(dip_freq, np.nan)
    dip_sigma = np.full_like(dip_freq, np.nan)
    fit_freq_GHz = np.full((len(qubit_names), num_fit_points), np.nan)
    fit_freq_Hz = np.full_like(fit_freq_GHz, np.nan)
    fit_params = np.full((len(qubit_names), len(fit_param_names)), np.nan)
    decouple_offsets = np.full(len(qubit_names), np.nan)
    fit_flux = None

    fit_results = {}
    for qubit_index, qubit in enumerate(qubit_names):
        dip_points = detect_resonator_dips(
            ds,
            qubit=qubit,
            data_var=data_var,
            qubit_dim=qubit_dim,
            flux_dim=flux_dim,
            freq_dim=freq_dim,
            edge_points=edge_points,
            smooth_window=smooth_window,
            prominence_fraction=prominence_fraction,
            min_distance=min_distance,
            max_dips_per_flux=max_dips_per_flux,
        )
        _store_dip_points_in_arrays(
            dip_points,
            flux_values,
            qubit_index,
            dip_freq,
            dip_iq_abs,
            dip_prominence,
            dip_sigma,
            flux_dim,
            freq_dim,
        )

        fit = fit_hamiltonian_dips(
            ds,
            dip_points,
            qubit=qubit,
            data_var=data_var,
            qubit_dim=qubit_dim,
            flux_dim=flux_dim,
            freq_dim=freq_dim,
            freq_ref_coord=freq_ref_coord,
            weighted=weighted,
            weight_column=weight_column,
            weight_power=weight_power,
            normalize_weights=normalize_weights,
            loss=loss,
            f_scale=f_scale,
            num_fit_points=num_fit_points,
            enforce_fc_max_above_fr=enforce_fc_max_above_fr,
            min_fc_max_above_fr_GHz=min_fc_max_above_fr_GHz,
            **fit_kwargs,
        )

        if fit_flux is None:
            fit_flux = np.asarray(fit["flux_fit"], dtype=float)
        fit_freq_GHz[qubit_index, :] = np.asarray(fit["freq_fit_abs_GHz"], dtype=float)
        fit_freq_Hz[qubit_index, :] = np.asarray(fit["freq_fit_Hz"], dtype=float)
        fit_params[qubit_index, :] = [float(fit["params"].loc[name]) for name in fit_param_names]
        decouple_offsets[qubit_index] = float(fit["params"].loc["flux_offset"])
        dip_freq_abs_GHz[qubit_index, :] = float(fit["freq_ref_GHz"]) + dip_freq[qubit_index, :] / 1e9

        fit_results[qubit] = _serializable_fit_result(fit, dip_points)

    if fit_flux is None:
        fit_flux = np.linspace(float(ds[flux_dim].min()), float(ds[flux_dim].max()), num_fit_points)

    ds = ds.assign_coords(
        {
            "coupler_fit_flux": fit_flux,
            "coupler_fit_flux_mV": ("coupler_fit_flux", fit_flux * 1e3),
            "hamiltonian_fit_param": fit_param_names,
        }
    )
    ds = ds.assign(
        {
            "resonator_dip_freq": ([qubit_dim, flux_dim], dip_freq),
            "resonator_dip_freq_abs_GHz": ([qubit_dim, flux_dim], dip_freq_abs_GHz),
            "resonator_dip_IQ_abs": ([qubit_dim, flux_dim], dip_iq_abs),
            "resonator_dip_prominence": ([qubit_dim, flux_dim], dip_prominence),
            "resonator_dip_sigma": ([qubit_dim, flux_dim], dip_sigma),
            "hamiltonian_fit_freq_GHz": ([qubit_dim, "coupler_fit_flux"], fit_freq_GHz),
            "hamiltonian_fit_freq": ([qubit_dim, "coupler_fit_flux"], fit_freq_Hz),
            "hamiltonian_fit_params": ([qubit_dim, "hamiltonian_fit_param"], fit_params),
            "coupler_decouple_offset": ([qubit_dim], decouple_offsets),
        }
    )

    ds.resonator_dip_freq.attrs.update(long_name="Detected resonator dip detuning", units="Hz")
    ds.resonator_dip_freq_abs_GHz.attrs.update(long_name="Detected resonator dip frequency", units="GHz")
    ds.resonator_dip_IQ_abs.attrs.update(long_name="Detected dip IQ amplitude")
    ds.resonator_dip_prominence.attrs.update(long_name="Detected dip prominence")
    ds.hamiltonian_fit_freq_GHz.attrs.update(long_name="Hamiltonian fitted resonator branch", units="GHz")
    ds.hamiltonian_fit_freq.attrs.update(long_name="Hamiltonian fitted resonator branch detuning", units="Hz")
    ds.coupler_decouple_offset.attrs.update(long_name="Fitted coupler decouple offset", units="V")
    ds.hamiltonian_fit_params.attrs.update(
        long_name="Hamiltonian fit parameters",
        description="Parameters are f_r, fc_max, g, flux_period, flux_offset. Frequencies are in GHz; flux values are in V.",
    )

    finite_offsets = decouple_offsets[np.isfinite(decouple_offsets)]
    fit_results["decouple_offset_mean"] = float(np.mean(finite_offsets)) if finite_offsets.size else np.nan
    ds, selection = select_best_fit_by_r_squared(
        ds,
        qubits=qubit_names,
        qubit_dim=qubit_dim,
        flux_dim=flux_dim,
        min_valid_dips=fit_selection_min_valid_dips,
    )
    fit_results["selected_decouple_qubit"] = selection["selected_qubit"]
    fit_results["selected_decouple_offset"] = selection["selected_decouple_offset"]
    fit_results["fit_quality"] = selection["quality"]
    return ds, fit_results


def select_best_fit_by_r_squared(
    ds: xr.Dataset,
    qubits=None,
    qubit_dim: str = "qubit",
    flux_dim: str = "flux",
    fit_flux_dim: str = "coupler_fit_flux",
    dip_freq_var: str = "resonator_dip_freq_abs_GHz",
    dip_prominence_var: str = "resonator_dip_prominence",
    fit_freq_var: str = "hamiltonian_fit_freq_GHz",
    decouple_var: str = "coupler_decouple_offset",
    min_valid_dips: int = 5,
) -> Tuple[xr.Dataset, dict]:
    """Select the best per-qubit fit using prominence-weighted R squared."""
    for var_name in [dip_freq_var, fit_freq_var, decouple_var]:
        if var_name not in ds:
            raise KeyError(f"Dataset does not contain analysis variable {var_name!r}.")

    qubit_names = _normalize_qubit_names(ds, qubits, qubit_dim)
    r_squared = np.full(len(qubit_names), np.nan)
    unweighted_r_squared = np.full(len(qubit_names), np.nan)
    mean_abs_residual_MHz = np.full(len(qubit_names), np.nan)
    valid_dips = np.zeros(len(qubit_names), dtype=int)

    for qubit_index, qubit in enumerate(qubit_names):
        dip_freq = ds[dip_freq_var].sel({qubit_dim: qubit})
        fit_at_dips = ds[fit_freq_var].sel({qubit_dim: qubit}).interp({fit_flux_dim: dip_freq[flux_dim]})
        y = np.asarray(dip_freq.values, dtype=float)
        y_fit = np.asarray(fit_at_dips.values, dtype=float)
        valid = np.isfinite(y) & np.isfinite(y_fit)
        if valid.sum() == 0:
            continue

        weights = _fit_quality_weights(ds, dip_prominence_var, qubit_dim, qubit, valid)
        y_valid = y[valid]
        y_fit_valid = y_fit[valid]
        valid_dips[qubit_index] = int(valid.sum())
        r_squared[qubit_index] = _weighted_r_squared(y_valid, y_fit_valid, weights)
        unweighted_r_squared[qubit_index] = _weighted_r_squared(y_valid, y_fit_valid, None)
        mean_abs_residual_MHz[qubit_index] = float(np.nanmean(np.abs(y_fit_valid - y_valid)) * 1e3)

    selectable = np.isfinite(r_squared) & (valid_dips >= min_valid_dips)
    if not selectable.any():
        selectable = np.isfinite(r_squared)

    if selectable.any():
        selected_index = int(np.nanargmax(np.where(selectable, r_squared, np.nan)))
        selected_qubit = qubit_names[selected_index]
        selected_decouple_offset = float(ds[decouple_var].sel({qubit_dim: selected_qubit}))
    else:
        selected_index = None
        selected_qubit = None
        selected_decouple_offset = np.nan

    ds = ds.assign(
        {
            "coupler_fit_r_squared": ([qubit_dim], r_squared),
            "coupler_fit_unweighted_r_squared": ([qubit_dim], unweighted_r_squared),
            "coupler_fit_mean_abs_residual_MHz": ([qubit_dim], mean_abs_residual_MHz),
            "coupler_fit_valid_dips": ([qubit_dim], valid_dips),
            "selected_coupler_decouple_offset": selected_decouple_offset,
        }
    )
    ds.coupler_fit_r_squared.attrs.update(long_name="Prominence-weighted R squared between detected dips and fitted curve")
    ds.coupler_fit_unweighted_r_squared.attrs.update(long_name="Unweighted R squared between detected dips and fitted curve")
    ds.coupler_fit_mean_abs_residual_MHz.attrs.update(
        long_name="Mean absolute residual between detected dips and fitted curve",
        units="MHz",
    )
    ds.selected_coupler_decouple_offset.attrs.update(
        long_name="Selected coupler decouple offset from highest R squared fit",
        units="V",
        selected_qubit="" if selected_qubit is None else selected_qubit,
    )

    quality = {}
    for qubit_index, qubit in enumerate(qubit_names):
        quality[qubit] = {
            "r_squared": float(r_squared[qubit_index]),
            "unweighted_r_squared": float(unweighted_r_squared[qubit_index]),
            "mean_abs_residual_MHz": float(mean_abs_residual_MHz[qubit_index]),
            "valid_dips": int(valid_dips[qubit_index]),
        }

    return ds, {
        "selected_qubit": selected_qubit,
        "selected_index": selected_index,
        "selected_decouple_offset": selected_decouple_offset,
        "quality": quality,
    }


def _fit_quality_weights(
    ds: xr.Dataset,
    dip_prominence_var: str,
    qubit_dim: str,
    qubit: str,
    valid_mask: np.ndarray,
) -> Optional[np.ndarray]:
    if dip_prominence_var not in ds:
        return None

    weights = np.asarray(ds[dip_prominence_var].sel({qubit_dim: qubit}).values, dtype=float)
    weights = weights[valid_mask]
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, np.nan)
    if not np.isfinite(weights).any():
        return None

    scale = np.nanmedian(weights)
    if np.isfinite(scale) and scale > 0:
        weights = weights / scale
    return weights


def _weighted_r_squared(y: np.ndarray, y_fit: np.ndarray, weights: Optional[np.ndarray]) -> float:
    y = np.asarray(y, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    finite = np.isfinite(y) & np.isfinite(y_fit)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        finite &= np.isfinite(weights) & (weights > 0)
    if finite.sum() < 2:
        return np.nan

    if weights is None:
        y_mean = np.nanmean(y[finite])
        ss_res = np.nansum(np.square(y[finite] - y_fit[finite]))
        ss_tot = np.nansum(np.square(y[finite] - y_mean))
    else:
        y_mean = np.average(y[finite], weights=weights[finite])
        ss_res = np.sum(weights[finite] * np.square(y[finite] - y_fit[finite]))
        ss_tot = np.sum(weights[finite] * np.square(y[finite] - y_mean))

    if not np.isfinite(ss_tot) or ss_tot <= 0:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def _ensure_analysis_coordinates(
    ds: xr.Dataset,
    freq_ref_coord: str,
    flux_dim: str,
    freq_dim: str,
) -> xr.Dataset:
    if freq_ref_coord == "freq_GHz" and freq_ref_coord not in ds.coords:
        if "freq_full" not in ds.coords:
            raise KeyError(
                "Dataset does not contain 'freq_GHz' or 'freq_full'. Add an absolute resonator "
                "frequency coordinate before fitting."
            )
        ds = ds.assign_coords(freq_GHz=ds.freq_full / 1e9)

    if "flux_mV" not in ds.coords and flux_dim in ds.coords:
        ds = ds.assign_coords(flux_mV=ds[flux_dim] * 1e3)

    if freq_dim not in ds.coords:
        raise KeyError(f"Dataset does not contain frequency coordinate {freq_dim!r}.")
    if flux_dim not in ds.coords:
        raise KeyError(f"Dataset does not contain flux coordinate {flux_dim!r}.")
    return ds


def _normalize_qubit_names(ds: xr.Dataset, qubits, qubit_dim: str) -> list:
    if qubits is None:
        return [str(qubit) for qubit in ds[qubit_dim].values]

    qubit_names = []
    for qubit in qubits:
        qubit_names.append(getattr(qubit, "name", str(qubit)))
    return qubit_names


def _store_dip_points_in_arrays(
    dip_points: pd.DataFrame,
    flux_values: np.ndarray,
    qubit_index: int,
    dip_freq: np.ndarray,
    dip_iq_abs: np.ndarray,
    dip_prominence: np.ndarray,
    dip_sigma: np.ndarray,
    flux_dim: str,
    freq_dim: str,
) -> None:
    for _, row in dip_points.iterrows():
        flux = float(row[flux_dim])
        flux_index = int(np.nanargmin(np.abs(flux_values - flux)))
        prominence = float(row["prominence"])
        current_prominence = dip_prominence[qubit_index, flux_index]
        if np.isfinite(current_prominence) and current_prominence >= prominence:
            continue

        dip_freq[qubit_index, flux_index] = float(row[freq_dim])
        dip_iq_abs[qubit_index, flux_index] = float(row["IQ_abs"])
        dip_prominence[qubit_index, flux_index] = prominence
        dip_sigma[qubit_index, flux_index] = float(row.get("sigma", np.nan))


def _serializable_fit_result(fit: dict, dip_points: pd.DataFrame) -> dict:
    summary = fit["summary"]["value"].to_dict()
    params = fit["params"].to_dict()
    return {
        "branch": fit["branch"],
        "decouple_offset": float(params["flux_offset"]),
        "detected_dips": int(len(dip_points)),
        "fit_score": float(fit["fit_score"]),
        "params": {key: float(value) for key, value in params.items()},
        "summary": {key: float(value) for key, value in summary.items()},
        "weighted": bool(fit["weighted"]),
    }


def detect_resonator_dips(
    ds: xr.Dataset,
    qubit: str,
    data_var: str = "IQ_abs",
    qubit_dim: str = "qubit",
    flux_dim: str = "flux",
    freq_dim: str = "freq",
    edge_points: int = 8,
    smooth_window: int = 21,
    prominence_fraction: float = 0.08,
    min_distance: int = 8,
    max_dips_per_flux: Optional[int] = 1,
) -> pd.DataFrame:
    """Detect local minima in resonator spectroscopy data."""
    if data_var not in ds:
        raise KeyError(f"Dataset does not contain data variable {data_var!r}.")

    data = ds[data_var].sel({qubit_dim: qubit}).transpose(flux_dim, freq_dim)
    freq = np.asarray(data[freq_dim].values, dtype=float)
    flux = np.asarray(data[flux_dim].values, dtype=float)
    iq_abs = np.asarray(data.values, dtype=float)

    if iq_abs.ndim != 2:
        raise ValueError(
            f"Expected {data_var!r} selected by {qubit_dim}={qubit!r} to be 2D, "
            f"got shape {iq_abs.shape}."
        )

    window = _valid_savgol_window(smooth_window, iq_abs.shape[1])
    if window is not None:
        smoothed = savgol_filter(iq_abs, window_length=window, polyorder=2, axis=1, mode="interp")
    else:
        smoothed = iq_abs

    dip_points = _detect_dips(
        iq_abs=iq_abs,
        smoothed=smoothed,
        freq=freq,
        flux=flux,
        qubit=qubit,
        edge_points=edge_points,
        prominence_fraction=prominence_fraction,
        min_distance=min_distance,
        max_dips_per_flux=max_dips_per_flux,
    )
    return dip_points


def _valid_savgol_window(requested_window: int, n_freq: int) -> Optional[int]:
    """Return a valid odd Savitzky-Golay window, or None to skip smoothing."""
    if requested_window < 5 or n_freq < 5:
        return None

    window = min(int(requested_window), n_freq if n_freq % 2 == 1 else n_freq - 1)
    if window % 2 == 0:
        window -= 1

    if window < 5:
        return None
    return window


def _detect_dips(
    iq_abs: np.ndarray,
    smoothed: np.ndarray,
    freq: np.ndarray,
    flux: np.ndarray,
    qubit: str,
    edge_points: int,
    prominence_fraction: float,
    min_distance: int,
    max_dips_per_flux: Optional[int],
) -> pd.DataFrame:
    dip_rows = []

    for flux_idx, trace in enumerate(smoothed):
        if np.isfinite(trace).sum() < 5:
            continue

        span = np.nanpercentile(trace, 95) - np.nanpercentile(trace, 5)
        if not np.isfinite(span) or span <= 0:
            continue

        peak_idx, props = find_peaks(
            -trace,
            prominence=prominence_fraction * span,
            distance=min_distance,
        )
        keep = (peak_idx >= edge_points) & (peak_idx < trace.size - edge_points)
        peak_idx = peak_idx[keep]
        prominences = props["prominences"][keep]

        if peak_idx.size == 0:
            continue

        strongest = np.argsort(prominences)[::-1]
        if max_dips_per_flux is not None:
            strongest = strongest[:max_dips_per_flux]

        for order_idx in strongest:
            freq_idx = peak_idx[order_idx]
            dip_rows.append(
                {
                    "qubit": qubit,
                    "flux": flux[flux_idx],
                    "freq": freq[freq_idx],
                    "IQ_abs": iq_abs[flux_idx, freq_idx],
                    "prominence": prominences[order_idx],
                    "sigma": 1.0 / prominences[order_idx],
                }
            )

    return pd.DataFrame(
        dip_rows,
        columns=["qubit", "flux", "freq", "IQ_abs", "prominence", "sigma"],
    )


def coupler_frequency_GHz(flux_array, fc_max, flux_period, flux_offset):
    """Return the tunable coupler frequency in GHz."""
    phase = np.pi * (np.asarray(flux_array, dtype=float) - flux_offset) / flux_period
    return fc_max * np.sqrt(np.abs(np.cos(phase)))


def hamiltonian_eigenbranch_GHz(
    flux_array,
    f_r,
    fc_max,
    g,
    flux_period,
    flux_offset,
    branch="resonator_like",
):
    """Return eigenfrequency from a vectorized 2x2 Hamiltonian diagonalization."""
    flux_array = np.asarray(flux_array, dtype=float)
    f_c = coupler_frequency_GHz(flux_array, fc_max, flux_period, flux_offset)

    hamiltonian = np.zeros((flux_array.size, 2, 2), dtype=float)
    hamiltonian[:, 0, 0] = f_r
    hamiltonian[:, 1, 1] = f_c
    hamiltonian[:, 0, 1] = g
    hamiltonian[:, 1, 0] = g

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

    if branch == "lower":
        return eigenvalues[:, 0]
    if branch == "upper":
        return eigenvalues[:, 1]
    if branch != "resonator_like":
        raise ValueError(f"Unknown branch: {branch}")

    resonator_weights = np.abs(eigenvectors[:, 0, :]) ** 2
    branch_index = np.argmax(resonator_weights, axis=1)
    return eigenvalues[np.arange(flux_array.size), branch_index]


def fit_hamiltonian_dips(
    ds: xr.Dataset,
    dip_points: pd.DataFrame,
    qubit: str,
    data_var: str = "IQ_abs",
    qubit_dim: str = "qubit",
    flux_dim: str = "flux",
    freq_dim: str = "freq",
    freq_ref_coord: str = "freq_GHz",
    freq_ref_GHz: Optional[float] = None,
    initial_guesses: Optional[np.ndarray] = None,
    lower_bounds: Optional[np.ndarray] = None,
    upper_bounds: Optional[np.ndarray] = None,
    branch: str = "resonator_like",
    weighted: bool = True,
    weight_column: str = "prominence",
    weight_power: float = 1.0,
    normalize_weights: bool = True,
    loss: str = "soft_l1",
    f_scale: float = 0.30,
    max_nfev: int = 2000,
    num_fit_points: int = 2000,
    enforce_fc_max_above_fr: bool = True,
    min_fc_max_above_fr_GHz: float = 1e-6,
) -> dict:
    """Fit detected resonator dips to the 2x2 Hamiltonian model."""
    if dip_points.empty:
        raise ValueError("dip_points is empty. Run dip detection before fitting.")
    if data_var not in ds:
        raise KeyError(f"Dataset does not contain data variable {data_var!r}.")

    data = ds[data_var].sel({qubit_dim: qubit}).transpose(flux_dim, freq_dim)
    freq_ref_GHz = _get_freq_ref_GHz(data, freq_dim, freq_ref_coord, freq_ref_GHz)

    fit_data = dip_points.dropna(subset=[flux_dim, freq_dim, "prominence"]).copy()
    if fit_data.empty:
        raise ValueError("No finite dip points remain after dropping NaNs.")
    if "sigma" not in fit_data and "prominence" in fit_data:
        fit_data["sigma"] = 1.0 / fit_data["prominence"]

    fit_data["freq_abs_GHz"] = freq_ref_GHz + fit_data[freq_dim] / 1e9

    flux_data = fit_data[flux_dim].to_numpy(dtype=float)
    freq_data_GHz = fit_data["freq_abs_GHz"].to_numpy(dtype=float)
    fit_weights = _fit_weights(
        fit_data,
        weighted=weighted,
        weight_column=weight_column,
        weight_power=weight_power,
        normalize_weights=normalize_weights,
    )
    fit_data["fit_weight"] = fit_weights

    if lower_bounds is None:
        lower_bounds = np.array([freq_ref_GHz - 0.05, 4.50, 1e-4, 0.05, -0.60], dtype=float)
    if upper_bounds is None:
        upper_bounds = np.array([freq_ref_GHz + 0.05, 12.0, 0.50, 1.50, 0.60], dtype=float)

    param_names = ["f_r", "fc_max", "g", "flux_period", "flux_offset"]
    initial_guess_info = {}
    if initial_guesses is None:
        initial_guesses, initial_guess_info = _hamiltonian_initial_guesses_from_dips(
            fit_data,
            freq_ref_GHz,
            flux_dim,
            freq_dim,
        )
    initial_guesses = _clip_initial_guesses_to_bounds(initial_guesses, lower_bounds, upper_bounds)
    if enforce_fc_max_above_fr:
        fit_lower_bounds, fit_upper_bounds = _hamiltonian_gap_parameter_bounds(
            lower_bounds,
            upper_bounds,
            min_fc_max_above_fr_GHz,
        )
        fit_initial_guesses = _physical_to_gap_parameter_guesses(
            initial_guesses,
            fit_lower_bounds,
            fit_upper_bounds,
        )
    else:
        fit_lower_bounds = lower_bounds
        fit_upper_bounds = upper_bounds
        fit_initial_guesses = initial_guesses

    freq_scale = max(np.nanstd(freq_data_GHz), 1e-3)

    def residuals(fit_params):
        params = _gap_parameter_to_physical_params(fit_params) if enforce_fc_max_above_fr else fit_params
        predicted = hamiltonian_eigenbranch_GHz(flux_data, *params, branch=branch)
        return fit_weights * (predicted - freq_data_GHz) / freq_scale

    candidates = []
    for guess in np.asarray(fit_initial_guesses, dtype=float):
        result = least_squares(
            residuals,
            guess,
            bounds=(fit_lower_bounds, fit_upper_bounds),
            loss=loss,
            f_scale=f_scale,
            max_nfev=max_nfev,
        )
        score = np.mean(np.abs(result.fun))
        candidates.append((score, result))

    fit_score, fit_result = min(candidates, key=lambda item: item[0])
    if enforce_fc_max_above_fr:
        fit_result.gap_parameter_x = fit_result.x.copy()
        fit_result.x = _gap_parameter_to_physical_params(fit_result.x)
    fit_params = pd.Series(fit_result.x, index=param_names, name="value")

    fit_data["freq_fit_abs_GHz"] = hamiltonian_eigenbranch_GHz(flux_data, *fit_result.x, branch=branch)
    fit_data["freq_fit"] = (fit_data["freq_fit_abs_GHz"] - freq_ref_GHz) * 1e9
    fit_data["residual_MHz"] = (fit_data["freq_fit_abs_GHz"] - fit_data["freq_abs_GHz"]) * 1e3

    flux_fit = np.linspace(float(data[flux_dim].min()), float(data[flux_dim].max()), num_fit_points)
    freq_fit_abs_GHz = hamiltonian_eigenbranch_GHz(flux_fit, *fit_result.x, branch=branch)
    freq_fit_Hz = (freq_fit_abs_GHz - freq_ref_GHz) * 1e9

    fit_summary = pd.concat(
        [
            fit_params,
            pd.Series(
                {
                    "freq_ref_GHz": freq_ref_GHz,
                    "weighted": bool(weighted),
                    "enforce_fc_max_above_fr": bool(enforce_fc_max_above_fr),
                    "min_fc_max_above_fr_GHz": min_fc_max_above_fr_GHz,
                    "fc_max_minus_f_r_GHz": fit_params["fc_max"] - fit_params["f_r"],
                    "fit_score": fit_score,
                    "initial_flux_period": initial_guess_info.get("flux_period", np.nan),
                    "initial_flux_offset": initial_guess_info.get("flux_offset", np.nan),
                    "mean_fit_weight": np.mean(fit_weights),
                    "mean_abs_residual_MHz": fit_data["residual_MHz"].abs().mean(),
                    "median_abs_residual_MHz": fit_data["residual_MHz"].abs().median(),
                    "p90_abs_residual_MHz": fit_data["residual_MHz"].abs().quantile(0.90),
                },
                name="value",
            ),
        ]
    ).to_frame()

    return {
        "branch": branch,
        "data": fit_data,
        "fit_score": fit_score,
        "flux_fit": flux_fit,
        "freq_fit_abs_GHz": freq_fit_abs_GHz,
        "freq_fit_Hz": freq_fit_Hz,
        "freq_ref_GHz": freq_ref_GHz,
        "freq_scale": freq_scale,
        "initial_guess_info": initial_guess_info,
        "initial_guesses": initial_guesses,
        "params": fit_params,
        "result": fit_result,
        "summary": fit_summary,
        "weighted": bool(weighted),
        "weights": fit_weights,
    }


def _get_freq_ref_GHz(data: xr.DataArray, freq_dim: str, freq_ref_coord: str, freq_ref_GHz: Optional[float]) -> float:
    if freq_ref_GHz is not None:
        return float(freq_ref_GHz)

    freq_axis_Hz = np.asarray(data[freq_dim].values, dtype=float)
    zero_freq_idx = int(np.nanargmin(np.abs(freq_axis_Hz)))
    if freq_ref_coord not in data.coords:
        raise KeyError(
            f"Could not find coordinate {freq_ref_coord!r}. Pass freq_ref_GHz explicitly if the dataset "
            "does not include an absolute frequency coordinate."
        )
    return float(data[freq_ref_coord].isel({freq_dim: zero_freq_idx}))


def _fit_weights(
    fit_data: pd.DataFrame,
    weighted: bool,
    weight_column: str,
    weight_power: float,
    normalize_weights: bool,
) -> np.ndarray:
    if not weighted:
        return np.ones(len(fit_data), dtype=float)

    if weight_column not in fit_data:
        raise KeyError(f"Cannot use weights because dip_points does not contain {weight_column!r}.")

    values = fit_data[weight_column].to_numpy(dtype=float)
    if weight_column == "sigma":
        weights = 1.0 / values
    else:
        weights = values

    weights = np.where(np.isfinite(weights) & (weights > 0), weights, np.nan)
    fill_value = np.nanmedian(weights)
    if not np.isfinite(fill_value) or fill_value <= 0:
        fill_value = 1.0
    weights = np.nan_to_num(weights, nan=fill_value, posinf=fill_value, neginf=fill_value)
    weights = np.power(weights, weight_power)

    if normalize_weights:
        scale = np.nanmedian(weights)
        if np.isfinite(scale) and scale > 0:
            weights = weights / scale

    return weights


def _hamiltonian_initial_guesses_from_dips(
    fit_data: pd.DataFrame,
    freq_ref_GHz: float,
    flux_dim: str,
    freq_dim: str,
) -> Tuple[np.ndarray, dict]:
    """Build deterministic Hamiltonian initial guesses from the detected dip curve."""
    initial_flux = _estimate_flux_period_and_offset_from_dips(fit_data, flux_dim, freq_dim)
    flux_period = initial_flux["flux_period"]
    flux_offset = initial_flux["flux_offset"]

    freq_abs_GHz = freq_ref_GHz + fit_data[freq_dim].to_numpy(dtype=float) / 1e9
    finite_freq = freq_abs_GHz[np.isfinite(freq_abs_GHz)]
    if finite_freq.size:
        freq_span_GHz = float(np.nanmax(finite_freq) - np.nanmin(finite_freq))
        median_freq_GHz = float(np.nanmedian(finite_freq))
        max_freq_GHz = float(np.nanmax(finite_freq))
    else:
        freq_span_GHz = 0.0
        median_freq_GHz = freq_ref_GHz
        max_freq_GHz = freq_ref_GHz

    f_r_candidates = _unique_preserve_order([freq_ref_GHz, median_freq_GHz, freq_ref_GHz + 0.0002])
    fc_max_candidates = _unique_preserve_order(
        [
            max(freq_ref_GHz + 0.05, max_freq_GHz + 0.05),
            freq_ref_GHz + 0.25,
            freq_ref_GHz + 0.50,
            freq_ref_GHz + 1.00,
        ]
    )
    g_base = max(0.005, min(0.08, 0.25 * max(freq_span_GHz, 0.02)))
    g_candidates = _unique_preserve_order([g_base, 0.5 * g_base, 2.0 * g_base, 0.02])

    period_candidates = _unique_preserve_order(
        [
            flux_period,
            2.0 * flux_period,
            4.0 * flux_period,
            initial_flux["flux_span"],
        ]
    )

    guesses = []
    for period in period_candidates:
        if not np.isfinite(period) or period <= 0:
            continue
        for fc_max in fc_max_candidates:
            for g in g_candidates[:2]:
                guesses.append([f_r_candidates[0], fc_max, g, period, flux_offset])
    for f_r in f_r_candidates[1:]:
        guesses.append([f_r, fc_max_candidates[0], g_candidates[0], flux_period, flux_offset])
    guesses.append([f_r_candidates[0], fc_max_candidates[-1], g_candidates[-1], flux_period, flux_offset])

    return np.asarray(guesses, dtype=float), initial_flux


def _estimate_flux_period_and_offset_from_dips(fit_data: pd.DataFrame, flux_dim: str, freq_dim: str) -> dict:
    curve = fit_data.dropna(subset=[flux_dim, freq_dim]).copy()
    if "prominence" in curve:
        idx = curve.groupby(flux_dim)["prominence"].idxmax()
        curve = curve.loc[idx]

    curve = curve.sort_values(flux_dim)
    flux = curve[flux_dim].to_numpy(dtype=float)
    freq = curve[freq_dim].to_numpy(dtype=float)
    finite = np.isfinite(flux) & np.isfinite(freq)
    flux = flux[finite]
    freq = freq[finite]

    flux_span = float(np.nanmax(flux) - np.nanmin(flux)) if flux.size else 0.0
    fallback_period = flux_span if flux_span > 0 else 0.58
    fallback_offset = float(np.nanmean(flux)) if flux.size else 0.0

    if flux.size < 3 or flux_span <= 0:
        return {
            "flux_period": fallback_period,
            "flux_offset": fallback_offset,
            "flux_span": fallback_period,
            "gradient_peak_fluxes": [],
            "gradient_peak_values": [],
        }

    gradient = np.abs(np.gradient(freq, flux))
    selected = _select_top_separated_gradient_indices(flux, gradient)
    peak_fluxes = np.sort(flux[selected])
    peak_values = gradient[selected]

    if peak_fluxes.size >= 2:
        flux_period = float(abs(peak_fluxes[-1] - peak_fluxes[0]))
        flux_offset = float(np.mean([peak_fluxes[0], peak_fluxes[-1]]))
        if not np.isfinite(flux_period) or flux_period <= 0:
            flux_period = fallback_period
    elif peak_fluxes.size == 1:
        flux_period = fallback_period
        flux_offset = float(peak_fluxes[0])
    else:
        flux_period = fallback_period
        flux_offset = fallback_offset

    return {
        "flux_period": flux_period,
        "flux_offset": flux_offset,
        "flux_span": fallback_period,
        "gradient_peak_fluxes": peak_fluxes.tolist(),
        "gradient_peak_values": peak_values.tolist(),
    }


def _select_top_separated_gradient_indices(flux: np.ndarray, gradient: np.ndarray, max_points: int = 2) -> np.ndarray:
    finite = np.isfinite(flux) & np.isfinite(gradient)
    valid_indices = np.flatnonzero(finite)
    if valid_indices.size == 0:
        return np.array([], dtype=int)

    flux_span = float(np.nanmax(flux[finite]) - np.nanmin(flux[finite]))
    min_flux_separation = 0.10 * flux_span if flux_span > 0 else 0.0
    peak_indices, _ = find_peaks(gradient[finite], distance=max(1, valid_indices.size // 8))
    candidate_indices = valid_indices[peak_indices]
    if candidate_indices.size < max_points:
        candidate_indices = valid_indices

    candidate_indices = candidate_indices[np.argsort(gradient[candidate_indices])[::-1]]
    selected = []
    for idx in candidate_indices:
        if all(abs(flux[idx] - flux[prev]) >= min_flux_separation for prev in selected):
            selected.append(idx)
        if len(selected) == max_points:
            break

    if len(selected) < max_points and min_flux_separation > 0:
        for idx in candidate_indices:
            if idx not in selected:
                selected.append(idx)
            if len(selected) == max_points:
                break

    return np.asarray(selected, dtype=int)


def _clip_initial_guesses_to_bounds(
    initial_guesses: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    guesses = np.asarray(initial_guesses, dtype=float)
    guesses = np.atleast_2d(guesses)
    lower_bounds = np.asarray(lower_bounds, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)
    margin = 1e-9 * np.maximum(1.0, np.abs(upper_bounds - lower_bounds))
    return np.clip(guesses, lower_bounds + margin, upper_bounds - margin)


def _hamiltonian_gap_parameter_bounds(
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    min_fc_max_above_fr_GHz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bounds for [f_r, fc_max - f_r, g, flux_period, flux_offset]."""
    lower_bounds = np.asarray(lower_bounds, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)
    min_gap = float(min_fc_max_above_fr_GHz)
    if min_gap < 0:
        raise ValueError("min_fc_max_above_fr_GHz must be non-negative.")

    gap_lower = max(min_gap, lower_bounds[1] - lower_bounds[0])
    gap_upper = upper_bounds[1] - upper_bounds[0]
    if gap_upper <= gap_lower:
        raise ValueError(
            "Cannot enforce fc_max > f_r with the supplied bounds. "
            f"Need upper_bounds[1] - upper_bounds[0] > {gap_lower:g} GHz."
        )

    transformed_lower = lower_bounds.copy()
    transformed_upper = upper_bounds.copy()
    transformed_lower[1] = gap_lower
    transformed_upper[1] = gap_upper
    return transformed_lower, transformed_upper


def _physical_to_gap_parameter_guesses(
    initial_guesses: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    guesses = np.asarray(initial_guesses, dtype=float)
    transformed = guesses.copy()
    transformed[:, 1] = guesses[:, 1] - guesses[:, 0]
    return _clip_initial_guesses_to_bounds(transformed, lower_bounds, upper_bounds)


def _gap_parameter_to_physical_params(params: np.ndarray) -> np.ndarray:
    physical = np.asarray(params, dtype=float).copy()
    physical[1] = physical[0] + physical[1]
    return physical


def _unique_preserve_order(values) -> list:
    unique = []
    for value in values:
        if not np.isfinite(value):
            continue
        if not any(np.isclose(value, existing) for existing in unique):
            unique.append(float(value))
    return unique


@dataclass
class BranchDecoupleResult:
    qubit: str
    detrended: np.ndarray
    freq_GHz: np.ndarray
    offset_mV: float
    offset_V: float
    wider_label: str
    minima_flux_mV: np.ndarray
    d_m1_m2_mV: float
    d_m2_m3_mV: float


@dataclass
class PairDecoupleResult:
    pair_name: str
    qubits: list[str]
    flux_mV: np.ndarray
    branches: dict[str, BranchDecoupleResult]
    selected_qubit: str
    selected_branch: str
    decouple_offset_mV: float
    decouple_offset_V: float


@dataclass
class DecoupleOffsetAnalysis:
    pairs: dict[str, PairDecoupleResult]
    pair_order: list[str]

    def decouple_offsets_V(self) -> dict[str, float]:
        return {name: self.pairs[name].decouple_offset_V for name in self.pair_order}


def detrend_linear_per_flux(image, fit_min_percentile: float = 35) -> np.ndarray:
    """Linear detrend along frequency, independently for each coupler-flux row."""
    image = np.asarray(image, dtype=float)
    freq_axis = np.linspace(-1.0, 1.0, image.shape[1])
    detrended = np.full_like(image, np.nan)

    for row_index, trace in enumerate(image):
        finite = np.isfinite(trace)
        if finite.sum() < 2:
            detrended[row_index] = trace
            continue

        threshold = np.nanpercentile(trace[finite], fit_min_percentile)
        fit_mask = finite & (trace >= threshold)
        if fit_mask.sum() < 2:
            fit_mask = finite

        coeff = np.polyfit(freq_axis[fit_mask], trace[fit_mask], deg=1)
        background = np.polyval(coeff, freq_axis)
        detrended[row_index] = trace - background + np.nanmedian(background)

    return detrended


def initial_decouple_offset_from_detrended(
    flux_mV: np.ndarray,
    detrended: np.ndarray,
    prominence_fraction: float = 0.05,
    distance: int = 5,
) -> dict:
    """Estimate branch offset from the three deepest row-contrast minima."""
    row_span = np.nanmax(detrended, axis=1) - np.nanmin(detrended, axis=1)
    dip_idx, dip_props = find_peaks(
        -row_span,
        prominence=prominence_fraction * np.nanmax(row_span),
        distance=distance,
    )
    if len(dip_idx) < 3:
        raise ValueError(f"find_peaks found only {len(dip_idx)} minima; lower prominence or distance.")

    prominence = dip_props["prominences"]
    top3_local = np.argsort(prominence)[-3:]
    top3_flux_mV = flux_mV[np.sort(dip_idx[top3_local])]

    d_m1_m2, d_m2_m3 = np.diff(top3_flux_mV)
    if d_m1_m2 >= d_m2_m3:
        wider_label = "M1-M2"
        wider_flux_mV = top3_flux_mV[0], top3_flux_mV[1]
    else:
        wider_label = "M2-M3"
        wider_flux_mV = top3_flux_mV[1], top3_flux_mV[2]

    offset_mV = 0.5 * (wider_flux_mV[0] + wider_flux_mV[1])
    return {
        "offset_mV": float(offset_mV),
        "offset_V": float(offset_mV * 1e-3),
        "wider_label": wider_label,
        "minima_flux_mV": top3_flux_mV,
        "d_m1_m2_mV": float(d_m1_m2),
        "d_m2_m3_mV": float(d_m2_m3),
    }


def select_offset_closer_to_zero(branch_offsets_mV: dict[str, float], qubit_names: Sequence[str]) -> tuple[str, str]:
    """Pick the branch whose |offset| is smaller (closer to 0 V)."""
    top_q, bottom_q = qubit_names[0], qubit_names[1]
    if abs(branch_offsets_mV[top_q]) <= abs(branch_offsets_mV[bottom_q]):
        return top_q, "top"
    return bottom_q, "bottom"


def analyze_branch_decouple_offset(
    pair_ds: xr.Dataset,
    qubit: str,
    flux_mV: np.ndarray,
    fit_min_percentile: float = 35,
    prominence_fraction: float = 0.05,
    distance: int = 5,
) -> BranchDecoupleResult:
    trace = pair_ds.IQ_abs.sel(qubit=qubit).transpose("flux", "freq")
    freq_GHz = (
        trace.freq_GHz.values
        if "freq_GHz" in trace.coords
        else pair_ds.freq_GHz.sel(qubit=qubit).values
    )
    detrended = detrend_linear_per_flux(np.asarray(trace.values, dtype=float), fit_min_percentile)
    offset = initial_decouple_offset_from_detrended(
        flux_mV,
        detrended,
        prominence_fraction=prominence_fraction,
        distance=distance,
    )
    return BranchDecoupleResult(
        qubit=qubit,
        detrended=detrended,
        freq_GHz=freq_GHz,
        **offset,
    )


def analyze_pair_decouple_offsets(
    ds: xr.Dataset,
    qp,
    fit_min_percentile: float = 35,
    prominence_fraction: float = 0.05,
    distance: int = 5,
) -> PairDecoupleResult:
    pair_ds = dataset_for_pair_analysis(ds, qp)
    flux_mV = pair_ds.flux.values * 1e3
    qubits = qubit_names_for_pair(qp)

    branches = {
        qubit: analyze_branch_decouple_offset(
            pair_ds,
            qubit,
            flux_mV,
            fit_min_percentile=fit_min_percentile,
            prominence_fraction=prominence_fraction,
            distance=distance,
        )
        for qubit in qubits
    }
    branch_offsets_mV = {qubit: branches[qubit].offset_mV for qubit in qubits}
    selected_qubit, selected_branch = select_offset_closer_to_zero(branch_offsets_mV, qubits)
    selected = branches[selected_qubit]

    return PairDecoupleResult(
        pair_name=qp.name,
        qubits=qubits,
        flux_mV=flux_mV,
        branches=branches,
        selected_qubit=selected_qubit,
        selected_branch=selected_branch,
        decouple_offset_mV=selected.offset_mV,
        decouple_offset_V=selected.offset_V,
    )


def analyze_decouple_offsets(
    ds: xr.Dataset,
    qubit_pairs,
    fit_min_percentile: float = 35,
    prominence_fraction: float = 0.05,
    distance: int = 5,
) -> DecoupleOffsetAnalysis:
    """Analyze all qubit pairs and pick |offset|-closest-to-zero decouple point per pair."""
    pair_order = [qp.name for qp in qubit_pairs]
    pairs = {
        qp.name: analyze_pair_decouple_offsets(
            ds,
            qp,
            fit_min_percentile=fit_min_percentile,
            prominence_fraction=prominence_fraction,
            distance=distance,
        )
        for qp in qubit_pairs
    }
    return DecoupleOffsetAnalysis(pairs=pairs, pair_order=pair_order)


def format_decouple_offset_summary(analysis: DecoupleOffsetAnalysis) -> str:
    lines = ["Best decouple_offset per pair (|offset| closer to 0):"]
    for pair_name in analysis.pair_order:
        pair = analysis.pairs[pair_name]
        top_q, bottom_q = pair.qubits
        top_mV = pair.branches[top_q].offset_mV
        bottom_mV = pair.branches[bottom_q].offset_mV
        lines.append(
            f"  {pair_name}: top {top_q} = {top_mV:.2f} mV | "
            f"bottom {bottom_q} = {bottom_mV:.2f} mV -> "
            f"decouple_offset = {pair.decouple_offset_mV:.2f} mV "
            f"({pair.decouple_offset_V:.6f} V) from {pair.selected_branch} ({pair.selected_qubit})"
        )
    return "\n".join(lines)
