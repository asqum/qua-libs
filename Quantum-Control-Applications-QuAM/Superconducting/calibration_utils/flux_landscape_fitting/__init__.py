"""Flux landscape fitting utilities (Savitzky–Golay + sliding-window FFT)."""

from .analysis import (
    ANALYSIS_PRESETS,
    FluxLandscapeFit,
    fit_coupler_zeropoint_pair,
    fit_coupler_zeropoint_to_legacy_results,
    get_analysis_fit_config,
)
from .plotting import plot_contrast_cut_debug, plot_coupler_zeropoint_maps

__all__ = [
    "ANALYSIS_PRESETS",
    "FluxLandscapeFit",
    "fit_coupler_zeropoint_pair",
    "fit_coupler_zeropoint_to_legacy_results",
    "get_analysis_fit_config",
    "plot_contrast_cut_debug",
    "plot_coupler_zeropoint_maps",
]
