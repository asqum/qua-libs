"""JY's own readout-point selection for 07d.

Operator instruction 2026-10-02: 07d must judge, at every swept point, how
close the ground and excited IQ shots are to two clean blobs, and pick the
readout point from that, not from fidelity alone. 07b then rejected every
qubit for long IQ tails although 07d had reported 97-98% fidelity.

Two blobs means every shot lies on one of two round Gaussian clouds. A ground
shot that sits on the excited blob (thermal population, decay during readout)
is still on a blob and is counted by fidelity, not here. What this measure
counts is a shot on *neither* blob: smear between them, a third cloud or a
long streak, which is what a too-strong readout produces.

Operator decision 2026-10-20: the point is chosen by JY's own score from the
raw shots, not by the Qualibrate node's fidelity. The score weighs whether the
two blobs separate above shots on neither blob, ``separation_weight`` :
``outlier_weight`` (3:1, 5:1 from 2026-10-21): ``score = 5 x blob-overlap error + 1 x off-blob
share``, lowest wins. The blob-overlap error is the 1D Gaussian overlap of two
blobs ``d`` sigma apart, ``0.5 erfc(d / (2 sqrt 2))``. Fidelity is only
reported.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.special import erfc

# Median of a 2D isotropic Gaussian's radius, in units of sigma: sqrt(2 ln 2).
_RAYLEIGH_MEDIAN = math.sqrt(2.0 * math.log(2.0))


def two_blob_metrics(
    dataset: Any,
    rules: dict[str, Any],
    max_factors: dict[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Select each qubit's readout point from fidelity and two-blob shape.

    Operator choice 2026-10-03: points above ``max_factors[qubit]`` (the 02c
    dressed-limit power) are never chosen. The best-fidelity point is replaced
    by the best two-blob point only when that costs at most
    ``fidelity_tolerance_percent``; otherwise the fidelity optimum is kept and
    07b's two-blob test judges it.
    """

    sigma_cut = float(rules.get("outlier_sigma", 3.0))
    max_outliers = float(rules.get("max_outlier_fraction", 0.03))
    radius = int(rules.get("neighborhood_points", 1))
    tolerance = float(rules.get("fidelity_tolerance_percent", 2.0))
    separation_weight = rules.get("separation_weight")
    if separation_weight is not None:
        separation_weight = float(separation_weight)
    outlier_weight = float(rules.get("outlier_weight", 1.0))
    expected = math.exp(-0.5 * sigma_cut**2)
    dims = ("qubit", "run", "freq", "amp", "duration")
    arrays = {
        name: np.asarray(dataset[name].transpose(*dims).values, dtype=float)
        for name in ("I_g", "Q_g", "I_e", "Q_e")
    }
    fidelity_all = np.asarray(
        dataset["fidelity"].transpose("qubit", "freq", "amp", "duration").values,
        dtype=float,
    )
    freqs = np.asarray(dataset["freq"].values, dtype=float)
    amps = np.asarray(dataset["amp"].values, dtype=float)
    durations = np.asarray(dataset["duration"].values, dtype=float)
    results: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(dataset["qubit"].values):
        g = arrays["I_g"][index] + 1j * arrays["Q_g"][index]
        e = arrays["I_e"][index] + 1j * arrays["Q_e"][index]
        outliers, separation = _point_outlier_fraction(g, e, sigma_cut)
        pooled = _neighborhood_mean(outliers, radius)
        fidelity = fidelity_all[index]
        cap = (max_factors or {}).get(str(name))
        allowed = np.ones(fidelity.shape, dtype=bool)
        if cap is not None and math.isfinite(cap):
            allowed &= (amps <= cap)[None, :, None]
        entry: dict[str, Any] = {
            "outlier_sigma": sigma_cut,
            "gaussian_outlier_fraction": expected,
            "max_outlier_fraction": max_outliers,
            "fidelity_tolerance_percent": tolerance,
            "max_amplitude_factor": cap,
            "total_points": int(fidelity.size),
        }
        if not allowed.any():
            entry["selected"] = None
            entry["reason"] = "no swept point lies at or below the dressed-limit power"
            results[str(name)] = entry
            continue
        unconstrained = np.unravel_index(
            int(np.argmax(np.where(allowed, fidelity, -np.inf))), fidelity.shape
        )
        entry["unconstrained"] = _describe(
            unconstrained, freqs, amps, durations, fidelity, pooled, separation
        )
        acceptable = allowed & np.isfinite(pooled) & (pooled <= max_outliers)
        entry["acceptable_points"] = int(acceptable.sum())
        chosen = unconstrained
        entry["two_blob"] = False
        if separation_weight is not None:
            overlap = _neighborhood_mean(blob_overlap_error(separation), radius)
            score = separation_weight * overlap + outlier_weight * pooled
            pool = acceptable & np.isfinite(score)
            entry["selection"] = "separation_weighted"
            entry["separation_weight"] = separation_weight
            entry["outlier_weight"] = outlier_weight
            if not pool.any():
                # Nothing within the off-blob limit: the best score still
                # goes to 07b, which judges the shape.
                pool = allowed & np.isfinite(score)
                entry["least_outlier_fraction"] = float(
                    np.nanmin(np.where(allowed, pooled, np.nan))
                )
            else:
                entry["two_blob"] = True
            if pool.any():
                chosen = np.unravel_index(
                    int(np.argmin(np.where(pool, score, np.inf))), score.shape
                )
            entry["selected"] = _describe(
                chosen, freqs, amps, durations, fidelity, pooled, separation
            )
            entry["selected"]["blob_overlap_error"] = float(overlap[chosen])
            entry["selected"]["score"] = float(score[chosen])
            top = (
                amps.size - 1
                if cap is None
                else int(np.max(np.nonzero(amps <= cap)[0]))
            )
            entry["selected"]["at_amplitude_top"] = bool(chosen[1] == top)
            results[str(name)] = entry
            continue
        if acceptable.any():
            blob = np.unravel_index(
                int(np.argmax(np.where(acceptable, fidelity, -np.inf))), fidelity.shape
            )
            cost = float(fidelity[unconstrained] - fidelity[blob])
            entry["two_blob_fidelity_cost_percent"] = cost
            if cost <= tolerance:
                chosen = blob
                entry["two_blob"] = True
        else:
            entry["least_outlier_fraction"] = float(
                np.nanmin(np.where(allowed, pooled, np.nan))
            )
        entry["selected"] = _describe(
            chosen, freqs, amps, durations, fidelity, pooled, separation
        )
        top = (
            amps.size - 1
            if cap is None
            else int(np.max(np.nonzero(amps <= cap)[0]))
        )
        entry["selected"]["at_amplitude_top"] = bool(chosen[1] == top)
        results[str(name)] = entry
    return results


def blob_overlap_error(separation_sigma: np.ndarray) -> np.ndarray:
    """Share of shots misassigned between two Gaussian blobs d sigma apart."""

    return 0.5 * erfc(np.asarray(separation_sigma, dtype=float) / (2.0 * math.sqrt(2.0)))


def _point_outlier_fraction(
    g: np.ndarray, e: np.ndarray, sigma_cut: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per point: share of shots on neither blob, and blob separation in sigma."""

    centre_g = np.median(g.real, axis=0) + 1j * np.median(g.imag, axis=0)
    centre_e = np.median(e.real, axis=0) + 1j * np.median(e.imag, axis=0)
    own = np.concatenate([np.abs(g - centre_g), np.abs(e - centre_e)], axis=0)
    sigma = np.median(own, axis=0) / _RAYLEIGH_MEDIAN
    sigma = np.where(sigma > 0, sigma, np.nan)
    shots = np.concatenate([g, e], axis=0)
    nearest = np.minimum(np.abs(shots - centre_g), np.abs(shots - centre_e))
    outliers = np.mean(nearest / sigma > sigma_cut, axis=0)
    separation = np.abs(centre_e - centre_g) / sigma
    return outliers, separation


def cloud_outlier_fractions(
    g: np.ndarray, e: np.ndarray, sigma_cut: float
) -> dict[str, float]:
    """For one set of g and e shots: the share of each cloud on neither blob.

    Also reports how many ground shots sit on the excited blob and vice versa.
    Those are readout errors (thermal population, decay), which fidelity
    already counts, and they do not make a cloud fail the two-blob test.
    """

    centre_g = complex(np.median(g.real), np.median(g.imag))
    centre_e = complex(np.median(e.real), np.median(e.imag))
    own = np.concatenate([np.abs(g - centre_g), np.abs(e - centre_e)])
    sigma = float(np.median(own)) / _RAYLEIGH_MEDIAN
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("zero or non-finite cloud width")

    def neither(shots: np.ndarray) -> float:
        nearest = np.minimum(np.abs(shots - centre_g), np.abs(shots - centre_e))
        return float(np.mean(nearest / sigma > sigma_cut))

    return {
        "separation_sigma": abs(centre_e - centre_g) / sigma,
        "g_on_neither_blob": neither(g),
        "e_on_neither_blob": neither(e),
        "g_on_e_blob": float(np.mean(np.abs(g - centre_e) < np.abs(g - centre_g))),
        "e_on_g_blob": float(np.mean(np.abs(e - centre_g) < np.abs(e - centre_e))),
    }


def _neighborhood_mean(values: np.ndarray, radius: int) -> np.ndarray:
    """Average over +-radius in frequency and amplitude, same duration."""

    if radius <= 0:
        return values
    total = np.zeros_like(values)
    count = np.zeros_like(values)
    n_freq, n_amp = values.shape[0], values.shape[1]
    for df in range(-radius, radius + 1):
        for da in range(-radius, radius + 1):
            f0, f1 = max(0, -df), min(n_freq, n_freq - df)
            a0, a1 = max(0, -da), min(n_amp, n_amp - da)
            block = values[f0 + df : f1 + df, a0 + da : a1 + da]
            finite = np.isfinite(block)
            total[f0:f1, a0:a1] += np.where(finite, block, 0.0)
            count[f0:f1, a0:a1] += finite
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def _describe(
    index: tuple[int, ...],
    freqs: np.ndarray,
    amps: np.ndarray,
    durations: np.ndarray,
    fidelity: np.ndarray,
    pooled: np.ndarray,
    separation: np.ndarray,
) -> dict[str, Any]:
    f, a, d = (int(value) for value in index)
    return {
        "frequency_offset_hz": float(freqs[f]),
        "amplitude_factor": float(amps[a]),
        "duration_ns": int(round(float(durations[d]))),
        "fidelity_percent": float(fidelity[f, a, d]),
        "outlier_fraction": float(pooled[f, a, d]),
        "separation_sigma": float(separation[f, a, d]),
    }
