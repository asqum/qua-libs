"""Flux-calibration evidence for 02b, 03b and 06a.

Operator request 2026-10-08. Every number gated on here comes from the saved
dataset or the node's own fit results; the thresholds live in
`rules/policies.yaml` under `analysis.02b`, `analysis.03b` and `analysis.06a`.

02b classifies each qubit from its resonator trace -- the dip frequency at every
flux point -- against a robust cosine fit:

* ``tunable``: the trace follows the cosine and the node's idle offset sits on
  an interior extremum. Its offsets commit and it goes on to 03b/06a.
* ``flat``: the resonator does not move with flux. Resolved, nothing commits,
  and the qubit skips 03b and 06a.
* ``avoided_crossing``: a run of consecutive flux points jumps far off the
  cosine and comes back (a coupler or neighbour crossing the resonator).
  Resolved, nothing commits, the qubit skips 03b and 06a, and the operator is
  asked to tune the neighbouring couplers' flux offsets.
* ``failed``: no visible dip or an unusable fit; the node retries it.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

FLUX_NODES = frozenset({"02b", "03b", "06a"})

TUNABLE = "tunable"
FLAT = "flat"
AVOIDED_CROSSING = "avoided_crossing"
FAILED = "failed"
RESOLVED_02B_CLASSES = frozenset({TUNABLE, FLAT, AVOIDED_CROSSING})


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _scalar(value: Any) -> float | None:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if _finite(value):
        return float(value)
    return None


def _robust_sigma(values: Any) -> float:
    import numpy as np

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def _r_squared(values: Any, fitted: Any) -> float:
    import numpy as np

    values = np.asarray(values, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    total = float(np.sum((values - values.mean()) ** 2))
    residual = float(np.sum((values - fitted) ** 2))
    if total <= 0:
        return 1.0 if residual <= 0 else 0.0
    return 1.0 - residual / total


def _qubit_trace(dataset: Any, variable: str, name: str, outer: str, inner: str) -> Any:
    signal = dataset[variable]
    if "qubit" in signal.dims:
        signal = signal.sel(qubit=name)
    return signal.transpose(outer, inner)


def cosine_fit(x: Any, y: Any, mask: Any | None = None) -> dict[str, Any]:
    """Least-squares a*cos(2 pi f x) + b*sin(2 pi f x) + c over a grid of f.

    The grid reaches periods four times longer than the sweep, so a window
    showing only part of a period is still fitted (it looks quadratic).
    """

    import numpy as np

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    use = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        use &= np.asarray(mask, dtype=bool)
    if use.sum() < 5:
        return {"error": "too few points"}
    span = float(x.max() - x.min()) or 1.0
    highest = max(1.0, use.sum() / 4.0) / span
    best: dict[str, Any] | None = None
    for frequency in np.linspace(0.25 / span, highest, 600):
        phase = 2 * np.pi * frequency * x[use]
        design = np.column_stack([np.cos(phase), np.sin(phase), np.ones(phase.size)])
        coefficients, *_ = np.linalg.lstsq(design, y[use], rcond=None)
        error = float(np.sum((y[use] - design @ coefficients) ** 2))
        if best is None or error < best["sse"]:
            best = {"sse": error, "frequency": float(frequency), "coefficients": coefficients}
    assert best is not None
    a, b, c = (float(value) for value in best["coefficients"])
    frequency = best["frequency"]

    def model(values: Any) -> Any:
        phase = 2 * np.pi * frequency * np.asarray(values, dtype=float)
        return a * np.cos(phase) + b * np.sin(phase) + c

    return {
        "frequency": frequency,
        "amplitude": math.hypot(a, b),
        "phase": math.atan2(-b, a),
        "offset": c,
        "model": model,
        "r_squared": _r_squared(y[use], model(x[use])),
    }


def _robust_cosine_fit(x: Any, y: Any, step: float) -> tuple[dict[str, Any], Any]:
    """Fit, drop points far off the fit, refit; return the fit and inliers."""

    import numpy as np

    mask = np.isfinite(y)
    fit = cosine_fit(x, y, mask)
    for _ in range(3):
        if fit.get("error"):
            break
        residual = y - fit["model"](x)
        sigma = _robust_sigma(residual[mask])
        threshold = max(4.0 * sigma, 2.0 * step)
        new_mask = np.isfinite(y) & (np.abs(residual) <= threshold)
        if new_mask.sum() < 5 or np.array_equal(new_mask, mask):
            break
        mask = new_mask
        fit = cosine_fit(x, y, mask)
    return fit, mask


def _outlier_segments(flags: Any) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start = None
    for index, flagged in enumerate(list(flags) + [False]):
        if flagged and start is None:
            start = index
        elif not flagged and start is not None:
            segments.append((start, index - 1))
            start = None
    return segments


def classify_02b_trace(
    flux: Any,
    frequency: Any,
    magnitude: Any,
    rules: dict[str, Any],
    node_fit: dict[str, Any] | None,
) -> dict[str, Any]:
    """Classify one qubit's resonator-vs-flux map.

    `magnitude` is |IQ| with shape (flux, frequency); the resonance is the
    minimum of each row.
    """

    import numpy as np

    flux = np.asarray(flux, dtype=float)
    frequency = np.asarray(frequency, dtype=float)
    magnitude = np.asarray(magnitude, dtype=float)
    step = float(np.median(np.abs(np.diff(frequency)))) if frequency.size > 1 else 1.0
    span_v = float(flux.max() - flux.min()) if flux.size else 0.0
    result: dict[str, Any] = {"frequency_step_hz": step}

    peak = frequency[np.nanargmin(magnitude, axis=1)]
    depths = []
    for row in magnitude:
        # Point-to-point noise: the spread of the row itself is dominated by
        # the dip and the background slope, not by noise (02b, 2026-10-08).
        noise = _robust_sigma(np.diff(row)) / np.sqrt(2.0)
        depths.append(
            (float(np.nanmedian(row)) - float(np.nanmin(row))) / noise
            if noise > 0
            else float("inf")
        )
    dip_snr = float(np.median(depths))
    result["dip_snr"] = dip_snr
    result["trace"] = {"flux": flux, "peak": peak}
    if not dip_snr >= float(rules.get("min_dip_snr", 4.0)):
        result["classification"] = FAILED
        result["reason"] = (
            f"resonator dip is not visible (median dip SNR {dip_snr:.3g} < "
            f"{float(rules.get('min_dip_snr', 4.0)):g})"
        )
        return result

    fit, inliers = _robust_cosine_fit(flux, peak, step)
    if fit.get("error"):
        result["classification"] = FAILED
        result["reason"] = f"cosine fit failed ({fit['error']})"
        return result
    fitted = fit["model"](flux)
    swing = 2.0 * float(fit["amplitude"])
    inlier_r2 = _r_squared(peak[inliers], fitted[inliers])
    trace_span = float(np.percentile(peak, 95) - np.percentile(peak, 5))
    result.update(
        {
            "swing_hz": swing,
            "trace_span_hz": trace_span,
            "period_v": 1.0 / float(fit["frequency"]),
            "r_squared": inlier_r2,
            "inlier_fraction": float(np.mean(inliers)),
        }
    )
    result["trace"]["fit"] = fitted

    residual = peak - fitted
    sigma = _robust_sigma(residual[inliers])
    threshold = max(
        float(rules.get("crossing_outlier_sigma", 5.0)) * sigma,
        float(rules.get("crossing_min_steps", 4.0)) * step,
        float(rules.get("crossing_min_fraction_of_swing", 0.25)) * swing,
    )
    minimum_points = int(rules.get("crossing_min_points", 2))
    crossings = []
    edge_outliers = []
    for first, last in _outlier_segments(np.abs(residual) > threshold):
        length = last - first + 1
        if length < minimum_points or length > 0.3 * flux.size:
            continue
        window = residual[first:last + 1]
        segment = {
            "flux_from_v": float(flux[first]),
            "flux_to_v": float(flux[last]),
            "max_excursion_hz": float(window[np.argmax(np.abs(window))]),
            "points": int(length),
        }
        # Operator rule 2026-10-09: a crossing must leave the cosine and come
        # back inside the window. A run touching the first or last flux point
        # is a settling transient at the sweep edge (q8-q10, run #334).
        if first == 0 or last == flux.size - 1:
            edge_outliers.append(segment)
            continue
        crossings.append(segment)
    result["crossings"] = crossings
    if edge_outliers:
        result["edge_outliers"] = edge_outliers
    if crossings:
        result["classification"] = AVOIDED_CROSSING
        where = ", ".join(
            f"{item['flux_from_v']:.3g}..{item['flux_to_v']:.3g} V" for item in crossings
        )
        result["reason"] = f"avoided crossing near {where}"
        return result

    if (
        swing < float(rules.get("flat_max_swing_steps", 3.0)) * step
        and trace_span < float(rules.get("flat_max_span_steps", 4.0)) * step
    ):
        result["classification"] = FLAT
        result["reason"] = (
            f"no flux dependence (cosine swing {swing / 1e3:.3g} kHz over the "
            f"{span_v:g} V sweep)"
        )
        return result

    problems = []
    if inlier_r2 < float(rules.get("min_fit_r_squared", 0.8)):
        problems.append(
            f"cosine fit R^2 {inlier_r2:.3g} is below "
            f"{float(rules.get('min_fit_r_squared', 0.8)):g}"
        )
    offset = _scalar((node_fit or {}).get("offset"))
    period = _scalar((node_fit or {}).get("dv_phi0"))
    if offset is None:
        problems.append("the node reported no idle offset")
    else:
        edge = min(offset - flux.min(), flux.max() - offset) / span_v if span_v else 0
        result["idle_offset_v"] = offset
        result["edge_fraction"] = float(edge)
        if edge < float(rules.get("min_edge_fraction", 0.05)):
            problems.append(
                f"idle offset {offset:.4g} V lies within "
                f"{float(rules.get('min_edge_fraction', 0.05)):.0%} of a sweep edge"
            )
        else:
            # The idle offset must sit on an extremum of the fitted cosine.
            # The model is A cos(2 pi f x + phase); its slope is ~ sin(...).
            slope = abs(
                math.sin(2 * math.pi * float(fit["frequency"]) * offset + float(fit["phase"]))
            )
            result["idle_offset_slope"] = slope
            if slope > 0.3:
                problems.append(
                    f"idle offset {offset:.4g} V is not at an extremum of the "
                    "resonator trace"
                )
    if period is None or period <= 0:
        problems.append("the node reported no positive flux period")
    if problems:
        result["classification"] = FAILED
        result["reason"] = "; ".join(problems)
    else:
        result["classification"] = TUNABLE
    return result


def analyze_03b_qubit(
    flux: Any,
    frequency: Any,
    signal: Any,
    rules: dict[str, Any],
    node_fit: dict[str, Any] | None,
) -> dict[str, Any]:
    """Check a qubit-spectroscopy-vs-flux map against the node's parabola."""

    import numpy as np

    flux = np.asarray(flux, dtype=float)
    frequency = np.asarray(frequency, dtype=float)
    signal = np.asarray(signal, dtype=float)
    span_v = float(flux.max() - flux.min()) if flux.size else 0.0
    result: dict[str, Any] = {}
    # 2026-10-09: a single-sample argmax over a wide (300 MHz) window often
    # lands on noise (run #383, q3/q5 R^2 ~0.5 for clean parabolas). Smooth
    # along frequency over about one linewidth before locating the line.
    width = max(1, int(rules.get("peak_smoothing_points", 5)))
    kernel = np.ones(width) / width
    peaks, visible = [], []
    for row in signal:
        centred = row - np.nanmedian(row)
        smoothed = np.convolve(np.nan_to_num(centred), kernel, mode="same")
        deviation = np.abs(smoothed)
        noise = _robust_sigma(smoothed)
        index = int(np.nanargmax(deviation))
        snr = float(deviation[index] / noise) if noise > 0 else float("inf")
        peaks.append(float(frequency[index]))
        visible.append(snr >= float(rules.get("min_peak_snr", 4.0)))
    peaks_array = np.asarray(peaks)
    visible_array = np.asarray(visible, dtype=bool)
    result["visible_fraction"] = float(visible_array.mean()) if visible_array.size else 0.0
    problems = []
    if result["visible_fraction"] < float(rules.get("min_visible_fraction", 0.5)):
        problems.append(
            f"qubit line visible at only {result['visible_fraction']:.0%} of flux points"
        )
    elif visible_array.sum() >= 4:
        # Points that jumped to a neighbour's line or a two-photon feature are
        # not the parabola; refit without them, keeping at least 70%.
        inliers = visible_array.copy()
        step = float(np.median(np.abs(np.diff(frequency)))) if frequency.size > 1 else 0.0
        for _ in range(3):
            coefficients = np.polyfit(flux[inliers], peaks_array[inliers], 2)
            residual = peaks_array - np.polyval(coefficients, flux)
            limit = max(3 * _robust_sigma(residual[inliers]), 3 * step)
            candidate = visible_array & (np.abs(residual) <= limit)
            if candidate.sum() < max(4, 0.7 * visible_array.sum()):
                break
            inliers = candidate
        coefficients = np.polyfit(flux[inliers], peaks_array[inliers], 2)
        fitted = np.polyval(coefficients, flux[inliers])
        r2 = _r_squared(peaks_array[inliers], fitted)
        result["r_squared"] = r2
        result["inlier_fraction"] = float(inliers.sum() / visible_array.sum())
        result["parabola_curvature_hz_per_v2"] = float(coefficients[0])
        if coefficients[0] != 0:
            result["vertex_v"] = float(-coefficients[1] / (2 * coefficients[0]))
        if r2 < float(rules.get("min_fit_r_squared", 0.7)):
            problems.append(
                f"parabola R^2 {r2:.3g} is below {float(rules.get('min_fit_r_squared', 0.7)):g}"
            )
    shift = _scalar((node_fit or {}).get("flux_shift"))
    quad = _scalar((node_fit or {}).get("quad_term"))
    drive = _scalar((node_fit or {}).get("drive_freq"))
    result.update({"flux_shift_v": shift, "quad_term": quad, "drive_freq_hz": drive})
    if shift is None or quad is None or drive is None:
        problems.append("the node fit produced no finite sweet spot")
    else:
        if quad >= 0:
            problems.append(
                f"quadratic term {quad:.3g} Hz/V^2 is not negative (no sweet-spot maximum)"
            )
        edge = min(shift - flux.min(), flux.max() - shift) / span_v if span_v else 0
        result["edge_fraction"] = float(edge)
        if edge < float(rules.get("min_edge_fraction", 0.05)):
            problems.append(
                f"sweet spot {shift:.4g} V lies within "
                f"{float(rules.get('min_edge_fraction', 0.05)):.0%} of a sweep edge"
            )
        half = (float(frequency.max()) - float(frequency.min())) / 2
        centre = (float(frequency.max()) + float(frequency.min())) / 2
        if abs(drive - centre) > half:
            problems.append("sweet-spot frequency lies outside the frequency window")
        # 2026-10-09: the node's peak finder can fit a neighbour's faint line
        # (q8 in run #381: node vertex -0.032 V, the tracked line's 1.43 V).
        vertex = result.get("vertex_v")
        tolerance = float(rules.get("max_vertex_disagreement_fraction", 0.1)) * span_v
        if vertex is not None and abs(float(vertex) - shift) > tolerance:
            problems.append(
                f"node sweet spot {shift:.4g} V disagrees with the tracked line's "
                f"vertex {float(vertex):.4g} V"
            )
    result["fit_successful"] = not problems
    if problems:
        result["reason"] = "; ".join(problems)
    return result


def _fringe_frequency(times_us: Any, values: Any) -> tuple[float | None, float]:
    """Dominant oscillation frequency (MHz) of one Ramsey trace, and its SNR."""

    import numpy as np

    times = np.asarray(times_us, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.size < 8 or not np.all(np.isfinite(values)):
        return None, 0.0
    dt = float(np.median(np.diff(times)))
    centred = values - values.mean()
    padded = 16 * times.size
    spectrum = np.abs(np.fft.rfft(centred, n=padded))
    frequencies = np.fft.rfftfreq(padded, d=dt)
    spectrum[0] = 0
    index = int(np.argmax(spectrum))
    floor = float(np.median(spectrum[1:])) or 1e-12
    return float(frequencies[index]), float(spectrum[index] / floor)


def analyze_06a_qubit(
    flux: Any,
    times_us: Any,
    state: Any,
    rules: dict[str, Any],
    node_fit: dict[str, Any] | None,
    flux_span: float | None,
) -> dict[str, Any]:
    """Check the Ramsey-vs-flux parabola against the node's flux offset."""

    import numpy as np

    flux = np.asarray(flux, dtype=float)
    result: dict[str, Any] = {}
    fringes, valid = [], []
    for row in np.asarray(state, dtype=float):
        value, strength = _fringe_frequency(times_us, row)
        fringes.append(value if value is not None else np.nan)
        valid.append(value is not None and value > 0 and strength >= 3.0)
    fringe_array = np.asarray(fringes, dtype=float)
    valid_array = np.asarray(valid, dtype=bool)
    result["valid_fraction"] = float(valid_array.mean()) if valid_array.size else 0.0
    problems = []
    if result["valid_fraction"] < float(rules.get("min_valid_flux_fraction", 0.5)):
        problems.append(
            f"Ramsey fringes resolved at only {result['valid_fraction']:.0%} of flux points"
        )
    elif valid_array.sum() >= 4:
        coefficients = np.polyfit(flux[valid_array], fringe_array[valid_array], 2)
        r2 = _r_squared(
            fringe_array[valid_array], np.polyval(coefficients, flux[valid_array])
        )
        result["r_squared"] = r2
        if r2 < float(rules.get("min_fit_r_squared", 0.5)):
            problems.append(
                f"fringe-frequency parabola R^2 {r2:.3g} is below "
                f"{float(rules.get('min_fit_r_squared', 0.5)):g}"
            )
    offset = _scalar((node_fit or {}).get("flux_offset"))
    frequency_offset = _scalar((node_fit or {}).get("freq_offset"))
    result.update({"flux_offset_v": offset, "freq_offset_hz": frequency_offset})
    if offset is None or frequency_offset is None:
        problems.append("the node fit produced no finite flux offset")
    elif flux_span:
        limit = float(rules.get("max_offset_fraction_of_span", 0.4)) * float(flux_span)
        if abs(offset) > limit:
            problems.append(
                f"flux offset {offset * 1e3:.3g} mV is beyond {limit * 1e3:.3g} mV "
                "(too close to the edge of the flux window)"
            )
    result["fit_successful"] = not problems
    if problems:
        result["reason"] = "; ".join(problems)
    return result


def flux_dataset_metrics(
    node_id: str,
    dataset: Any,
    data: dict[str, Any] | None,
    rules: dict[str, Any],
) -> dict[str, Any]:
    """Per-qubit flux evidence; plotting traces are kept under `_traces`."""

    import numpy as np

    data = data if isinstance(data, dict) else {}
    fit_results = data.get("fit_results") if isinstance(data.get("fit_results"), dict) else {}
    parameters = data.get("initial_parameters") or {}
    names = (
        [str(name) for name in dataset.coords["qubit"].values]
        if "qubit" in dataset.coords
        else []
    )
    qubits: dict[str, Any] = {}
    traces: dict[str, Any] = {}
    for name in names:
        node_fit = fit_results.get(name) if isinstance(fit_results.get(name), dict) else {}
        try:
            if node_id == "02b":
                variable = "IQ_abs" if "IQ_abs" in dataset.data_vars else None
                if variable is None:
                    raise ValueError("dataset has no IQ_abs")
                trace = _qubit_trace(dataset, variable, name, "flux", "freq")
                metrics = classify_02b_trace(
                    dataset.coords["flux"].values,
                    dataset.coords["freq"].values,
                    trace.values,
                    rules,
                    node_fit,
                )
                trace_data = metrics.pop("trace", {})
                traces[name] = {
                    **trace_data,
                    "frequency": np.asarray(dataset.coords["freq"].values, dtype=float),
                    "magnitude": np.asarray(trace.values, dtype=float),
                }
            elif node_id == "03b":
                variable = "I" if "I" in dataset.data_vars else "IQ_abs"
                trace = _qubit_trace(dataset, variable, name, "flux", "freq")
                metrics = analyze_03b_qubit(
                    dataset.coords["flux"].values,
                    dataset.coords["freq"].values,
                    trace.values,
                    rules,
                    node_fit,
                )
            else:
                trace = _qubit_trace(dataset, "state", name, "flux", "idle_time")
                metrics = analyze_06a_qubit(
                    dataset.coords["flux"].values,
                    dataset.coords["idle_time"].values,
                    trace.values,
                    rules,
                    node_fit,
                    _scalar(parameters.get("flux_span")),
                )
        except Exception as exc:  # pragma: no cover - reported per qubit
            metrics = {"error": f"{name} flux dataset could not be analysed: {exc}"}
        qubits[name] = metrics
    return {"qubits": qubits, "_traces": traces}


def flux_fit_quality(
    node_id: str,
    metrics: dict[str, Any],
    targets: list[str],
) -> tuple[dict[str, Any], list[str]]:
    """Per-qubit results and failure reasons for a flux node."""

    results: dict[str, Any] = {}
    failures: list[str] = []
    by_qubit = metrics.get("qubits", {}) if isinstance(metrics, dict) else {}
    for name in targets:
        item = by_qubit.get(name)
        if not isinstance(item, dict):
            failures.append(f"{name} is missing from the {node_id} dataset.")
            continue
        if item.get("error"):
            failures.append(str(item["error"]))
            continue
        if node_id == "02b":
            classification = item.get("classification")
            results[name] = {
                "classification": classification,
                "resolved": classification in RESOLVED_02B_CLASSES,
                "fit_successful": classification == TUNABLE,
                "flux_tunable": classification == TUNABLE,
                "idle_offset_v": item.get("idle_offset_v"),
                "period_v": item.get("period_v"),
            }
            if classification == FAILED:
                failures.append(f"{name} 02b {item.get('reason', 'failed')}.")
        else:
            results[name] = {
                key: item.get(key)
                for key in ("fit_successful", "flux_shift_v", "flux_offset_v", "quad_term", "r_squared")
                if key in item
            }
            if not item.get("fit_successful"):
                failures.append(f"{name} {node_id} {item.get('reason', 'fit failed')}.")
    return results, failures


def avoided_crossing_notices(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Operator notices for qubits whose 02b trace shows an avoided crossing."""

    notices = []
    for name, item in sorted((metrics.get("qubits") or {}).items()):
        if not isinstance(item, dict) or item.get("classification") != AVOIDED_CROSSING:
            continue
        where = ", ".join(
            f"{crossing['flux_from_v']:.3g}..{crossing['flux_to_v']:.3g} V"
            for crossing in item.get("crossings", [])
        )
        notices.append(
            {
                "kind": "avoided_crossing",
                "qubit": name,
                "crossings": item.get("crossings", []),
                "message": {
                    "zh-Hant": (
                        f"{name} 的 02b 共振腔 vs flux 在 {where} 出現 avoided crossing"
                        "（頻率突然變很高或很低再從另一側回來）。這通常是相鄰 coupler"
                        "與其交叉造成：請調整相鄰 coupler 的 flux offset 後，再於新的"
                        "工作流程中重做 flux 校正。此 qubit 本次不更新 flux，並略過 03b／06a。"
                    ),
                    "en": (
                        f"{name}'s 02b resonator-vs-flux trace shows an avoided crossing "
                        f"near {where} (the frequency swings far up or down and returns "
                        "from the other side). A neighbouring coupler is usually crossing "
                        "it: tune the neighbouring couplers' flux offsets, then repeat "
                        "flux calibration in a new workflow. This qubit's flux is not "
                        "updated now and it skips 03b/06a."
                    ),
                },
            }
        )
    return notices


def render_02b_plots(
    output_root: Path,
    snapshot_name: str,
    metrics: dict[str, Any],
) -> list[str]:
    """Annotated resonator-vs-flux plot for every avoided-crossing qubit."""

    traces = metrics.get("_traces") or {}
    flagged = [
        name
        for name, item in (metrics.get("qubits") or {}).items()
        if isinstance(item, dict) and item.get("classification") == AVOIDED_CROSSING
        and name in traces
    ]
    if not flagged:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        import numpy as np
    except Exception:  # pragma: no cover - plotting is advisory
        return []
    import re

    output_root.mkdir(parents=True, exist_ok=True)
    rendered = []
    for name in flagged:
        trace = traces[name]
        flux = np.asarray(trace["flux"], dtype=float)
        frequency = np.asarray(trace["frequency"], dtype=float) / 1e6
        fig, axis = plt.subplots(figsize=(6.6, 4.4))
        axis.pcolormesh(
            flux, frequency, np.asarray(trace["magnitude"], dtype=float).T,
            shading="auto", cmap="viridis",
        )
        axis.plot(flux, np.asarray(trace["peak"]) / 1e6, "w.", ms=3, label="resonance")
        if "fit" in trace:
            axis.plot(flux, np.asarray(trace["fit"]) / 1e6, "r--", lw=1.2, label="cosine fit")
        for crossing in metrics["qubits"][name].get("crossings", []):
            axis.axvspan(
                crossing["flux_from_v"], crossing["flux_to_v"],
                color="orange", alpha=0.35, label="avoided crossing",
            )
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), loc="upper right", fontsize=7)
        axis.set_xlabel("Flux (V)")
        axis.set_ylabel("Detuning (MHz)")
        axis.set_title(f"02b {name}: avoided crossing - tune neighbouring couplers")
        fig.tight_layout()
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", snapshot_name)
        output = output_root / f"{safe}-02b-{name}-avoided-crossing.png"
        fig.savefig(output, dpi=140)
        plt.close(fig)
        rendered.append(str(output.resolve()))
    return rendered
