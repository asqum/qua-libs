from __future__ import annotations

import math
from typing import Any

from .analysis_02c_absence import assess_bare_only_absence


DEFAULT_02C_RULES: dict[str, Any] = {
    "plateau_tail_fraction": 0.20,
    "min_plateau_points": 4,
    "plateau_min_coverage": 0.55,
    "min_frequency_separation_steps": 4.0,
    "plateau_band_steps": 4.0,
    "plateau_band_fraction_of_separation": 0.25,
    "max_plateau_width_steps": 8.0,
    "max_plateau_width_fraction_of_separation": 0.55,
    "min_depletion_points": 2,
    "min_transition_monotonic_fraction": 0.70,
    "transition_monotonic_slack_fraction": 0.10,
    "min_classified_plateau_points": 2,
    "require_plateau_stability": False,
    "require_plateau_width": False,
    "require_depletion_points": False,
    "require_transition_monotonic": False,
    "frequency_edge_fraction": 0.05,
    "selected_power_tolerance_steps": 2.0,
    "smoothing_window_points": 3,
    "bare_only_max_trace_width_steps": 2.0,
    "bare_only_bare_match_steps": 3.0,
    "bare_only_required_min_power_dbm": -50.0,
    "bare_only_max_frequency_span_mhz": 10.0,
    "bare_only_max_frequency_step_mhz": 0.1,
    "bare_only_min_num_averages": 200,
    "bare_only_min_readout_length_ns": 1000,
    "bare_only_min_low_power_snr": 5.0,
    "bare_only_min_power_points": 30,
    # Operator rule 2026-10-05: the frequency window must be about 2-5x the
    # bare-dressed separation so the shift is visible; 0 disables a bound.
    "min_window_to_separation_ratio": 0.0,
    "max_window_to_separation_ratio": 0.0,
    "recommended_window_to_separation_ratio": 3.5,
    # A smeared or unstable low-power dressed plateau (e.g. a noise band read
    # as dressed) blocks, so the power window can be shifted and rescanned.
    "require_dressed_plateau_quality": False,
}


def analyze_02c_transitions(
    dataset: Any,
    configured_rules: dict[str, Any] | None = None,
    run_parameters: dict[str, Any] | None = None,
    qubit_context: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate dressed/bare separability in a 02c power sweep.

    Power is sorted from low power (more-negative dBm) to high power.  A run
    passes when the low-power dressed frequency and the high-power bare
    frequency are telling apart: they must be separated by the required
    frequency distance, each must be observed on its own classifiable points,
    and neither may sit on a frequency-sweep edge.  The power segment in
    between, where a point is neither clearly dressed nor clearly bare, is
    discarded; the proposed readout power is the highest power that is still
    classified as dressed.  Plateau stability, plateau width, depletion-point
    count, and transition monotonicity are reported as advisory metrics and
    only block when their ``require_*`` rule is enabled.
    """
    import numpy as np

    rules = dict(DEFAULT_02C_RULES)
    if configured_rules:
        rules.update(configured_rules)

    required_coordinates = {"power_dbm", "freq"}
    missing = sorted(required_coordinates - set(dataset.coords))
    if missing:
        return {
            "error": f"02c dataset is missing coordinates: {', '.join(missing)}.",
            "qubits": {},
            "rules": rules,
        }

    if "rr_min_response" in dataset.data_vars:
        trace = dataset["rr_min_response"]
        trace_source = "rr_min_response"
    else:
        source_name = next(
            (
                name
                for name in ("IQ_abs_norm", "IQ_abs")
                if name in dataset.data_vars
            ),
            None,
        )
        if source_name is None:
            return {
                "error": "02c dataset has no rr_min_response or IQ amplitude data.",
                "qubits": {},
                "rules": rules,
            }
        trace = dataset[source_name].idxmin(dim="freq")
        trace_source = f"{source_name}.idxmin(freq)"

    qubit_names = (
        [str(value) for value in dataset.coords["qubit"].values]
        if "qubit" in dataset.coords
        else ["unknown"]
    )
    frequency_axis = np.asarray(dataset.coords["freq"].values, dtype=float)
    power_axis = np.asarray(dataset.coords["power_dbm"].values, dtype=float)
    results: dict[str, Any] = {}
    for name in qubit_names:
        selected = trace.sel(qubit=name) if "qubit" in trace.dims else trace
        values = np.asarray(selected.values, dtype=float).reshape(-1)
        base_frequency = _base_frequency(dataset, name)
        full = _analyze_trace(
            power_axis,
            values,
            frequency_axis,
            base_frequency,
            rules,
            trace_source,
        )
        # Operator rule 2026-10-11: low-power rows whose dip is lost in the
        # noise are ignored; the dressed plateau is read from the lowest power
        # that still shows the resonance clearly.
        row_snr = _power_row_snr(dataset, name)
        kept = _usable_power_mask(power_axis, row_snr, rules)
        transition = full
        if kept is not None and not bool(np.all(kept)):
            transition = _analyze_trace(
                power_axis[kept],
                values[kept],
                frequency_axis,
                base_frequency,
                rules,
                trace_source,
            )
            transition["ignored_low_power_rows"] = int(np.count_nonzero(~kept))
            transition["usable_min_power_dbm"] = float(np.min(power_axis[kept]))
        if row_snr is not None:
            transition["power_row_snr"] = [round(float(value), 2) for value in row_snr]
        # Absence is judged on the whole sweep (its own low-power SNR rule),
        # and never when the usable rows show a transition.
        absence = assess_bare_only_absence(
            dataset,
            name,
            full,
            run_parameters,
            (qubit_context or {}).get(name, {}),
            rules,
        )
        if transition is not full and transition.get("classification") == "dressed_transition":
            absence = {
                key: value
                for key, value in absence.items()
                if key not in {"classification", "skip_subsequent_experiments"}
            }
        transition.update(absence)
        if transition.get("skip_subsequent_experiments") is True:
            transition["validation_failures"] = []
        results[name] = transition
    return {"qubits": results, "rules": rules}


def _power_row_snr(dataset: Any, name: str) -> Any:
    """Per-power contrast of the resonance dip over the point-to-point noise.

    (90th percentile - minimum) of the 3-point-smoothed |IQ| row, divided by
    the robust point-to-point noise. None when the dataset has no |IQ|.
    """
    import numpy as np

    if "IQ_abs" not in dataset.data_vars:
        return None
    data = dataset["IQ_abs"]
    if "qubit" in data.dims:
        data = data.sel(qubit=name)
    if set(data.dims) != {"power_dbm", "freq"}:
        return None
    rows = np.asarray(data.transpose("power_dbm", "freq").values, dtype=float)
    snr = []
    for row in rows:
        if row.size < 5 or not np.all(np.isfinite(row)):
            snr.append(0.0)
            continue
        smooth = np.convolve(row, np.ones(3) / 3, mode="same")[1:-1]
        diffs = np.diff(row)
        noise = 1.4826 * float(np.median(np.abs(diffs - np.median(diffs)))) / math.sqrt(2)
        contrast = float(np.percentile(smooth, 90) - np.min(smooth))
        snr.append(contrast / noise if noise > 0 else float("inf"))
    return np.asarray(snr, dtype=float)


def _usable_power_mask(power: Any, row_snr: Any, rules: dict[str, Any]) -> Any:
    """Keep the powers from the lowest one where the dip is clearly visible.

    The cut is the lowest power from which two consecutive rows reach
    ``min_power_row_snr``; everything below it is ignored. None disables it.
    """
    import numpy as np

    threshold = rules.get("min_power_row_snr")
    if row_snr is None or threshold is None or float(threshold) <= 0:
        return None
    power = np.asarray(power, dtype=float).reshape(-1)
    if row_snr.size != power.size:
        return None
    order = np.argsort(power)
    good = row_snr[order] >= float(threshold)
    cut = None
    for index in range(len(good) - 1):
        if good[index] and good[index + 1]:
            cut = index
            break
    if cut is None:
        return None
    keep_sorted = np.zeros(power.size, dtype=bool)
    keep_sorted[cut:] = True
    kept = np.zeros(power.size, dtype=bool)
    kept[order] = keep_sorted
    return kept


def selected_readout_power_dbm(result: Any) -> float | None:
    """Read an explicitly recorded physical power from a 02c fit result.

    Qualibrate's interactive state-update recorder suppresses assignments while
    capturing them. Consequently, ``full_scale_power_dbm`` and ``amplitude``
    in a node result can describe different states and must not be combined to
    reconstruct the sweep power. The saved derivative trace is authoritative.
    """
    if not isinstance(result, dict):
        return None
    for key in ("optimal_power_dbm", "readout_power_dbm", "power_dbm"):
        value = result.get(key)
        if _finite(value):
            return float(value)
    return None


def node_selected_readout_powers_dbm(
    dataset: Any, derivative_threshold_hz_per_dbm: Any
) -> dict[str, float | None]:
    """Reproduce the protected node's first derivative-threshold crossing.

    The selected power is the first power-grid coordinate whose saved smoothed
    derivative is below the configured threshold. This is also the y-position
    of the red horizontal line in the node's result figure.
    """
    import numpy as np

    if (
        not _finite(derivative_threshold_hz_per_dbm)
        or "power_dbm" not in dataset.coords
        or "rr_min_response_diff_avg" not in dataset.data_vars
    ):
        return {}
    threshold = float(derivative_threshold_hz_per_dbm)
    power = np.asarray(dataset.coords["power_dbm"].values, dtype=float).reshape(-1)
    derivative = dataset["rr_min_response_diff_avg"]
    qubit_names = (
        [str(value) for value in dataset.coords["qubit"].values]
        if "qubit" in dataset.coords
        else ["unknown"]
    )
    selected: dict[str, float | None] = {}
    for name in qubit_names:
        values = derivative.sel(qubit=name) if "qubit" in derivative.dims else derivative
        values = np.asarray(values.values, dtype=float).reshape(-1)
        if values.size != power.size:
            selected[name] = None
            continue
        crossings = np.flatnonzero(
            np.isfinite(power) & np.isfinite(values) & (values < threshold)
        )
        selected[name] = float(power[crossings[0]]) if crossings.size else None
    return selected


def _analyze_trace(
    powers: Any,
    frequencies: Any,
    frequency_axis: Any,
    base_frequency: float | None,
    rules: dict[str, Any],
    trace_source: str,
) -> dict[str, Any]:
    import numpy as np

    power = np.asarray(powers, dtype=float).reshape(-1)
    tracked = np.asarray(frequencies, dtype=float).reshape(-1)
    finite = np.isfinite(power) & np.isfinite(tracked)
    power = power[finite]
    tracked = tracked[finite]
    order = np.argsort(power)
    power = power[order]
    tracked = tracked[order]
    failures: list[str] = []
    advisories: list[str] = []

    def _record(blocking: bool, message: str) -> None:
        (failures if blocking else advisories).append(message)

    min_plateau_points = max(int(rules["min_plateau_points"]), 2)
    min_depletion_points = max(int(rules["min_depletion_points"]), 1)
    minimum_samples = 2 * min_plateau_points + min_depletion_points
    if power.size < minimum_samples:
        return {
            "error": f"at least {minimum_samples} finite power samples are required",
            "validation_failures": [
                f"only {power.size} finite power samples are available"
            ],
            "trace_source": trace_source,
        }

    unique_frequency = np.sort(
        np.unique(np.asarray(frequency_axis, dtype=float)[np.isfinite(frequency_axis)])
    )
    frequency_steps = np.diff(unique_frequency)
    frequency_steps = frequency_steps[frequency_steps > 0]
    if unique_frequency.size < 2 or frequency_steps.size == 0:
        return {
            "error": "frequency sweep coordinates do not increase",
            "validation_failures": ["frequency sweep coordinates do not increase"],
            "trace_source": trace_source,
        }
    frequency_step = float(np.median(frequency_steps))
    sweep_min = float(unique_frequency[0])
    sweep_max = float(unique_frequency[-1])
    sweep_span = sweep_max - sweep_min

    power_steps = np.diff(power)
    power_steps = power_steps[power_steps > 0]
    if power_steps.size == 0:
        return {
            "error": "power sweep coordinates do not increase",
            "validation_failures": ["power sweep coordinates do not increase"],
            "trace_source": trace_source,
        }
    power_step = float(np.median(power_steps))

    window = max(int(rules["smoothing_window_points"]), 1)
    if window % 2 == 0:
        window += 1
    smoothed = _rolling_median(tracked, window)
    tail_count = max(
        min_plateau_points,
        int(math.ceil(power.size * float(rules["plateau_tail_fraction"]))),
    )
    tail_count = min(tail_count, (power.size - min_depletion_points) // 2)

    dressed_frequency = float(np.median(smoothed[:tail_count]))
    bare_frequency = float(np.median(smoothed[-tail_count:]))
    signed_separation = bare_frequency - dressed_frequency
    separation = abs(signed_separation)
    minimum_separation = float(rules["min_frequency_separation_steps"]) * frequency_step
    if separation < minimum_separation:
        failures.append(
            "bare and dressed plateaus are not distinct by the required frequency separation"
        )

    band = max(
        float(rules["plateau_band_steps"]) * frequency_step,
        float(rules["plateau_band_fraction_of_separation"]) * separation,
    )
    dressed_tail = smoothed[:tail_count]
    bare_tail = smoothed[-tail_count:]
    dressed_coverage = float(np.mean(np.abs(dressed_tail - dressed_frequency) <= band))
    bare_coverage = float(np.mean(np.abs(bare_tail - bare_frequency) <= band))
    minimum_coverage = float(rules["plateau_min_coverage"])
    stability_blocks = bool(rules["require_plateau_stability"])
    dressed_quality_blocks = bool(rules["require_dressed_plateau_quality"])
    if dressed_coverage < minimum_coverage:
        _record(
            stability_blocks or dressed_quality_blocks,
            "low-power dressed plateau is not stable",
        )
    if bare_coverage < minimum_coverage:
        _record(stability_blocks, "high-power bare plateau is not stable")

    dressed_width = _central_width(dressed_tail)
    bare_width = _central_width(bare_tail)
    maximum_width = max(
        float(rules["max_plateau_width_steps"]) * frequency_step,
        float(rules["max_plateau_width_fraction_of_separation"]) * separation,
    )
    width_blocks = bool(rules["require_plateau_width"])
    if dressed_width > maximum_width:
        _record(
            width_blocks or dressed_quality_blocks,
            "low-power dressed plateau width is too large",
        )
    if bare_width > maximum_width:
        _record(width_blocks, "high-power bare plateau width is too large")

    window_ratio = sweep_span / separation if separation > 0 else math.inf
    minimum_ratio = float(rules["min_window_to_separation_ratio"])
    maximum_ratio = float(rules["max_window_to_separation_ratio"])
    recommended_span_hz = (
        float(rules["recommended_window_to_separation_ratio"]) * separation
    )
    if separation >= minimum_separation:
        if maximum_ratio > 0 and window_ratio > maximum_ratio:
            failures.append(
                f"frequency window is {window_ratio:.1f}x the bare-dressed "
                f"separation; narrow it to {minimum_ratio:g}-{maximum_ratio:g}x "
                f"(about {recommended_span_hz / 1e6:.2f} MHz) centred between them"
            )
        elif minimum_ratio > 0 and window_ratio < minimum_ratio:
            failures.append(
                f"frequency window is only {window_ratio:.1f}x the bare-dressed "
                f"separation; widen it to {minimum_ratio:g}-{maximum_ratio:g}x "
                f"(about {recommended_span_hz / 1e6:.2f} MHz)"
            )

    dressed_edge_fraction = _edge_fraction(
        dressed_frequency, sweep_min, sweep_max
    )
    bare_edge_fraction = _edge_fraction(bare_frequency, sweep_min, sweep_max)
    minimum_edge = float(rules["frequency_edge_fraction"])
    if dressed_edge_fraction < minimum_edge:
        failures.append("dressed plateau lies within 5% of a frequency-sweep edge")
    if bare_edge_fraction < minimum_edge:
        failures.append("bare plateau lies within 5% of a frequency-sweep edge")

    dressed_end: int | None = None
    bare_start: int | None = None
    depletion_points = 0
    monotonic_fraction = 0.0
    dressed_classified = 0
    bare_classified = 0
    ambiguous_classified = 0
    min_classified = max(int(rules["min_classified_plateau_points"]), 1)
    if separation >= minimum_separation:
        normalized = (smoothed - dressed_frequency) / signed_separation
        band_fraction = min(band / separation, 0.45)
        dressed_mask = np.abs(normalized) <= band_fraction
        bare_mask = np.abs(normalized - 1.0) <= band_fraction
        # A point that is neither clearly dressed nor clearly bare belongs to
        # the discarded middle segment; it is counted but never selected.
        ambiguous_classified = int(np.count_nonzero(~dressed_mask & ~bare_mask))
        # The usable dressed part ends at the first sign of punch-out: either a
        # confirmed exit from the dressed frequency, or the first point that has
        # already jumped to the bare frequency, whichever comes first. Nothing
        # above that boundary is selectable, so a bimodal transition cannot pull
        # the proposed power toward the punch-out edge.
        transition_start = _first_confirmed_false(
            dressed_mask, tail_count, min_classified
        )
        bare_points = np.flatnonzero(bare_mask)
        first_bare = int(bare_points[0]) if bare_points.size else None
        boundaries = [
            value for value in (transition_start, first_bare) if value is not None
        ]
        dressed_boundary = min(boundaries) if boundaries else int(power.size)
        if dressed_boundary < min_plateau_points:
            failures.append(
                "the dressed frequency already ends inside the low-power tail; "
                "lower min_power_dbm"
            )
        dressed_indices = np.flatnonzero(dressed_mask[:dressed_boundary])
        dressed_classified = int(dressed_indices.size)
        if dressed_classified < min_classified:
            failures.append(
                "too few low-power points could be classified as the dressed frequency"
            )
        else:
            dressed_end = int(dressed_indices[-1])

        bare_start = _confirmed_run_start(
            bare_mask, dressed_boundary, min_classified
        )
        if bare_start is None:
            failures.append(
                "too few high-power points could be classified as the bare frequency"
            )
        else:
            bare_classified = int(np.count_nonzero(bare_mask[bare_start:]))

        if dressed_end is not None and bare_start is not None:
            transition = normalized[dressed_end : bare_start + 1]
            interior = transition[
                (transition > band_fraction)
                & (transition < 1.0 - band_fraction)
            ]
            depletion_points = int(interior.size)
            if depletion_points < min_depletion_points:
                _record(
                    bool(rules["require_depletion_points"]),
                    "depletion region has too few intermediate-frequency points",
                )
            differences = np.diff(transition)
            if differences.size:
                slack = float(rules["transition_monotonic_slack_fraction"])
                monotonic_fraction = float(np.mean(differences >= -slack))
            if monotonic_fraction < float(rules["min_transition_monotonic_fraction"]):
                _record(
                    bool(rules["require_transition_monotonic"]),
                    "depletion-region frequency does not move consistently from dressed to bare as power increases",
                )

    dressed_limit = (
        float(power[dressed_end]) if dressed_end is not None else None
    )
    selected_tolerance = (
        float(rules["selected_power_tolerance_steps"]) * power_step
    )
    absolute_dressed = (
        float(base_frequency + dressed_frequency)
        if base_frequency is not None
        else None
    )
    absolute_bare = (
        float(base_frequency + bare_frequency)
        if base_frequency is not None
        else None
    )
    return {
        "trace_source": trace_source,
        "base_frequency_hz": base_frequency,
        "samples": int(power.size),
        "low_power_dbm": float(power[0]),
        "high_power_dbm": float(power[-1]),
        "power_step_db": power_step,
        "dressed_frequency_offset_hz": dressed_frequency,
        "bare_frequency_offset_hz": bare_frequency,
        "dressed_frequency_hz": absolute_dressed,
        "bare_frequency_hz": absolute_bare,
        "bare_dressed_separation_hz": separation,
        "frequency_window_hz": sweep_span,
        "window_to_separation_ratio": (
            window_ratio if math.isfinite(window_ratio) else None
        ),
        "recommended_frequency_span_mhz": round(recommended_span_hz / 1e6, 3),
        "midpoint_frequency_offset_hz": (dressed_frequency + bare_frequency) / 2.0,
        "dressed_plateau_coverage": dressed_coverage,
        "bare_plateau_coverage": bare_coverage,
        "dressed_plateau_width_hz": dressed_width,
        "bare_plateau_width_hz": bare_width,
        "dressed_frequency_edge_fraction": dressed_edge_fraction,
        "bare_frequency_edge_fraction": bare_edge_fraction,
        "depletion_point_count": depletion_points,
        "transition_monotonic_fraction": monotonic_fraction,
        "dressed_classified_point_count": dressed_classified,
        "bare_classified_point_count": bare_classified,
        "ambiguous_point_count": ambiguous_classified,
        "dressed_power_limit_dbm": dressed_limit,
        "selected_power_tolerance_db": selected_tolerance,
        "validation_failures": failures,
        "advisory_notes": advisories,
    }


def _base_frequency(dataset: Any, qubit: str) -> float | None:
    import numpy as np

    if "freq_full" not in dataset.coords:
        return None
    full = dataset.coords["freq_full"]
    if "qubit" in full.dims:
        full = full.sel(qubit=qubit)
    frequency = np.asarray(dataset.coords["freq"].values, dtype=float)
    full_values = np.asarray(full.values, dtype=float)
    if full_values.shape != frequency.shape:
        return None
    difference = full_values - frequency
    finite = difference[np.isfinite(difference)]
    return float(np.median(finite)) if finite.size else None


def _rolling_median(values: Any, window: int) -> Any:
    import numpy as np

    source = np.asarray(values, dtype=float)
    half = window // 2
    return np.asarray(
        [
            np.median(source[max(0, index - half) : min(source.size, index + half + 1)])
            for index in range(source.size)
        ],
        dtype=float,
    )


def _central_width(values: Any) -> float:
    import numpy as np

    source = np.asarray(values, dtype=float)
    return float(np.percentile(source, 90) - np.percentile(source, 10))


def _edge_fraction(value: float, minimum: float, maximum: float) -> float:
    span = maximum - minimum
    if span <= 0:
        return 0.0
    return max(0.0, min(value - minimum, maximum - value) / span)


def _first_confirmed_false(mask: Any, start: int, count: int) -> int | None:
    import numpy as np

    values = np.asarray(mask, dtype=bool)
    for index in range(start, values.size - count + 1):
        if not bool(np.any(values[index : index + count])):
            return index
    return None


def _confirmed_run_start(mask: Any, start: int, count: int) -> int | None:
    """First index at or after ``start`` that begins ``count`` true samples."""
    import numpy as np

    values = np.asarray(mask, dtype=bool)
    for index in range(max(start, 0), values.size - count + 1):
        if bool(np.all(values[index : index + count])):
            return index
    return None


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
