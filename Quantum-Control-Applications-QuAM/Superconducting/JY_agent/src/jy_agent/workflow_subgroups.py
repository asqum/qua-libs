from __future__ import annotations

import math
from typing import Any, Iterable

from .util import json_loads
from .workflow_02c import resolve_02c_targets


SUBGROUP_NODES = frozenset(
    {
        "02x",
        "02a",
        "02c",
        "02b",
        "03a",
        "03b",
        "04",
        "05",
        "07b",
        "06",
        "06a",
        "06b",
        "10a",
        "05st",
        "06st_t2star",
        "06st_t2e",
    }
)
# Retained as an empty set for import compatibility. Every calibration node now
# follows shared-first-batch then unresolved-subgroup retry; no node silently
# expands a target-local retry back to the full workflow set.
FIXED_TARGET_NODES: frozenset[str] = frozenset()
# First completed run of these nodes must multiplex every currently-active
# target with the same parameters. Later retries must omit already-resolved
# qubits and should still group any remaining unresolved qubits that can
# share the recovery settings. 03a keeps its own first-coarse-all-targets
# gate. Statistics nodes may split earlier when wait windows are incompatible.
SHARED_FIRST_BATCH_NODES = frozenset(
    {"02x", "02a", "02c", "02b", "03b", "04", "05", "07b", "06", "06a", "06b", "10a"}
)
# Flux calibration (operator 2026-10-08). Their resolution never gates the
# rest of the sequence: a qubit that is flat, crosses a coupler or fails here
# still runs 03a, 04, ... with its old flux settings.
FLUX_NODES = frozenset({"02b", "03b", "06a"})
DEFAULT_02X_MIN_ROBUST_SNR = 3.0
DEFAULT_02A_MIN_ROBUST_SNR = 3.0
DEFAULT_03A_MIN_ROBUST_SNR = 6.0
DEFAULT_03A_MIN_EDGE_FRACTION = 0.05
DEFAULT_03A_MAX_NOISE_CONFIRMATION_AVERAGES = 2000
DEFAULT_03A_FINAL_MIN_FWHM_HZ = 500_000.0
DEFAULT_03A_FINAL_MAX_FWHM_HZ = 16_000_000.0
# Fallbacks only; the service passes analysis."04"/"05".min_robust_snr from
# rules/policies.yaml. Kept equal to those values.
DEFAULT_04_MIN_ROBUST_SNR = 7.0
DEFAULT_05_MIN_ROBUST_SNR = 15.0


def resolve_node_targets(
    node_id: str,
    run_rows: Iterable[dict[str, Any]],
    min_03a_snr: float = DEFAULT_03A_MIN_ROBUST_SNR,
    min_04_snr: float = DEFAULT_04_MIN_ROBUST_SNR,
    min_05_snr: float = DEFAULT_05_MIN_ROBUST_SNR,
    min_03a_fwhm_hz: float = DEFAULT_03A_FINAL_MIN_FWHM_HZ,
    max_03a_fwhm_hz: float = DEFAULT_03A_FINAL_MAX_FWHM_HZ,
    min_02x_snr: float = DEFAULT_02X_MIN_ROBUST_SNR,
    min_02a_snr: float = DEFAULT_02A_MIN_ROBUST_SNR,
) -> set[str]:
    """Return targets with final per-qubit evidence for one node."""
    rows = list(run_rows)
    if node_id == "02c":
        resolved, _ = resolve_02c_targets(rows)
        return resolved
    if node_id not in SUBGROUP_NODES - {"02c"}:
        return set()
    if node_id == "03a":
        return _resolve_03a_targets(
            rows,
            require_fine=True,
            min_snr=min_03a_snr,
            min_fwhm_hz=min_03a_fwhm_hz,
            max_fwhm_hz=max_03a_fwhm_hz,
        )
    return _resolve_usable_targets(
        node_id,
        rows,
        require_fine_03a=False,
        min_03a_snr=min_03a_snr,
        min_04_snr=min_04_snr,
        min_05_snr=min_05_snr,
        min_03a_fwhm_hz=min_03a_fwhm_hz,
        max_03a_fwhm_hz=max_03a_fwhm_hz,
        min_02x_snr=min_02x_snr,
        min_02a_snr=min_02a_snr,
    )


def resolve_02b_tunable_targets(run_rows: Iterable[dict[str, Any]]) -> set[str]:
    """Qubits whose latest decided 02b result says the resonator tunes.

    Only these go on to 03b and 06a (operator rule 2026-10-08); flat and
    avoided-crossing qubits skip them.
    """

    latest: dict[str, bool] = {}
    for row in run_rows:
        if row.get("status") not in {None, "completed"}:
            continue
        if row.get("analysis_status") not in {"pass", "needs_review"}:
            continue
        analysis = json_loads(row.get("analysis_json"), {})
        results = analysis.get("fit_quality", {}).get("results", {})
        if not isinstance(results, dict):
            continue
        for name, fit in results.items():
            if isinstance(fit, dict) and fit.get("resolved") is True:
                latest[str(name)] = fit.get("flux_tunable") is True
    return {name for name, tunable in latest.items() if tunable}


def resolve_03a_candidate_targets(
    run_rows: Iterable[dict[str, Any]],
    min_snr: float = DEFAULT_03A_MIN_ROBUST_SNR,
) -> set[str]:
    """Return credible 03a candidates after explicit human rejections."""
    candidates, _ = _03a_candidate_walk(list(run_rows), min_snr)
    return candidates


def _03a_candidate_walk(
    rows: list[dict[str, Any]], min_snr: float
) -> tuple[set[str], set[str]]:
    """Return (candidates, dismissed) after walking the 03a rows in order.

    Operator rule 2026-10-20 (q5/q7 window-shift lock): a decided low-power
    refinement or fine scan that no longer shows a credible line for a
    candidate dismisses it -- the coarse line vanished with power, so it was
    multi-photon -- and the qubit may leave the window like any unresolved one.
    """
    candidates: set[str] = set()
    dismissed: set[str] = set()
    for row in rows:
        credible = _resolve_usable_targets(
            "03a", [row], require_fine_03a=False, min_03a_snr=min_snr
        )
        candidates.update(credible)
        dismissed.difference_update(credible)
        parameters = json_loads(row.get("parameters_json"), {})
        names = parameters.get("qubits", [])
        names = [str(name) for name in names] if isinstance(names, list) else []
        if row.get("decision") == "manual_review":
            candidates.difference_update(names)
            continue
        analysis = json_loads(row.get("analysis_json"), {})
        if (
            row.get("decision")
            and row.get("status") in {None, "completed"}
            and row.get("analysis_status") in {"pass", "needs_review", "failed"}
            and analysis.get("03a_stage") in {"refinement_candidate", "fine"}
        ):
            # Vanished means the trace lost its line (SNR below the floor),
            # not a fine line held back for other reasons such as its
            # pending upward check.
            metrics_by_name = analysis.get("dataset_metrics", {}).get("qubits", {})
            if not isinstance(metrics_by_name, dict):
                metrics_by_name = {}
            vanished = set()
            for name in names:
                metrics = metrics_by_name.get(name)
                snr = metrics.get("robust_snr") if isinstance(metrics, dict) else None
                if (
                    name in candidates
                    and name not in credible
                    and _finite(snr)
                    and float(snr) < float(min_snr)
                ):
                    vanished.add(name)
            candidates.difference_update(vanished)
            dismissed.update(vanished)
    return candidates, dismissed


def _resolve_03a_targets(
    rows: list[dict[str, Any]],
    *,
    require_fine: bool,
    min_snr: float,
    min_fwhm_hz: float = DEFAULT_03A_FINAL_MIN_FWHM_HZ,
    max_fwhm_hz: float = DEFAULT_03A_FINAL_MAX_FWHM_HZ,
) -> set[str]:
    resolved: set[str] = set()
    for row in rows:
        resolved.update(
            _resolve_usable_targets(
                "03a",
                [row],
                require_fine_03a=require_fine,
                min_03a_snr=min_snr,
                min_03a_fwhm_hz=min_fwhm_hz,
                max_03a_fwhm_hz=max_fwhm_hz,
            )
        )
        if row.get("decision") == "manual_review":
            parameters = json_loads(row.get("parameters_json"), {})
            names = parameters.get("qubits", [])
            if isinstance(names, list):
                resolved.difference_update(str(name) for name in names)
    return resolved


def resolve_03a_shift_ready_targets(
    run_rows: Iterable[dict[str, Any]],
    *,
    min_snr: float = DEFAULT_03A_MIN_ROBUST_SNR,
    min_edge_fraction: float = DEFAULT_03A_MIN_EDGE_FRACTION,
    max_noise_confirmation_averages: int = (
        DEFAULT_03A_MAX_NOISE_CONFIRMATION_AVERAGES
    ),
) -> set[str]:
    """Return unresolved targets allowed to move to another LO window.

    Edge-limited features may shift immediately. Interior low-SNR traces must
    first be reconfirmed with at least the configured maximum averages. A
    candidate dismissed by a low-power scan may shift immediately too: the
    full-power coarse scan already covered the window.
    """
    rows = list(run_rows)
    candidates, dismissed = _03a_candidate_walk(rows, min_snr)
    latest: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") not in {None, "completed"}:
            continue
        if row.get("analysis_status") not in {"pass", "needs_review", "failed"}:
            continue
        analysis = json_loads(row.get("analysis_json"), {})
        parameters = json_loads(row.get("parameters_json"), {})
        metrics_by_name = analysis.get("dataset_metrics", {}).get("qubits", {})
        names = parameters.get("qubits", [])
        if not isinstance(names, list):
            names = []
        for name in names:
            metrics = (
                metrics_by_name.get(name, {})
                if isinstance(metrics_by_name, dict)
                else {}
            )
            latest[str(name)] = (parameters, metrics)

    ready: set[str] = set()
    for name, (parameters, metrics) in latest.items():
        if name in candidates:
            continue
        edge = metrics.get("edge_fraction") if isinstance(metrics, dict) else None
        averages = parameters.get("num_averages")
        edge_limited = _finite(edge) and float(edge) < float(min_edge_fraction)
        sufficiently_averaged = (
            isinstance(averages, int)
            and not isinstance(averages, bool)
            and averages >= int(max_noise_confirmation_averages)
        )
        if edge_limited or sufficiently_averaged or name in dismissed:
            ready.add(name)
    return ready


def _resolve_usable_targets(
    node_id: str,
    rows: list[dict[str, Any]],
    *,
    require_fine_03a: bool,
    min_03a_snr: float,
    min_04_snr: float = DEFAULT_04_MIN_ROBUST_SNR,
    min_05_snr: float = DEFAULT_05_MIN_ROBUST_SNR,
    min_03a_fwhm_hz: float = DEFAULT_03A_FINAL_MIN_FWHM_HZ,
    max_03a_fwhm_hz: float = DEFAULT_03A_FINAL_MAX_FWHM_HZ,
    min_02x_snr: float = DEFAULT_02X_MIN_ROBUST_SNR,
    min_02a_snr: float = DEFAULT_02A_MIN_ROBUST_SNR,
) -> set[str]:
    resolved: set[str] = set()
    for row in rows:
        if row.get("status") not in {None, "completed"}:
            continue
        if row.get("analysis_status") not in {"pass", "needs_review"}:
            continue
        analysis = json_loads(row.get("analysis_json"), {})
        if (
            node_id == "03a"
            and require_fine_03a
            and analysis.get("03a_stage") != "fine"
        ):
            continue
        results = analysis.get("fit_quality", {}).get("results", {})
        # Resonator analysis may intentionally omit structured fit results and
        # rely on the per-qubit sweep metrics as its authoritative evidence.
        if results is None and node_id in {"02x", "02a"}:
            results = {}
        metrics = analysis.get("dataset_metrics", {}).get("qubits", {})
        if not isinstance(results, dict) or not isinstance(metrics, dict):
            continue
        names = set(results) | {
            str(name) for name in metrics if isinstance(name, str)
        }
        for name in names:
            fit = results.get(name, {})
            if not isinstance(fit, dict):
                continue
            if not _fit_is_usable(node_id, fit):
                continue
            evidence = metrics.get(name)
            if node_id in {
                "02b",
                "03b",
                "06a",
                "07b",
                "06",
                "06b",
                "10a",
                "05st",
                "06st_t2star",
                "06st_t2e",
            }:
                resolved.add(str(name))
                continue
            if node_id == "05":
                if _snr_evidence_is_usable(evidence, min_05_snr):
                    resolved.add(str(name))
                continue
            if node_id == "03a" and require_fine_03a:
                width = (
                    evidence.get("feature_fwhm_hz")
                    if isinstance(evidence, dict)
                    else None
                )
                if not _finite(width) or not (
                    float(min_03a_fwhm_hz)
                    <= float(width)
                    <= float(max_03a_fwhm_hz)
                ):
                    continue
            min_snr = {
                "02x": min_02x_snr,
                "02a": min_02a_snr,
                "03a": min_03a_snr,
                "04": min_04_snr,
                "05": min_05_snr,
            }.get(node_id, 5.0)
            if not _sweep_evidence_is_usable(evidence, min_snr):
                continue
            resolved.add(str(name))
    return resolved


def _fit_is_usable(node_id: str, fit: dict[str, Any]) -> bool:
    if node_id in {"02x", "02a"}:
        if fit.get("fit_successful") is False:
            return False
        for key in (
            "RO_frequency",
            "frequency",
            "resonance_frequency",
            "freq",
        ):
            if _finite(fit.get(key)):
                return True
        # Resonator nodes may rely on sweep metrics when the protected node
        # omits a structured frequency field.
        return True
    if node_id == "03a":
        return fit.get("fit_successful") is True and _finite(
            fit.get("drive_freq")
        )
    if node_id == "04":
        amplitude = fit.get("Pi_amplitude")
        return _finite(amplitude) and float(amplitude) != 0.0
    if node_id == "05":
        t1 = fit.get("t1_seconds")
        relative_uncertainty = fit.get("relative_uncertainty")
        r_squared = fit.get("r_squared")
        coverage = fit.get("coverage_lifetimes")
        samples_per_lifetime = fit.get("samples_per_lifetime")
        return (
            _finite(t1)
            and float(t1) > 0
            and _finite(relative_uncertainty)
            and 0 <= float(relative_uncertainty) < 0.25
            and _finite(r_squared)
            and float(r_squared) >= 0.9
            and _finite(coverage)
            and float(coverage) >= 3.5
            and _finite(samples_per_lifetime)
            and float(samples_per_lifetime) >= 2
            and fit.get("fit_at_search_boundary") is not True
        )
    if node_id == "07b":
        fidelity = fit.get("readout_fidelity")
        return (
            fit.get("fit_successful") is True
            and _finite(fidelity)
            and 0.5 <= float(fidelity) <= 1
        )
    if node_id in {"06", "06b"}:
        lifetime = fit.get("coherence_seconds")
        relative_uncertainty = fit.get("relative_uncertainty")
        coverage = fit.get("coverage_lifetimes")
        samples = fit.get("samples_per_lifetime")
        return (
            fit.get("fit_successful") is True
            and _finite(lifetime)
            and float(lifetime) > 0
            and _finite(relative_uncertainty)
            and 0 <= float(relative_uncertainty) < 0.25
            and _finite(coverage)
            and float(coverage) >= 3.5
            and _finite(samples)
            and float(samples) >= 2
            and (
                node_id != "06b"
                or (
                    # The R-squared floor is policy (analysis.06b.min_r_squared,
                    # 0.8 since 2026-10-12) and already decided fit_successful.
                    _finite(fit.get("r_squared"))
                    and fit.get("fit_at_search_boundary") is not True
                )
            )
        )
    if node_id == "10a":
        epc = fit.get("EPC")
        epg = fit.get("EPG")
        return (
            fit.get("fit_successful") is True
            and _finite(epc)
            and 0 <= float(epc) <= 1
            and _finite(epg)
            and 0 <= float(epg) <= 1
        )
    if node_id == "02b":
        # Tunable, flat and avoided-crossing qubits are all decided.
        return fit.get("resolved") is True
    if node_id in {"03b", "06a"}:
        return fit.get("fit_successful") is True
    if node_id in {"05st", "06st_t2star", "06st_t2e"}:
        return (
            fit.get("fit_successful") is True
            and fit.get("histo_num") == 100
            and fit.get("raw_iteration_refit_required") is False
        )
    return False


def _sweep_evidence_is_usable(metrics: Any, min_snr: float) -> bool:
    if not isinstance(metrics, dict) or metrics.get("error"):
        return False
    edge = metrics.get("edge_fraction")
    snr = metrics.get("robust_snr")
    return (
        _finite(edge)
        and float(edge) >= DEFAULT_03A_MIN_EDGE_FRACTION
        and _finite(snr)
        and float(snr) >= float(min_snr)
    )


def _snr_evidence_is_usable(metrics: Any, min_snr: float) -> bool:
    if not isinstance(metrics, dict) or metrics.get("error"):
        return False
    snr = metrics.get("robust_snr")
    return _finite(snr) and float(snr) >= float(min_snr)


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
