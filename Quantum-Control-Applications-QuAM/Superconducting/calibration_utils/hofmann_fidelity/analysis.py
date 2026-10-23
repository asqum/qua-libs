"""
Hofmann complementary classical fidelities of a two-qubit CZ gate (analysis for 95_hofmann_fidelity).

Method (H. F. Hofmann, arXiv:quant-ph/0409083)
    Eight product inputs form two mutually unbiased sets, prepared from |00>, sent through the gate under test
    and measured in the basis of their set::

        set A (measure X on the control, Z on the target):  |+0>  |-0>  |+1>  |-1>
        set B (measure Z on the control, X on the target):  |0+>  |0->  |1+>  |1->

    For an ideal CZ every input ends in exactly one computational outcome of its measurement basis, so the
    fidelity F_k of input k is the fraction of shots that land on that outcome. With equal weights::

        F_A = mean(F_A1..F_A4)      F_B = mean(F_B1..F_B4)
        F_A + F_B - 1  <=  F_pro  <=  min(F_A, F_B)

    F_pro is the process (entanglement) fidelity with respect to CZ = diag(1, 1, 1, -1). The two outer terms
    are *bounds on* F_pro (valid for any CPTP map if the inputs are prepared and measured as specified), they
    are not F_pro itself. The binomial standard errors reported next to them are statistical uncertainties of
    the measured fractions and are a separate thing from the width of the Hofmann interval.
    ``C_ML = (F_A + F_B) / 2 = (1 + F_lower) / 2`` is the score that the DGX retuning loop maximises.

Conventions (identical to 40b_Bell_state_tomography and 90_cz_process_tomography)
    * Qubit 0 = ``qubit_pair.qubit_control``, qubit 1 = ``qubit_pair.qubit_target``. A joint outcome is the bit
      string "c t" and its histogram bin is ``2 * c + t`` (bins 00, 01, 10, 11).
    * ``readout_state``: 0 -> |g> (Z = +1), 1 -> |e>. With GEF readout 2 -> |f> (leaked).
    * Pulses: x180 = R_x(pi), y90 = R_y(+pi/2), -y90 = R_y(-pi/2), R_a(t) = exp(-i t sigma_a / 2).
      |0> no pulse, |1> x180, |+> y90, |-> -y90. X-basis readout = -y90 followed by the Z readout.
    * The CZ pulse is assumed to contain its calibrated virtual-Z (frame) corrections, so that the target is
      exactly diag(1, 1, 1, -1).

Leaked shots (GEF readout) are kept in the shot total and never in a histogram bin, so they count as failures
and are not post-selected away.

This module is pure numpy: it imports nothing from QUA or QuAM and can be tested without hardware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "BASIS_A",
    "BASIS_B",
    "CONVENTIONS",
    "HOFMANN_INPUTS",
    "INPUT_LABELS",
    "MEAS_PULSES",
    "N_INPUTS",
    "OUTCOME_LABELS",
    "PREP_PULSES",
    "HofmannInput",
    "aggregate_fidelities",
    "analyse_shots",
    "controlled_phase_unitary",
    "hofmann_fidelities",
    "histograms_from_states",
    "readout_corrected_success",
    "result_to_flat_dict",
    "simulate_histograms",
    "validate_input_table",
]

CONVENTIONS = (
    "qubit 0 = control, qubit 1 = target; joint outcome 'c t' -> bin 2c + t (00, 01, 10, 11); readout 0 = |g>, 1 = |e>, "
    "2 = |f> (GEF only, counted as failure); |0> no pulse, |1> x180, |+> y90, |-> -y90; X readout = -y90 then Z readout; "
    "F_k = fraction of shots on the ideal-CZ outcome of input k; F_A, F_B = equal-weight means of set A (X on control, Z on "
    "target) and set B (Z on control, X on target); F_lower = F_A + F_B - 1 (unclipped); F_upper = min(F_A, F_B); "
    "C_ML = (F_A + F_B) / 2; sigma = binomial standard error; readout correction = inv(qp.confusion) @ p "
    "(confusion[measured][prepared])."
)

OUTCOME_LABELS: Tuple[str, ...] = ("00", "01", "10", "11")

# State letter -> pulse that prepares it from |0> (None: no pulse)
PREP_PULSES: Dict[str, Optional[str]] = {"0": None, "1": "x180", "+": "y90", "-": "-y90"}
# Measurement letter -> rotation applied before the Z readout (None: no pulse)
MEAS_PULSES: Dict[str, Optional[str]] = {"Z": None, "X": "-y90"}

_PULSE_ROTATIONS = {"x180": ("x", math.pi), "x90": ("x", math.pi / 2), "y90": ("y", math.pi / 2), "-y90": ("y", -math.pi / 2)}
_PAULI = {
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
}
_I2 = np.eye(2, dtype=complex)
_CZ = np.diag([1, 1, 1, -1]).astype(complex)


def _pulse_unitary(pulse: Optional[str]) -> np.ndarray:
    if pulse is None:
        return _I2
    axis, angle = _PULSE_ROTATIONS[pulse]
    return math.cos(angle / 2) * _I2 - 1j * math.sin(angle / 2) * _PAULI[axis]


def controlled_phase_unitary(delta: float = 0.0) -> np.ndarray:
    """diag(1, 1, 1, -exp(i delta)): a CZ with a controlled-phase error ``delta`` (for tests and simulations)."""
    return np.diag([1, 1, 1, -np.exp(1j * delta)]).astype(complex)


@dataclass(frozen=True)
class HofmannInput:
    """One of the eight inputs: preparation, measurement basis and the outcome an ideal CZ gives."""

    label: str
    prep: Tuple[str, str]  # state letters of (control, target), e.g. ("+", "0")
    measure: Tuple[str, str]  # measurement letters of (control, target), e.g. ("X", "Z")
    expected: Tuple[int, int]  # ideal-CZ outcome bits (control, target)

    @property
    def expected_index(self) -> int:
        return 2 * self.expected[0] + self.expected[1]

    @property
    def ket(self) -> str:
        return "".join(self.prep)

    @property
    def basis(self) -> str:
        return "".join(self.measure)


def _row(label: str, prep: str, measure: str, expected: Tuple[int, int]) -> HofmannInput:
    return HofmannInput(label, (prep[0], prep[1]), (measure[0], measure[1]), expected)


# The protocol table. ``validate_input_table`` re-derives every expected outcome from the CZ matrix and the
# pulse definitions above, so a typo in this table fails at import and not on the first hardware run.
HOFMANN_INPUTS: Tuple[HofmannInput, ...] = (
    _row("A1", "+0", "XZ", (0, 0)),
    _row("A2", "-0", "XZ", (1, 0)),
    _row("A3", "+1", "XZ", (1, 1)),
    _row("A4", "-1", "XZ", (0, 1)),
    _row("B1", "0+", "ZX", (0, 0)),
    _row("B2", "0-", "ZX", (0, 1)),
    _row("B3", "1+", "ZX", (1, 1)),
    _row("B4", "1-", "ZX", (1, 0)),
)
N_INPUTS = len(HOFMANN_INPUTS)
INPUT_LABELS: Tuple[str, ...] = tuple(entry.label for entry in HOFMANN_INPUTS)
BASIS_A = slice(0, 4)
BASIS_B = slice(4, 8)


def _outcome_probabilities(entry: HofmannInput, unitary: np.ndarray = _CZ) -> np.ndarray:
    """Joint outcome probabilities [00, 01, 10, 11] of ``entry`` through ``unitary`` with ideal pulses."""
    ket0 = np.array([1, 0], dtype=complex)
    state = np.kron(_pulse_unitary(PREP_PULSES[entry.prep[0]]) @ ket0, _pulse_unitary(PREP_PULSES[entry.prep[1]]) @ ket0)
    measurement = np.kron(_pulse_unitary(MEAS_PULSES[entry.measure[0]]), _pulse_unitary(MEAS_PULSES[entry.measure[1]]))
    return np.abs(measurement @ unitary @ state) ** 2


def validate_input_table() -> None:
    """Raise ``ValueError`` unless each ideal-CZ output of the table is a single outcome equal to ``expected``."""
    if len(HOFMANN_INPUTS) != 8 or len(set(INPUT_LABELS)) != 8:
        raise ValueError("the Hofmann protocol has eight distinct inputs")
    for entry in HOFMANN_INPUTS:
        probabilities = _outcome_probabilities(entry)
        index = int(np.argmax(probabilities))
        if not math.isclose(probabilities[index], 1.0, abs_tol=1e-9) or index != entry.expected_index:
            raise ValueError(
                f"input {entry.label} (|{entry.ket}> measured as {entry.basis}): expected outcome {entry.expected} "
                f"disagrees with the simulated ideal CZ (probabilities {np.round(probabilities, 4).tolist()})"
            )
    for basis, measures in (("A", ("X", "Z")), ("B", ("Z", "X"))):
        if any(tuple(e.measure) != measures for e in HOFMANN_INPUTS if e.label[0] == basis):
            raise ValueError(f"every input of set {basis} must be measured as {''.join(measures)}")


# --------------------------------------------------------------------------------------
# Counts -> fidelities
# --------------------------------------------------------------------------------------


def aggregate_fidelities(F, sigma) -> Dict[str, object]:
    """Set means, Hofmann bounds and ML score from per-input values ``F`` and errors ``sigma`` (both shape (8,)).

    Also used for readout-corrected values, which may leave [0, 1]. ``F_lower`` is not clipped.
    """
    F = np.asarray(F, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if F.shape != (N_INPUTS,) or sigma.shape != (N_INPUTS,):
        raise ValueError(f"F and sigma must have shape ({N_INPUTS},), got {F.shape} and {sigma.shape}")
    F_A, F_B = float(F[BASIS_A].mean()), float(F[BASIS_B].mean())
    sigma_A = float(np.sqrt(np.sum(sigma[BASIS_A] ** 2)) / 4)
    sigma_B = float(np.sqrt(np.sum(sigma[BASIS_B] ** 2)) / 4)
    return {
        "F": F,
        "sigma": sigma,
        "F_A": F_A,
        "F_B": F_B,
        "F_lower": F_A + F_B - 1.0,
        "F_upper": min(F_A, F_B),
        "C_ML": 0.5 * (F_A + F_B),
        "sigma_F_A": sigma_A,
        "sigma_F_B": sigma_B,
        "sigma_F_lower": float(np.hypot(sigma_A, sigma_B)),
        # first order: the error of the set that currently attains the minimum
        "sigma_F_upper": sigma_A if F_A <= F_B else sigma_B,
        "sigma_C_ML": float(0.5 * np.hypot(sigma_A, sigma_B)),
    }


def hofmann_fidelities(success, shots) -> Dict[str, object]:
    """Fidelities and bounds from the eight success counts and the eight shot totals.

    ``success[k]`` is the number of shots of input k that gave its ideal outcome and ``shots[k]`` the number
    taken (a scalar is broadcast). Every input has the same weight in F_A / F_B regardless of its shot count.
    """
    s = np.asarray(success, dtype=float)
    n = np.asarray(shots, dtype=float)
    if n.ndim == 0:
        n = np.full(N_INPUTS, float(n))
    if s.shape != (N_INPUTS,) or n.shape != (N_INPUTS,):
        raise ValueError(f"expected {N_INPUTS} success counts and shot totals, got shapes {s.shape} and {n.shape}")
    if not (np.all(np.isfinite(s)) and np.all(np.isfinite(n))) or np.any(s != np.floor(s)) or np.any(n != np.floor(n)):
        raise ValueError("counts must be finite whole numbers")
    if np.any(n <= 0):
        bad = [INPUT_LABELS[i] for i in np.flatnonzero(n <= 0)]
        raise ValueError(f"zero shots for input(s) {bad}: the fidelity is undefined")
    if np.any(s < 0) or np.any(s > n):
        raise ValueError("success counts must satisfy 0 <= success <= shots for every input")
    F = s / n
    sigma = np.sqrt(F * (1.0 - F) / n)  # binomial standard error of each fraction
    return aggregate_fidelities(F, sigma)


def histograms_from_states(state_control, state_target) -> Tuple[np.ndarray, np.ndarray]:
    """Joint-outcome histograms and leaked-shot counts from single-shot readouts.

    Args:
        state_control, state_target: integer readouts of shape ``(n_shots, 8)`` (shots x inputs, in the order of
            ``INPUT_LABELS``); values 0/1 for g/e and 2 for |f> when GEF readout was used.

    Returns:
        ``histograms`` of shape (8, 4), the counts of the bins 00, 01, 10, 11 (leaked shots excluded), and
        ``leaked`` of shape (8,), the number of shots in which either qubit was read out in |f>.
    """
    sc = np.asarray(state_control)
    st = np.asarray(state_target)
    if sc.ndim != 2 or sc.shape != st.shape or sc.shape[1] != N_INPUTS or sc.shape[0] < 1:
        raise ValueError(f"readouts must have shape (n_shots, {N_INPUTS}), got {sc.shape} and {st.shape}")
    if not (np.all(np.isin(sc, (0, 1, 2))) and np.all(np.isin(st, (0, 1, 2)))):
        raise ValueError("readout values must be 0, 1 or 2")
    sc = sc.astype(int)
    st = st.astype(int)
    leaked = (sc == 2) | (st == 2)
    histograms = np.zeros((N_INPUTS, 4), dtype=np.int64)
    for k in range(N_INPUTS):
        keep = ~leaked[:, k]
        histograms[k] = np.bincount(2 * sc[keep, k] + st[keep, k], minlength=4)
    return histograms, leaked.sum(axis=0).astype(np.int64)


def readout_corrected_success(histograms, confusion) -> Tuple[np.ndarray, np.ndarray]:
    """Success probabilities after inverting the readout confusion matrix, with propagated standard errors.

    This is the correction of 40b_Bell_state_tomography without its clipping, ``inv(qp.confusion) @ p`` with
    ``confusion[measured][prepared]`` and bins ordered 00, 01, 10, 11. It is an unconstrained linear inversion,
    so the result can leave [0, 1]; the standard error follows from the multinomial covariance of the raw
    histogram. It assumes no leaked shots (2-state readout).
    """
    h = np.asarray(histograms, dtype=float)
    matrix = np.asarray(confusion, dtype=float)
    if h.shape != (N_INPUTS, 4):
        raise ValueError(f"histograms must have shape ({N_INPUTS}, 4), got {h.shape}")
    if matrix.shape != (4, 4):
        raise ValueError(f"confusion matrix must be 4x4, got {matrix.shape}")
    totals = h.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("cannot correct a histogram without shots")
    p = h / totals
    expected = np.array([entry.expected_index for entry in HOFMANN_INPUTS])
    a = np.linalg.inv(matrix)[expected]  # row of the inverse that produces each input's expected bin, (8, 4)
    corrected = np.einsum("kj,kj->k", a, p)
    variance = (np.einsum("kj,kj->k", a**2, p) - corrected**2) / totals[:, 0]
    return corrected, np.sqrt(np.clip(variance, 0.0, None))


def analyse_shots(state_control, state_target, confusion=None) -> Dict[str, object]:
    """Complete analysis of one qubit pair from its single-shot readouts.

    Args:
        state_control, state_target: readouts of shape ``(n_shots, 8)``, see ``histograms_from_states``.
        confusion: optional ``qp.confusion`` (4x4, ``confusion[measured][prepared]``). When given, a
            readout-corrected evaluation is added next to the raw one. Only valid without leaked shots.

    Returns a dict with ``histograms`` (8, 4), ``leaked`` (8,), ``shots_per_state``, ``raw`` (the output of
    ``hofmann_fidelities``) and, with a confusion matrix, ``corrected`` (``aggregate_fidelities`` of the
    corrected success probabilities). Raw values are the primary result and never use the correction.
    """
    histograms, leaked = histograms_from_states(state_control, state_target)
    n_shots = int(np.asarray(state_control).shape[0])
    expected = np.array([entry.expected_index for entry in HOFMANN_INPUTS])
    success = histograms[np.arange(N_INPUTS), expected]
    raw = hofmann_fidelities(success, np.full(N_INPUTS, n_shots))

    out: Dict[str, object] = {
        "histograms": histograms,
        "leaked": leaked,
        "shots_per_state": n_shots,
        "success": success,
        "raw": raw,
        "leakage_probability": float(leaked.sum() / (N_INPUTS * n_shots)),
    }
    if confusion is not None:
        if leaked.any():
            raise ValueError("the confusion matrix is 2-state: pass confusion=None when shots were read out in |f>")
        corrected, sigma = readout_corrected_success(histograms, confusion)
        out["corrected"] = aggregate_fidelities(corrected, sigma)
    return out


def result_to_flat_dict(result: Dict[str, object], prefix: str = "") -> Dict[str, float]:
    """Scalars and per-input values of a ``hofmann_fidelities`` / ``aggregate_fidelities`` result as plain floats."""
    out: Dict[str, float] = {}
    for key in ("F_A", "F_B", "F_lower", "F_upper", "C_ML", "sigma_F_A", "sigma_F_B", "sigma_F_lower", "sigma_F_upper", "sigma_C_ML"):
        out[f"{prefix}{key}"] = float(result[key])
    for label, f, s in zip(INPUT_LABELS, result["F"], result["sigma"]):
        out[f"{prefix}F_{label}"] = float(f)
        out[f"{prefix}sigma_{label}"] = float(s)
    return out


# --------------------------------------------------------------------------------------
# Simulator (tests and offline checks)
# --------------------------------------------------------------------------------------


def simulate_histograms(unitary: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Multinomial joint-outcome counts, shape (8, 4), of ``unitary`` with ideal preparation and measurement."""
    histograms = np.zeros((N_INPUTS, 4), dtype=np.int64)
    for k, entry in enumerate(HOFMANN_INPUTS):
        probabilities = _outcome_probabilities(entry, unitary)
        histograms[k] = rng.multinomial(shots, probabilities / probabilities.sum())
    return histograms


# Fail at import, not on the first hardware run, if the table above has a typo.
validate_input_table()
