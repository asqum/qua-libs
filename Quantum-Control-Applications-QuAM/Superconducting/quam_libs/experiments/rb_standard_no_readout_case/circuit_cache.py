"""
Circuit cache and RB reference helpers for the RB nodes with gate-only packed circuits (no readout case).

* Own cache (read/write): ``<repo>/.rb_cache_no_readout_case`` -- circuits WITHOUT the readout opcode 66.
  The file key carries the single-qubit basis and a cache version, and the payload is stamped with the same values
  and checked on load.
* Original cache (READ ONLY): ``<repo>/.rb_cache`` written by 70b/70c -- circuits WITH opcode 66 appended; they are
  re-used here (66 stripped) so already transpiled circuits need not be generated again. Circuits that do not fit the
  XZ basis (e.g. contain analog y gates) are treated as a cache miss. Nothing is ever written into the original cache.
* RB reference (``store_reference`` / ``get_reference_alpha``): the standard-RB node stores its decay under its own
  extras key, so 70b's ``StandardRB`` entry is never overwritten; the interleaved-RB node reads it and refuses a
  reference of a different basis.
"""
from pathlib import Path

from quam_libs.experiments.rb_standard.rb_utils import rb_save, rb_try_load
from quam_libs.experiments.rb_standard_no_readout_case.packing import SINGLE_QUBIT_BASIS, circuits_to_compact

READOUT_OPCODE = 66
CACHE_VERSION = 1  # bump when the stored circuit format changes
REFERENCE_EXTRAS_KEY = "StandardRB_no_readout_case"


def own_cache_dir(calibration_graph_dir: Path) -> Path:
    return Path(calibration_graph_dir).resolve().parent / ".rb_cache_no_readout_case"


def original_cache_dir(calibration_graph_dir: Path) -> Path:
    return Path(calibration_graph_dir).resolve().parent / ".rb_cache"


def own_cache_key(key: str, basis: str = SINGLE_QUBIT_BASIS) -> str:
    return f"{key}_{basis}_v{CACHE_VERSION}"


def strip_readout(circuits_with_readout: list[list[int]]) -> list[list[int]]:
    """Remove the trailing readout opcode of every circuit (the opcode never appears elsewhere)."""
    stripped = []
    for circuit in circuits_with_readout:
        if circuit and circuit[-1] == READOUT_OPCODE:
            circuit = circuit[:-1]
        if READOUT_OPCODE in circuit:
            raise ValueError("Unexpected readout opcode in the middle of a cached circuit.")
        stripped.append(list(circuit))
    return stripped


def _fits_basis(circuits: list[list[int]]) -> bool:
    try:
        circuits_to_compact(circuits)
    except ValueError:
        return False
    return True


def load_circuits(calibration_graph_dir: Path, key: str, basis: str = SINGLE_QUBIT_BASIS) -> dict | None:
    """
    Returns a dict with ``circuits_as_ints`` (no opcode 66, XZ-basis opcodes) and, if known,
    ``average_layers_per_clifford``, or None if no usable cache entry exists.
    """
    cached = rb_try_load(own_cache_dir(calibration_graph_dir), own_cache_key(key, basis))
    if (
        cached is not None
        and cached.get("single_qubit_basis") == basis
        and cached.get("cache_version") == CACHE_VERSION
        and _fits_basis(cached["circuits_as_ints"])
    ):
        return cached

    original = rb_try_load(original_cache_dir(calibration_graph_dir), key)
    if original is not None:
        try:
            stripped = strip_readout(original["circuits_as_ints"])
        except ValueError:
            return None
        if _fits_basis(stripped):
            converted = dict(original)
            converted["circuits_as_ints"] = stripped
            return converted
    return None


def save_circuits(calibration_graph_dir: Path, key: str, payload: dict, basis: str = SINGLE_QUBIT_BASIS) -> None:
    """Write to the OWN cache only. ``payload['circuits_as_ints']`` must not contain opcode 66."""
    stamped = dict(payload, single_qubit_basis=basis, cache_version=CACHE_VERSION)
    rb_save(own_cache_dir(calibration_graph_dir), own_cache_key(key, basis), stamped)


def store_reference(qubit_pair, alpha: float, error_per_clifford: float, basis: str = SINGLE_QUBIT_BASIS) -> None:
    """Standard RB: remember the decay of this pair together with the basis it was measured in."""
    qubit_pair.extras[REFERENCE_EXTRAS_KEY] = {
        "alpha": float(alpha),
        "error_per_clifford": float(error_per_clifford),
        "single_qubit_basis": basis,
    }


def get_reference_alpha(qubit_pair, basis: str = SINGLE_QUBIT_BASIS) -> float:
    """Interleaved RB: the stored standard-RB alpha of the SAME basis; raises if missing or from another basis."""
    if REFERENCE_EXTRAS_KEY not in qubit_pair.extras:
        raise ValueError(
            f"{qubit_pair.id}: no '{REFERENCE_EXTRAS_KEY}' reference in qubit_pair.extras. "
            "Run side_project/70k_two_qubit_standard_rb_smaller_gap first (70b's 'StandardRB' entry is not used)."
        )
    reference = qubit_pair.extras[REFERENCE_EXTRAS_KEY]
    if reference["single_qubit_basis"] != basis:
        raise ValueError(
            f"{qubit_pair.id}: reference was measured in basis {reference['single_qubit_basis']}, "
            f"but this run uses {basis}. Re-run the standard RB in the same basis."
        )
    return float(reference["alpha"])
