"""
Packed gate-only circuits for 2Q RB (host side, no QUA).

Data layout:

    packed = [n_circuits, len0, g, g, ..., len1, g, g, ..., ...]

* Circuits contain gate opcodes only; there is NO readout marker. The readout/save/reset block runs once per
  circuit, after its gate loop, so an empty circuit (len 0) is still read out.
* A circuit is never split between two input-stream chunks; every chunk is a self-contained packed array.

Opcodes used on the QUA side (compact layout; basis XZ = virtual Z):

    0..29 : control action * 6 + target action,   actions = [sx, x, rz90, rz180, rz270] (+ "none" for the target)
    30..34: target-only action (control does nothing), same order
    35    : idle (wait 20 cycles on both qubits)
    36    : CZ
    37    : idle_2q

The calibration nodes still encode circuits with ``circuit_utils.circuit_to_layer_ints`` (local layout:
``control * 8 + target`` with sx 0, x 1, sy 2, y 3, rz90 4, rz180 5, rz270 6, none 7; 64 CZ, 65 idle_2q). That layout is
translated to the compact one here. Analog y gates (local ids 2 and 3) are not part of the XZ basis and are rejected.
"""
from __future__ import annotations

from typing import Iterable

SINGLE_QUBIT_BASIS = "XZ"  # only the virtual-Z basis is implemented (basis gates rz, sx, x, cz)

LOCAL_READOUT_OPCODE = 66
COMPACT_IDLE = 35
COMPACT_CZ = 36
COMPACT_IDLE_2Q = 37
N_COMPACT_OPCODES = 38
PAD_OPCODE = COMPACT_IDLE  # harmless no-op used to pad input-stream chunks

# local single-qubit gate id -> index into the compact action list [sx, x, rz90, rz180, rz270]
_LOCAL_TO_ACTION = {0: 0, 1: 1, 4: 2, 5: 3, 6: 4}
_LOCAL_NONE = 7
_N_ACTIONS = 5


def local_to_compact(opcode: int) -> int:
    """Translate one opcode of ``circuit_utils`` (local layout) to the compact opcode of the QUA switch."""
    if opcode == LOCAL_READOUT_OPCODE:
        raise ValueError("The readout opcode 66 is not allowed: circuits must be gate-only for the packed format.")
    if opcode == 64:
        return COMPACT_CZ
    if opcode == 65:
        return COMPACT_IDLE_2Q
    if opcode == 63:
        return COMPACT_IDLE
    if not 0 <= opcode < 63:
        raise ValueError(f"Unknown opcode {opcode}")
    control, target = divmod(opcode, 8)
    for gate in (control, target):
        if gate != _LOCAL_NONE and gate not in _LOCAL_TO_ACTION:
            raise ValueError(
                f"Opcode {opcode} uses the analog y gate (local id {gate}), which is not part of the XZ basis "
                "(basis_gates must be ['rz', 'sx', 'x', 'cz'])."
            )
    if control == _LOCAL_NONE:
        return 30 + _LOCAL_TO_ACTION[target]
    target_index = _N_ACTIONS if target == _LOCAL_NONE else _LOCAL_TO_ACTION[target]
    return _LOCAL_TO_ACTION[control] * 6 + target_index


def circuits_to_compact(circuits_as_ints: Iterable[Iterable[int]]) -> list[list[int]]:
    return [[local_to_compact(op) for op in circuit] for circuit in circuits_as_ints]


def pack_circuits(circuits: list[list[int]]) -> list[int]:
    """[n_circuits, len0, gates..., len1, gates..., ...]"""
    packed = [len(circuits)]
    for circuit in circuits:
        packed.append(len(circuit))
        packed.extend(circuit)
    return packed


def unpack_circuits(packed: list[int]) -> list[list[int]]:
    """Inverse of ``pack_circuits`` (ignores padding after the last circuit)."""
    n, idx, circuits = packed[0], 1, []
    for _ in range(n):
        length = packed[idx]
        circuits.append(list(packed[idx + 1 : idx + 1 + length]))
        idx += 1 + length
    return circuits


def packed_size(circuits: list[list[int]]) -> int:
    return 1 + sum(1 + len(c) for c in circuits)


def build_packed_chunks(
    circuits: list[list[int]],
    circuit_depths: list[int],
    num_circuits_per_length: int,
    max_chunk_ints: int,
) -> tuple[list[list[list[int]]], int]:
    """
    Pack circuits into self-contained chunks for the input-stream path.

    Circuits are grouped strictly by depth (a chunk never mixes two depths) and greedily filled up to
    ``max_chunk_ints`` ints (header included); a circuit is never split.

    Returns ``(chunks_per_depth, declared_size)`` where every chunk is a packed array and ``declared_size`` is the
    length of the longest chunk (size of the input stream array; shorter chunks are padded with ``PAD_OPCODE``).
    """
    expected = len(circuit_depths) * num_circuits_per_length
    if len(circuits) != expected:
        raise ValueError(
            f"{len(circuits)} circuits do not match circuit_lengths x num_circuits_per_length "
            f"({len(circuit_depths)} x {num_circuits_per_length} = {expected})."
        )
    chunks_per_depth: list[list[list[int]]] = []
    declared_size = 0
    for depth_idx, depth in enumerate(circuit_depths):
        start = depth_idx * num_circuits_per_length
        sub_chunks: list[list[int]] = []
        current: list[list[int]] = []
        for circ_idx, circuit in enumerate(circuits[start : start + num_circuits_per_length]):
            if packed_size([circuit]) > max_chunk_ints:
                raise ValueError(
                    f"Single circuit too large for an input-stream chunk: depth={depth} Cliffords, "
                    f"circuit_index={circ_idx}, packed ints={packed_size([circuit])}, max_chunk_ints={max_chunk_ints}."
                )
            if current and packed_size(current + [circuit]) > max_chunk_ints:
                sub_chunks.append(pack_circuits(current))
                current = []
            current.append(circuit)
        if current:
            sub_chunks.append(pack_circuits(current))
        declared_size = max([declared_size] + [len(c) for c in sub_chunks])
        chunks_per_depth.append(sub_chunks)
    return chunks_per_depth, declared_size


def validate_single_array_budget(circuits: list[list[int]], max_chunk_ints: int) -> None:
    """Fail fast (before QUA compilation) if the one packed array would exceed the OPX variable budget."""
    size = packed_size(circuits)
    if size > max_chunk_ints:
        raise ValueError(
            f"Packed RB sequence has {size} ints, which exceeds max_chunk_ints={max_chunk_ints} of the "
            "non-input-stream path. Set use_input_stream=True, reduce circuit_lengths, or reduce num_circuits_per_length."
        )
