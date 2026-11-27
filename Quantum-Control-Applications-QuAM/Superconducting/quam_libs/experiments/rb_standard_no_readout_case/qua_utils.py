"""
QUA program for 2Q RB with gate-only packed circuits (no readout case in the gate switch).

Differences to ``quam_libs/experiments/rb_standard/qua_utils.py`` (used by 70b / 70c):

1. Gate-only packed circuits ``[n_circuits, len0, gates..., len1, gates..., ...]`` (see ``packing.py``). There is no
   readout marker (opcode 66) any more.
2. The inner loop is only ``switch_(gate, unsafe=True)`` over the gate opcodes 0..37 (compact layout, no analog y
   cases because the basis is XZ). ``readout_save_and_reset`` runs once per circuit, after its gate loop; empty
   circuits are still read out.
3. The CZ sits alone in its case: flux pulses (+ the phase-correcting frame rotations), then a global ``align()``
   and a global ``wait(4)`` on all elements. No XY pulse is ever played inside the CZ case. With
   ``cz_align_elements=False`` (default) the align calls inside the CZ macro (before / between flux and frame
   rotations) are not issued. With ``cz_align_elements=True`` the original ``CZGate.execute()`` (with its aligns) is
   used, followed by the same ``align()`` + ``wait(4)``.
4. Single-qubit basis: XZ (virtual Z) only.

Nothing in this module is used by 70b / 70c / 70f / 70g; the original helpers are only imported, never modified.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from qm.qua import *
from qm.qua._expressions import QuaVariable
from qualang_tools.units import unit
from qualibrate import QualibrationNode

from quam_libs.components import QuAM, TransmonPair
from quam_libs.macros import active_reset, active_reset_gef, align, assign, reset_frame

# Import only: original helpers that do not depend on the readout opcode.
from quam_libs.experiments.rb_standard.qua_utils import measure_two_qubit_state

from quam_libs.experiments.rb_standard_no_readout_case.packing import (
    PAD_OPCODE,
    build_packed_chunks,
    circuits_to_compact,
    pack_circuits,
    validate_single_array_budget,
)

# Clock cycles (4 ns each) waited on all elements at the end of the CZ case (QUA wait() counts cycles).
CZ_WAIT_CYCLES = 4

# compact action index -> ("play", pulse) | ("frame", angle in rad); same order as packing.py [sx, x, rz90, rz180, rz270]
_ACTIONS = [
    ("play", "x90"),
    ("play", "x180"),
    ("frame", np.pi / 2),
    ("frame", np.pi),
    ("frame", 3 * np.pi / 2),
]


def gate_layer_actions(opcode: int):
    """(control action, target action) of the compact opcodes 0..34; None = nothing on that qubit."""
    if opcode < 30:
        control, target = divmod(opcode, 6)
        return _ACTIONS[control], (_ACTIONS[target] if target < 5 else None)
    return None, _ACTIONS[opcode - 30]


def _apply_action(qubit, action, frame_wait_cycles: int = 0):
    if action is None:
        return
    kind, value = action
    if kind == "play":
        qubit.xy.play(value)
    else:
        qubit.xy.frame_rotation(value)
        if frame_wait_cycles > 0:
            # Workaround for pulse overlap with switch_(unsafe=True) + frame rotations: occupy the rotated element.
            qubit.xy.wait(frame_wait_cycles)


def play_cz(qubit_pair: TransmonPair, align_elements: bool = False):
    """
    CZ alone in its case (no XY pulse): flux pulses + frame rotations, then a global align() and a global
    wait(CZ_WAIT_CYCLES) on all elements.
    """
    gate = qubit_pair.gates["Cz"]
    if align_elements or type(gate).__name__ != "CZGate":
        gate.execute()  # original macro (with its aligns); also used for gate classes with extra compensation
    else:
        gate.qubit_control.z.play(gate.flux_pulse_control_label, validate=False)
        if gate.coupler_flux_pulse is not None:
            gate.coupler.play(gate.coupler_flux_pulse_label, validate=False)
        frame_rotation_2pi(gate.phase_shift_control, gate.qubit_control.xy.name)
        frame_rotation_2pi(gate.phase_shift_target, gate.qubit_target.xy.name)
    align()
    wait(CZ_WAIT_CYCLES)


def play_gate(
    gate: QuaVariable,
    qubit_pair: TransmonPair,
    cz_align_elements: bool = False,
    frame_wait_cycles: int = 0,
):
    """
    Play one 2Q gate layer: ``switch_(gate, unsafe=True)`` over the compact opcodes 0..37, nothing else.

    ``frame_wait_cycles`` > 0 adds ``xy.wait(frame_wait_cycles)`` (clock cycles) after every frame rotation of the
    single-qubit cases, on the rotated element only.
    """
    with switch_(gate, unsafe=True):
        for opcode in range(35):
            control_action, target_action = gate_layer_actions(opcode)
            with case_(opcode):
                _apply_action(qubit_pair.qubit_control, control_action, frame_wait_cycles)
                _apply_action(qubit_pair.qubit_target, target_action, frame_wait_cycles)
        with case_(35):  # idle
            qubit_pair.qubit_control.wait(20)
            qubit_pair.qubit_target.wait(20)
        with case_(36):  # CZ
            play_cz(qubit_pair, cz_align_elements)
        with case_(37):  # idle_2q
            qubit_pair.qubit_control.wait(qubit_pair.qubit_control.xy.operations["x180"].length // 4)
            qubit_pair.qubit_target.wait(qubit_pair.qubit_target.xy.operations["x180"].length // 4)


def readout_save_and_reset(
    node: QualibrationNode,
    qubit_pair: TransmonPair,
    state: QuaVariable,
    state_control: QuaVariable,
    state_target: QuaVariable,
    state_st,
    reset_type: Literal["thermal", "active", "active_gef"],
):
    """Once per circuit, after the gate loop (the body of the former case 66)."""
    align()
    wait(4)

    measure_two_qubit_state(node, qubit_pair, state, state_control, state_target)
    save(state, state_st)

    # Initialize the qubits
    if reset_type == "active":
        active_reset(qubit_pair.qubit_control, "readout")
        active_reset(qubit_pair.qubit_target, "readout")
    elif reset_type == "active_gef":
        active_reset_gef(qubit_pair.qubit_control, "readout")
        active_reset_gef(qubit_pair.qubit_target, "readout")
    else:
        qubit_pair.qubit_control.resonator.wait(qubit_pair.qubit_control.thermalization_time // 4)
        qubit_pair.qubit_target.resonator.wait(qubit_pair.qubit_target.thermalization_time // 4)
    # Reset the frame of the qubits in order not to accumulate rotations
    reset_frame(qubit_pair.qubit_control.xy.name, qubit_pair.qubit_target.xy.name)

    align()


class QuaProgramHandler:
    """
    Same constructor and public methods as ``rb_standard.qua_utils.QuaProgramHandler``.

    ``circuits_as_ints`` are the circuits as encoded by ``circuit_utils.circuit_to_layer_ints`` (local opcode layout),
    WITHOUT the readout opcode 66 (it raises if one is present). They are translated to the compact opcodes and
    packed here. Saved data layout (stream buffers) is identical to the original handler.
    """

    def __init__(
        self,
        node: QualibrationNode,
        num_pairs: int,
        circuits_as_ints: list[list[int]],
        machine: QuAM,
        qubit_pairs: list[TransmonPair],
        max_sequence_length: int = 6000,
    ):
        self.u = unit(coerce_to_integer=True)
        self.node = node
        self.num_pairs = num_pairs
        self.circuits_as_ints = circuits_as_ints
        self.machine = machine
        self.qubit_pairs = qubit_pairs
        self.max_sequence_length = max_sequence_length
        self.readout_mode = getattr(node.parameters, "readout_mode", "ge")
        self.reset_mode = getattr(node.parameters, "reset_type_thermal_or_active", "active")
        self.use_input_stream = getattr(node.parameters, "use_input_stream", False)
        self.max_chunk_ints = getattr(node.parameters, "max_chunk_ints", 15000)
        self.cz_align_elements = getattr(node.parameters, "cz_align_elements", False)
        self.frame_wait_cycles = getattr(node.parameters, "frame_rotation_wait_cycles", 0)
        if self.frame_wait_cycles and self.frame_wait_cycles < 4:
            raise ValueError("frame_rotation_wait_cycles must be 0 (off) or >= 4 clock cycles (QUA minimum wait).")

        self.compact_circuits = circuits_to_compact(circuits_as_ints)  # raises on opcode 66 / analog y gates
        self.total_circuits = len(self.compact_circuits)

        self.chunks_per_depth = None
        self.declared_size = None
        self.n_sub_chunks = None
        if self.use_input_stream:
            self.chunks_per_depth, self.declared_size = build_packed_chunks(
                self.compact_circuits,
                list(node.parameters.circuit_lengths),
                node.parameters.num_circuits_per_length,
                self.max_chunk_ints,
            )
            self.n_sub_chunks = sum(len(depth_chunks) for depth_chunks in self.chunks_per_depth)
        else:
            validate_single_array_budget(self.compact_circuits, self.max_chunk_ints)

    @property
    def total_pushes(self) -> int:
        if self.n_sub_chunks is None:
            raise RuntimeError("total_pushes is only defined when use_input_stream is True.")
        return self.n_sub_chunks * self.num_pairs

    # ------------------------------------------------------------------ shared pieces
    def _initialize_pair(self, qubit_pair: TransmonPair):
        self.machine.set_all_fluxes(
            flux_point=self.node.parameters.flux_point_joint_or_independent, target=qubit_pair.qubit_control
        )
        if self.reset_mode == "active":
            active_reset(qubit_pair.qubit_control, "readout")
            active_reset(qubit_pair.qubit_target, "readout")
        elif self.reset_mode == "active_gef":
            active_reset_gef(qubit_pair.qubit_control, "readout")
            active_reset_gef(qubit_pair.qubit_target, "readout")
        else:
            qubit_pair.qubit_control.resonator.wait(qubit_pair.qubit_control.thermalization_time * self.u.ns)
            qubit_pair.qubit_target.resonator.wait(qubit_pair.qubit_target.thermalization_time * self.u.ns)
        # Align the two elements to play the sequence after qubit initialization
        align()

    def _play_packed(self, packed, qubit_pair, state, state_control, state_target, state_st, idx, n_circ, c, length, k):
        """Dispatcher for one packed array: [n, len0, gates..., len1, gates..., ...]. Leaves ``idx`` after the last gate."""
        assign(idx, 0)
        assign(n_circ, packed[idx])
        assign(idx, idx + 1)
        with for_(c, 0, c < n_circ, c + 1):
            assign(length, packed[idx])
            assign(idx, idx + 1)
            with for_(k, 0, k < length, k + 1):
                play_gate(packed[idx], qubit_pair, self.cz_align_elements, self.frame_wait_cycles)
                assign(idx, idx + 1)
            readout_save_and_reset(self.node, qubit_pair, state, state_control, state_target, state_st, self.reset_mode)

    def _declare_loop_vars(self):
        return {name: declare(int) for name in ("n", "j", "c", "k", "idx", "n_circ", "length", "state", "state_control", "state_target")}

    # ------------------------------------------------------------------ input-stream path
    def _get_qua_program_with_input_stream(self):
        n_sub_chunks = self.n_sub_chunks
        with program() as rb:
            n_st = declare_stream()
            sequence = declare_input_stream(int, name="sequence", size=self.declared_size)
            state_st = [declare_stream() for _ in range(self.num_pairs)]
            push_counter = declare(int, value=0)

            for i, qubit_pair in enumerate(self.qubit_pairs):
                v = self._declare_loop_vars()
                self._initialize_pair(qubit_pair)

                # Chunk outer, shot inner: one advance (one host push) per chunk; all shots replay the chunk.
                with for_(v["j"], 0, v["j"] < n_sub_chunks, v["j"] + 1):
                    advance_input_stream(sequence)
                    assign(push_counter, push_counter + 1)
                    save(push_counter, n_st)

                    with for_(v["n"], 0, v["n"] < self.node.parameters.num_averages, v["n"] + 1):
                        self._play_packed(
                            sequence, qubit_pair, v["state"], v["state_control"], v["state_target"], state_st[i],
                            v["idx"], v["n_circ"], v["c"], v["length"], v["k"],
                        )

                align()  # align between pairs. No multiplexing support for the moment.

            with stream_processing():
                n_st.save("iteration")
                for i in range(len(self.qubit_pairs)):
                    state_st[i].buffer(self.node.parameters.num_circuits_per_length).buffer(
                        self.node.parameters.num_averages
                    ).buffer(len(self.node.parameters.circuit_lengths)).save(f"state{i + 1}")
        return rb

    def _padded_chunks(self) -> list[list[int]]:
        return [
            chunk + [PAD_OPCODE] * (self.declared_size - len(chunk))
            for depth_chunks in self.chunks_per_depth
            for chunk in depth_chunks
        ]

    def get_all_padded_chunks_for_all_pairs(self) -> list[list[int]]:
        if not self.use_input_stream:
            raise RuntimeError("get_all_padded_chunks_for_all_pairs called but use_input_stream is False.")
        return self._padded_chunks() * len(self.qubit_pairs)

    def push_all_chunks(self, job) -> None:
        for chunk in self.get_all_padded_chunks_for_all_pairs():
            job.push_to_input_stream("sequence", chunk)

    # ------------------------------------------------------------------ single-array path
    def _get_qua_program_without_input_stream(self):
        packed = pack_circuits(self.compact_circuits)
        with program() as rb:
            n_st = declare_stream()
            packed_qua = declare(int, value=packed)
            state_st = [declare_stream() for _ in range(self.num_pairs)]

            for i, qubit_pair in enumerate(self.qubit_pairs):
                v = self._declare_loop_vars()
                self._initialize_pair(qubit_pair)

                with for_(v["n"], 0, v["n"] < self.node.parameters.num_averages, v["n"] + 1):
                    self._play_packed(
                        packed_qua, qubit_pair, v["state"], v["state_control"], v["state_target"], state_st[i],
                        v["idx"], v["n_circ"], v["c"], v["length"], v["k"],
                    )
                    save(v["n"], n_st)

                align()  # align between pairs. No multiplexing support for the moment.

            with stream_processing():
                n_st.save("iteration")
                for i in range(len(self.qubit_pairs)):
                    state_st[i].buffer(self.node.parameters.num_circuits_per_length).buffer(
                        len(self.node.parameters.circuit_lengths)
                    ).buffer(self.node.parameters.num_averages).save(f"state{i + 1}")
        return rb

    def get_qua_program(self):
        if self.use_input_stream:
            return self._get_qua_program_with_input_stream()
        return self._get_qua_program_without_input_stream()
