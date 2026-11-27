"""
QUA program for 2Q RB with gate-only packed circuits, every circuit's gate loop starting at index 0 of a buffer.

The compiler uses its fast path for ``switch_(unsafe=True)`` only when the array index is the loop variable of a
``for_`` starting at the constant 0 (the loop end may be a runtime variable). In simulation ``for_(i, 0, i < length)``
over ``switch_(buf[i])`` gives 0 ns between two single-qubit layers and 16 ns per frame-only layer, against 20 ns and
56 ns when the loop starts at any other index of the packed array. Each circuit's gates are therefore copied from the
packed array into ``buf`` before its gate loop (qubits idle in the ground state after the reset, nothing is played
during the copy).

The CZ case ends with ``wait(CZ_WAIT_CYCLES)`` on the control and target xy only. Without that xy wait the next
layer's XY pulse starts 16 ns before the end of the CZ flux pulse (simulated, unsafe switch with this fast gate loop).
The macro's aligns are still issued when ``cz_align_elements`` is True. Single-qubit cases, frame waits, readout /
reset and the saved data come from ``rb_standard_no_readout_case.qua_utils``.

Used by side_project/70k_two_qubit_standard_rb_smaller_gap.py and 70l_two_qubit_interleaved_rb_smaller_gap.py.
"""
from __future__ import annotations

from qm.qua import *
from qm.qua._expressions import QuaVariable

from quam_libs.components import TransmonPair
from quam_libs.macros import assign

from quam_libs.experiments.rb_standard_no_readout_case.qua_utils import (
    CZ_WAIT_CYCLES,
    QuaProgramHandler as _BaseQuaProgramHandler,
    _apply_action,
    gate_layer_actions,
    readout_save_and_reset,
)


def play_cz(qubit_pair: TransmonPair, align_elements: bool = False):
    """CZ alone in its case: flux pulses + frame rotations, then wait(CZ_WAIT_CYCLES) on the control and target xy."""
    gate = qubit_pair.gates["Cz"]
    if align_elements or type(gate).__name__ != "CZGate":
        gate.execute()  # original macro (with its aligns); also used for gate classes with extra compensation
    else:
        gate.qubit_control.z.play(gate.flux_pulse_control_label, validate=False)
        if gate.coupler_flux_pulse is not None:
            gate.coupler.play(gate.coupler_flux_pulse_label, validate=False)
        frame_rotation_2pi(gate.phase_shift_control, gate.qubit_control.xy.name)
        frame_rotation_2pi(gate.phase_shift_target, gate.qubit_target.xy.name)
    qubit_pair.qubit_control.xy.wait(CZ_WAIT_CYCLES)
    qubit_pair.qubit_target.xy.wait(CZ_WAIT_CYCLES)


def play_gate(gate: QuaVariable, qubit_pair: TransmonPair, cz_align_elements: bool = False, frame_wait_cycles: int = 0):
    """One 2Q gate layer: ``switch_(gate, unsafe=True)`` over the compact opcodes 0..37 (CZ case above)."""
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


class QuaProgramHandler(_BaseQuaProgramHandler):
    """Same constructor, public methods and saved data as the base handler; the gate loop and CZ case differ."""

    def _declare_loop_vars(self):
        v = super()._declare_loop_vars()
        self._buf = declare(int, size=max([len(c) for c in self.compact_circuits] + [1]))
        self._gate = declare(int)
        return v

    def _play_packed(self, packed, qubit_pair, state, state_control, state_target, state_st, idx, n_circ, c, length, k):
        """Dispatcher for one packed array: [n, len0, gates..., len1, gates..., ...]. Leaves ``idx`` after the last gate."""
        buf, gate = self._buf, self._gate
        assign(idx, 0)
        assign(n_circ, packed[idx])
        assign(idx, idx + 1)
        with for_(c, 0, c < n_circ, c + 1):
            assign(length, packed[idx])
            assign(idx, idx + 1)
            with for_(k, 0, k < length, k + 1):
                assign(buf[k], packed[idx])
                assign(idx, idx + 1)
            with for_(gate, 0, gate < length, gate + 1):
                play_gate(buf[gate], qubit_pair, self.cz_align_elements, self.frame_wait_cycles)
            readout_save_and_reset(self.node, qubit_pair, state, state_control, state_target, state_st, self.reset_mode)
