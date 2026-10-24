"""
Export a calibrated composed_CZGate of a qubit pair as a plain CZGate named "Cz_Custom" (one WaveformPulse per line).

    - What will happen:
        1. The flux waveform the composed gate plays is rebuilt sample by sample:
           1 ns mode:    sample j = flux_pulse_control.amplitude * extras['qb_amps'][j]   (coupler: amplitude * extras['cp_amps'][j])
           segment mode: segment k = the segment pulse samples * extras['qb_amps'][k]      (same for the coupler)
        2. qp.gates["Cz_Custom"] = CZGate with WaveformPulse(waveform_I=samples) on q_control.z and the coupler, the pulse ids
           and the phase corrections of the composed gate. An existing Cz_Custom is overwritten.
        3. The generated QUA config is checked: its Cz_Custom samples must equal the rebuilt waveform.
        4. If save=True, the state is saved. qp.gates["Cz"] is never changed (point it to Cz_Custom by hand if wanted).

    - Notes:
        * Cz_Custom plays like any CZGate: execute(), execute_dgx(), execute_dgx_vRatis() (amplitude_scale scales the whole
          waveform; keep cz_duration=None, an arbitrary waveform cannot be compressed).
        * A WaveformPulse has no amplitude / flat_length: nodes that read gates[...].flux_pulse_control.amplitude
          (32a, 33a, 33b, 64c, ...) do not work on Cz_Custom.
        * Samples are in V relative to the port offset, as the amplitudes of the other CZ pulses.
"""
import numpy as np
from quam.components.pulses import WaveformPulse

from quam_libs.components import QuAM
from quam_libs.components.gates import CZGate, composed_CZGate

CUSTOM_GATE_NAME = "Cz_Custom"
# %%


def _plain(value):
    """QuAM containers (QuamDict / QuamList, which keep a parent) -> plain python objects."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "items"):
        return {k: _plain(v) for k, v in value.items()}
    if hasattr(value, "__iter__"):
        return [_plain(v) for v in value]
    return value


def _segment_samples(pulse) -> np.ndarray:
    """Real 1 ns samples of one segment pulse (constant pulses are expanded to their length)."""
    waveform = np.asarray(pulse.calculate_waveform())
    if np.iscomplexobj(waveform):
        waveform = waveform.real
    if waveform.ndim == 0:
        waveform = np.full(int(pulse.length), float(waveform))
    return waveform.astype(float)


def composed_samples(gate: composed_CZGate) -> tuple:
    """(q samples, c samples or None) in V, as gate.execute() plays them with amplitude_scale = 1."""
    n = gate.n_amplitudes
    q_scales = np.asarray(gate.extras.get("qb_amps", [1.0] * n), float)
    c_scales = np.asarray(gate.extras.get("cp_amps", [1.0] * n), float)
    if len(q_scales) != n or len(c_scales) != n:
        raise ValueError(f"qb_amps / cp_amps have {len(q_scales)} / {len(c_scales)} entries, expected {n}")
    if gate.one_ns_mode:
        q = float(gate.flux_pulse_control.amplitude) * q_scales
        c = float(gate.coupler_flux_pulse.amplitude) * c_scales if gate.coupler_flux_pulse is not None else None
    else:
        q = np.concatenate([_segment_samples(gate.flux_pulse_control) * s for s in q_scales])
        c = (np.concatenate([_segment_samples(gate.coupler_flux_pulse) * s for s in c_scales])
             if gate.coupler_flux_pulse is not None else None)
    return q, c


def _config_samples(config: dict, element: str, operation: str) -> np.ndarray:
    pulse = config["pulses"][config["elements"][element]["operations"][operation]]
    wf = config["waveforms"][pulse["waveforms"]["single"]]
    return np.asarray(wf["samples"], float) if wf["type"] == "arbitrary" else np.full(int(pulse["length"]), float(wf["sample"]))


def export_composed_to_custom(
    machine: QuAM,
    qubit_pair: str,
    composed_gate: str,
    save: bool = False,
    save_path=None,
) -> CZGate:
    """Build qp.gates["Cz_Custom"] as a CZGate with WaveformPulse lines reproducing qp.gates[composed_gate].

    Args:
        machine: loaded QuAM.
        qubit_pair: e.g. "coupler_q4_q5".
        composed_gate: source composed_CZGate in qp.gates, e.g. "composed_cz_dflat32".
        save: call machine.save(save_path) after adding the gate.
        save_path: where to save; None uses the active QuAM state path.
    """
    qp = machine.qubit_pairs[qubit_pair]
    if composed_gate not in qp.gates:
        raise KeyError(f"'{composed_gate}' not in {qubit_pair}.gates: {list(qp.gates)}")
    source = qp.gates[composed_gate]
    if not isinstance(source, composed_CZGate):
        raise TypeError(f"{qubit_pair}.gates['{composed_gate}'] is {type(source).__name__}, not a composed_CZGate")

    q, c = composed_samples(source)
    if len(q) % 4 or len(q) < 16 or (c is not None and len(c) != len(q)):
        raise ValueError(f"waveform length {len(q)} (coupler {None if c is None else len(c)}): need equal lengths, a multiple of 4, >= 16")

    gate = CZGate(
        flux_pulse_control=WaveformPulse(id=source.flux_pulse_control.id or "flux_pulse_control",
                                         waveform_I=[float(x) for x in q]),
        coupler_flux_pulse=(WaveformPulse(id=source.coupler_flux_pulse.id or "coupler_flux_pulse",
                                          waveform_I=[float(x) for x in c]) if c is not None else None),
        phase_shift_control=float(source.phase_shift_control),
        phase_shift_target=float(source.phase_shift_target),
        extras={"source_gate": composed_gate, **{k: _plain(v) for k, v in source.extras.items() if k != "seg_num"}},
    )
    if CUSTOM_GATE_NAME in qp.gates:
        del qp.gates[CUSTOM_GATE_NAME]
    qp.gates[CUSTOM_GATE_NAME] = gate

    # The generated QUA config must play exactly these samples
    config = machine.generate_config()
    gate = qp.gates[CUSTOM_GATE_NAME]
    if not np.allclose(_config_samples(config, qp.qubit_control.z.name, gate.flux_pulse_control_label), q, rtol=0, atol=1e-12):
        raise RuntimeError(f"config samples of {gate.flux_pulse_control_label} differ from the composed waveform")
    if c is not None and not np.allclose(_config_samples(config, qp.coupler.name, gate.coupler_flux_pulse_label), c, rtol=0, atol=1e-12):
        raise RuntimeError(f"config samples of {gate.coupler_flux_pulse_label} differ from the composed waveform")

    if save:
        machine.save(save_path)
    return gate


# %%
if __name__ == "__main__":
    target_qubit_pair: str = "coupler_q4_q5"
    source_composed_gate: str = "composed_cz_dflat32_rb211"

    machine = QuAM.load()
    gate = export_composed_to_custom(machine, target_qubit_pair, source_composed_gate, save=True)
    q = np.asarray(gate.flux_pulse_control.waveform_I)
    c = np.asarray(gate.coupler_flux_pulse.waveform_I) if gate.coupler_flux_pulse is not None else np.zeros(0)
    print(
        f"{target_qubit_pair}.gates['{CUSTOM_GATE_NAME}'] <- {source_composed_gate}: {len(q)} ns, "
        f"q {q.min():+.6f} .. {q.max():+.6f} V, c {c.min() if len(c) else 0:+.6f} .. {c.max() if len(c) else 0:+.6f} V, "
        f"phases {gate.phase_shift_control:.6f} / {gate.phase_shift_target:.6f} (config checked; Cz unchanged)"
    )
