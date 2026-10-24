"""
Convert an existing CZ operation of a qubit pair into a 1 ns composed_CZGate (one amplitude per ns).

    - What will happen:
        1. The qubit / coupler flux waveforms of `operation` are sampled at 1 ns (exactly the samples sent to the OPX).
        2. A composed_CZGate with one_ns_mode=True is built so that its execute() reproduces that waveform:
           base amplitude = the largest-magnitude sample of each line, extras['qb_amps'] / extras['cp_amps'] = sample / base.
        3. The phase corrections of `operation` are copied over (re-calibrate them, the gate timing is not guaranteed identical).
        4. The gate is stored as qp.gates[<gate_name>] (default "composed_cz") and, if save=True, the state is saved.

    - Notes:
        * Sample j of a line = base amplitude * scale_j. With the DGX notebook convention scale = 2 * action,
          action_j = extras['qb_amps'][j] / 2 reproduces the source waveform.
        * amp() limits |base * scale| / basis_amplitude < 2; for the notebook range |scale| <= 2 this means |base| < basis_amplitude.
"""
import warnings

import numpy as np
from quam.components.pulses import SquarePulse

from quam_libs.components import QuAM
from quam_libs.components.gates import composed_CZGate
# %%

def _sample_pulse(pulse) -> np.ndarray:
    """Real 1 ns samples of a flux pulse (constant pulses are expanded to their length)."""
    waveform = np.asarray(pulse.calculate_waveform())
    if np.iscomplexobj(waveform):
        if np.any(np.abs(waveform.imag) > 1e-12):
            raise ValueError(f"Pulse {pulse.get_reference()} is complex; a flux line needs a real waveform")
        waveform = waveform.real
    if waveform.ndim == 0:
        waveform = np.full(int(pulse.length), float(waveform))
    return waveform.astype(float)


def _base_and_scales(samples: np.ndarray, line: str) -> tuple:
    """base = largest-magnitude sample (keeps its sign), scales = samples / base."""
    peak_idx = int(np.argmax(np.abs(samples)))
    base = float(samples[peak_idx])
    if base == 0.0:
        warnings.warn(f"{line} waveform is all zeros; its base amplitude is set to 0 and it will output nothing")
        return 0.0, [0.0] * len(samples)
    return base, (samples / base).tolist()


def compose_cz_from_operation(
    machine: QuAM,
    qubit_pair: str,
    operation: str,
    gate_name: str = "composed_cz",
    basis_amplitude: float = 0.25,
    strict_timing: bool = True,
    overwrite: bool = True,
    save: bool = False,
    save_path=None,
) -> composed_CZGate:
    """Build qp.gates[gate_name] as a 1 ns composed_CZGate reproducing qp.gates[operation].

    Args:
        machine: loaded QuAM.
        qubit_pair: e.g. "coupler_q4_q5".
        operation: source CZ operation in qp.gates, e.g. "Cz_flattop_gaussian" or "Cz".
        gate_name: name of the new gate in qp.gates.
        basis_amplitude: V, non-zero sample of the 4 ns basis waveforms.
        strict_timing: compile the flux plays with strict_timing_ (recommended).
        overwrite: replace an existing qp.gates[gate_name].
        save: call machine.save(save_path) after adding the gate.
        save_path: where to save; None uses the active QuAM state path.
    """
    qp = machine.qubit_pairs[qubit_pair]
    if operation not in qp.gates:
        raise KeyError(f"'{operation}' not in {qubit_pair}.gates: {list(qp.gates)}")
    if gate_name in qp.gates and not overwrite:
        raise KeyError(f"{qubit_pair}.gates['{gate_name}'] already exists (use overwrite=True)")
    source = qp.gates[operation]

    q_samples = _sample_pulse(source.flux_pulse_control)
    c_samples = _sample_pulse(source.coupler_flux_pulse) if getattr(source, "coupler_flux_pulse", None) is not None else None

    # Both lines start together in the source gate; pad to a common length that is a multiple of 4 ns.
    lengths = [len(q_samples)] + ([len(c_samples)] if c_samples is not None else [])
    tot_len = int(np.ceil(max(lengths) / 4) * 4)
    q_samples = np.pad(q_samples, (0, tot_len - len(q_samples)))
    if c_samples is not None:
        c_samples = np.pad(c_samples, (0, tot_len - len(c_samples)))

    q_base, q_scales = _base_and_scales(q_samples, f"{operation} qubit")
    if c_samples is not None:
        c_base, c_scales = _base_and_scales(c_samples, f"{operation} coupler")
    for line, base in (("qubit", q_base), ("coupler", c_base if c_samples is not None else 0.0)):
        if abs(base) >= basis_amplitude:
            warnings.warn(
                f"|{line} base| = {abs(base):.4f} V >= basis_amplitude {basis_amplitude} V: "
                "scales up to 2 (DGX notebook range) would exceed the amp() range"
            )

    q_id = source.flux_pulse_control.id or "flux_pulse_control"
    gate = composed_CZGate(
        flux_pulse_control=SquarePulse(length=4, amplitude=q_base, id=q_id),
        coupler_flux_pulse=(
            SquarePulse(length=4, amplitude=c_base, id=source.coupler_flux_pulse.id or "coupler_flux_pulse")
            if c_samples is not None else None
        ),
        tot_len=tot_len,
        one_ns_mode=True,
        basis_amplitude=basis_amplitude,
        strict_timing=strict_timing,
        phase_shift_control=float(getattr(source, "phase_shift_control", 0.0)),
        phase_shift_target=float(getattr(source, "phase_shift_target", 0.0)),
        extras={
            "seg_num": tot_len,
            "qb_amps": q_scales,
            "cp_amps": c_scales if c_samples is not None else [0.0] * tot_len,
            "source_operation": operation,
        },
    )
    if gate_name in qp.gates:
        del qp.gates[gate_name]
    qp.gates[gate_name] = gate

    if save:
        machine.save(save_path)
    return qp.gates[gate_name]


# %%
if __name__ == "__main__":
    target_qubit_pair: str = "coupler_q4_q5"
    source_operation: str = "Cz_flattop_gaussian"

    machine = QuAM.load()
    gate = compose_cz_from_operation(machine, target_qubit_pair, source_operation, save=True)
    print(
        f"{target_qubit_pair}.gates['composed_cz'] <- {source_operation}: tot_len={gate.tot_len} ns, "
        f"q base {gate.flux_pulse_control.amplitude:+.6f} V, "
        f"c base {gate.coupler_flux_pulse.amplitude if gate.coupler_flux_pulse else 0:+.6f} V"
    )
