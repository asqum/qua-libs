from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Dict, Any
from dataclasses import field
from copy import copy

from qm.qua import align, amp, declare, fixed, frame_rotation_2pi, play, strict_timing_

from quam.components.pulses import Pulse
from quam.core import quam_dataclass, QuamComponent
from quam.utils import string_reference as str_ref


__all__ = ["TwoQubitGate", "CZGate", "CZWithCompensationGate", "CNotGate_TC", "CNotGate_CT", "composed_CZGate"]


@quam_dataclass
class TwoQubitGate(QuamComponent, ABC):
    
    extras: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def transmon_pair(self):
        from ..transmon_pair import TransmonPair

        if isinstance(self.parent, TransmonPair):
            return self.parent
        elif hasattr(self.parent, "parent") and isinstance(self.parent.parent, TransmonPair):
            return self.parent.parent
        else:
            raise AttributeError("TwoQubitGate is not attached to a QubitPair. 2Q_gate: {self}")

    @property
    def qubit_control(self):
        return self.transmon_pair.qubit_control

    @property
    def qubit_target(self):
        return self.transmon_pair.qubit_target
    
    @property
    def coupler(self):
        return self.transmon_pair.coupler    

    def __call__(self):
        self.execute()


@quam_dataclass
class CZGate(TwoQubitGate):
    """CZ Operation for a qubit pair"""

    # Pulses will be added to qubit elements
    # The reason we don't add "flux_to_q1" directly to q1.z is because it is part of
    # the CZ operation, i.e. it is only applied as part of a CZ operation

    flux_pulse_control: Pulse
    coupler_flux_pulse: Pulse = None
    
    pre_wait: int = 4

    phase_shift_control: float = 0.0
    phase_shift_target: float = 0.0

    @property
    def gate_label(self) -> str:
        try:
            return self.parent.get_attr_name(self)
        except AttributeError:
            return "CZ"

    @property
    def flux_pulse_control_label(self) -> str:
        if self.flux_pulse_control.id is not None:
            pulse_label = self.flux_pulse_control.id
        else:
            pulse_label = "flux_pulse_control"

        return f"{self.gate_label}{str_ref.DELIMITER}{pulse_label}"

    @property
    def coupler_flux_pulse_label(self) -> str:
        if self.coupler_flux_pulse.id is not None:
            pulse_label = self.coupler_flux_pulse.id
        else:
            pulse_label = "coupler_flux_pulse"

        return f"{self.gate_label}{str_ref.DELIMITER}{pulse_label}"

    def execute(self, amplitude_scale=None, coupler_amplitude_scale= None):        
        self.transmon_pair.align()
        
        # self.qubit_control.xy.wait(self.pre_wait)
        # self.qubit_target.xy.wait(self.pre_wait)
        
        self.qubit_control.z.play(
            self.flux_pulse_control_label,
            validate=False,
            amplitude_scale=amplitude_scale,
            #duration=self.flux_pulse_control.length//4
        )
        
        if self.coupler_flux_pulse is not None:
            self.coupler.play(
                self.coupler_flux_pulse_label,
                validate=False,
                amplitude_scale=coupler_amplitude_scale,
                #duration=self.coupler_flux_pulse.length//4
            )
        
        self.transmon_pair.align()
        frame_rotation_2pi(self.phase_shift_control, self.qubit_control.xy.name)
        frame_rotation_2pi(self.phase_shift_target, self.qubit_target.xy.name)
        self.transmon_pair.align()
        
    def execute_dgx(self, amplitude_scale=None, phase_shift_control = None, phase_shift_target =None):        
        if phase_shift_control is None:
            phase_shift_control = self.phase_shift_control
        if phase_shift_target is None:
            phase_shift_target = self.phase_shift_target

        self.transmon_pair.align()
        
        # self.qubit_control.xy.wait(self.pre_wait)
        # self.qubit_target.xy.wait(self.pre_wait)
        
        self.qubit_control.z.play(
            self.flux_pulse_control_label,
            validate=False,
            amplitude_scale=amplitude_scale,
        )
        
        if self.coupler_flux_pulse is not None:
            self.coupler.play(
                self.coupler_flux_pulse_label,
                validate=False
            )
        
        self.transmon_pair.align()
        frame_rotation_2pi(phase_shift_control, self.qubit_control.xy.name)
        frame_rotation_2pi(phase_shift_target, self.qubit_target.xy.name)
        self.transmon_pair.align()
    
    def execute_dgx_vRatis(self, qubit_flux_scale=None, phase_shift_control = None, phase_shift_target =None, coupler_flux_scale=None, cz_duration=None, amp_N:int=1):        
        if phase_shift_control is None:
            phase_shift_control = self.phase_shift_control
        if phase_shift_target is None:
            phase_shift_target = self.phase_shift_target
        if qubit_flux_scale is None:
            qubit_flux_scale = 1.0
        if coupler_flux_scale is None:
            coupler_flux_scale = 1.0


        if amp_N %2 != 1: # avoid a even amp_N was given 
            amp_N = 1

        self.transmon_pair.align()
        
        # self.qubit_control.xy.wait(self.pre_wait)
        # self.qubit_target.xy.wait(self.pre_wait)
        for _ in range(amp_N):
            self.qubit_control.z.play(
                self.flux_pulse_control_label,
                duration = cz_duration,
                validate=False,
                amplitude_scale=qubit_flux_scale,
            )
            
            if self.coupler_flux_pulse is not None:
                self.coupler.play(
                    self.coupler_flux_pulse_label,
                    validate=False,
                    duration=cz_duration,
                    amplitude_scale=coupler_flux_scale,
                )
        
        self.transmon_pair.align()

        for _ in range(amp_N):
            frame_rotation_2pi(phase_shift_control, self.qubit_control.xy.name)
            frame_rotation_2pi(phase_shift_target, self.qubit_target.xy.name)
        self.transmon_pair.align()


    @property
    def config_settings(self):
        return {"after": [self.qubit_control.z]}

    def apply_to_config(self, config: dict) -> None:
        pulse = copy(self.flux_pulse_control)
        pulse.id = self.flux_pulse_control_label
        pulse.parent = None  # Reset parent so it can be attached to new parent
        pulse.parent = self.qubit_control.z

        if self.flux_pulse_control_label in self.qubit_control.z.operations:
            raise ValueError(
                "Pulse name already exists in pulse operations. "
                f"Channel: {self.qubit_control.z.get_reference()}, "
                f"Pulse: {self.flux_pulse_control.get_reference()}, "
                f"Pulse name: {self.flux_pulse_control_label}"
            )

        pulse.apply_to_config(config)

        element_config = config["elements"][self.qubit_control.z.name]
        element_config["operations"][self.flux_pulse_control_label] = pulse.pulse_name

        if self.coupler_flux_pulse is not None:
            pulse = copy(self.coupler_flux_pulse)
            pulse.id = self.coupler_flux_pulse_label
            pulse.parent = None  # Reset parent so it can be attached to new parent
            pulse.parent = self.coupler

            pulse.apply_to_config(config)

        element_config_coupler = config["elements"][self.coupler.name]
        element_config_coupler["operations"][self.coupler_flux_pulse_label] = pulse.pulse_name

@quam_dataclass
class CZWithCompensationGate(CZGate):
    compensations : list  = None # Extra qubits that are not part of the gate but are shifted when the gate to be executed. Expected format: [{"qubit": qubit_object,"shift" : float, "phase" : float}]

    def execute(self, *args, **kwargs):
        if not self.compensations:
            super().execute(*args, **kwargs)
            return

        self.transmon_pair.align()
                 
        pulse_duration = self.flux_pulse_control.length // 4 + 10
        
        for compensation in self.compensations:
            qubit = compensation["qubit"]
            # Assume amplitude is 100 mV by default
            qubit.z.play(f"const", duration=pulse_duration, 
                         amplitude_scale=compensation["shift"] / qubit.z.operations["const"].amplitude)
            frame_rotation_2pi(compensation["phase"], qubit.xy.name)
            qubit.xy.play("x180", amplitude_scale=0.0, duration=4)
        
        super().execute(*args, **kwargs)

        self.transmon_pair.align()

@quam_dataclass
class CNotGate_CT(TwoQubitGate):
    
    def execute(self):
        self.qubit_target.gates["Hadamard"]()
        self.qubit_pair.gates["Cz"]()
        self.qubit_target.gates["Hadamard"]()
        
@quam_dataclass
class CNotGate_TC(TwoQubitGate):
    def execute(self):
        self.qubit_control.gates["Hadamard"]()
        self.qubit_control.align(self.qubit_target)
        self.qubit_pair.gates["Cz"]()
        self.qubit_control.align(self.qubit_target)
        self.qubit_control.gates["Hadamard"]()
        
        
@quam_dataclass
class SWAP_Coupler_Gate(TwoQubitGate):
    """SWAP Operation for a qubit pair"""

    # Pulses will be added to qubit elements
    # The reason we don't add "flux_to_q1" directly to q1.z is because it is part of
    # the CZ operation, i.e. it is only applied as part of a CZ operation

    flux_pulse_control: Pulse
    coupler_pulse_control: Pulse

    phase_shift_control: float = 0.0
    phase_shift_target: float = 0.0    
    pre_wait: int = 10
    post_wait: int = 20

    @property
    def gate_label(self) -> str:
        try:
            return self.parent.get_attr_name(self)
        except AttributeError:
            return "SWAP_Coupler_Gate"

    @property
    def flux_pulse_control_label(self) -> str:
        if self.flux_pulse_control.id is not None:
            pulse_label = self.flux_pulse_control.id
        else:
            pulse_label = "flux_pulse_control"

        return f"{self.gate_label}{str_ref.DELIMITER}{pulse_label}"

    @property
    def coupler_pulse_control_label(self) -> str:
        if self.coupler_pulse_control.id is not None:
            pulse_label = self.coupler_pulse_control.id
        else:
            pulse_label = "coupler_pulse_control"

        return f"{self.gate_label}{str_ref.DELIMITER}{pulse_label}"

    def execute(self, amplitude_scale=None, coupler_pulse_scale=None):        
        self.transmon_pair.align()
        
        # self.qubit_control.xy.wait(self.pre_wait)
        # self.qubit_target.xy.wait(self.pre_wait)
        
        self.qubit_control.z.play(
            self.flux_pulse_control_label,
            validate=False,
            amplitude_scale=amplitude_scale,
        )
        self.coupler.play(
            self.coupler_pulse_control_label,
            validate=False,
            amplitude_scale=coupler_pulse_scale,
        )
        
        self.transmon_pair.align()
        self.qubit_control.xy.wait(self.post_wait)
        self.transmon_pair.align()

    @property
    def config_settings(self):
        return {"after": [self.qubit_control.z]}

    def apply_to_config(self, config: dict) -> None:
        pulse_control = copy(self.flux_pulse_control)
        pulse_control.id = self.flux_pulse_control_label
        pulse_control.parent = None  # Reset parent so it can be attached to new parent
        pulse_control.parent = self.qubit_control.z
        
        pulse_coupler = copy(self.coupler_pulse_control)
        pulse_coupler.id = self.coupler_pulse_control_label
        pulse_coupler.parent = None  # Reset parent so it can be attached to new parent
        pulse_coupler.parent = self.coupler

        if self.flux_pulse_control_label in self.qubit_control.z.operations:
            raise ValueError(
                "Pulse name already exists in pulse operations. "
                f"Channel: {self.qubit_control.z.get_reference()}, "
                f"Pulse: {self.flux_pulse_control.get_reference()}, "
                f"Pulse name: {self.flux_pulse_control_label}"
            )

        pulse_control.apply_to_config(config)
        pulse_coupler.apply_to_config(config)

        element_config_control = config["elements"][self.qubit_control.z.name]
        element_config_control["operations"][self.flux_pulse_control_label] = pulse_control.pulse_name
        element_config_coupler = config["elements"][self.coupler.name]
        element_config_coupler["operations"][self.coupler_pulse_control_label] = pulse_coupler.pulse_name



@quam_dataclass
class composed_CZGate(TwoQubitGate):
    """CZ whose flux waveform has an independent amplitude per segment (default) or per 1 ns sample.

    Segment mode (one_ns_mode=False):
        `flux_pulse_control` / `coupler_flux_pulse` are one segment each (e.g. a 4 ns SquarePulse);
        `n_amplitudes = tot_len // segment_length` segments are played back-to-back on q_control.z
        and the coupler, segment k scaled by its own factor.

    1 ns mode (one_ns_mode=True):
        `n_amplitudes = tot_len` (one value per ns, tot_len a multiple of 4). Four alias elements per
        line share the physical port of q_control.z / the coupler; alias r plays a 4 ns basis
        waveform whose only non-zero sample (`basis_amplitude`) sits at position r. Played in
        parallel, their sum gives four independent 1 ns samples per 4 ns block. Sample j has voltage
        `flux_pulse_control.amplitude * scale_j` (coupler: `coupler_flux_pulse.amplitude * scale_j`),
        i.e. the same meaning a scale has in segment mode. Requires 1 GS/s on both ports, and
        |amplitude * scale / basis_amplitude| < 2 (the amp() range). The aliases exist only in the
        generated QUA config, not in the QuAM state.

    Amplitude factors (one per segment / per ns):
        execute():      extras['qb_amps'][k] * amplitude_scale, extras['cp_amps'][k] * coupler_amplitude_scale
        execute_dgx():  q_amp_scales[k], c_amp_scales[k]  (python floats or QUA expressions)
    """

    # Pulses will be added to qubit elements
    # The reason we don't add "flux_to_q1" directly to q1.z is because it is part of
    # the CZ operation, i.e. it is only applied as part of a CZ operation

    flux_pulse_control: Pulse
    coupler_flux_pulse: Pulse = None

    pre_wait: int = 4

    phase_shift_control: float = 0.0
    phase_shift_target: float = 0.0

    tot_len: int = 40

    one_ns_mode: bool = False      # True: one amplitude per 1 ns via 4 parallel basis aliases per line
    basis_amplitude: float = 0.25  # V, the non-zero sample of each 4 ns basis waveform (1 ns mode)
    strict_timing: bool = False    # 1 ns mode: wrap the flux plays in strict_timing_ (compiler-guaranteed gapless)

    @property
    def n_amplitudes(self) -> int:
        """Number of amplitude factors per line: tot_len in 1 ns mode, otherwise the number of segments."""
        if self.one_ns_mode:
            if self.tot_len <= 0 or self.tot_len % 4:
                raise ValueError(f"{self.gate_label}: tot_len must be a positive multiple of 4 ns in 1 ns mode, got {self.tot_len}")
            return self.tot_len
        return self.n_segments

    @property
    def segment_length(self) -> int:
        """Length of one flux segment in ns (taken from flux_pulse_control)."""
        return int(self.flux_pulse_control.length)

    @property
    def n_segments(self) -> int:
        """Number of segments played per gate; validates the segment / total lengths."""
        seg_len = self.segment_length
        if seg_len <= 0 or seg_len % 4:
            raise ValueError(f"{self.gate_label}: segment length must be a positive multiple of 4 ns, got {seg_len}")
        if self.tot_len % seg_len:
            raise ValueError(f"{self.gate_label}: tot_len={self.tot_len} ns is not a multiple of the segment length {seg_len} ns")
        if self.coupler_flux_pulse is not None and int(self.coupler_flux_pulse.length) != seg_len:
            raise ValueError(
                f"{self.gate_label}: coupler segment length {self.coupler_flux_pulse.length} ns "
                f"differs from qubit segment length {seg_len} ns"
            )
        return self.tot_len // seg_len

    @property
    def gate_label(self) -> str:
        try:
            return self.parent.get_attr_name(self)
        except AttributeError:
            return "CZ"

    @property
    def flux_pulse_control_label(self) -> str:
        if self.flux_pulse_control.id is not None:
            pulse_label = self.flux_pulse_control.id
        else:
            pulse_label = "flux_pulse_control"

        return f"{self.gate_label}{str_ref.DELIMITER}{pulse_label}"

    @property
    def coupler_flux_pulse_label(self) -> str:
        if self.coupler_flux_pulse.id is not None:
            pulse_label = self.coupler_flux_pulse.id
        else:
            pulse_label = "coupler_flux_pulse"

        return f"{self.gate_label}{str_ref.DELIMITER}{pulse_label}"

    def _amplitude_scales(self, scales, name: str) -> list:
        """Return one amplitude factor per segment / per ns; None means 1.0 everywhere."""
        n = self.n_amplitudes
        if scales is None:
            return [1.0] * n
        if len(scales) != n:
            unit = "per ns" if self.one_ns_mode else "per segment"
            raise ValueError(f"{self.gate_label}: {name} has {len(scales)} entries, expected {n} (one {unit})")
        return list(scales)

    # ---------- 1 ns mode: names of the config-only basis aliases ----------
    def _basis_prefix(self) -> str:
        return f"{self.transmon_pair.name}.{self.gate_label}"

    def _alias_names(self, line: str) -> list:
        """line: 'q' (q_control.z port) or 'c' (coupler port)."""
        return [f"{self._basis_prefix()}.{line}{r}" for r in range(4)]

    def _basis_pulse_name(self, r: int) -> str:
        return f"{self._basis_prefix()}.basis{r}.pulse"

    def _basis_waveform_name(self, r: int) -> str:
        return f"{self._basis_prefix()}.basis{r}.wf"

    def _lanes(self) -> list:
        lanes = self._alias_names("q")
        if self.coupler_flux_pulse is not None:
            lanes += self._alias_names("c")
        return lanes

    def _anchors(self) -> list:
        """Elements of the pair that the aliases must be aligned with (same set as TransmonPair.align)."""
        qc, qt = self.qubit_control, self.qubit_target
        anchors = [qc.xy.name, qc.z.name, qc.resonator.name, qt.xy.name, qt.z.name, qt.resonator.name]
        if self.coupler is not None:
            anchors.append(self.coupler.name)
        return anchors

    def _play_samples(self, q_scales: list, c_scales: list) -> None:
        """1 ns mode: per 4 ns block, the 4 aliases of each line play their basis in parallel."""
        q_gain = self.flux_pulse_control.amplitude / self.basis_amplitude
        c_gain = self.coupler_flux_pulse.amplitude / self.basis_amplitude if self.coupler_flux_pulse is not None else None
        q_alias, c_alias = self._alias_names("q"), self._alias_names("c")
        lanes = self._lanes()

        align(*self._anchors(), *lanes)
        with strict_timing_() if self.strict_timing else nullcontext():
            for k in range(self.tot_len // 4):
                for r in range(4):
                    j = 4 * k + r
                    play("basis" * amp(q_gain * q_scales[j]), q_alias[r])
                    if c_gain is not None:
                        play("basis" * amp(c_gain * c_scales[j]), c_alias[r])
        align(*self._anchors(), *lanes)

    def _play_flux(self, q_scales: list, c_scales: list) -> None:
        if self.one_ns_mode:
            self._play_samples(q_scales, c_scales)
        else:
            self._play_segments(q_scales, c_scales)

    def _play_segments(self, q_scales: list, c_scales: list) -> None:
        """Play all segments back-to-back on q_control.z and (if present) the coupler."""
        for q_scale, c_scale in zip(q_scales, c_scales):
            self.qubit_control.z.play(
                self.flux_pulse_control_label,
                validate=False,
                amplitude_scale=q_scale,
            )
            if self.coupler_flux_pulse is not None:
                self.coupler.play(
                    self.coupler_flux_pulse_label,
                    validate=False,
                    amplitude_scale=c_scale,
                )
    

    def execute(self, amplitude_scale: float | None = None, coupler_amplitude_scale: float | None = None):
        """Play the gate with the per-segment factors stored in extras['qb_amps'] / extras['cp_amps']."""
        if amplitude_scale is None:
            amplitude_scale = 1.0
        if coupler_amplitude_scale is None:
            coupler_amplitude_scale = 1.0

        self.transmon_pair.align()

        q_scales = [s * amplitude_scale for s in self._amplitude_scales(self.extras.get("qb_amps"), "extras['qb_amps']")]
        c_scales = [s * coupler_amplitude_scale for s in self._amplitude_scales(self.extras.get("cp_amps"), "extras['cp_amps']")]

        self.transmon_pair.align()
        self._play_flux(q_scales, c_scales)
        self.transmon_pair.align()
        frame_rotation_2pi(self.phase_shift_control, self.qubit_control.xy.name)
        frame_rotation_2pi(self.phase_shift_target, self.qubit_target.xy.name)
        self.transmon_pair.align()

    def execute_dgx(self, q_amp_scales: list | None = None, c_amp_scales: list | None = None,
                    phase_shift_control=None, phase_shift_target=None):
        """Play the gate with externally supplied factors, one per segment / per ns (python floats or QUA expressions)."""
        if phase_shift_control is None:
            phase_shift_control = self.phase_shift_control
        if phase_shift_target is None:
            phase_shift_target = self.phase_shift_target

        self.transmon_pair.align()

        q_scales = self._amplitude_scales(q_amp_scales, "q_amp_scales")
        c_scales = self._amplitude_scales(c_amp_scales, "c_amp_scales")

        self.transmon_pair.align()
        self._play_flux(q_scales, c_scales)
        self.transmon_pair.align()
        frame_rotation_2pi(phase_shift_control, self.qubit_control.xy.name)
        frame_rotation_2pi(phase_shift_target, self.qubit_target.xy.name)
        self.transmon_pair.align()

    @property
    def config_settings(self):
        # 1 ns mode reads the z / coupler element ports and port settings from the config.
        after = [self.qubit_control.z, getattr(self.qubit_control.z, "opx_output", None)]
        if self.coupler is not None:
            after += [self.coupler, getattr(self.coupler, "opx_output", None)]
        return {"after": [c for c in after if isinstance(c, QuamComponent)]}

    @staticmethod
    def _lookup(mapping: dict, key):
        return mapping[key] if key in mapping else mapping[str(key)]

    def _element_port(self, config: dict, channel) -> tuple:
        """Physical (con, fem, port) of a single-input LF channel; checks the 1 GS/s requirement."""
        element = config["elements"][channel.name]
        if "singleInput" not in element:
            raise ValueError(f"{self.gate_label}: 1 ns mode needs a singleInput element, '{channel.name}' is not")
        port = tuple(element["singleInput"]["port"])
        con, fem, port_id = port
        port_cfg = self._lookup(self._lookup(config["controllers"][con]["fems"], fem)["analog_outputs"], port_id)
        if float(port_cfg.get("sampling_rate", 1e9)) != 1e9:
            raise ValueError(f"{self.gate_label}: 1 ns mode needs sampling_rate=1e9 on {port}, got {port_cfg['sampling_rate']}")
        return port

    def _add_basis_to_config(self, config: dict) -> None:
        """1 ns mode: register the 4 basis waveforms/pulses and the alias elements (config only)."""
        for name, base in (("flux_pulse_control", self.flux_pulse_control),
                           ("coupler_flux_pulse", self.coupler_flux_pulse)):
            if base is not None and abs(base.amplitude) / self.basis_amplitude >= 2:
                raise ValueError(
                    f"{self.gate_label}: |{name}.amplitude| / basis_amplitude = "
                    f"{abs(base.amplitude) / self.basis_amplitude:.3f} must be < 2 (amp() range)"
                )
        lines = [("q", self.qubit_control.z)]
        if self.coupler_flux_pulse is not None:
            lines.append(("c", self.coupler))
        ports = {line: self._element_port(config, channel) for line, channel in lines}
        if len(set(ports.values())) != len(ports):
            raise ValueError(f"{self.gate_label}: q_control.z and the coupler must use different ports in 1 ns mode")

        for r in range(4):
            wf, pulse = self._basis_waveform_name(r), self._basis_pulse_name(r)
            if wf in config["waveforms"] or pulse in config["pulses"]:
                raise ValueError(f"{self.gate_label}: basis waveform/pulse name collision: {wf} / {pulse}")
            config["waveforms"][wf] = {"type": "arbitrary",
                                       "samples": [self.basis_amplitude if n == r else 0.0 for n in range(4)]}
            config["pulses"][pulse] = {"operation": "control", "length": 4, "waveforms": {"single": wf}}
        for line, _ in lines:
            for r, alias in enumerate(self._alias_names(line)):
                if alias in config["elements"]:
                    raise ValueError(f"{self.gate_label}: element name collision: {alias}")
                # Only the port is shared: offsets, delay, filters and crosstalk stay on the physical port.
                config["elements"][alias] = {"singleInput": {"port": ports[line]},
                                             "operations": {"basis": self._basis_pulse_name(r)}}

    @staticmethod
    def _add_pulse_to_element(config: dict, pulse_template: Pulse, label: str, channel) -> None:
        """Register a copy of pulse_template on `channel` under `label` in the QUA config only."""
        if label in channel.operations:
            raise ValueError(
                "Pulse name already exists in pulse operations. "
                f"Channel: {channel.get_reference()}, "
                f"Pulse: {pulse_template.get_reference()}, "
                f"Pulse name: {label}"
            )
        pulse = copy(pulse_template)
        pulse.id = label
        pulse.parent = None  # Reset parent so it can be attached to new parent
        pulse.parent = channel
        pulse.apply_to_config(config)
        config["elements"][channel.name]["operations"][label] = pulse.pulse_name

    def apply_to_config(self, config: dict) -> None:
        self.n_amplitudes  # validate lengths before generating the config
        # The segment pulses stay registered in both modes, so switching modes never breaks references to them.
        self._add_pulse_to_element(config, self.flux_pulse_control, self.flux_pulse_control_label, self.qubit_control.z)
        if self.coupler_flux_pulse is not None:
            self._add_pulse_to_element(config, self.coupler_flux_pulse, self.coupler_flux_pulse_label, self.coupler)
        if self.one_ns_mode:
            self._add_basis_to_config(config)