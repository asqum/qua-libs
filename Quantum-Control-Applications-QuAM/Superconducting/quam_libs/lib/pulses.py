# from quam.core import quam_dataclass
# from quam.components.pulses import Pulse
# import numpy as np


# @quam_dataclass
# class DragPulseCosine(Pulse):
#     """
#     Creates Cosine based DRAG waveforms that compensate for the leakage and for the AC stark shift.

#     These DRAG waveforms has been implemented following the next Refs.:
#     Chen et al. PRL, 116, 020501 (2016)
#     https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.116.020501
#     and Chen's thesis
#     https://web.physics.ucsb.edu/~martinisgroup/theses/Chen2018.pdf

#     :param float amplitude: The amplitude in volts.
#     :param int length: The pulse length in ns.
#     :param float alpha: The DRAG coefficient.
#     :param float anharmonicity: f_21 - f_10 - The differences in energy between the 2-1 and the 1-0 energy levels, in Hz.
#     :param float detuning: The frequency shift to correct for AC stark shift, in Hz.
#     :return: Returns a tuple of two lists. The first list is the I waveform (real part) and the second is the
#         Q waveform (imaginary part)
#     """

#     axis_angle: float
#     amplitude: float
#     alpha: float
#     anharmonicity: float
#     detuning: float = 0.0
#     subtracted: bool = True

#     def waveform_function(self):
#         from qualang_tools.config.waveform_tools import drag_cosine_pulse_waveforms

#         I, Q = drag_cosine_pulse_waveforms(
#             amplitude=self.amplitude,
#             length=self.length,
#             alpha=self.alpha,
#             anharmonicity=self.anharmonicity,
#             detuning=self.detuning,
#             subtracted=self.subtracted,
#         )
#         I, Q = np.array(I), np.array(Q)

#         I_rot = I * np.cos(self.axis_angle) - Q * np.sin(self.axis_angle)
#         Q_rot = I * np.sin(self.axis_angle) + Q * np.cos(self.axis_angle)

#         return I_rot + 1.0j * Q_rot

# @quam_dataclass
# class FluxPulse(Pulse):
#     """Flux pulse QuAM component.

#     Args:
#         length (int): The total length of the pulse in samples, including zero padding.
#         digital_marker (str, list, optional): The digital marker to use for the pulse.
#         amplitude (float): The amplitude of the pulse in volts.
#     """

#     amplitude: float
#     zero_padding: int = 0

#     def waveform_function(self):
#         waveform = self.amplitude * np.ones(self.length)
#         if self.zero_padding:
#             if self.zero_padding > self.length:
#                 raise ValueError(
#                     f"Flux pulse zero padding ({self.zero_padding} ns) exceeds " f"pulse length ({self.length} ns)."
#                 )
#             waveform[-self.zero_padding :] = 0
#         return waveform

# @quam_dataclass
# class SNZPulse(Pulse):
#     amplitude: float
#     step_amplitude: float
#     step_length: int
#     spacing : int

#     def __post_init__(self):
#         self.length -= self.length % 4

#     def waveform_function(self):
#         rect_duration = (self.length - 4 - 2 * self.step_length - self.spacing) // 2
#         waveform = [self.amplitude] * rect_duration
#         waveform += [self.step_amplitude] * self.step_length
#         waveform += [0] * self.spacing
#         waveform += [-self.step_amplitude] * self.step_length
#         waveform += [-self.amplitude] * rect_duration
#         waveform += [0.0] * (self.length - len(waveform))

#         return waveform



# ============================== ^ Originally in qiskit_to_qua ^ ==============================================



# ============================== V from as_quam_qualibrate V ===========================================
from quam.core import quam_dataclass
from quam.components.pulses import Pulse
import numpy as np


@quam_dataclass
class DragPulseCosine(Pulse):
    """
    Creates Cosine based DRAG waveforms that compensate for the leakage and for the AC stark shift.

    These DRAG waveforms has been implemented following the next Refs.:
    Chen et al. PRL, 116, 020501 (2016)
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.116.020501
    and Chen's thesis
    https://web.physics.ucsb.edu/~martinisgroup/theses/Chen2018.pdf

    :param float amplitude: The amplitude in volts.
    :param int length: The pulse length in ns.
    :param float alpha: The DRAG coefficient.
    :param float anharmonicity: f_21 - f_10 - The differences in energy between the 2-1 and the 1-0 energy levels, in Hz.
    :param float detuning: The frequency shift to correct for AC stark shift, in Hz.
    :return: Returns a tuple of two lists. The first list is the I waveform (real part) and the second is the
        Q waveform (imaginary part)
    """

    axis_angle: float
    amplitude: float
    alpha: float
    anharmonicity: float
    detuning: float = 0.0
    subtracted: bool = True

    def waveform_function(self):
        from qualang_tools.config.waveform_tools import drag_cosine_pulse_waveforms

        I, Q = drag_cosine_pulse_waveforms(
            amplitude=self.amplitude,
            length=self.length,
            alpha=self.alpha,
            anharmonicity=self.anharmonicity,
            detuning=self.detuning,
            subtracted=self.subtracted,
        )
        I, Q = np.array(I), np.array(Q)

        I_rot = I * np.cos(self.axis_angle) - Q * np.sin(self.axis_angle)
        Q_rot = I * np.sin(self.axis_angle) + Q * np.cos(self.axis_angle)

        return I_rot + 1.0j * Q_rot

@quam_dataclass
class FluxPulse(Pulse):
    """Flux pulse QuAM component.

    Args:
        length (int): The total length of the pulse in samples, including zero padding.
        digital_marker (str, list, optional): The digital marker to use for the pulse.
        amplitude (float): The amplitude of the pulse in volts.
    """

    amplitude: float
    zero_padding: int = 0

    def waveform_function(self):
        waveform = self.amplitude * np.ones(self.length)
        if self.zero_padding:
            if self.zero_padding > self.length:
                raise ValueError(
                    f"Flux pulse zero padding ({self.zero_padding} ns) exceeds " f"pulse length ({self.length} ns)."
                )
            waveform[-self.zero_padding :] = 0
        return waveform

@quam_dataclass
class SNZPulse(Pulse):
    amplitude: float
    step_amplitude: float
    step_length: int
    spacing : int

    def __post_init__(self):
        self.length -= self.length % 4

    def waveform_function(self):
        rect_duration = (self.length - 4 - 2 * self.step_length - self.spacing) // 2
        waveform = [self.amplitude] * rect_duration
        waveform += [self.step_amplitude] * self.step_length
        waveform += [0] * self.spacing
        waveform += [-self.step_amplitude] * self.step_length
        waveform += [-self.amplitude] * rect_duration
        waveform += [0.0] * (self.length - len(waveform))

        return waveform
    
@quam_dataclass
class CosineBipolarPulse(Pulse):
    """Slepian bipolar pulse QUAM component.

    Args:
        length (int): The total length of the pulse in samples.
        amplitude (float): The amplitude of the pulse in volts.
        axis_angle (float, optional): IQ axis angle of the output pulse in radians.
            If None (default), the pulse is meant for a single channel or the I port
                of an IQ channel
            If not None, the pulse is meant for an IQ channel (0 is X, pi/2 is Y).
        flat_length (int): The length of the pulse's flat top in samples.
            The rise and fall lengths are calculated from the total length and the
            flat length.
    """

    amplitude: float
    axis_angle: float = None
    flat_length: int

    def waveform_function(self):
        # Helper segment generators (length 0 returns empty array)
        def halfcos_up(n: int):
            if n <= 0:
                return np.array([])
            t = np.arange(n) / n
            return 0.5 * (1 - np.cos(np.pi * t))  # 0 -> 1

        def halfcos_down(n: int):
            if n <= 0:
                return np.array([])
            t = np.arange(n) / n
            return 0.5 * (1 + np.cos(np.pi * t))  # 1 -> 0

        def cos_switch(n: int):
            if n <= 0:
                return np.array([])
            t = np.arange(n) / n
            return np.cos(np.pi * t)  # +1 -> -1

        L = int(self.length)
        F = int(self.flat_length)
        if F > L:
            raise ValueError(
                f"CosineBipolarPulse.flat_length ({F}) cannot exceed total length ({L})."
            )

        remaining = L - F
        if remaining <= 0:
            rise_len = switch_len = fall_len = 0
        else:
            base = remaining // 3
            extra = remaining % 3
            rise_len = base + (1 if extra > 0 else 0)
            switch_len = base
            fall_len = base + (1 if extra > 1 else 0)

        flat_pos_len = F // 2 + (F % 2)  # positive half gets extra sample if odd
        flat_neg_len = F // 2

        A = float(self.amplitude)

        seg_rise = A * halfcos_up(rise_len)
        seg_flat_pos = A * np.ones(flat_pos_len)
        seg_switch = A * cos_switch(switch_len)
        seg_flat_neg = -A * np.ones(flat_neg_len)
        seg_fall = -A * halfcos_down(fall_len)

        p = np.concatenate(
            [seg_rise, seg_flat_pos, seg_switch, seg_flat_neg, seg_fall]
        )

        current_len = len(p)
        if current_len < L:
            pad_total = L - current_len
            pad_front = pad_total // 2
            pad_back = pad_total - pad_front
            p = np.concatenate([np.zeros(pad_front), p, np.zeros(pad_back)])
        elif current_len > L:
            trim_total = current_len - L
            trim_front = trim_total // 2
            trim_back = trim_total - trim_front
            p = p[trim_front: current_len - trim_back]

        if self.axis_angle is not None:
            p = p * np.exp(1j * self.axis_angle)

        return p.tolist()
    
@quam_dataclass
class CosineFlatTopPulse(Pulse):
    """
    Cosine flat-top pulse (unipolar, smooth rise/fall).

    Args:
        length (int): Total pulse duration in samples.
        amplitude (float): Peak amplitude of the pulse.
        axis_angle (float, optional): IQ axis angle in radians.
            If None (default), pulse is real and drives a single channel.
            If set, pulse becomes complex-valued for IQ outputs.
        flat_length (int): Number of samples at full amplitude (the flat-top region).
    """

    amplitude: float
    axis_angle: float = None
    flat_length: int = 0  # default no flat top

    def waveform_function(self):
        L = int(self.length)
        F = int(self.flat_length)
        if F > L:
            raise ValueError(
                f"CosineFlatTopPulse.flat_length ({F}) cannot exceed total length ({L})."
            )

        # Remaining samples divided into rise and fall
        remaining = L - F
        rise_len = remaining // 2
        fall_len = remaining - rise_len

        def halfcos_up(n: int):
            if n <= 0:
                return np.array([])
            t = np.arange(n) / n
            return 0.5 * (1 - np.cos(np.pi * t))  # 0 → 1

        def halfcos_down(n: int):
            if n <= 0:
                return np.array([])
            t = np.arange(n) / n
            return 0.5 * (1 + np.cos(np.pi * t))  # 1 → 0

        A = float(self.amplitude)
        seg_rise = A * halfcos_up(rise_len)
        seg_flat = A * np.ones(F)
        seg_fall = A * halfcos_down(fall_len)

        # Concatenate waveform
        p = np.concatenate([seg_rise, seg_flat, seg_fall])

        # Ensure exact total length (pad/trim if needed)
        current_len = len(p)
        if current_len < L:
            p = np.pad(p, (0, L - current_len))
        elif current_len > L:
            p = p[:L]

        # Apply axis angle for IQ output if provided
        if self.axis_angle is not None:
            p = p * np.exp(1j * self.axis_angle)

        return p.tolist()