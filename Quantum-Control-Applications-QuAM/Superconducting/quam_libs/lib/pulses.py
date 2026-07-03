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


def drag_slepian_pulse_waveforms(
    amplitude: float,
    length: int,
    alpha: float,
    anharmonicity: float,
    detuning: float = 0.0,
    time_bandwidth: float = 4.0,
    slepian_order: int = 0,
):
    """Slepian-envelope DRAG waveforms (leakage + AC-Stark compensation).

    Same DRAG construction as ``drag_cosine_pulse_waveforms``, with a normalized
    DPSS (Slepian) envelope instead of a cosine lobe.
    """
    from scipy.signal.windows import dpss

    length = int(length)
    if alpha != 0 and anharmonicity == 0:
        raise ValueError("Cannot create a DRAG pulse with `anharmonicity=0`")

    nw = _effective_dpss_bandwidth(length, time_bandwidth)
    w = dpss(length, nw, Kmax=slepian_order + 1)[slepian_order]
    w = w / np.max(np.abs(w))

    slepian_wave = amplitude * w
    der_wave = amplitude * np.gradient(w, 1e-9)

    t = np.arange(length, dtype=int)
    z = slepian_wave.astype(complex)
    if alpha != 0:
        z += 1j * der_wave * (alpha / (anharmonicity - 2 * np.pi * detuning))
        z *= np.exp(1j * 2 * np.pi * detuning * t * 1e-9)

    return z.real.tolist(), z.imag.tolist()


@quam_dataclass
class DragPulseSlepian(Pulse):
    """
    Slepian-envelope DRAG pulse for XY gates.

    :param float amplitude: The amplitude in volts.
    :param int length: The pulse length in ns.
    :param float alpha: The DRAG coefficient.
    :param float anharmonicity: f_21 - f_10 in Hz.
    :param float detuning: AC-Stark correction in Hz.
    :param float time_bandwidth: DPSS half-bandwidth product NW.
    :param int slepian_order: DPSS sequence index (0 = first-order Slepian).
    :param float axis_angle: IQ rotation angle in radians.
    """

    axis_angle: float
    amplitude: float
    alpha: float
    anharmonicity: float
    detuning: float = 0.0
    time_bandwidth: float = 4.0
    slepian_order: int = 0

    def waveform_function(self):
        I, Q = drag_slepian_pulse_waveforms(
            amplitude=self.amplitude,
            length=self.length,
            alpha=self.alpha,
            anharmonicity=self.anharmonicity,
            detuning=self.detuning,
            time_bandwidth=self.time_bandwidth,
            slepian_order=self.slepian_order,
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
class CosineFluxPulse(Pulse):
    """Pure cosine-lobe flux pulse (0 → amplitude → 0).

    Args:
        length (int): Total pulse length in samples, including zero padding.
        amplitude (float): Peak amplitude of the pulse.
        zero_padding (int): Number of samples at the end set to zero.
    """

    amplitude: float
    zero_padding: int = 0

    def waveform_function(self):
        if self.zero_padding > self.length:
            raise ValueError(
                f"Flux pulse zero padding ({self.zero_padding} ns) exceeds "
                f"pulse length ({self.length} ns)."
            )

        active_length = self.length - self.zero_padding
        if active_length <= 0:
            return np.zeros(self.length)

        t = np.arange(active_length)

        waveform = 0.5 * self.amplitude * (
            1 - np.cos(2 * np.pi * t / (active_length - 1))
        )

        if self.zero_padding:
            waveform = np.concatenate(
                [waveform, np.zeros(self.zero_padding)]
            )

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


def _effective_dpss_bandwidth(length: int, time_bandwidth: float) -> float:
    """Clip NW so scipy dpss(M, NW) satisfies NW < M/2."""
    return min(time_bandwidth, length / 2.0 - 1e-9)


def _slepian_segment(
    length: int,
    time_bandwidth: float,
    slepian_order: int,
    rising: bool = True,
) -> np.ndarray:
    """First-order (or k-th) DPSS edge: 0 -> 1 (rise) or 1 -> 0 (fall)."""
    if length <= 0:
        return np.array([])
    if length == 1:
        return np.array([1.0])

    from scipy.signal.windows import dpss

    dpss_length = 2 * length - 1
    nw = _effective_dpss_bandwidth(dpss_length, time_bandwidth)
    w_full = dpss(dpss_length, nw, Kmax=slepian_order + 1)[slepian_order]
    w = w_full[:length].astype(float)
    w = w / w[-1]
    if not rising:
        w = w[::-1]
    return w


@quam_dataclass
class SlepianPulse(Pulse):
    """
    Unipolar flux pulse shaped by the first DPSS (Slepian) sequence.

    Args:
        length (int): Total pulse duration in samples.
        amplitude (float): Peak amplitude of the pulse.
        time_bandwidth (float): DPSS half-bandwidth product NW (concentration).
            Automatically reduced for short pulses (scipy requires NW < M/2).
        slepian_order (int): DPSS sequence index. 0 = first-order Slepian.
        axis_angle (float, optional): IQ axis angle in radians.
    """

    amplitude: float
    time_bandwidth: float = 4.0
    slepian_order: int = 0
    axis_angle: float = None

    def waveform_function(self):
        from scipy.signal.windows import dpss

        length = int(self.length)
        nw = _effective_dpss_bandwidth(length, self.time_bandwidth)
        w = dpss(length, nw, Kmax=self.slepian_order + 1)[self.slepian_order]
        w = w / np.max(np.abs(w))
        p = float(self.amplitude) * w

        if self.axis_angle is not None:
            p = p * np.exp(1j * self.axis_angle)

        return p.tolist()

@quam_dataclass
class FreeCosineBipolarPulse(Pulse):
    """
    Bipolar pulse defined by ratios, allowing asymmetric positive/negative lobes.

    Args:
        length (int): Total length in samples.
        amplitude (float): The amplitude of the pulse (V).
        pos_len_ratio (float): Ratio of total length allocated to the positive lobe.
            Range [0.0, 1.0]. E.g., 0.6 means first 60% is positive.
        neg_amp_scal (float): Ratio of the amplitude for negative pole, means the amplitude for negative
            pole will be amp*neg_amp_scal. 
        flat_length_ratio (float): Ratio of length within each lobe that is flat.
            Range [0.0, 1.0]. E.g., 0.9 means 90% of the lobe is flat top.
        axis_angle (float, optional): IQ axis angle in radians.
    """

    amplitude: float
    axis_angle: float = None
    flat_length_ratio: float
    neg_amp_scal:float
    pos_len_ratio:float

    def waveform_function(self):
        # --- Helper functions ---
        # 0 -> 1 (Rising edge)
        def halfcos_up(n: int):
            if n <= 0: return np.array([])
            t = np.arange(n) / n
            return 0.5 * (1 - np.cos(np.pi * t))

        # 1 -> 0 (Falling edge)
        def halfcos_down(n: int):
            if n <= 0: return np.array([])
            t = np.arange(n) / n
            return 0.5 * (1 + np.cos(np.pi * t))

        L = int(self.length)
        A = float(self.amplitude)
        
        # 1. calc the length of postive pole and negative pole
        # flip_point_ratio = 0.6  --> pos_total_len = 0.6 * L
        
        pos_total_len = int(np.round(L * self.pos_len_ratio))
        neg_total_len = L - pos_total_len

        # 2. calc flat length
        # flat_length_ratio = 0.9 
        pos_flat_len = int(np.round(pos_total_len * self.flat_length_ratio))
        neg_flat_len = int(np.round(neg_total_len * self.flat_length_ratio))

        # 3. calc transition length
        
        
        # [Positive Lobe]
        pos_remain = pos_total_len - pos_flat_len
        len_rise = pos_remain // 2                    # 0 -> A
        len_fall_to_zero = pos_remain - len_rise      # A -> 0 

        # [Negative Lobe]
        neg_remain = neg_total_len - neg_flat_len
        len_fall_from_zero = neg_remain // 2          # 0 -> -A 
        len_return_zero = neg_remain - len_fall_from_zero # -A -> 0

        # 4. Create waveform
        # Part A: Positive Side
        seg_rise = A * halfcos_up(len_rise)
        seg_flat_pos = A * np.ones(pos_flat_len)
        seg_switch_1 = A * halfcos_down(len_fall_to_zero) # 1 -> 0

        # Part B: Negative Side
        # halfcos_up 
        seg_switch_2 = -A * self.neg_amp_scal * halfcos_up(len_fall_from_zero) 
        seg_flat_neg = -A * self.neg_amp_scal * np.ones(neg_flat_len)
        # halfcos_down 
        seg_return = -A * self.neg_amp_scal * halfcos_down(len_return_zero)

        # 5. connect
        p = np.concatenate([
            seg_rise, 
            seg_flat_pos, 
            seg_switch_1,  # Cross 0 (Pos end)
            seg_switch_2,  # Cross 0 (Neg start)
            seg_flat_neg, 
            seg_return
        ])

        # 6. Padding / Trimming
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

        # 7. IQ rotation
        if self.axis_angle is not None:
            p = p * np.exp(1j * self.axis_angle)

        return p.tolist()
    

## ===== Trying to split bipolar to different segment so that we can esily modify it in QUA =====
@quam_dataclass
class FlatPulse(Pulse):
    """Flat segment in bipolar.

    Args:
        length (int): The total length of the pulse in samples.
        amplitude (float): The amplitude of the pulse in volts.
    """

    amplitude: float

    def waveform_function(self):
        waveform = self.amplitude * np.ones(self.length)
        return waveform.tolist()
@quam_dataclass  
class HalfCosineRisePulse(Pulse):
    """
    cos (0 -> Amplitude)
    which is the halfcos_up * A segment in FreeCosineBipolarPulse
    """
    amplitude: float
    def waveform_function(self):
        
        if self.length <= 0:
            return np.array([])
        t = np.arange(self.length) / self.length
        
        p = self.amplitude * 0.5 * (1 - np.cos(np.pi * t))
        
        return p.tolist()
@quam_dataclass  
class HalfCosineFallPulse(Pulse):
    """
    cos (Amplitude -> 0)
    which is the halfcos_down * A segment in FreeCosineBipolarPulse
    """
    amplitude: float
    def waveform_function(self):
        if self.length <= 0:
            return np.array([])
        t = np.arange(self.length) / self.length
        p = self.amplitude * 0.5 * (1 + np.cos(np.pi * t))
        return p.tolist()

@quam_dataclass  
class CosineRatioFlatTopPulse(Pulse):
    """
    Cosine flat-top pulse (unipolar, smooth rise/fall).

    Args:
        length (int): Total pulse duration in samples.
        amplitude (float): Peak amplitude of the pulse.
        axis_angle (float, optional): IQ axis angle in radians.
            If None (default), pulse is real and drives a single channel.
            If set, pulse becomes complex-valued for IQ outputs.
        flat_length_ratio (float): ratio of samples at full amplitude (the flat-top region).
    """

    amplitude: float
    axis_angle: float = None
    flat_ratio: float = 0.9  # default no flat top

    def waveform_function(self):
        
        L = int(self.length)
        F = int(self.flat_ratio*L)

        if self.flat_ratio > 1.0:
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



@quam_dataclass  
class aSWAPPulse(Pulse):
    """
    The aSWAP pulse shape

    Args:
        length (int): Total pulse duration in samples.
        amplitude (float): Peak amplitude of the pulse.
        axis_angle (float, optional): IQ axis angle in radians.
            If None (default), pulse is real and drives a single channel.
            If set, pulse becomes complex-valued for IQ outputs.
        slope_direction (Literal[1, -1]): The sign of the slope when choosing the amplitude for the aSWAP pulse. It can be either 1 or -1.
    """

    amplitude: float
    axis_angle: float = None
    slope_direction: int = 1 # 1 for positive slope, -1 for negative slope
    truncate_len:int


    def waveform_function(self):
        totL = int(self.length)
        tL = int(self.truncate_len)

        if tL > totL:
            tL = totL


        S = int(self.slope_direction)

        if S == 1:
            p = np.linspace(0, self.amplitude, totL)
        else:
            p = np.linspace(self.amplitude, 0, totL)


        first_part = p[:tL]
        zero_part = np.zeros(totL - tL)
        p = np.concatenate((first_part, zero_part))

        # Apply axis angle for IQ output if provided
        if self.axis_angle is not None:
            p = p * np.exp(1j * self.axis_angle)

        return p.tolist()


@quam_dataclass  
class ArbWaveformPulse(Pulse):
    """
    The Arbitrary waveform pulse shape

    Args:
        annotation: str, a note for a better remind
        sn: str, serial number for a better specification
        I_samples: list,
        Q_samples: list,
        axis_angle: float, IQ axis angle in radians.
    """
    annotation: str
    sn:str
    I_samples: list
    Q_samples: list
    axis_angle: float


    def waveform_function(self):
         


        I = np.array(self.I_samples)
        Q = np.array(self.Q_samples)

        I_rot = I * np.cos(self.axis_angle) - Q * np.sin(self.axis_angle)
        Q_rot = I * np.sin(self.axis_angle) + Q * np.cos(self.axis_angle)

        IQ_signal = I_rot + 1.0j * Q_rot

        return IQ_signal