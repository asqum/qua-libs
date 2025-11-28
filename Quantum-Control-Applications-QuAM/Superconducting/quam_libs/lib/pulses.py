from quam.core import quam_dataclass
from quam.components.pulses import Pulse
import numpy as np
from dataclasses import field


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
class SquarePulse(Pulse):
    amplitude: float
    pre_padding: int = 0
    post_padding: int = 0

    def waveform_function(self):
        L  = int(self.length)
        PL = int(self.pre_padding)
        PR = int(self.post_padding)

        if PL < 0 or PR < 0:
            raise ValueError("pre_padding and post_padding must be >= 0.")
        if PL + PR > L:
            raise ValueError(
                f"Total zero padding ({PL + PR}) exceeds pulse length ({L})."
            )

        active_len = L - (PL + PR)
        active = self.amplitude * np.ones(active_len)

        w = np.concatenate([
            np.zeros(PL),
            active,
            np.zeros(PR),
        ])

        return w
    
@quam_dataclass
class CosineFlatTopPulse(Pulse):
    """
    [pre_padding] + cosine rise + flat + cosine fall + [post_padding (+ any rounding zeros)]
    """

    amplitude: float
    axis_angle: float = None
    flat_length: int = 0
    smoothing_time: int = 0          # rise + fall
    pre_padding: int = 0
    post_padding: int = 0
    length: int = field(default="#./inferred_total_length", init=False)

    @property
    def inferred_total_length(self) -> int:
        active = self.flat_length + self.smoothing_time
        base_total = active + self.pre_padding + self.post_padding
        # keep multiple-of-4 behaviour, extra zeros go at the end
        return int(np.ceil(base_total / 4) * 4)

    def waveform_function(self):
        def halfcos(n: int):
            if n <= 0:
                return np.array([])
            t = np.arange(n) / n
            return 0.5 * (1 - np.cos(np.pi * t))

        L  = int(self.length)
        F  = int(self.flat_length)
        S  = int(self.smoothing_time)
        PL = int(self.pre_padding)
        PR = int(self.post_padding)

        if F < 0 or S < 0:
            raise ValueError("flat_length and smoothing_time must be >= 0.")
        if PL < 0 or PR < 0:
            raise ValueError("pre_padding and post_padding must be >= 0.")

        active = F + S
        base_total = PL + active + PR
        if base_total > L:
            raise ValueError(
                f"pre + active + post = {base_total} exceeds total length={L}."
            )

        # any extra samples from rounding go into the *effective* post padding
        extra = L - base_total
        PR_eff = PR + extra

        # split rise/fall
        if S == 0:
            rise_len = fall_len = 0
        else:
            rise_len = S // 2
            fall_len = S - rise_len

        A = float(self.amplitude)
        seg_rise = A * halfcos(rise_len)
        seg_flat = A * np.ones(F)
        seg_fall = A * halfcos(fall_len)[::-1]

        pre_zeros  = np.zeros(PL)
        post_zeros = np.zeros(PR_eff)

        p = np.concatenate([pre_zeros, seg_rise, seg_flat, seg_fall, post_zeros])

        if self.axis_angle is not None:
            p = p * np.exp(1j * self.axis_angle)

        return p.tolist()

@quam_dataclass
class SlepianFlatTopPulse(Pulse):
    """
    [pre_padding] + slepian ramp up + flat + slepian ramp down + [post_padding (+ rounding)]
    """

    amplitude: float
    axis_angle: float = None
    flat_length: int = 64
    smoothing_time: int = 24       # rise + fall
    pre_padding: int = 0
    post_padding: int = 0
    slepian_NW: float = 2.0
    length: int = field(default="#./inferred_total_length", init=False)

    @property
    def inferred_total_length(self) -> int:
        active = self.flat_length + self.smoothing_time
        base_total = active + self.pre_padding + self.post_padding
        return int(np.ceil(base_total / 4) * 4)

    def waveform_function(self):
        L  = int(self.length)
        F  = int(self.flat_length)
        S  = int(self.smoothing_time)
        PL = int(self.pre_padding)
        PR = int(self.post_padding)

        if F < 0 or S < 0:
            raise ValueError("flat_length and smoothing_time must be >= 0.")
        if PL < 0 or PR < 0:
            raise ValueError("pre_padding and post_padding must be >= 0.")

        active = F + S
        base_total = PL + active + PR
        if base_total > L:
            raise ValueError(
                f"pre + active + post = {base_total} exceeds total length={L}."
            )

        extra = L - base_total
        PR_eff = PR + extra

        # split smoothing_time
        if S == 0:
            rise_len = fall_len = 0
        else:
            rise_len = S // 2
            fall_len = S - rise_len

        def slepian_edge(n: int):
            if n <= 0:
                return np.array([])
            try:
                from scipy.signal.windows import dpss
                w = dpss(2 * n, self.slepian_NW)  # length 2n
                w_half = w[:n]
                return (w_half - w_half[0]) / (w_half.max() - w_half[0] + 1e-15)
            except ImportError:
                t = np.arange(n) / n
                return 0.5 * (1 - np.cos(np.pi * t))

        A = float(self.amplitude)
        seg_rise = A * slepian_edge(rise_len)
        seg_flat = A * np.ones(F)
        seg_fall = A * slepian_edge(fall_len)[::-1]

        pre_zeros  = np.zeros(PL)
        post_zeros = np.zeros(PR_eff)

        p = np.concatenate([pre_zeros, seg_rise, seg_flat, seg_fall, post_zeros])

        if self.axis_angle is not None:
            p = p * np.exp(1j * self.axis_angle)

        return p.tolist()


@quam_dataclass
class SlepianPulse(Pulse):
    """
    Pure Slepian (DPSS) envelope with asymmetric padding:

        waveform = [pre_padding zeros][smooth Slepian envelope][post_padding zeros]
    """

    amplitude: float
    slepian_NW: float = 2.5
    axis_angle: float = None
    pre_padding: int = 0
    post_padding: int = 0
    length: int = 64

    def waveform_function(self):
        L  = int(self.length)
        PL = int(self.pre_padding)
        PR = int(self.post_padding)

        if PL < 0 or PR < 0:
            raise ValueError("pre_padding and post_padding must be >= 0.")
        if PL + PR > L:
            raise ValueError(
                f"Total zero padding ({PL + PR}) exceeds pulse length ({L})."
            )

        L_active = L - (PL + PR)
        if L_active <= 0:
            raise ValueError(
                f"Active Slepian portion {L_active} <= 0 "
                f"(length={L}, pre_padding={PL}, post_padding={PR})"
            )

        try:
            from scipy.signal.windows import dpss
            w = dpss(L_active, NW=self.slepian_NW, Kmax=1, sym=True)[0]
        except ImportError:
            t = np.arange(L_active) / L_active
            w = 0.5 * (1 - np.cos(2 * np.pi * t))

        # shift endpoints to zero
        offset = w[0]
        w = w - offset
        w[0] = 0.0
        w[-1] = 0.0

        max_val = np.max(np.abs(w))
        if max_val > 0:
            w = (w / max_val) * self.amplitude

        pre_zeros  = np.zeros(PL)
        post_zeros = np.zeros(PR)

        w = np.concatenate([pre_zeros, w, post_zeros])

        if self.axis_angle is not None:
            w = w * np.exp(1j * self.axis_angle)

        assert len(w) == L, f"Waveform length {len(w)} != expected {L}"

        return w.tolist()

@quam_dataclass
class CosineBipolarPulse(Pulse):
    """
    CosineBipolarPulse QUAM component.

    Generates a net-zero pulse with two symmetric cosine-shaped lobes.
    Minimizes DC offset and long-timescale distortions. Waveform: smooth cosine
    rise to positive flat section, cosine switch to negative flat, ends with
    symmetric cosine rise. Positive and negative flat regions are equal length, so
    the area is zero.

    Net-zero property helps against slow baseline drifts and long-memory effects.
    Smooth transitions reduce spectral leakage and high-frequency noise — suitable
    for sensitive quantum control.

    Increasing the total length with constant flat length makes longer, smoother
    rises/falls and switching, further reducing high-frequency content.

    Args:
        amplitude (float): Peak amplitude (V).
        axis_angle (float, optional): IQ axis angle in radians. If None, use for
            a single channel or I of IQ; if not None, use for IQ (0 is X, pi/2 is Y).
        flat_length (int): Flat region length (must be even and ≤ total length).
            Split equally between positive and negative.
        smoothing_time (int): Total length of rise, switch, and fall segments
            (samples). Default 0 for abrupt transitions. Increasing this smooths
            edges, reducing high-frequency content.
        post_zero_padding_time (int): Additional zeros appended after the pulse
            (samples). Default 0.
    """

    amplitude: float
    axis_angle: float = None
    flat_length: int
    smoothing_time: int = 0
    post_zero_padding_time: int = 0
    length: int = field(default="#./inferred_total_length", init=False)

    @property
    def inferred_total_length(self) -> int:
        return int(np.ceil((self.flat_length + self.smoothing_time + self.post_zero_padding_time) / 4) * 4)

    def waveform_function(self):
        # Helper segment generators (length 0 returns empty array)
        def halfcos(n: int):
            if n <= 0:
                return np.array([])
            t = np.arange(n) / n
            return 0.5 * (1 - np.cos(np.pi * t))

        def cos_switch(n: int):
            """
            Endpoint-exclusive cosine from +1 to -1 with zero discrete sum.
            Uses midpoint sampling: theta_k = (k + 0.5)*pi/n, k=0..n-1.
            """
            if n <= 0:
                return np.array([])
            k = np.arange(n, dtype=float)
            theta = (k + 0.5) * np.pi / n
            return np.cos(theta)  # strictly between +1 and -1, antisymmetric -> net zero

        L = int(self.length)
        F = int(self.flat_length)

        if F > L:
            raise ValueError(f"CosineBipolarPulse.flat_length={F} cannot exceed total length={L}.")
        if F % 2 != 0:
            raise ValueError(
                f"CosineBipolarPulse.flat_length={F} must be an even number to split " "equally into + and - halves."
            )
        if L - (self.smoothing_time + F) < 0:
            raise ValueError(
                f"CosineBipolarPulse.smoothing_time + flat_length ="
                f" {self.smoothing_time + F} exceeds total length={L}."
            )

        if self.smoothing_time == 0:
            rise_len = switch_len = fall_len = 0
        else:
            base = self.smoothing_time // 4
            extra = self.smoothing_time % 4
            rise_len = base + (1 if extra in (2, 3) else 0)
            switch_len = 2 * base + (1 if extra in (1, 3) else 0)
            fall_len = base + (1 if extra in (2, 3) else 0)

        flat_pos_len = F // 2
        flat_neg_len = F // 2

        A = float(self.amplitude)

        seg_rise = A * halfcos(rise_len)
        seg_flat_pos = A * np.ones(flat_pos_len)
        seg_switch = A * cos_switch(switch_len)
        seg_flat_neg = -A * np.ones(flat_neg_len)
        seg_fall = -A * halfcos(fall_len)[::-1]
        zero_padding = np.zeros(L - (self.smoothing_time + F))

        p = np.concatenate([seg_rise, seg_flat_pos, seg_switch, seg_flat_neg, seg_fall, zero_padding])

        if self.axis_angle is not None:
            p = p * np.exp(1j * self.axis_angle)

        return p.tolist()