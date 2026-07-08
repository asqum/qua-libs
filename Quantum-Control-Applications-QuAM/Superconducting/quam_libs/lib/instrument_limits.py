from dataclasses import dataclass
from typing import Union

from quam.components.channels import IQChannel, MWChannel

# Hard-coded multiplex setup: five qubits share one MW readout port.
num_qubits_sharing_mw_readout = 5
max_readout_wf_amplitude = 1.0 / num_qubits_sharing_mw_readout  # 0.2

opx1000_full_scale_powers_dbm: tuple[int, ...] = tuple(range(-11, 17))


@dataclass(frozen=True)
class InstrumentLimits:
    max_wf_amplitude: float
    max_x180_wf_amplitude: float
    units: str


def instrument_limits(channel: Union[IQChannel, MWChannel]) -> InstrumentLimits:
    if not (isinstance(channel, IQChannel) ^ isinstance(channel, MWChannel)):
        raise TypeError(
            f"Expected channel to be type IQChannel xor MWChannel for type checking, got {type(channel)}."
        )

    if isinstance(channel, MWChannel):
        limits = InstrumentLimits(
            max_wf_amplitude=1,  # MW-FEM max normalized amplitude
            max_x180_wf_amplitude=1.0,  # A subjective "safe" value for x180 pulses
            units="(scaled by `full_scale_power_dbm`)"
        )
    elif isinstance(channel, IQChannel):
        limits = InstrumentLimits(
            max_wf_amplitude=0.5,  # OPX+ and LF-FEM not in amplified-mode
            max_x180_wf_amplitude=0.3,  # A subjective "safe" value for x180 pulses
            units="V"
        )
    else:
        raise TypeError()

    return limits