from dataclasses import dataclass
from typing import Union

from quam.components.channels import IQChannel, MWChannel

NUM_QUBITS_SHARING_MW_READOUT = 5
MAX_READOUT_WF_AMPLITUDE = 1.0 / NUM_QUBITS_SHARING_MW_READOUT
OPX1000_FULL_SCALE_POWERS_DBM = tuple(range(-11, 17, 3))


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
            max_x180_wf_amplitude=0.6,  # A subjective "safe" value for x180 pulses
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