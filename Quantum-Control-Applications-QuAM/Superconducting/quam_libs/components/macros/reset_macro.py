from typing import Literal, Union

from quam import quam_dataclass
from quam.components.macro import QubitMacro
from quam.components.pulses import Pulse, ReadoutPulse

from quam_libs.components.macros.get_pulse_name import get_pulse_name

__all__ = ["ResetMacro"]


@quam_dataclass
class ResetMacro(QubitMacro):
    reset_type: Literal["active", "thermalize"] = "active"
    pi_pulse: Union[Pulse, str] = "x180"
    readout_pulse: Union[ReadoutPulse, str] = "readout"
    max_attempts: int = 15
    thermalize_time: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.max_attempts > 0, "max_attempts must be greater than 0"

    def apply(self, **kwargs) -> None:

        pi_pulse: Pulse = (
            self.pi_pulse
            if isinstance(self.pi_pulse, Pulse)
            else self.qubit.get_pulse(self.pi_pulse)
        )
        readout_pulse: ReadoutPulse = (
            self.readout_pulse
            if isinstance(self.readout_pulse, Pulse)
            else self.qubit.get_pulse(self.readout_pulse)
        )
        if self.reset_type == "active":
            from quam_libs.macros import active_reset
            active_reset(self.qubit, pi_pulse_name=get_pulse_name(pi_pulse),
                         readout_pulse_name=get_pulse_name(readout_pulse),
                         max_attempts=self.max_attempts)
            # self.qubit.reset_qubit_active(
            #     pi_pulse_name=get_pulse_name(pi_pulse),
            #     readout_pulse_name=get_pulse_name(readout_pulse),
            #     max_attempts=self.max_attempts,
            # )
        else:
            # Thermalize the qubit
            self.qubit.wait(self.thermalize_time // 4)
