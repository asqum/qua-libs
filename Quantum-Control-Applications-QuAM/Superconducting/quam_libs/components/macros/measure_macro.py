from typing import Union

from qm.qua import *
from quam import quam_dataclass
from quam.components.macro import QubitMacro
from quam.components.pulses import ReadoutPulse, Pulse
from quam_libs.components import ReadoutResonatorIQ
from quam_libs.components.macros.get_pulse_name import get_pulse_name

__all__ = ["MeasureMacro"]


@quam_dataclass
class MeasureMacro(QubitMacro):
    pulse: Union[ReadoutPulse, str] = "readout"

    def apply(self, **kwargs) -> QuaVariableType:
        state: QuaVariableType = kwargs.get("state", declare(bool))
        qua_vars = kwargs.get("qua_vars", (declare(fixed), declare(fixed)))
        pulse: ReadoutPulse = (
            self.pulse if isinstance(self.pulse, Pulse) else self.qubit.get_pulse(self.pulse)
        )

        resonator: ReadoutResonatorIQ = self.qubit.resonator
        resonator.measure(get_pulse_name(pulse), qua_vars=qua_vars)
        I, Q = qua_vars
        assign(state, I > pulse.threshold)
        wait(self.qubit.resonator.depletion_time // 4, self.qubit.resonator.name)

        return state
