from quam import quam_dataclass
from quam.components.macro import QubitMacro


__all__ = ["VirtualZMacro"]

@quam_dataclass
class VirtualZMacro(QubitMacro):
    def apply(self, angle: float) -> None:
        self.qubit.xy.frame_rotation(-angle)
