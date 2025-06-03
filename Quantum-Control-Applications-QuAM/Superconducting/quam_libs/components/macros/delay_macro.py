from quam import quam_dataclass
from quam.components.macro import QubitMacro


__all__ = ["DelayMacro"]

@quam_dataclass
class DelayMacro(QubitMacro):
    def apply(self, duration) -> None:
        self.qubit.wait(duration)
