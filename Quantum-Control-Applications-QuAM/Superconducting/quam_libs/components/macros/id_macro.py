from quam import quam_dataclass
from quam.components.macro import QubitMacro


__all__ = ["IdMacro"]

@quam_dataclass
class IdMacro(QubitMacro):
    """
    Identity macro for a qubit.
    This macro does not perform any operation on the qubit.
    It is used to ensure that the qubit is in a valid state.
    """

    def apply(self, **kwargs) -> None:
        # No operation is performed
        self.qubit.align()
