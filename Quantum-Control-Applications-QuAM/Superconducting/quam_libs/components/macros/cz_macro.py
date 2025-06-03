from quam.core import quam_dataclass
from quam.components.macro import QubitPairMacro

from quam_libs.components.gates import CZGate


__all__ = ["CZMacro"]

@quam_dataclass
class CZMacro(QubitPairMacro):
    cz_gate: CZGate

    def apply(self, **kwargs):
        self.cz_gate.execute()
