from qiskit.circuit.library import get_standard_gate_name_mapping as gate_map
from qiskit.circuit import QuantumCircuit, Gate
import numpy as np

class SYdgGate(Gate):
    def __init__(self, label=None):
        super().__init__("sydg", 1, [], label=label)

    def _define(self):
        qc = QuantumCircuit(1)
        qc.ry(-np.pi / 2, 0)
        self.definition = qc

    def inverse(self, annotated: bool = False):
        qc = QuantumCircuit(1)
        qc.ry(np.pi / 2, 0)
        return qc.to_gate()

    def __eq__(self, other):
        return isinstance(other, SYdgGate)

    def __array__(self, dtype=None, copy=None):
        return gate_map()["ry"](-np.pi / 2).__array__(dtype=dtype, copy=copy)
