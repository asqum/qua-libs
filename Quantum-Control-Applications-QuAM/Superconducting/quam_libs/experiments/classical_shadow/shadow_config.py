from dataclasses import dataclass, field
from typing import Literal, List, Union, Optional, Dict, Callable, Any, TYPE_CHECKING

from qiskit.circuit.library import SXGate, ZGate
from ...components import Transmon, QuAM
from ..two_qubit_xeb import QUAGate, QUAGateSet
from qiskit.circuit import QuantumCircuit
from .additional_gates import SYdgGate
from qiskit.quantum_info import Operator
import numpy as np


@dataclass
class ShadowConfig:
    """
    Configuration for the classical shadow experiment.

    Args:

    shadow_size: int = Number of shots/snapshots to construct the shadow
    shots_per_snapshot: int = Number of shots per snapshot
    input_state_circuit: Callable[[Any], QuantumCircuit] = Input state circuit
    qubits: List[Transmon] = Qubits to measure
    measurement_basis: Optional[Dict[int, QuantumCircuit]] = Circuits for random unitaries to be applied before measurement (optional, if not provided, Pauli measurements are done through conditional plays of the measurement basis rotations)
    input_state_circuit_kwargs: Optional[Dict[str, Any]] = Input state circuit kwargs
    readout_qubits: Optional[List[Transmon]] = Readout qubits
    readout_pulse_name: str = Readout pulse name
    reset_method: Literal["active", "cooldown"] = Reset method

    """

    shadow_size: int
    shots_per_snapshot: int
    input_state_circuit: Callable[[Any], QuantumCircuit]
    qubits: List[Transmon]
    measurement_basis: Optional[Dict[int, QuantumCircuit]] = None
    input_state_circuit_kwargs: Optional[Dict[str, Any]] = None
    readout_qubits: Optional[List[Transmon]] = None
    readout_pulse_name: str = "readout"
    reset_method: Literal["active", "cooldown"] = "cooldown"
    reset_kwargs: Optional[Dict[str, Union[float, str, int]]] = field(
        default_factory=lambda: {
            "cooldown_time": 20,
            "max_tries": None,
            "pi_pulse": None,
        }
    )
    gate_indices: List[List[int]] | np.ndarray | None = None
    save_dir: str = ""
    should_save_data: bool = True
    data_folder_name: Optional[str] = None
    generate_new_data: bool = True
    seed: int = 1234

    def __post_init__(self):
        self.n_qubits = len(self.qubits)
        self.dim = 2**self.n_qubits

        if self.gate_indices is not None:
            if isinstance(self.gate_indices, list):
                self.gate_indices = np.array(self.gate_indices)
            self.gate_indices = self.gate_indices.astype(int)
            if self.gate_indices.ndim != 2:
                raise ValueError("gate_indices must be a 2D array")
            if self.gate_indices.shape[1] != self.n_qubits:
                raise ValueError(
                    "gate_indices must have the same number of columns as the number of qubits"
                )
            if self.gate_indices.shape[0] != self.shadow_size:
                raise ValueError(
                    "gate_indices must have the same number of rows as the shadow size"
                )
            # Check if there is no index that is negative or higher than length of dictionary of macros
            if self.measurement_basis is not None:
                if not isinstance(self.measurement_basis, dict):
                    raise ValueError("measurement_basis must be a dictionary")
                if not all(
                    isinstance(key, int) and key >= 0
                    for key in self.measurement_basis.keys()
                ):
                    raise ValueError("measurement_basis keys must be positive integers")
                if not all(
                    isinstance(value, QuantumCircuit)
                    for value in self.measurement_basis.values()
                ):
                    raise ValueError(
                        "measurement_basis values must be QuantumCircuit objects"
                    )
                if any(
                    index < 0 or index > len(self.measurement_basis) - 1
                    for index in self.gate_indices.flatten()
                ):
                    raise ValueError(
                        "gate_indices must contain only indices that are within the range of the measurement basis"
                    )
                for i in range(len(self.measurement_basis)):
                    try:
                        op = Operator(self.measurement_basis[i])
                        assert op.num_qubits == 1, (
                            "All unitary operations in the measurement basis must be single qubit operations, but {i} has {op.num_qubits} qubits"
                        )
                        assert op.is_unitary(), (
                            "All unitary operations in the measurement basis must be unitary, but {i} is not"
                        )
                    except Exception as e:
                        raise ValueError(
                            f"Measurement basis {i} is not a valid unitary operation (make sure to not add the measurement instruction): {e}"
                        )

    def as_dict(self):
        """
        Return the ShadowConfig object as a dictionary
        """
        config_dict = {
            "shadow_size": self.shadow_size,
            "measurement_basis": self.measurement_basis,
            "qubits": [
                qubit.name if isinstance(qubit, Transmon) else qubit
                for qubit in self.qubits
            ],
            "seed": self.seed,
        }
        return config_dict

    @property
    def random_unitary_set(self):
        """
        Return the unitary operations of the measurement basis
        """
        if self.measurement_basis is None:
            return {
                0: SXGate().to_matrix(),
                1: SYdgGate().to_matrix(),
                2: ZGate().to_matrix(),
            }
        return {
            i: Operator(self.measurement_basis[i]).to_matrix()
            for i in range(len(self.measurement_basis))
        }

    @classmethod
    def from_dict(cls, config_dict: Dict, machine: Optional[QuAM] = None):
        """
        Create a ShadowConfig object from a dictionary
        """
        qubits_names = config_dict["qubits"]
        qubits = [
            machine.qubits[name] if machine is not None else name
            for name in qubits_names
        ]
        config_dict["qubits"] = qubits
        config_dict["measurement_basis"] = QUAGateSet(config_dict["measurement_basis"])
        return cls(**config_dict)
