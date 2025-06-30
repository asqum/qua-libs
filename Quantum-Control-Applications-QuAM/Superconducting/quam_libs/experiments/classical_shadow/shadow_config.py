from dataclasses import dataclass, field
from typing import Literal, List, Union, Optional, Dict, Callable, Any
from ...components import Transmon, TransmonPair, QuAM
from ..two_qubit_xeb import QUAGate, QUAGateSet
from qiskit.circuit import QuantumCircuit
@dataclass
class ShadowConfig:
    """
    Configuration for the classical shadow experiment.

    Args:
        
    """
    shadow_size: int
    shots_per_snapshot: int
    input_state_prep_macro: Callable[[Any], None]
    input_state_circuit: Callable[[Any], QuantumCircuit]
    measurement_basis: Union[str, Dict[int, QUAGate]]
    qubits: List[Transmon]
    input_state_prep_macro_kwargs: Optional[Dict[str, Any]] = None
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
    save_dir: str = ""
    should_save_data: bool = True
    data_folder_name: Optional[str] = None
    generate_new_data: bool = True
    seed: int = 1234
    
    def __post_init__(self):
        self.n_qubits = len(self.qubits)
        self.dim = 2**self.n_qubits
        self.measurement_basis = QUAGateSet(self.measurement_basis)
        
    def as_dict(self):
        """
        Return the ShadowConfig object as a dictionary
        """
        config_dict = {
            "shadow_size": self.shadow_size,
            "measurement_basis": self.measurement_basis,
            "qubits": [qubit.name if isinstance(qubit, Transmon) else qubit for qubit in self.qubits],
            "seed": self.seed,
        }
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict, machine: Optional[QuAM] = None):
        """
        Create a ShadowConfig object from a dictionary
        """
        qubits_names = config_dict["qubits"]
        qubits = [machine.qubits[name] if machine is not None else name for name in qubits_names]
        config_dict["qubits"] = qubits
        config_dict["measurement_basis"] = QUAGateSet(config_dict["measurement_basis"])
        return cls(**config_dict)