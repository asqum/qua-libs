import copy
import hashlib
import json
import pickle
from pathlib import Path
from typing import Literal
from matplotlib import pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.quantum_info import random_clifford
from qiskit.quantum_info import Clifford
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.circuit.library import *
from qiskit.quantum_info import Operator
from more_itertools import flatten


from scipy.optimize import curve_fit
import xarray
from tqdm.auto import tqdm

EPS = 1e-8


def rb_cache_key(
    seed: int,
    circuit_lengths: list[int] | tuple[int, ...],
    num_circuits_per_length: int,
    *,
    target_gate: str | None = None,
) -> str:
    """Match qualibration_graphs two_qubit_interleaved_rb cache key format."""
    payload = {
        "seed": seed,
        "circuit_lengths": sorted(circuit_lengths),
        "num_circuits_per_length": num_circuits_per_length,
    }
    if target_gate is not None:
        payload["target_gate"] = target_gate
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def rb_try_load(cache_dir: Path, key: str) -> dict | None:
    cache_path = cache_dir / f"{key}.pkl"
    if not cache_path.exists():
        return None
    with cache_path.open("rb") as handle:
        return pickle.load(handle)


def rb_save(cache_dir: Path, key: str, payload: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{key}.pkl"
    with cache_path.open("wb") as handle:
        pickle.dump(payload, handle)


def rb_decay_curve(x, A, alpha, B):
    """
    Exponential decay model for RB fidelity.

    Args:
        x (array-like): Circuit depths.
        A (float): Amplitude of the decay.
        alpha (float): Decay constant.
        B (float): Offset of the curve.

    Returns:
        np.ndarray: Calculated decay curve.
    """
    return A * alpha**x + B


class RBBase:
    
    def __init__(self, circuit_lengths: list[int], num_circuits_per_length: int, basis_gates: list[str] = ['cz', 'rz', 'sx', 'x'], 
                 num_qubits: int = 2, reduce_to_1q_cliffords: bool = False, seed: int | None = None,
                 show_progress: bool = True):
        
        self.num_qubits = num_qubits
        self.circuit_lengths = circuit_lengths
        self.num_circuits_per_length = num_circuits_per_length
        
        self.basis_gates = basis_gates
        self.seed = seed if seed is not None else np.random.randint(0, 1000000)
        self.rolling_seed = copy.deepcopy(self.seed)
        self.reduce_to_1q_cliffords = reduce_to_1q_cliffords
        self.show_progress = show_progress
    
    def generate_circuits_and_transpile(self, interleaved: bool = False):
        
        self.circuits = self.generate_circuits(interleaved)
        items = self.circuits.items()
        if self.show_progress:
            items = tqdm(items, total=len(self.circuits), desc="Transpiling RB circuit depths", unit="depth")
        self.transpiled_circuits = {
            length: self.transpile_circuits(circuits, depth=length) for length, circuits in items
        }
        
    def generate_circuits_per_length(self, length: int, interleaved: bool = False, progress=None) -> list[QuantumCircuit]:
        
        circuits = []
        
        for _ in range(self.num_circuits_per_length):
            qc = QuantumCircuit(self.num_qubits)
            clifford_product = Clifford(qc)  # Identity Clifford
            
            # Apply random Clifford gates
            for _ in range(length):
                if self.reduce_to_1q_cliffords and self.num_qubits == 2:
                    qc_temp = QuantumCircuit(2)
                    cliff = random_clifford(1, self.rolling_seed)
                    self.rolling_seed += 1
                    qc_temp.append(cliff, (0,))
                    cliff = random_clifford(1, self.rolling_seed)
                    self.rolling_seed += 1
                    qc_temp.append(cliff, (1,))
                    cliff = Clifford(qc_temp)
                else:
                    cliff = random_clifford(self.num_qubits, self.rolling_seed)
                    self.rolling_seed += 1
                
                qc.append(cliff, range(self.num_qubits))
                
                if interleaved:
                    if not hasattr(self, 'target_gate_instruction'):
                        raise AttributeError("The attribute 'target_gate_instruction' is not defined in the class.")
                    
                    qc.append(self.target_gate_instruction, range(self.num_qubits))
                    cliff = Clifford(self.target_gate_instruction) @ cliff
                
                clifford_product = cliff @ clifford_product  # Update the total Clifford
                if progress is not None:
                    progress.update(1)
            
            if length > 0:
                # Append the inverse Clifford
                inverse_clifford = clifford_product.adjoint()
                qc.append(inverse_clifford, range(self.num_qubits))
            
            # # Verify that the quantum circuit is an identity operator up to a phase
            # unitary = Operator(qc).data
            # identity = np.eye(unitary.shape[0])
            # # Normalize the unitary to remove global phase
            # unitary_normalized = unitary / np.linalg.det(unitary)**(1/unitary.shape[0])
            # assert np.allclose(unitary_normalized, identity, atol=1e-8), "Circuit is not an identity operator up to a phase."
            
            circuits.append(qc)
        
        return circuits
    
    def generate_circuits(self, interleaved: bool = False) -> dict[int, list[QuantumCircuit]]:
        
        total_cliffords = self.num_circuits_per_length * sum(self.circuit_lengths)
        pbar = (
            tqdm(total=total_cliffords, desc="Generating RB Cliffords", unit="cliff")
            if self.show_progress
            else None
        )
        try:
            circuits = {}
            for length in self.circuit_lengths:
                circuits[length] = self.generate_circuits_per_length(length, interleaved, progress=pbar)
            return circuits
        finally:
            if pbar is not None:
                pbar.close()
    
    def transpile_circuits(self, circuits: list[QuantumCircuit], depth: int | None = None) -> list[QuantumCircuit]:
        """Transpile full RB circuits in one Qiskit call per circuit (much faster than per-Clifford)."""
        circuit_iter = circuits
        if self.show_progress:
            label = f"Transpiling depth={depth}" if depth is not None else "Transpiling RB circuits"
            circuit_iter = tqdm(circuits, desc=label, leave=False, unit="circ")

        return [
            transpile(qc, basis_gates=self.basis_gates, optimization_level=1)
            for qc in circuit_iter
        ]

    def transpile_per_clifford(self, circuits: list[QuantumCircuit]) -> list[QuantumCircuit]:
        """Legacy path: transpile each Clifford gate separately (slow, kept for reference)."""
        transpiled_circuits = []
        
        for qc in circuits:
            transp_circ = QuantumCircuit(self.num_qubits)
            for instruction in qc:
                qc_per_inst = QuantumCircuit(len(instruction.qubits))
                qc_per_inst.append(instruction)
                
                if isinstance(instruction.operation, Clifford):
                    # if optimization level is > 1 one might get fractional angles
                    transpiled_gate = transpile(qc_per_inst, basis_gates=self.basis_gates, optimization_level=1)
                else:
                    transpiled_gate = qc_per_inst.copy()
                
                transp_circ = transp_circ.compose(transpiled_gate, front=False)
            
            transpiled_circuits.append(transp_circ)
        
        return transpiled_circuits
    
    def count_num_gates(self) -> int:
        return sum([len(qc) for qc in flatten(self.transpiled_circuits.values())])
    
    def plot_with_fidelity(self, data: xarray, num_averages: int):
        """
        Plots the RB fidelity as a function of circuit depth, including a fit to an exponential decay model.
        The fitted curve is overlaid with the raw data points.
        """
        A, alpha, B = self.fit_exponential(data, num_averages)
        fidelity = self.get_fidelity(alpha)

        plt.figure()
        plt.plot(self.circuit_lengths, self.get_decay_curve(data, num_averages), "o", label="Data")
        plt.plot(
            self.circuit_lengths,
            rb_decay_curve(np.array(self.circuit_lengths), A, alpha, B),
            "-",
            label=f"Fidelity={fidelity * 100:.3f}%\nalpha={alpha:.4f}",
        )
        plt.xlabel("Circuit Depth")
        plt.ylabel("Fidelity")
        plt.title("2Q Randomized Benchmarking Fidelity")
        plt.legend()
        plt.show()

    def fit_exponential(self, data: xarray, num_averages: int):
        """
        Fits the decay curve of the RB data to an exponential model.

        Returns:
            tuple: Fitted parameters (A, alpha, B) where:
                - A is the amplitude.
                - alpha is the decay constant.
                - B is the offset.
        """
        decay_curve = self.get_decay_curve(data, num_averages)

        popt, _ = curve_fit(rb_decay_curve, self.circuit_lengths, decay_curve, p0=[0.75, -0.1, 0.25], maxfev=10000)
        A, alpha, B = popt

        return A, alpha, B

    def get_fidelity(self, alpha):
        """
        Calculates the average fidelity per Clifford based on the decay constant.

        Args:
            alpha (float): Decay constant from the exponential fit.

        Returns:
            float: Estimated average fidelity per Clifford.
        """
        n_qubits = 2  # Assuming 2 qubits as per the context
        d = 2**n_qubits
        r = 1 - alpha - (1 - alpha) / d
        fidelity = 1 - r

        return fidelity

    def get_decay_curve(self, data: xarray, num_averages: int):
        """
        Calculates the decay curve from the RB data.

        Returns:
            np.ndarray: Decay curve representing the fidelity as a function of circuit depth.
        """
        return (data.state == 0).sum(("qubit", "num_circuits_per_length", "N")) / (self.num_circuits_per_length * num_averages)


class StandardRB(RBBase):
    
    def __init__(self, amplification_lengths: list[int], num_circuits_per_length: int, basis_gates: list[str] = ['cz', 'rz', 'sx', 'x'], 
                 num_qubits: int = 2, reduce_to_1q_cliffords: bool = False, seed: int | None = None,
                 show_progress: bool = True):
        
        super().__init__(amplification_lengths, num_circuits_per_length, basis_gates, num_qubits, reduce_to_1q_cliffords, seed, show_progress)
        
        self.generate_circuits_and_transpile()

class InterleavedRB(RBBase):
    
    def __init__(self, target_gate: Literal['cz', 'idle_2q'], amplification_lengths: list[int], num_circuits_per_length: int, basis_gates: list[str] = ['cz', 'rz', 'sx', 'x'], 
                 num_qubits: int = 2, reduce_to_1q_cliffords: bool = False, seed: int | None = None,
                 show_progress: bool = True):
        
        self.target_gate = target_gate
        self.target_gate_instruction = self.target_gate_to_instruction()
        
        super().__init__(amplification_lengths, num_circuits_per_length, basis_gates, num_qubits, reduce_to_1q_cliffords, seed, show_progress)
        
        self.generate_circuits_and_transpile(interleaved=True)
    
    def target_gate_to_instruction(self) -> Instruction:
        
        qc = QuantumCircuit(2)
        if self.target_gate == 'cz':
            qc.cz(0, 1)
        elif self.target_gate == 'idle_2q':
            qc.id((0, 1))
        else:
            raise ValueError(f"Target gate {self.target_gate} not supported")
        
        instruction = qc.to_instruction()
        instruction.name = self.target_gate
        return instruction

    
   