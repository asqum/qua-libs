import warnings
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import Statevector
from qualang_tools.results import DataHandler

from quam_libs.components import QuAM, Transmon
from qm.qua import *
from qm import SimulationConfig, QuantumMachinesManager, generate_qua_script, Program
from qm.jobs.running_qm_job import RunningQmJob
from qm.jobs.simulated_job import SimulatedJob
from .shadow_config import ShadowConfig

from qualang_tools.units import unit

from ..two_qubit_xeb.macros import qua_declaration, reset_qubit, binary

u = unit(coerce_to_integer=True)

def qiskit_to_qua_macro(circuit: QuantumCircuit, machine: QuAM):
    qubit_pairs_mapping = {qubit_pair.name: (machine.active_qubits.index(qubit_pair.qubit_control), machine.active_qubits.index(qubit_pair.qubit_target)) for qubit_pair in machine.active_qubit_pairs}
    
    coupling_map = CouplingMap(list(qubit_pairs_mapping.values()))
    qc = transpile(circuit, basis_gates=["sx", "x", "rz", "cz"], coupling_map=coupling_map)
    qubit_indices = {qubit: qc.find_bit(qubit).index for i, qubit in enumerate(qc.qubits)}
    for instruction in qc.data:
        try:
            qubits = instruction.qubits
            if len(qubits) == 2:
                qubit_control = machine.active_qubits[qubit_indices[qubits[0]]]
                qubit_target = machine.active_qubits[qubit_indices[qubits[1]]]
                qubit_pair = qubit_control @ qubit_target
                qubit_pair.apply(instruction.operation.name, *instruction.operation.params)
            elif len(qubits) == 1:
                qubit = machine.active_qubits[qubit_indices[qubits[0]]]
                qubit.apply(instruction.operation.name, *instruction.operation.params)
            else:
                raise ValueError(f"Unsupported number of qubits: {len(qubits)}")
        except Exception as e:
            print(f"Error processing instruction: {instruction}")
            raise e
   
class ClassicalShadow:
    def __init__(
        self,
        config: ShadowConfig,
        machine: QuAM,
    ):
        """
        Initialize the ClassicalShadow experiment.

        Args:
            config (ShadowConfig): Configuration for the classical shadow experiment.
            machine (QuAM): QuAM object containing the qubits and qubit pairs used in the experiment.
        """
        self.config = config
        self.machine = machine
        self.data_handler = DataHandler(name="classical_shadow",
                                        root_data_folder=self.config.save_dir)
        
    
    def cs_prog(self, simulate: bool = False) -> Program:
        
        """
        Generate the QUA script for the classical shadow experiment.
        """
        n_qubits = self.config.n_qubits
        dim = self.config.dim
        random_gates = len(self.config.measurement_basis)
        ge_thresholds = [qubit.resonator.operations[self.config.readout_pulse_name].threshold 
                         for qubit in self.config.qubits]
        
        with program() as cs_prog:
            I, I_st, Q, Q_st = qua_declaration(n_qubits=n_qubits, 
                                               readout_elements=[qubit.resonator for qubit in self.config.qubits],)
            random_basis = declare(int, size=n_qubits)
            random_basis_stream = declare_stream()
            state = declare(bool, size=self.config.n_qubits)
            state_int = declare(int, value=0)
            state_int_stream = declare_stream()
            r = Random(seed=self.config.seed)
            i = declare(int)
            j = declare(int)
            shot = declare(int)
            if self.config.gate_indices is not None:
                gate_indices = [declare(int,
                                         value=self.config.gate_indices[:, n].tolist()) for n in range(n_qubits)]
            
            self.machine.apply_all_flux_to_min()
            self.machine.apply_all_couplers_to_min()
            
            if simulate:
                for qubit in self.config.qubits:
                    qubit.xy.update_frequency(0)
                    
            with for_(i, 0, i < self.config.shadow_size, i + 1):
                # Possible wait time before the experiment
                # wait(...)
                if self.config.gate_indices is not None:
                    for n in range(n_qubits):
                        assign(random_basis[n], gate_indices[n][i])
                        save(random_basis[n], random_basis_stream)
                else:
                    # Sample random basis (assumed to be local measurements)
                    with for_(j, 0, j < n_qubits, j + 1):
                        assign(random_basis[j], r.rand_int(random_gates))
                        save(random_basis[j], random_basis_stream)

                with for_(shot, 0, shot < self.config.shots_per_snapshot, shot + 1):
                    # Prepare state
                    if self.config.input_state_circuit_kwargs: #is not None:
                        qiskit_to_qua_macro(self.config.input_state_circuit(**self.config.input_state_circuit_kwargs), self.machine)
                    else:
                        qiskit_to_qua_macro(self.config.input_state_circuit(), self.machine)
                    align()
                    # for q, qubit, in enumerate(self.config.qubits):
                    #     with switch_(random_basis[q], unsafe=False):
                    #         # Apply the random basis rotation
                    #         for k in range(random_gates):
                    #             with case_(k):
                    #                 self.config.measurement_basis[k].gate_macro(qubit)
                    # align()
                    # Readout
                    for q, qubit, in enumerate(self.config.qubits):
                        # Replace switch case with conditional plays of the measurement basis rotations
                        qubit.xy.play("x90", condition=random_basis[q] == 0)
                        qubit.xy.play("-y90", condition=random_basis[q] == 1)
                        
                        
                    # Play the readout on the other resonator to measure in the same condition as when optimizing readout
                    for other_qubit in self.config.readout_qubits:
                        if other_qubit.resonator not in [qubit.resonator for qubit in self.config.qubits]:
                            other_qubit.resonator.play("readout")
                    for q, qubit, in enumerate(self.config.qubits):
                        # qubit.align()
                        qubit.resonator.measure(self.config.readout_pulse_name,
                                                qua_vars=(I[q], Q[q]))
                        # State Estimation: returned as integer
                        assign(state[q], I[q] > ge_thresholds[q])
                        assign(state_int, state_int + (1<<q) * Cast.to_int(state[q]))

                        reset_qubit(self.config.reset_method,
                                    qubit,
                                    threshold=ge_thresholds[q],
                                    **self.config.reset_kwargs)
                    save(state_int, state_int_stream)
                    assign(state_int, 0)
                
            with stream_processing():
                random_basis_stream.buffer(n_qubits).save_all("random_basis")
                state_int_stream.buffer(self.config.shots_per_snapshot).save_all("state_int")
        
        return cs_prog
    
    def run(self, simulate: bool = False,
            simulation_config: Optional[SimulationConfig] = None,
            qmm_cloud_simulator: Optional[QuantumMachinesManager] = None,
            **simulate_kwargs):
        config = self.machine.generate_config()
        if simulation_config is None:
            simulation_config = SimulationConfig(
                duration=10_000
            )
        cs_prog = self.cs_prog(simulate=simulate)
        if simulate and qmm_cloud_simulator is not None:
            qmm = qmm_cloud_simulator
        else:
            qmm = self.machine.connect()
        
        qm = qmm.open_qm(config)
        if simulate:
            with open("debug.py", "w+") as f:
                f.write(generate_qua_script(cs_prog, config))
            job = qm.simulate(cs_prog, simulate=simulation_config, **simulate_kwargs)
            
        elif self.config.generate_new_data:
            job = qm.execute(cs_prog)
        else:
            warnings.warn("No new data will be generated. Please set generate_new_data to True to generate new data.")
            return 
        
        return ClassicalShadowJob(job, self.config, self.data_handler)
        
            
class ClassicalShadowJob:
    def __init__(self, job: RunningQmJob | SimulatedJob, config: ShadowConfig, data_handler: DataHandler):
        """
        Initialize the ClassicalShadowJob object.

        Args:
            job (RunningQmJob | SimulatedJob): The job object returned by the QUA program.
            config (ShadowConfig): Configuration for the classical shadow experiment.
            data_handler (DataHandler): Data handler for saving and processing results.
        """
        self.job = job
        self._result_handles = self.job.result_handles
        self._result_handles.wait_for_all_values()
        self.config = config
        self.data_handler = data_handler
        self._gate_indices = np.zeros((self.config.shadow_size, self.config.n_qubits), dtype=int)
        
        
    def _get_circuits(self):
        """
        Get the circuits from the job.
        """
        shadow_size = self.config.shadow_size
        gates = self._result_handles["random_basis"].fetch_all()['value']
        input_state_circuit = self.config.input_state_circuit(**self.config.input_state_prep_macro_kwargs)
        for i in range(shadow_size):
            for j in range(self.config.n_qubits):
                self._gate_indices[i, j] = gates[i][j]
                
        circuits = [QuantumCircuit(self.config.n_qubits) for _ in range(shadow_size)]
        for i in range(shadow_size):
            circuits[i].compose(input_state_circuit, inplace=True)
            for j in range(self.config.n_qubits):
                circuits[i].append(self.config.measurement_basis[self._gate_indices[i, j]].gate,
                                   [j])
        return circuits
    
    def result(self):
        """
        Get the result of the job.
        """
        state_ints = self._result_handles["state_int"].fetch_all()['value']
        bitstrings = []
        for i, state_int in enumerate(state_ints):
            # Count all occurences of each bitstring and build a dictionary of counted bitstrings
            counts = {binary(i, self.config.n_qubits): 0 for i in range(self.config.dim)}
            for j in range(len(state_int)):
                bitstring = binary(state_int[j], self.config.n_qubits)
                counts[bitstring] += 1
            bitstrings.append(counts)

        return [(bitstring, self._gate_indices[i]) for i, bitstring in enumerate(bitstrings)]
    
    
    def ideal_result(self):
        """
        Get the ideal results of the job.
        """
        circuits = self._get_circuits()
        results = []
        for i, circuit in enumerate(circuits):
            state = Statevector(circuit)
            probs = state.probabilities_dict()
            results.append((probs, self._gate_indices[i]))
            
        return results
            
            
            
            
     