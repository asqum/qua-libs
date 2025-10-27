from quam_libs.components import QuAM, Transmon
from quam_libs.experiments.classical_shadow import ClassicalShadow, ShadowConfig, SYdgGate
from quam_libs.experiments.two_qubit_xeb.qua_gate import QUAGate
from qualang_tools.units import unit
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import SXGate, ZGate
from qm import generate_qua_script
import numpy as np

u = unit(coerce_to_integer=True)
machine = QuAM.load()
qubits  = machine.active_qubits
readout_qubit_indices = [0, 1, 2, 3, 4]
readout_qubits = [qubits[i] for i in readout_qubit_indices]
target_qubit_indices = [0]
target_qubits = [qubits[i] for i in target_qubit_indices]

measurement_basis = {i: QuantumCircuit(1) for i in range(3)}
# Create the measurement basis circuits
# 0: U =  H
# 1: U =  Sdg@H
# 2: U =  I
measurement_basis[0].h(0)
measurement_basis[1].sdg(0)
measurement_basis[1].h(0)


def input_state_circuit(*, wait_duration: int) -> QuantumCircuit:
    # TODO: Add input state Qiskit QuantumCircuit here
    qc = QuantumCircuit(len(target_qubits))
    for i in range(len(target_qubits)):
        qc.x(i)
    qc.delay(wait_duration, target_qubits, unit='ns')
    return qc

shadow_size = 100 # Number of shots/snapshots to construct the shadow
seed = 1234
np.random.seed(seed)
# Define custom snapshots here if needed (otherwise, sampling is done in real time)
gate_indices = np.random.randint(0, 3, (shadow_size, len(target_qubits)))
wait_duration = 0.1*u.us

input_circuit_kwargs = {"wait_duration": wait_duration}
shadow_config = ShadowConfig(shadow_size=shadow_size,
                             shots_per_snapshot=128,
                            input_state_circuit=input_state_circuit,
                            measurement_basis=measurement_basis,
                            qubits=target_qubits,
                            readout_qubits=readout_qubits,
                            readout_pulse_name="readout",
                            reset_method="cooldown", #"active",
                            reset_kwargs={"cooldown_time": 80*u.us,
                                          "max_tries": 5,
                                          "pi_pulse": "x180"},
                            input_state_circuit_kwargs=input_circuit_kwargs,
                            # gate_indices=gate_indices,
                            seed=seed,
                             )

shadow_exp = ClassicalShadow(shadow_config, machine)

# print("Generating QUA script...")
# print(generate_qua_script(shadow_exp.cs_prog(simulate=False)))
job = shadow_exp.run()
# Each element in the results corresponds to a snapshot of the shadow (with a different random basis, and the counts
# for each bitstring sampled per snapshot).
results = job.result() # [({"010": 2, "110": 3, ...}, [0, 1, 2]), ({"101": 5, "100": 4, ...}, [2, 0, 1]), ...]
ideal_results = job.ideal_result()

print("Results:")
print(results)
print("Ideal results:")
print(ideal_results)

# Accessing the unitary operations associated to the measurement basis
gate_dict = shadow_config.random_unitary_set
print("Gate dictionary:")
print(gate_dict)

# Result format: List of (bitstring, random_gate_indices) of size shadow_size

#TODO: Add post-processing for shadow tomography below
    