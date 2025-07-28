from quam_libs.components import QuAM, Transmon
from quam_libs.experiments.classical_shadow import ClassicalShadow, ShadowConfig, SYdgGate
from quam_libs.experiments.two_qubit_xeb.qua_gate import QUAGate
from qualang_tools.units import unit
from qiskit.circuit import QuantumCircuit
from qm import generate_qua_script
import numpy as np

u = unit(coerce_to_integer=True)
machine = QuAM.load()
qubits  = machine.active_qubits
readout_qubit_indices = [0, 1, 2, 3, 4]
readout_qubits = [qubits[i] for i in readout_qubit_indices]
target_qubit_indices = [0]
target_qubits = [qubits[i] for i in target_qubit_indices]


def sx_macro(qubit: Transmon):
    qubit.xy.play("x90")
    
def sy_macro(qubit: Transmon):
    qubit.xy.play("-y90")
    
def z_macro(qubit: Transmon):
    qubit.wait(4)
    
def input_state_macro(*, angle):
    # RY(angle)
    q0 = target_qubits[0]
    q0.xy.play("x90")
    q0.xy.frame_rotation(angle + np.pi)
    q0.xy.play("x90")
    q0.xy.frame_rotation2pi(0.5)
   
def input_state_circuit(*, angle: float) -> QuantumCircuit:
    qc = QuantumCircuit(1)
    qc.ry(angle, 0)
    
    return qc

measurement_basis = {0: QUAGate("sx", sx_macro),
                     1: QUAGate(SYdgGate(), sy_macro),
                     2: QUAGate("z", z_macro)}

shadow_size = 10 # Number of shots/snapshots to construct the shadow
seed = 1234
np.random.seed(seed)
# Define custom snapshots here if needed (otherwise, sampling is done in real time)
gate_indices = np.random.randint(0, 3, (shadow_size, len(target_qubits)))
wait_duration = 0.1*u.us

input_macro_kwargs = {}
shadow_config = ShadowConfig(shadow_size=shadow_size,
                             shots_per_snapshot=128,
                            input_state_prep_macro=input_state_macro,
                            input_state_circuit=input_state_circuit,
                            measurement_basis=measurement_basis,
                            qubits=target_qubits,
                            readout_qubits=readout_qubits,
                            readout_pulse_name="readout",
                            reset_method="cooldown", #"active",
                            reset_kwargs={"cooldown_time": 80*u.us,
                                          "max_tries": 5,
                                          "pi_pulse": "x180"},
                            input_state_prep_macro_kwargs=input_macro_kwargs,
                            # gate_indices=gate_indices,
                            # num_angles=100,  # Specify number of angles to sample (creates: np.linspace(0, np.pi, num_angles))
                            seed=seed,
                            )

shadow_exp = ClassicalShadow(shadow_config, machine)

# print("Generating QUA script...")
# print(generate_qua_script(shadow_exp.cs_prog(simulate=False)))
job = shadow_exp.run()
# New axis: angle
# Each element in the results corresponds to a collection of snapshots for a given angle defining the input state.
results = job.result() # [({"010": 2, "110": 3, ...}, [0, 1, 2]), ({"101": 5, "100": 4, ...}, [2, 0, 1]), ...]
ideal_results = job.ideal_result()

print("Results:")
print(results)
print("Ideal results:")
print(ideal_results)

gate_dict = {i: qua_gate.gate for i, qua_gate in measurement_basis.items()}

# Result format: List of (bitstring, random_gate_indices) of size shadow_size

#TODO: Add post-processing for shadow tomography below
    