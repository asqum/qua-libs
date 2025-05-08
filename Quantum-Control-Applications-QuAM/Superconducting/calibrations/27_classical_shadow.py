import numpy as np
from quam_libs.components import QuAM, Transmon
from quam_libs.experiments.classical_shadow import ClassicalShadow, ShadowConfig, SYdgGate
from quam_libs.experiments.two_qubit_xeb.qua_gate import QUAGate
from qm.qua import *
from qualang_tools.units import unit

u = unit()
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
    qubit.wait(1)
    
def input_state_macro(*, wait_duration: int):
    q0 = target_qubits[0]
    q0.xy.play("x180")
    q0.wait(u.to_clock_cycles(wait_duration))
    
measurement_basis = {0: QUAGate("sx", sx_macro),
                     1: QUAGate(SYdgGate(), sy_macro),
                     2: QUAGate("z", z_macro)}

shadow_size = 1000 # Number of shots/snapshots to construct the shadow
wait_duration = 0.1*u.ms
input_macro_kwargs = {"wait_duration": wait_duration}
shadow_config = ShadowConfig(shadow_size=shadow_size,
                            input_state_prep_macro=input_state_macro,
                            measurement_basis=measurement_basis,
                            qubits=target_qubits,
                            readout_qubits=readout_qubits,
                            readout_pulse_name="readout",
                            reset_method="active",
                            reset_kwargs={"cooldown_time": 20*u.us,
                                          "max_tries": 10,
                                          "pi_pulse": "x180"},
                            input_state_prep_macro_kwargs=input_macro_kwargs,
                             )

shadow_exp = ClassicalShadow(shadow_config, machine)

job = shadow_exp.run()

results = job.result()

# Result format: List of (bitstring, random_gate_indices) of size shadow_size

#TODO: Add post-processing for shadow tomography below
    