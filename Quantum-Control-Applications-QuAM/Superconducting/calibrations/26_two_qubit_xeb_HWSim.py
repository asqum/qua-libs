import numpy as np
from quam_libs.components import QuAM, TransmonPair
from quam_libs.experiments.two_qubit_xeb import (
    XEBConfig,
    XEB,
    backend as fake_backend,
    QUAGate,
)

machine = QuAM.load()
qubits = machine.active_qubits
# Get the relevant QuAM components
readout_qubit_indices = [0,1,2,3,4]  # Indices of the target qubits
readout_qubits = [qubits[i] for i in readout_qubit_indices]
target_qubit_indices = [0,1]  # Indices of the target qubits
target_qubits = [qubits[i] for i in target_qubit_indices]
target_qubit_pairs = [
    qubit_pair
    for qubit_pair in machine.active_qubit_pairs
    if qubit_pair.qubit_control in target_qubits and qubit_pair.qubit_target in target_qubits

]

from qm.qua import frame_rotation_2pi, align, wait
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

def cz_gate(qubit_pair: TransmonPair):
    """
    CZ gate QUA macro: Add your own QUA code here implementing your CZ gate for any given qubit pair
    :param qubit_pair: TransmonPair instance on which to apply the gate
    :return: None
    """

    q1 = qubit_pair.qubit_control # flux-tuned-down-qubit (flux-tuned)
    q2 = qubit_pair.qubit_target # flux-tuned-up-qubit (to-meet-with)  

    if q1.id=="q5" and q2.id=="q4": phi_to_flux_tune, phi_to_meet_with = -0.278, -0.830
    if q1.id=="q3" and q2.id=="q4": phi_to_flux_tune, phi_to_meet_with = 0.384, 0.449
    if q1.id=="q3" and q2.id=="q2": phi_to_flux_tune, phi_to_meet_with = 0.484, 0.374
    if q1.id=="q1" and q2.id=="q2": phi_to_flux_tune, phi_to_meet_with = 0.880, -0.404 

    q1.z.play("cz%s_%s"%(q1.name.replace("q",""),q2.name.replace("q","")))
    qubit_pair.coupler.play("cz")
    align(q1.z.name, qubit_pair.coupler.name, q1.xy.name, q2.xy.name)
    # align()
    frame_rotation_2pi(phi_to_flux_tune, q1.xy.name)
    frame_rotation_2pi(phi_to_meet_with, q2.xy.name)
    align()
    wait(40 * u.ns)


cz_qua = QUAGate("cz", cz_gate)

xeb_config = XEBConfig(
    seqs=88, #128, #81,
    # depths=np.arange(1, 1200, 24),
    depths=np.arange(1, 32, 1),
    # depths=list(np.arange(1, 9, 1)),
    n_shots=512, #1000,
    readout_qubits=readout_qubits, 
    qubits=target_qubits,
    qubit_pairs=target_qubit_pairs,
    baseline_gate_name="x90",
    gate_set_choice="sw",
    two_qb_gate=cz_qua, #cz_qua, None
    save_dir="xeb_data/QCage_5q4c",
    should_save_data=False, #True,
    generate_new_data=True,
    disjoint_processing=False, #False,
    # reset_method="active",
    # reset_kwargs={"max_tries": 3, "pi_pulse": "x180"},
    reset_method="cooldown", #"active",
    reset_kwargs={"cooldown_time": 100, "max_tries": 3, "pi_pulse": "x180"},
)

print("target_qubits: %s" %[q.name for q in target_qubits]) 
# print("qubit_control: %s" %(qubits[0]@qubits[1]).qubit_control)
xeb_runtime = xeb_config.seqs * len(xeb_config.depths) * xeb_config.n_shots / (150 * 27 * 700) * 19.35
print("time required: %s min" % (xeb_runtime))


from qm import SimulationConfig

simulate = False  # Set to True to simulate the experiment with Qiskit Aer instead of running it on the QPU
xeb = XEB(xeb_config, machine=machine)
if simulate:
    job = xeb.simulate(backend=fake_backend)
else:
    job = xeb.run(simulate=True, simulation_config=SimulationConfig(duration=200000))  # If simulate is False, job is run on the QPU, else pulse output is simulated


job.plot_simulated_samples()



