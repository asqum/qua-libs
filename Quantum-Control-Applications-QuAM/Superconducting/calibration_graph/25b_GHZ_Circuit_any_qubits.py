"""
This program generates a multi-qubit GHZ state for an arbitrary number of qubits.
"""

# %% {Imports}
import re
from itertools import product
from typing import Optional, List, Literal

import matplotlib
import matplotlib.pyplot as plt
from collections import Counter

from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.units import unit
from qualibrate import NodeParameters, QualibrationNode
from quam_libs.components import QuAM
from quam_libs.macros import active_reset, readout_state

# matplotlib.use("TKAgg")


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ["q1","q2","q3","q4","q5"]
    control_qubit_post_cz_phase_corrections: List[float] = [0.263, 0.484, 0.384, -0.278] #None
    target_qubit_post_cz_phase_corrections: List[float] = [-0.459, 0.374, 0.449, -0.830] #None
    num_averages: int = 1000
    # flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    reset_type: Literal['active', 'thermal'] = "thermal"
    simulate: bool = False
    timeout: int = 100
    shots: int = 2048

node = QualibrationNode(
    name="GHZ_Circuit", parameters=Parameters()
)

###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handle units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
qubits = [machine.qubits[q] for q in node.parameters.qubits]

def get_qubit_index_from_name(qubit_name: str) -> int:
    """ Pattern-matching to extract qubit index from name string. """
    return int(re.search(r"(?<=q)\d+", qubit_name).group())

num_qubits_full = len(qubits)

qubit_pairs = []
for i in range(len(qubits) - 1):
    try:
        qubit_pair = (qubits[i] @ qubits[i+1])
    except:
        try:
            qubit_pair = (qubits[i+1] @ qubits[i])
        except:
            raise ValueError(f"No pair found between qubits {qubits[i].name} and {qubits[i+1].name}")

    qubit_pairs.append(qubit_pair)

readout_qubits = [qubit for qubit in machine.qubits.values() if qubit not in qubits]


with program() as ghz_circuit:
    state = [declare(int) for _ in range(len(qubits))]
    state_st = [declare_stream() for _ in range(len(qubits))]
    n = declare(int)
    n_st = declare_stream()

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()
    machine.apply_all_couplers_to_min()

    with for_(n, 0, n < node.parameters.shots, n+1):
        save(n, n_st)

        if node.parameters.reset_type == "active":
            for qubit in qubits:
                active_reset(qubit)
        else:
            if not node.parameters.simulate:
                wait(machine.thermalization_time * u.ns)

        align()

        for i, qp in enumerate(qubit_pairs):
            qc = qp.qubit_control
            qt = qp.qubit_target

            qc_index = int(get_qubit_index_from_name(qc.name))
            qt_index = int(get_qubit_index_from_name(qt.name))

            q_lower_index = qc if qc_index < qt_index else qt
            q_higher_index = qt if qc_index < qt_index else qc 
            
            if i == 0:
                q_lower_index.xy.play("y90")
            q_higher_index.xy.play("-y90")

            qp.align()

            # assumes a CZ gate exists with name e.g. `cz1_2`, when q1 is the control and q2 is the target.
            qc.z.play(f"cz{get_qubit_index_from_name(qc.name)}_{get_qubit_index_from_name(qt.name)}")
            qp.coupler.play("cz")
            wait(150 * u.ns)

            qp.align()

            qc.xy.frame_rotation_2pi(node.parameters.control_qubit_post_cz_phase_corrections[i])
            qt.xy.frame_rotation_2pi(node.parameters.target_qubit_post_cz_phase_corrections[i])

            q_higher_index.xy.play("y90")

        # play readout pulse on other qubits for multiplexing
        for readout_qubit in readout_qubits:
            readout_qubit.resonator.measure("readout")

        for i, qubit in enumerate(qubits):
            readout_state(qubit, state[i])
            save(state[i], state_st[i])

        save(n, n_st)

    with stream_processing():
        n_st.save("n")
        for i in range(len(qubits)):
            state_st[i].save_all(f"state_{qubits[i].name}")


if not node.parameters.simulate:
    qm = qmm.open_qm(config)
    job = qm.execute(ghz_circuit)
    # Get results from QUA program
    results = fetching_tool(job, ["n"] + [f"state_{qubits[i].name}" for i in range(len(qubits))], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  #  Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch results
        n, *states = results.fetch_all()

        # Progress bar
        progress_counter(n, node.parameters.shots, start_time=results.start_time)

        bitstrings = []
        for i in range(len(states[0])):
            bitstrings.append(''.join([str(s[i]) for s in states]))

        all_possible_bitstrings = [''.join(bits) for bits in product('01', repeat=len(qubits))]

        bitstring_counts = Counter(bitstrings)
        counts = [bitstring_counts.get(bit, 0) for bit in all_possible_bitstrings]

        total = sum(counts)
        normalized_counts = [count / total for count in counts] if total > 0 else [0] * len(counts)

        all_possible_bitstrings = [fr"$|{''.join(bits)}\rangle$" for bits in product('01', repeat=len(qubits))]
        plt.cla()
        plt.bar(all_possible_bitstrings, normalized_counts, color='blue', width=0.4)
        plt.xlabel(fr"States in the form $|{''.join([qubits[i].name for i in range(len(qubits))])}\rangle$")
        plt.ylabel("Normalized Count")
        plt.title(f"{len(qubits)}-qubit state probability after GHZ circuit.")
        plt.xticks(rotation=90)
        plt.tight_layout()

        plt.tight_layout()
        plt.pause(0.3)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
    plt.show()

    node.outcomes = {qp.name: "successful" for qp in qubit_pairs}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.results["figure"] = fig
    node.machine = machine
    node.save()

else:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=3_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, ghz_circuit, simulation_config)
    # Plot the simulated samples
    job.get_simulated_samples().con1.plot()
    plt.show()
