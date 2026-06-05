from typing import List
from compat import BasicOrchestrator
from compat import GraphParameters
from compat import QualibrationGraph
from compat import QualibrationLibrary
from time import time
""" Try faster than V1 """

#%% { Load nodes }
library = QualibrationLibrary.get_active_library()

class Parameters(GraphParameters):
    qubits: List[str] = None


# calibration parameters
qubits = ["q1", "q2", "q3", "q4", "q5"]
multiplexed = False
reset_type_thermal_or_active = "active"
SQRB_max_gate_num =900
pts_fit_RB = 40
Rabi_state_avg = 80


nodes = [
    "close_other_qms",
    "ramsey_flux_calibration_1",
    "ramsey_flux_calibration_2",
    "IQ_blobs_after",
    "SQ_RB",
]

g = QualibrationGraph(
    name="Ratis_Retuning_Graph",
    parameters=Parameters(qubits=qubits),
    nodes={
        "close_other_qms": library.nodes["00_Close_other_QMs"].copy(
            name="close_other_qms",
        ),
        "ramsey_flux_calibration_1": library.nodes["06a_Ramsey_vs_Flux_Calibration"].copy(
            flux_point_joint_or_independent="independent", multiplexed=multiplexed, num_averages=150, frequency_detuning_in_mhz = 4.0,
            min_wait_time_in_ns= 16, max_wait_time_in_ns= 1016, wait_time_step_in_ns = 10, flux_span= 0.09, flux_step = 0.0009
        ),
        "ramsey_flux_calibration_2": library.nodes["06a_Ramsey_vs_Flux_Calibration"].copy(
            flux_point_joint_or_independent="independent", multiplexed=multiplexed, num_averages=150, frequency_detuning_in_mhz = 4.0,
            min_wait_time_in_ns= 16, max_wait_time_in_ns= 1016, wait_time_step_in_ns = 10, flux_span= 0.02, flux_step = 0.0002
        ),
        "IQ_blobs_after": library.nodes["07b_IQ_Blobs"].copy(
            flux_point_joint_or_independent="independent",
            multiplexed=multiplexed,
            name="IQ_blobs_after",
            reset_type_thermal_or_active="thermal",
        ),
        "SQ_RB": library.nodes["10a_Single_Qubit_Randomized_Benchmarking"].copy(
            flux_point_joint_or_independent="independent",
            multiplexed=multiplexed, 
            delta_clifford=int(SQRB_max_gate_num/pts_fit_RB),
            max_circuit_depth=SQRB_max_gate_num,
            reset_type_thermal_or_active=reset_type_thermal_or_active,
            num_random_sequences=78,
            num_averages=100
        ),
    },
    connectivity=[(a, b) for a, b in zip(nodes, nodes[1:])],
    orchestrator=BasicOrchestrator(skip_failed=True),
)



#%% {Execution} 
start = time()
g.run(qubits=qubits)
end = time()
print(f"Elapse time: {round((end-start)/60,1)} mins")

# %%
