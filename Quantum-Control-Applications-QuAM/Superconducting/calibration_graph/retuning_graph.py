from typing import List
from qualibrate.orchestration.basic_orchestrator import BasicOrchestrator
from qualibrate.parameters import GraphParameters
from qualibrate.qualibration_graph import QualibrationGraph
from qualibrate.qualibration_library import QualibrationLibrary

library = QualibrationLibrary.get_active_library()


class Parameters(GraphParameters):
    qubits: List[str] = None


qubits = ["q1", "q2", "q3", "q4", "q5"]
multiplexed = True
reset_type_thermal_or_active = "active"


g = QualibrationGraph(
    name="Retuning_Graph",
    parameters=Parameters(qubits=qubits),
    nodes={
        "close_other_qms": library.nodes["00_Close_other_QMs"].copy(
            name="close_other_qms",
        ),
        "IQ_blobs": library.nodes["07b_IQ_Blobs"].copy(
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="IQ_blobs",
            reset_type_thermal_or_active="active",
        ),
        "ramsey_flux_calibration": library.nodes["08_Ramsey_vs_Flux_Calibration"].copy(
            flux_point_joint_or_independent="independent", multiplexed=multiplexed, name="Ramsey_Flux_Calibration"
        ),
        "power_rabi_x180": library.nodes["04_Power_Rabi"].copy(
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            operation_x180_or_any_90="x180",
            name="Power_Rabi_x180",
            reset_type_thermal_or_active=reset_type_thermal_or_active,
            min_amp_factor=0.98,
            max_amp_factor=1.02,
            amp_factor_step=0.002,
            max_number_rabi_pulses_per_sweep=300,
            update_x90=False,
            state_discrimination=True,
        ),
        "power_rabi_x90": library.nodes["04_Power_Rabi"].copy(
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            operation_x180_or_any_90="x90",
            name="Power_Rabi_x90",
            reset_type_thermal_or_active=reset_type_thermal_or_active,
            min_amp_factor=0.98,
            max_amp_factor=1.02,
            amp_factor_step=0.002,
            max_number_rabi_pulses_per_sweep=300,
            state_discrimination=True,
        ),
        "single_qubit_randomized_benchmarking": library.nodes["10a_Single_Qubit_Randomized_Benchmarking"].copy(
            flux_point_joint_or_independent="joint",
            multiplexed=False, 
            delta_clifford=100,
            num_random_sequences=1000,
            reset_type_thermal_or_active=reset_type_thermal_or_active,
            name="Single_Qubit_Randomized_Benchmarking"
        ),
    },
    connectivity=[
        ("close_other_qms", "IQ_blobs"),
        ("IQ_blobs", "ramsey_flux_calibration"),
        ("ramsey_flux_calibration", "power_rabi_x180"),
        ("power_rabi_x180", "power_rabi_x90"),
        ("power_rabi_x90", "single_qubit_randomized_benchmarking"),
    ],
    orchestrator=BasicOrchestrator(skip_failed=True),
)

g.run(qubits=qubits)