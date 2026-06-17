from typing import List
from quam_libs.compat import BasicOrchestrator
from quam_libs.compat import GraphParameters
from quam_libs.compat import QualibrationGraph
from quam_libs.compat import QualibrationLibrary

library = QualibrationLibrary.get_active_library()

class Parameters(GraphParameters):
    qubits: List[str] = None


# calibration parameters
qubits = ["q1", "q2", "q3", "q4", "q5"]
multiplexed = True
reset_type_thermal_or_active = "active"
SQRB_max_gate_num = 900
pts_fit_RB = 45
Rabi_state_avg = 80


nodes = [
    "close_other_qms",
    "IQ_blobs_before",
    "ramsey_flux_calibration_1",
    "ramsey_flux_calibration_2",
    "ramsey_flux_calibration_3",
    # "RO_freq",
    "RO_power",
    "power_rabi_x180_before",
    "power_rabi_x90_before",
    "DRAG_alpha",
    "power_rabi_x180_after",
    "power_rabi_x90_after",
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
        "IQ_blobs_before": library.nodes["07b_IQ_Blobs"].copy(
            flux_point_joint_or_independent="independent",
            multiplexed=multiplexed,
            name="IQ_blobs_before",
            reset_type_thermal_or_active="thermal",
        ),
        "ramsey_flux_calibration_1": library.nodes["06a_Ramsey_vs_Flux_Calibration"].copy(
            flux_point_joint_or_independent="independent", multiplexed=multiplexed, num_averages=120
        ),
        "ramsey_flux_calibration_2": library.nodes["06a_Ramsey_vs_Flux_Calibration"].copy(
            flux_point_joint_or_independent="independent", multiplexed=multiplexed, num_averages=120
        ),
        "ramsey_flux_calibration_3": library.nodes["06a_Ramsey_vs_Flux_Calibration"].copy(
            flux_point_joint_or_independent="independent", multiplexed=multiplexed, num_averages=120
        ),
        # "RO_freq": library.nodes["07a_Readout_Frequency_Optimization"].copy(
        #     flux_point_joint_or_independent="independent", multiplexed=multiplexed,
        #     num_averages=120 ,
        #     frequency_span_in_mhz=25,
        #     frequency_step_in_mhz=0.25
        # ),
        "RO_power": library.nodes["07c_Readout_Power_Optimization"].copy(
            flux_point_joint_or_independent="independent", multiplexed=multiplexed,reset_type_thermal_or_active="thermal",
            num_runs=2000,
            start_amp=0.8,
            end_amp=1.2,
            num_amps=50
        ),
        "power_rabi_x180_before": library.nodes["09_Power_Rabi_State"].copy(
            num_averages=Rabi_state_avg,
            flux_point_joint_or_independent="independent",
            operation_x180_or_any_90="x180",
            reset_type_thermal_or_active=reset_type_thermal_or_active,
            min_amp_factor=0.8,
            max_amp_factor=1.2,
            amp_factor_step=0.008,
            max_number_rabi_pulses_per_sweep=88
        ),
        "power_rabi_x90_before": library.nodes["09_Power_Rabi_State"].copy(
            num_averages=int(1.2*Rabi_state_avg),
            flux_point_joint_or_independent="independent",
            operation_x180_or_any_90="x90",
            reset_type_thermal_or_active=reset_type_thermal_or_active,
            min_amp_factor=0.8,
            max_amp_factor=1.2,
            amp_factor_step=0.008,
            max_number_rabi_pulses_per_sweep=88
        ),
        "DRAG_alpha": library.nodes["09c_DRAG_Calibration_180_90"].copy(
            flux_point_joint_or_independent="independent", multiplexed=multiplexed,reset_type_thermal_or_active=reset_type_thermal_or_active,
        ),
        "power_rabi_x180_after": library.nodes["09_Power_Rabi_State"].copy(
            num_averages=Rabi_state_avg,
            flux_point_joint_or_independent="independent",
            operation_x180_or_any_90="x180",
            reset_type_thermal_or_active=reset_type_thermal_or_active,
            min_amp_factor=0.95,
            max_amp_factor=1.05,
            amp_factor_step=0.002,
            max_number_rabi_pulses_per_sweep=200
        ),
        "power_rabi_x90_after": library.nodes["09_Power_Rabi_State"].copy(
            num_averages=int(1*Rabi_state_avg),
            flux_point_joint_or_independent="independent",
            operation_x180_or_any_90="x90",
            reset_type_thermal_or_active=reset_type_thermal_or_active,
            min_amp_factor=0.9,
            max_amp_factor=1.1,
            amp_factor_step=0.004,
            max_number_rabi_pulses_per_sweep=88
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
            num_random_sequences=73,
            num_averages=37
        ),
    },
    connectivity=[(a, b) for a, b in zip(nodes, nodes[1:])],
    orchestrator=BasicOrchestrator(skip_failed=True),
)



if __name__ == "__main__":  
    g.run(qubits=qubits)

