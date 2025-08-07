from typing import List, Optional
from qualibrate import QualibrationLibrary, QualibrationGraph, GraphParameters
from qualibrate.orchestration.basic_orchestrator import BasicOrchestrator


# Define graph target parameters
class Parameters(GraphParameters):
    qubits: Optional[List[str]] = None


library = QualibrationLibrary.get_active_library()

nodes = [
    "close_qms",
    "res_spec_vs_amp",
    "res_spec",
    "res_spec_vs_flux",
    "qubit_spec_vs_flux",
    "power_rabi_1",
    "ramsey_1",
    "power_rabi_2",
    "ramsey_2",
    "power_rabi_3",
    "ramsey_3",
    "readout_freq_opt",
    "readout_power_opt",
    "iq_blobs_1",
    "ramsey_vs_flux",
    "power_rabi_state_x180_1",
    "power_rabi_state_x180_2",
    "power_rabi_state_x180_3",
    "power_rabi_state_x90_1",
    "power_rabi_state_x90_2",
    "power_rabi_state_x90_3",
    "iq_blobs_2",
    "ramsey_vs_flux_2",
    "drag_calibration_x180_1",
    "drag_calibration_x90_1",
    "drag_calibration_x180_2",
    "drag_calibration_x90_2",
    "randomized_benchmarking",
]

# Create the QualibrationGraph
graph = QualibrationGraph(
    name="Bring-up Graph",  # Unique graph name
    parameters=Parameters(
        qubits=["q1", "q2", "q3", "q4", "q5"]
    ),  # Instantiate graph parameters
    nodes={  # Specify nodes used in the graph
        "close_qms": library.nodes["00_Close_other_QMs"],
        "res_spec": library.nodes["02a_Resonator_Spectroscopy"],
        "res_spec_vs_amp": library.nodes["02c_Resonator_Spectroscopy_vs_Amplitude"].copy(),
        "res_spec_vs_flux": library.nodes["02b_Resonator_Spectroscopy_vs_Flux"].copy(
            frequency_span_in_mhz=20,
            update_flux_min=False
        ),
        "qubit_spec_vs_flux": library.nodes["03b_Qubit_Spectroscopy_vs_Flux"].copy(),
        "power_rabi_1": library.nodes["04_Power_Rabi"],
        "ramsey_1": library.nodes["06_Ramsey"].copy(
            max_wait_time_in_ns=5000,
            log_or_linear_sweep='linear'
        ),
        "power_rabi_2": library.nodes["04_Power_Rabi"].copy(),
        "ramsey_2": library.nodes["06_Ramsey"].copy(
            max_wait_time_in_ns=5000,
            log_or_linear_sweep='linear'
        ),
        "power_rabi_3": library.nodes["04_Power_Rabi"].copy(),
        "ramsey_3": library.nodes["06_Ramsey"].copy(
            max_wait_time_in_ns=5000,
            log_or_linear_sweep='linear'
        ),
        "readout_freq_opt": library.nodes["07a_Readout_Frequency_Optimization"],
        "readout_power_opt": library.nodes["07c_Readout_Power_Optimization"],
        "iq_blobs_1": library.nodes["07b_IQ_Blobs"].copy(multiplexed=False),
        "ramsey_vs_flux": library.nodes["06a_Ramsey_vs_Flux_Calibration"].copy(),
        "power_rabi_state_x180_1": library.nodes["09_Power_Rabi_State"].copy(
            operation_x180_or_any_90="x180",
            reset_type_thermal_or_active="active"
        ),
        "power_rabi_state_x90_1": library.nodes["09_Power_Rabi_State"].copy(
            operation_x180_or_any_90="x90",
            reset_type_thermal_or_active="active"
        ),
        "power_rabi_state_x180_2": library.nodes["09_Power_Rabi_State"].copy(
            operation_x180_or_any_90="x180",
            max_number_rabi_pulses_per_sweep=400,
            min_amp_factor=0.95,
            max_amp_factor = 1.05,
            amp_factor_step=0.001,
            reset_type_thermal_or_active="active"
        ),
        "power_rabi_state_x90_2": library.nodes["09_Power_Rabi_State"].copy(
            operation_x180_or_any_90="x90",
            max_number_rabi_pulses_per_sweep=400,
            min_amp_factor=0.95,
            max_amp_factor=1.05,
            amp_factor_step=0.001,
            reset_type_thermal_or_active="active"
        ),
        "power_rabi_state_x180_3": library.nodes["09_Power_Rabi_State"].copy(
            operation_x180_or_any_90="x180",
            max_number_rabi_pulses_per_sweep=1000,
            min_amp_factor=0.98,
            max_amp_factor=1.02,
            amp_factor_step=0.0005,
            reset_type_thermal_or_active="active"
        ),
        "power_rabi_state_x90_3": library.nodes["09_Power_Rabi_State"].copy(
            operation_x180_or_any_90="x90",
            max_number_rabi_pulses_per_sweep=1000,
            min_amp_factor=0.98,
            max_amp_factor=1.02,
            amp_factor_step=0.0005,
            reset_type_thermal_or_active="active"
        ),
        "iq_blobs_2": library.nodes["07b_IQ_Blobs"].copy(multiplexed=False),
        # "stark_detuning_x180_1": library.nodes["09a_Stark_Detuning"].copy(
        #     operation_x180_or_any_90="x180",
        #     reset_type_thermal_or_active="thermal"
        # ),
        # "stark_detuning_x90_1": library.nodes["09a_Stark_Detuning"].copy(
        #     operation_x180_or_any_90="x90",
        #     reset_type_thermal_or_active="thermal"
        # ),
        # "stark_detuning_x180_2": library.nodes["09a_Stark_Detuning"].copy(
        #     operation_x180_or_any_90="x180",
        #     reset_type_thermal_or_active="thermal",
        #     max_number_pulses_per_sweep=200,
        #     frequency_span_in_mhz=2,
        #     frequency_step_in_mhz = 0.002
        # ),
        # "stark_detuning_x90_2": library.nodes["09a_Stark_Detuning"].copy(
        #     operation_x180_or_any_90="x90",
        #     reset_type_thermal_or_active="thermal",
        #     max_number_pulses_per_sweep=200,
        #     frequency_span_in_mhz=2,
        #     frequency_step_in_mhz=0.002
        # ),
        "ramsey_vs_flux_2": library.nodes["06a_Ramsey_vs_Flux_Calibration"].copy(),
        "drag_calibration_x180_1": library.nodes["09c_DRAG_Calibration_180_90"].copy(
            operation="x180",
            reset_type_thermal_or_active="active",
            num_averages=1000,
            min_amp_factor=0.5,
            max_amp_factor=1.5,
            amp_factor_step=0.05,
        ),
        "drag_calibration_x90_1": library.nodes["09c_DRAG_Calibration_180_90"].copy(
            operation="x90",
            reset_type_thermal_or_active="active",
            num_averages=1000,
            min_amp_factor=0.5,
            max_amp_factor=1.5,
            amp_factor_step=0.05,
        ),
        "drag_calibration_x180_2": library.nodes["09c_DRAG_Calibration_180_90"].copy(
            operation="x180",
            reset_type_thermal_or_active="active",
            num_averages=1000,
            min_amp_factor=0.9,
            max_amp_factor=1.1,
            amp_factor_step=0.005,
        ),
        "drag_calibration_x90_2": library.nodes["09c_DRAG_Calibration_180_90"].copy(
            operation="x90",
            reset_type_thermal_or_active="active",
            num_averages=1000,
            min_amp_factor=0.9,
            max_amp_factor=1.1,
            amp_factor_step=0.005,
        ),
        "randomized_benchmarking": library.nodes["10a_Single_Qubit_Randomized_Benchmarking"].copy(
            reset_type_thermal_or_active="active",
            max_circuit_depth=900
        )
    },
    # Specify directed relationships between graph nodes
    connectivity=[(a, b) for a, b in zip(nodes, nodes[1:])],
    # Specify orchestrator used to run the graph
    orchestrator=BasicOrchestrator(skip_failed=True),
)

# Run the calibration graph for qubits q1, q2, and q3
graph.run(qubits=None)
