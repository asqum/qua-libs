import time
import csv
from datetime import datetime
from pathlib import Path
from quam_libs.experiments.rb_standard.data_utils import InterleavedRBResult, RBResult
from datetime import datetime, timezone, timedelta
from typing import List, Literal, Optional
import xarray as xr
from quam_libs.lib.fit import fit_oscillation_decay_exp, oscillation_decay_exp
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from qm.qua import *
from qm import SimulationConfig
from qualang_tools.multi_user import qm_session
import numpy as np
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.units import unit
from qualibrate import NodeParameters, QualibrationNode
from quam_libs.experiments.rb.circuit_utils import layerize_quantum_circuit, process_circuit_to_integers
from quam_libs.experiments.rb.qua_utils import QuaProgramHandler
from quam_libs.lib.plot_utils import plot_samples
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.macros import qua_declaration, readout_state
from qualang_tools.loops import from_array
from quam_libs.components import QuAM
from quam_libs.experiments.rb.cloud_utils import write_sync_hook
from quam_libs.experiments.rb_standard.rb_utils import InterleavedRB, StandardRB
import matplotlib.pyplot as plt
from quam_libs.lib.plot_utils import QubitGrid, grid_iter, QubitPairGrid, grid_iter, grid_pair_names
from quam_libs.macros import active_reset

average_gates_per_2q_layer = 1.51

## 2Q_RB
def run_cz_tracking(pair_name:str, depth_squences:tuple[int]|None=None, target_operation:str='cz'):
    ### The start of the copy ### copy the node to here

    if depth_squences is None:
        depth_squences = (0,1,2,3,4,5,6,7,8,9,10,12,15)

    print("Run standard 2QRB")
    # Standard
    class Parameters(NodeParameters):
        qubit_pairs: Optional[List[str]] = [pair_name]#None
        circuit_lengths: tuple[int] = depth_squences
        num_circuits_per_length: int = 15
        num_averages: int = 150
        basis_gates: list[str] = ['rz', 'sx', 'x', 'cz'] 
        flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
        reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
        reduce_to_1q_cliffords: bool = 0
        use_input_stream: bool = False
        simulate: bool = False
        simulation_duration_ns: int = 10000
        load_data_id: Optional[int] = None
        timeout: int = 100
        seed: int = 0
        targets_name = "qubit_pairs"

    node = QualibrationNode[Parameters, QuAM](name="70b_two_qubit_standard_rb", parameters=Parameters())

    # %% {Initialize_QuAM_and_QOP}

    # Instantiate the QuAM class from the state file
    node.machine = QuAM.load()

    # Get the relevant QuAM components
    if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
        qubit_pairs = node.machine.active_qubit_pairs
    else:
        qubit_pairs = [node.machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]

    if len(qubit_pairs) == 0:
        raise ValueError("No qubit pairs selected")

    # Generate the OPX and Octave configurations

    # Open Communication with the QOP
    if node.parameters.load_data_id is None:
        qmm = node.machine.connect()

    config = node.machine.generate_config()


    # %% {Random circuit generation}

    standard_RB = StandardRB(
        amplification_lengths=node.parameters.circuit_lengths,
        num_circuits_per_length=node.parameters.num_circuits_per_length,
        basis_gates=node.parameters.basis_gates,
        reduce_to_1q_cliffords=node.parameters.reduce_to_1q_cliffords,
        num_qubits=2,
        seed=node.parameters.seed
    )

    transpiled_circuits = standard_RB.transpiled_circuits
    transpiled_circuits_as_ints = {}
    layerized_circuits = {}
    for l, circuits in transpiled_circuits.items():
        layerized_circuits[l] = [layerize_quantum_circuit(qc) for qc in circuits]
        transpiled_circuits_as_ints[l] = [process_circuit_to_integers(qc) for qc in layerized_circuits[l]]

    # to calculate the average number of 2q layers per Clifford
    average_layers_per_clifford = np.mean([np.mean([len(circ) for circ in circuits])/np.array(length+1) for length, circuits in transpiled_circuits_as_ints.items() if length > 0])

    circuits_as_ints = []
    for circuits_per_len in transpiled_circuits_as_ints.values():
        for circuit in circuits_per_len:
            circuit_with_measurement = circuit + [66] # readout
            circuits_as_ints.append(circuit_with_measurement)

    # %% {QUA_program}

    num_pairs = len(qubit_pairs)

    qua_program_handler = QuaProgramHandler(node, num_pairs, circuits_as_ints, node.machine, qubit_pairs)

    rb = qua_program_handler.get_qua_program()
    node.namespace = {"qua_program" : rb}

    # %% {Simulate_or_execute}
    if node.parameters.simulate:
        simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns//4)  # in clock cycles
        job = qmm.simulate(config, rb, simulation_config)
        samples = job.get_simulated_samples()

    elif node.parameters.load_data_id is None:
        # Prepare data for saving
        node.results = {}
        date_time = datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S")
        
        with qm_session(node.machine.qmm, config, timeout=node.parameters.timeout) as qm:
            if node.parameters.use_input_stream:
                num_sequences = len(qua_program_handler.sequence_lengths)
                circuits_as_ints_batched_padded = [batch + [0] * (qua_program_handler.max_current_sequence_length - len(batch)) for batch in qua_program_handler.circuits_as_ints_batched]    
                
                if node.machine.network['cloud']:
                    write_sync_hook(circuits_as_ints_batched_padded)

                    job = qm.execute(rb,
                            terminal_output=True,options={"sync_hook": "sync_hook.py"})
                else:
                    job = qm.execute(rb)
                    for id, batch in enumerate(circuits_as_ints_batched_padded):
                        job.push_to_input_stream("sequence", batch)
                        print(f"{id}/{num_sequences}: Received ")
            
            else:
                job = qm.execute(rb)
            
            results = fetching_tool(job, ["iteration"], mode="live")
            while results.is_processing():
                # Fetch results
                n = results.fetch_all()[0]
                # Progress bar
                progress_counter(n, node.parameters.num_averages, start_time=results.start_time)

        
    # %% {Plot and save if simulation}
    if node.parameters.simulate:
        qubit_names = [qubit_pair.qubit_control.name for qubit_pair in qubit_pairs] + [qubit_pair.qubit_target.name for qubit_pair in qubit_pairs]
        readout_lines = set([q[1] for q in qubit_names])
        fig = plot_samples(samples, qubit_names, readout_lines=list(readout_lines), xlim=(0,10000))
        
        # node.results["figure"] = fig
        # node.save()

    # %% {Data_fetching_and_dataset_creation}
    if node.parameters.load_data_id is None:
        ds = fetch_results_as_xarray(
        job.result_handles,
        qubit_pairs,
            { "sequence": range(node.parameters.num_circuits_per_length), "depths": list(node.parameters.circuit_lengths), "shots": range(node.parameters.num_averages)},
        )
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    # Add the dataset to the node
    node.results = {"ds": ds}
    # %% {Data_analysis and plotting}

    # Assume ds is your input dataset and ds['state'] is your DataArray
    state = ds['state']  # shape: (qubit, shots, sequence, depths)

    # Outcome labels for 2-qubit states
    labels = ["00", "01", "10", "11"]

    # Create a list of DataArrays: one for each outcome
    probs = [state == i for i in range(4)]

    # Stack along a new outcome dimension
    probs = xr.concat(probs, dim='outcome')

    # Assign outcome labels
    probs = probs.assign_coords(outcome=("outcome", labels))

    probs_00 = probs.sel(outcome="00")
    probs_00 = probs_00.rename({"shots": "average", "sequence": "repeat", "depths": "circuit_depth"})
    probs_00 = probs_00.transpose("qubit", "repeat", "circuit_depth", "average")


    probs_00 = probs_00.astype(int)

    ds_transposed = ds.rename({"shots": "average", "sequence": "repeat", "depths": "circuit_depth"})
    ds_transposed = ds_transposed.transpose("qubit", "repeat", "circuit_depth", "average")

    rb_result = {}
    s_fidelity = []
    for qp in qubit_pairs:
        
        rb_result[qp.id] = RBResult(
                circuit_depths=list(node.parameters.circuit_lengths),
                num_repeats=node.parameters.num_circuits_per_length,
                num_averages=node.parameters.num_averages,
                state=ds_transposed.sel(qubit=qp.name).state.data
            )
        
        # Fit the data and calculate all error and fidelity metrics
        rb_result[qp.id].fit(
            average_layers_per_clifford=average_layers_per_clifford,
            average_gates_per_2q_layer=average_gates_per_2q_layer
        )
        
        # Plot the results
        fig = rb_result[qp.id].plot_with_fidelity()
        
        fig.suptitle(f"2Q Randomized Benchmarking - {qp.name}")
        # node.add_node_info_subtitle(fig)
        fig.show()
        
        node.results[f"{qp.id}_figure_RB_decay"] = fig
        s_fidelity.append(rb_result[qp.id].fidelity)
    # %% {Update_state}
    with node.record_state_updates():
        for qp in qubit_pairs:
            qp.extras["StandardRB"] = {
                "error_per_clifford": 1 - rb_result[qp.id].fidelity, 
                "error_per_2q_layer": rb_result[qp.id].error_per_2q_layer,
                "error_per_gate": rb_result[qp.id].error_per_gate,
                "average_gate_fidelity": 1 - rb_result[qp.id].error_per_gate,
                "alpha": rb_result[qp.id].alpha}
    # %% {Save_results}
    node.save()
    qmm.close_all_qms() 


    print("Run interleaved 2QRB")
    # Interleaved
    # %% {Node_parameters}

    class Parameters(NodeParameters):
        qubit_pairs: Optional[List[str]] = [pair_name] #None
        circuit_lengths: tuple[int] = depth_squences # in number of cliffords
        num_circuits_per_length: int = 15
        num_averages: int = 150
        target_gate: str = target_operation # "idle_2q" or "cz" supported 
        basis_gates: list[str] = ['rz', 'sx', 'x', 'cz'] 
        flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
        reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
        reduce_to_1q_cliffords: bool = False
        use_input_stream: bool = False
        simulate: bool = False
        simulation_duration_ns: int = 10000
        load_data_id: Optional[int] = None
        timeout: int = 100
        seed: int = 0
        targets_name = "qubit_pairs"

    node = QualibrationNode(name="70c_two_qubit_interleaved_rb", parameters=Parameters())

    # %% {Initialize_QuAM_and_QOP}

    # Instantiate the QuAM class from the state file
    node.machine = QuAM.load()

    # Get the relevant QuAM components
    if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
        qubit_pairs = node.machine.active_qubit_pairs
    else:
        qubit_pairs = [node.machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]

    if len(qubit_pairs) == 0:
        raise ValueError("No qubit pairs selected")

    # Generate the OPX and Octave configurations

    # Open Communication with the QOP
    if node.parameters.load_data_id is None:
        qmm = node.machine.connect()

    config = node.machine.generate_config()


    # %% {Random circuit generation}

    interleaved_RB = InterleavedRB(
        target_gate=node.parameters.target_gate,
        amplification_lengths=node.parameters.circuit_lengths,
        num_circuits_per_length=node.parameters.num_circuits_per_length,
        basis_gates=node.parameters.basis_gates,
        num_qubits=2,
        reduce_to_1q_cliffords=node.parameters.reduce_to_1q_cliffords,
        seed=node.parameters.seed
    )

    transpiled_circuits = interleaved_RB.transpiled_circuits
    transpiled_circuits_as_ints = {}
    for l, circuits in transpiled_circuits.items():
        transpiled_circuits_as_ints[l] = [process_circuit_to_integers(layerize_quantum_circuit(qc)) for qc in circuits]

    circuits_as_ints = []
    for circuits_per_len in transpiled_circuits_as_ints.values():
        for circuit in circuits_per_len:
            circuit_with_measurement = circuit + [66] # readout
            circuits_as_ints.append(circuit_with_measurement)

    # %% {QUA_program}

    num_pairs = len(qubit_pairs)

    qua_program_handler = QuaProgramHandler(node, num_pairs, circuits_as_ints, node.machine, qubit_pairs)

    rb = qua_program_handler.get_qua_program()

    # %% {Simulate_or_execute}
    if node.parameters.simulate:
        simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns//4)  # in clock cycles
        job = qmm.simulate(config, rb, simulation_config)
        samples = job.get_simulated_samples()

    elif node.parameters.load_data_id is None:
        # Prepare data for saving
        node.results = {}
        date_time = datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S")
        
        with qm_session(node.machine.qmm, config, timeout=node.parameters.timeout) as qm:
            if node.parameters.use_input_stream:
                num_sequences = len(qua_program_handler.sequence_lengths)
                circuits_as_ints_batched_padded = [batch + [0] * (qua_program_handler.max_current_sequence_length - len(batch)) for batch in qua_program_handler.circuits_as_ints_batched]    
                
                if node.machine.network['cloud']:
                    write_sync_hook(circuits_as_ints_batched_padded)

                    job = qm.execute(rb,
                            terminal_output=True,options={"sync_hook": "sync_hook.py"})
                else:
                    job = qm.execute(rb)
                    for id, batch in enumerate(circuits_as_ints_batched_padded):
                        job.push_to_input_stream("sequence", batch)
                        print(f"{id}/{num_sequences}: Received ")
            
            else:
                job = qm.execute(rb)
            
            results = fetching_tool(job, ["iteration"], mode="live")
            while results.is_processing():
                # Fetch results
                n = results.fetch_all()[0]
                # Progress bar
                progress_counter(n, node.parameters.num_averages, start_time=results.start_time)

        
    # %% {Plot and save if simulation}
    if node.parameters.simulate:
        qubit_names = [qubit_pair.qubit_control.name for qubit_pair in qubit_pairs] + [qubit_pair.qubit_target.name for qubit_pair in qubit_pairs]
        readout_lines = set([q[1] for q in qubit_names])
        fig = plot_samples(samples, qubit_names, readout_lines=list(readout_lines), xlim=(0,10000))
        
        # node.results["figure"] = fig
        # node.save()

    # %% {Data_fetching_and_dataset_creation}
    if node.parameters.load_data_id is None:
        ds = fetch_results_as_xarray(
        job.result_handles,
        qubit_pairs,
            { "sequence": range(node.parameters.num_circuits_per_length), "depths": list(node.parameters.circuit_lengths), "shots": range(node.parameters.num_averages)},
        )
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    # Add the dataset to the node
    node.results = {"ds": ds}
    # %% {Data_analysis and plotting}

    # Assume ds is your input dataset and ds['state'] is your DataArray
    state = ds['state']  # shape: (qubit, shots, sequence, depths)

    # Outcome labels for 2-qubit states
    labels = ["00", "01", "10", "11"]

    # Create a list of DataArrays: one for each outcome
    probs = [state == i for i in range(4)]

    # Stack along a new outcome dimension
    probs = xr.concat(probs, dim='outcome')

    # Assign outcome labels
    probs = probs.assign_coords(outcome=("outcome", labels))

    probs_00 = probs.sel(outcome="00")
    probs_00 = probs_00.rename({"shots": "average", "sequence": "repeat", "depths": "circuit_depth"})
    probs_00 = probs_00.transpose("qubit", "repeat", "circuit_depth", "average")


    probs_00 = probs_00.astype(int)

    ds_transposed = ds.rename({"shots": "average", "sequence": "repeat", "depths": "circuit_depth"})
    ds_transposed = ds_transposed.transpose("qubit", "repeat", "circuit_depth", "average")

    rb_result = {}
    fidelity = []
    pairs = []
    for qp in qubit_pairs:

        rb_result[qp.id] = InterleavedRBResult(
            # standard_rb_alpha=node.machine.qubit_pairs[qp.id].macros["cz"].fidelity.get("StandardRB", 1).get("alpha", 1),
            standard_rb_alpha=qp.extras.StandardRB.alpha, # if "StandardRB" in qp.extras else 1,
            circuit_depths=list(node.parameters.circuit_lengths),
            num_repeats=node.parameters.num_circuits_per_length,
            num_averages=node.parameters.num_averages,
            state=ds_transposed.sel(qubit=qp.name).state.data
        )

        # Fit the data and calculate all error and fidelity metrics
        rb_result[qp.id].fit()
        
        # Plot the results
        fig = rb_result[qp.id].plot_with_fidelity()
        fig.suptitle(f"2Q {node.parameters.target_gate.upper()} Interleaved Randomized Benchmarking - {qp.name}")
        # node.add_node_info_subtitle(fig)
        fig.show()
        
        node.results[f"{qp.id}_figure_IRB_decay"] = fig
        fidelity.append(rb_result[qp.id].fidelity)
        pairs.append(qp.name)

    # with node.record_state_updates():
    #     for qp in qubit_pairs:
    #         qp.extras['Interleaved_RB'] = rb_result[qp.id].fidelity
    
    node.save()
    qmm.close_all_qms() 

    return s_fidelity[0], fidelity[0], pairs[0] # temporarily, 2QRB node can only run single pair


def run_offset_tracking(target_qs:Optional[List[str]], update_state:bool=False):
    ### The start of the copy ### copy the node to here
    # %% {Node_parameters}
    class Parameters(NodeParameters):
        qubits: Optional[List[str]] = target_qs
        num_averages: int = 500
        frequency_detuning_in_mhz: float = 8.0
        min_wait_time_in_ns: int = 16
        max_wait_time_in_ns: int = 500
        wait_time_step_in_ns: int = 10
        flux_span: float = 0.04
        flux_step: float = 0.001
        flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
        simulate: bool = False
        simulation_duration_ns: int = 2500
        timeout: int = 100
        load_data_id: Optional[int] = None
        multiplexed: bool = False

    node = QualibrationNode(name="06a_Ramsey_vs_Flux_Calibration", parameters=Parameters())


    # %% {Initialize_QuAM_and_QOP}
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Instantiate the QuAM class from the state file
    machine = QuAM.load()
    # Generate the OPX and Octave configurations
    config = machine.generate_config()
    # Open Communication with the QOP
    if node.parameters.load_data_id is None:
        qmm = machine.connect()

    # Get the relevant QuAM components
    if node.parameters.qubits is None or node.parameters.qubits == "":
        qubits = machine.active_qubits
    else:
        qubits = [machine.qubits[q] for q in node.parameters.qubits]
    num_qubits = len(qubits)


    # %% {QUA_program}
    n_avg = node.parameters.num_averages  # The number of averages

    # Dephasing time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
    idle_times = np.arange(
        node.parameters.min_wait_time_in_ns // 4,
        node.parameters.max_wait_time_in_ns // 4,
        node.parameters.wait_time_step_in_ns // 4,
    )

    # Detuning converted into virtual Z-rotations to observe Ramsey oscillation and get the qubit frequency
    detuning = int(1e6 * node.parameters.frequency_detuning_in_mhz)
    flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
    fluxes = np.arange(
        -node.parameters.flux_span / 2, node.parameters.flux_span / 2 + 0.001, step=node.parameters.flux_step
    )

    with program() as ramsey:
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
        init_state = declare(int)
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]
        t = declare(int)  # QUA variable for the idle time
        phi = declare(fixed)  # QUA variable for dephasing the second pi/2 pulse (virtual Z-rotation)
        flux = declare(fixed)  # QUA variable for the flux dc level
        reset_global_phase()

        machine.apply_all_couplers_to_min()
        for i, qubit in enumerate(qubits):
            # Bring the active qubits to the desired frequency point
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)
            if "c" in qubit.id: qubit.z.set_dc_offset(qubit.z.joint_offset) # for coupler-test case
            qubit.z.settle()
            qubit.align()   

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(flux, fluxes)):
                    with for_(*from_array(t, idle_times)):
                        # Read the state of the qubit before Ramsey starts
                        readout_state(qubit, init_state)
                        qubit.align()
                        # Rotate the frame of the second x90 gate to implement a virtual Z-rotation
                        # 4*tau because tau was in clock cycles and 1e-9 because tau is ns
                        assign(phi, Cast.mul_fixed_by_int(detuning * 1e-9, 4 * t))
                        # TODO: this has gaps and the Z rotation is not derived properly, is it okay still?
                        # Ramsey sequence
                        qubit.xy.play("x180", amplitude_scale=0.5)
                        qubit.align()
                        wait(20, qubit.z.name)
                        qubit.z.play("const", amplitude_scale=flux / qubit.z.operations["const"].amplitude, duration=t)
                        wait(20, qubit.z.name)
                        qubit.xy.frame_rotation_2pi(phi)
                        qubit.align()
                        qubit.xy.play("x180", amplitude_scale=0.5)

                        # Align the elements to measure after playing the qubit pulse.
                        align()
                        # Measure the state of the resonators
                        readout_state(qubit, state[i])
                        assign(state[i], init_state ^ state[i])
                        save(state[i], state_st[i])

                        # Reset the frame of the qubits in order not to accumulate rotations
                        reset_frame(qubit.xy.name)
                        qubit.align()

            if not node.parameters.multiplexed:
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                state_st[i].buffer(len(idle_times)).buffer(len(fluxes)).average().save(f"state{i + 1}")


    # %% {Simulate_or_execute}
    if node.parameters.simulate:
        # Simulates the QUA program for the specified duration
        simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
        job = qmm.simulate(config, ramsey, simulation_config)
        # Get the simulated samples and plot them for all controllers
        samples = job.get_simulated_samples()
        fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
        for i, con in enumerate(samples.keys()):
            plt.subplot(len(samples.keys()), 1, i + 1)
            samples[con].plot()
            plt.title(con)
        plt.tight_layout()
        # Save the figure
        node.results = {"figure": plt.gcf()}
        node.machine = machine
        node.save()

    elif node.parameters.load_data_id is None:
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(ramsey)
            results = fetching_tool(job, ["n"], mode="live")
            while results.is_processing():
                # Fetch results
                n = results.fetch_all()[0]
                # Progress bar
                progress_counter(n, n_avg, start_time=results.start_time)

    # %% {Data_fetching_and_dataset_creation}
    if not node.parameters.simulate:
        if node.parameters.load_data_id is None:
            ds = fetch_results_as_xarray(job.result_handles, qubits, {"idle_time": idle_times, "flux": fluxes})
            # Add the absolute time in µs to the dataset
            ds = ds.assign_coords(idle_time=4 * ds.idle_time / 1e3)
            ds.flux.attrs = {"long_name": "flux", "units": "V"}
            ds.idle_time.attrs = {"long_name": "idle time", "units": "µs"}
        else:
            ds, machine, json_data, qubits, node.parameters = load_dataset(node.parameters.load_data_id, parameters = node.parameters)
        # Add the dataset to the node
        node.results = {"ds": ds}

        # %% {Data_analysis}
        # TODO: explain the data analysis
        fit_data = fit_oscillation_decay_exp(ds.state, "idle_time")
        fit_data.attrs = {"long_name": "time", "units": "µs"}
        fitted = oscillation_decay_exp(
            ds.state.idle_time,
            fit_data.sel(fit_vals="a"),
            fit_data.sel(fit_vals="f"),
            fit_data.sel(fit_vals="phi"),
            fit_data.sel(fit_vals="offset"),
            fit_data.sel(fit_vals="decay"),
        )

        frequency = fit_data.sel(fit_vals="f")
        frequency.attrs = {"long_name": "frequency", "units": "MHz"}

        decay = fit_data.sel(fit_vals="decay")
        decay.attrs = {"long_name": "decay", "units": "nSec"}

        tau = 1 / fit_data.sel(fit_vals="decay")
        tau.attrs = {"long_name": "T2*", "units": "uSec"}

        frequency = frequency.where(frequency > 0, drop=True)

        fitvals = frequency.polyfit(dim="flux", deg=2)
        flux = frequency.flux
        a = {}
        flux_offset = {}
        freq_offset = {}
        for q in qubits:
            a[q.name] = float(-1e6 * fitvals.sel(qubit=q.name, degree=2).polyfit_coefficients.values)
            flux_offset[q.name] = float(
                (
                    -0.5
                    * fitvals.sel(qubit=q.name, degree=1).polyfit_coefficients
                    / fitvals.sel(qubit=q.name, degree=2).polyfit_coefficients
                ).values
            )
            freq_offset[q.name] = 1e6 * (flux_offset[q.name]**2 * float(fitvals.sel(qubit=q.name, degree=2).polyfit_coefficients.values) +
                                        flux_offset[q.name] * float(fitvals.sel(qubit=q.name, degree=1).polyfit_coefficients.values) + 
                                        float(fitvals.sel(qubit=q.name, degree=0).polyfit_coefficients.values)) - detuning

        # Save fitting results
        node.results["fit_results"] = {}
        for q in qubits:
            node.results["fit_results"][q.name] = {}
            node.results["fit_results"][q.name]["flux_offset"] = flux_offset[q.name]
            node.results["fit_results"][q.name]["freq_offset"] = freq_offset[q.name]
            node.results["fit_results"][q.name]["quad_term"] = a[q.name]

        # %% {Plotting}
        grid_names = [q.grid_location for q in qubits]
        grid = QubitGrid(ds, grid_names)
        for ax, qubit in grid_iter(grid):
            ds.sel(qubit=qubit["qubit"]).state.plot(ax=ax)
            ax.set_title(qubit["qubit"])
            ax.set_xlabel("Idle_time (uS)")
            ax.set_ylabel(" Flux (V)")
        grid.fig.suptitle("Ramsey freq. Vs. flux")
        plt.tight_layout()
        plt.show()
        node.results["figure_raw"] = grid.fig

        grid = QubitGrid(ds, grid_names)
        for ax, qubit in grid_iter(grid):
            fitted_freq = (
                fitvals.sel(qubit=qubit["qubit"], degree=2).polyfit_coefficients * flux**2
                + fitvals.sel(qubit=qubit["qubit"], degree=1).polyfit_coefficients * flux
                + fitvals.sel(qubit=qubit["qubit"], degree=0).polyfit_coefficients
            )
            frequency.sel(qubit=qubit["qubit"]).plot(marker=".", linewidth=0, ax=ax)
            ax.plot(flux, fitted_freq)
            ax.set_title(qubit["qubit"])
            ax.set_xlabel(" Flux (V)")
            print(f"The quad term for {qubit['qubit']} is {a[qubit['qubit']]/1e9:.3f} GHz/V^2")
            print(f"Flux offset for {qubit['qubit']} is {flux_offset[qubit['qubit']]*1e3:.1f} mV")
            print(f"Freq offset for {qubit['qubit']} is {freq_offset[qubit['qubit']]/1e6:.3f} MHz")
            print()
        grid.fig.suptitle("Ramsey freq. Vs. flux")
        plt.tight_layout()
        plt.show()
        node.results["figure"] = grid.fig

        # %% {Update_state}
        fluctactuation = []
        qubits_name = []
        if update_state:
            if node.parameters.load_data_id is None:
                with node.record_state_updates():
                    for qubit in qubits:
                        qubit.xy.intermediate_frequency -= freq_offset[qubit.name]
                        if flux_point == "independent":
                            qubit.z.independent_offset += flux_offset[qubit.name]
                            if "c" in qubit.id: # for coupler-test case
                                qubit.z.joint_offset += flux_offset[qubit.name]
                                qubit.z.independent_offset = qubit.z.joint_offset - qubit.phi0_voltage / 2 
                        elif flux_point == "joint":
                            qubit.z.joint_offset += flux_offset[qubit.name]
                        else:
                            raise RuntimeError(f"unknown flux_point")
                        qubit.freq_vs_flux_01_quad_term = float(a[qubit.name])
        else:
            for qubit in qubits:
                qubits_name.append(qubit.name)
                if flux_point == "independent":
                    fluctactuation.append(qubit.z.independent_offset + flux_offset[qubit.name])
                elif flux_point == "joint":
                    fluctactuation.append(qubit.z.joint_offset + flux_offset[qubit.name])
                else:
                    raise RuntimeError(f"unknown flux_point")


            # %% {Save_results}
            node.outcomes = {q.name: "successful" for q in qubits}
            node.results["initial_parameters"] = node.parameters.model_dump()
            node.machine = machine
            node.save()

            ### The end of the copy ###

    return fluctactuation, qubits_name


def run_Bell_fidelity(qubit_pair:Optional[List[str]], shots:int=3000):
    class Parameters(NodeParameters): 

        qubit_pairs: Optional[List[str]] = qubit_pair
        num_shots: int = shots
        flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
        reset_type: Literal['active', 'thermal'] = "active"
        simulate: bool = False
        timeout: int = 100
        load_data_id: Optional[int] = None


    node = QualibrationNode(
        name="40b_Bell_state_tomography", parameters=Parameters()
    )
    assert not (node.parameters.simulate and node.parameters.load_data_id is not None), "If simulate is True, load_data_id must be None, and vice versa."

    # %% {Initialize_QuAM_and_QOP}
    # Class containing tools to help handling units and conversions.
    u = unit(coerce_to_integer=True)
    # Instantiate the QuAM class from the state file
    machine = QuAM.load()

    # Get the relevant QuAM components
    if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
        qubit_pairs = machine.active_qubit_pairs
    else:
        qubit_pairs = [machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]

    num_qubit_pairs = len(qubit_pairs)

    # Generate the OPX and Octave configurations
    config = machine.generate_config()
    octave_config = machine.get_octave_config()
    # Open Communication with the QOP
    if node.parameters.load_data_id is None:
        qmm = machine.connect()
    # %%

    ####################
    # Helper functions #
    ####################
    from matplotlib.colors import LinearSegmentedColormap

    def plot_3d_hist_with_frame(data,ideal, title = ''):
        fig, axs = plt.subplots(1, 2, figsize=(8, 4), subplot_kw={'projection': '3d'})
        # Create a grid of positions for the bars
        xpos, ypos = np.meshgrid(np.arange(4) + 0.5, np.arange(4) + 0.5, indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = np.zeros_like(xpos)
        # Create a custom colormap with two distinct colors for positive and negative values
        colors = [(0.1, 0.1, 0.6), (0.55, 0.55, 1.0)]  # Light blue for positive, dark blue for negative
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

        # finding global min,max
        gmin = np.min([np.min(np.real(data)),np.min(np.imag(data)),np.min(np.real(ideal)),np.min(np.imag(ideal))])
        gmax = np.max([np.max(np.real(data)),np.max(np.imag(data)),np.max(np.real(ideal)),np.max(np.imag(ideal))])

        # Use the bar3d function with the 'color' parameter to color the bars
        for i in range(2):
            if i == 0:
                dz = np.real(data).ravel()
                dzi = np.real(ideal).ravel()
                axs[i].set_title('real')
            else:
                dz = np.imag(data).ravel()
                dzi = np.imag(ideal).ravel()
                axs[i].set_title('imaginary')            
            axs[i].bar3d(xpos, ypos, zpos, dx=0.4, dy=0.4, dz=dz, alpha= 1, color=cmap(np.sign(dz)))
            axs[i].bar3d(xpos, ypos, zpos, dx=0.4, dy=0.4, dz=dzi, alpha= 0.1, edgecolor = 'k')
            # Set tick labels for x and y axes
            axs[i].set_xticks(np.arange(1, 5))
            axs[i].set_yticks(np.arange(1, 5))
            axs[i].set_xticklabels(['00', '01', '10', '11'])
            axs[i].set_yticklabels(['00', '01', '10', '11'])
            axs[i].set_xticklabels(['00', '01', '10', '11'], rotation=45)
            axs[i].set_yticklabels(['00', '01', '10', '11'], rotation=45)
            axs[i].set_zlim([gmin,gmax])
        fig.suptitle(title)
        # Show the plot
        
        return fig


    def plot_3d_hist_with_frame_real(data,ideal, ax ):
        xpos, ypos = np.meshgrid(np.arange(4) + 0.5, np.arange(4) + 0.5, indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = np.zeros_like(xpos)
        # Create a custom colormap with two distinct colors for positive and negative values
        colors = [(0.1, 0.1, 0.6), (0.55, 0.55, 1.0)]  # Light blue for positive, dark blue for negative
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

        # finding global min,max
        gmin = np.min([np.min(np.real(data)),np.min(np.imag(data)),np.min(np.real(ideal)),np.min(np.imag(ideal))])
        gmax = np.max([np.max(np.real(data)),np.max(np.imag(data)),np.max(np.real(ideal)),np.max(np.imag(ideal))])

        # Use the bar3d function with the 'color' parameter to color the bars
        dz = np.real(data).ravel()
        dzi = np.real(ideal).ravel()
        ax.set_title('real')
        
        ax.bar3d(xpos, ypos, zpos, dx=0.4, dy=0.4, dz=dz, alpha= 1, color=cmap(np.sign(dz)))
        ax.bar3d(xpos, ypos, zpos, dx=0.4, dy=0.4, dz=dzi, alpha= 0.1, edgecolor = 'k')
        # Set tick labels for x and y axes
        ax.set_xticks(np.arange(1, 5))
        ax.set_yticks(np.arange(1, 5))
        ax.set_xticklabels(['00', '01', '10', '11'])
        ax.set_yticklabels(['00', '01', '10', '11'])
        ax.set_xticklabels(['00', '01', '10', '11'], rotation=45)
        ax.set_yticklabels(['00', '01', '10', '11'], rotation=45)
        ax.set_zlim([gmin,gmax])
        # Show the plot

    def plot_3d_hist_with_frame_imag(data,ideal, axs):
        xpos, ypos = np.meshgrid(np.arange(4) + 0.5, np.arange(4) + 0.5, indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = np.zeros_like(xpos)
        # Create a custom colormap with two distinct colors for positive and negative values
        colors = [(0.1, 0.1, 0.6), (0.55, 0.55, 1.0)]  # Light blue for positive, dark blue for negative
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

        # finding global min,max
        gmin = np.min([np.min(np.real(data)),np.min(np.imag(data)),np.min(np.real(ideal)),np.min(np.imag(ideal))])
        gmax = np.max([np.max(np.real(data)),np.max(np.imag(data)),np.max(np.real(ideal)),np.max(np.imag(ideal))])

        # Use the bar3d function with the 'color' parameter to color the bars
        dz = np.imag(data).ravel()
        dzi = np.imag(ideal).ravel()
        ax.set_title('imaginary')
        
        ax.bar3d(xpos, ypos, zpos, dx=0.4, dy=0.4, dz=dz, alpha= 1, color=cmap(np.sign(dz)))
        ax.bar3d(xpos, ypos, zpos, dx=0.4, dy=0.4, dz=dzi, alpha= 0.1, edgecolor = 'k')
        # Set tick labels for x and y axes
        ax.set_xticks(np.arange(1, 5))
        ax.set_yticks(np.arange(1, 5))
        ax.set_xticklabels(['00', '01', '10', '11'])
        ax.set_yticklabels(['00', '01', '10', '11'])
        ax.set_xticklabels(['00', '01', '10', '11'], rotation=45)
        ax.set_yticklabels(['00', '01', '10', '11'], rotation=45)
        ax.set_zlim([gmin,gmax])
        # Show the plot
        

    def flatten(data):
        if isinstance(data, tuple):
            if len(data) == 0:
                return ()
            else:
                return flatten(data[0]) + flatten(data[1:])
        else:
            return (data,)
        
    def generate_pauli_basis(n_qubits):    
        pauli = np.array([0,1,2,3])
        paulis = pauli
        for i in range(n_qubits-1):
            new_paulis = []
            for ps in paulis:
                for p in pauli:
                    new_paulis.append(flatten((ps, p)))
            paulis = new_paulis
        return paulis
            
    def gen_inverse_hadamard(n_qubits):
        H = np.array([[1,1],[1,-1]])/2
        for _ in range(n_qubits-1):
            H = np.kron(H, H)
        return np.linalg.inv(H)

    def get_pauli_data(da):

        pauli_basis = generate_pauli_basis(2)

        inverse_hadamard = gen_inverse_hadamard(2)

        # Create an xarray Dataset with dimensions and coordinates based on pauli_basis
        paulis_data = xr.Dataset(
            {
                "value": (["pauli_op"], np.zeros(len(pauli_basis))),
                "appearances": (["pauli_op"], np.zeros(len(pauli_basis), dtype=int))
            },
            coords={'pauli_op': [','.join(map(str, op)) for op in pauli_basis]}
        )

        for tomo_axis in da.coords['tomo_axis'].values:
            tomo_data = da.sel(tomo_axis = tomo_axis)
            pauli_data = inverse_hadamard @ tomo_data.data
            paulis = ["0,0", f"{tomo_axis[0]+1},0", f"0,{tomo_axis[1]+1}", f"{tomo_axis[0]+1},{tomo_axis[1]+1}"]
            for i, pauli in enumerate(paulis):
                paulis_data.value.loc[{'pauli_op': pauli}] += pauli_data[i]
                paulis_data.appearances.loc[{'pauli_op': pauli}] += 1
            
        paulis_data = xr.where(paulis_data.appearances != 0, paulis_data.value / paulis_data.appearances, paulis_data.value)
        
        return paulis_data


    def get_density_matrix(paulis_data):
        # 2Q
        # Define the Pauli matrices
        I = np.array([[1, 0], [0, 1]])
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])

        # Create a vector of the Pauli matrices
        pauli_matrices = [I, X, Y, Z]

        rho = np.zeros((4,4))

        for i, pauli_i in enumerate(pauli_matrices):
            for j, pauli_j in enumerate(pauli_matrices):
                rho = rho + 0.25*paulis_data.sel(pauli_op = f"{i},{j}").values * np.kron(pauli_i, pauli_j)
        
        return rho

    # %% {QUA_program}
    n_shots = node.parameters.num_shots  # The number of averages

    flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

    with program() as CPhase_Oscillations:
        n = declare(int)
        n_st = declare_stream()
        state_control = [declare(int) for _ in range(num_qubit_pairs)]
        state_target = [declare(int) for _ in range(num_qubit_pairs)]
        state = [declare(int) for _ in range(num_qubit_pairs)]
        state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
        state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
        state_st = [declare_stream() for _ in range(num_qubit_pairs)]
        # tomo_axis_control = declare(int)
        # tomo_axis_target = declare(int)
        
        for i, qp in enumerate(qubit_pairs):
            # Bring the active qubits to the minimum frequency point
            machine.set_all_fluxes(flux_point, qp.qubit_control)

            with for_(n, 0, n < n_shots, n + 1):
                save(n, n_st) 
                # with for_(tomo_axis_control, 0, tomo_axis_control < 3, tomo_axis_control + 1):
                    # with for_(tomo_axis_target, 0, tomo_axis_target < 3, tomo_axis_target + 1):
                for tomo_axis_control in [0,1,2]:
                    for tomo_axis_target in [0,1,2]:
                        # reset
                        if node.parameters.reset_type == "active":
                                active_reset(qp.qubit_control, "readout")
                                active_reset(qp.qubit_target, "readout") 
                        else:
                            wait(5*qp.qubit_control.thermalization_time * u.ns)
                        qp.align()
                        # Bell state
                        qp.qubit_control.xy.play("-y90")
                        qp.qubit_target.xy.play("y90")
                        qp.align()
                        qp.gates['Cz'].execute()
                        qp.align()
                        qp.qubit_control.xy.play("y90")
                        qp.align()
                        # tomography pulses
                        # with if_(tomo_axis_control == 0): #X axis
                        #     qp.qubit_control.xy.play("y90")
                        # with if_(tomo_axis_control == 1): #Y axis
                        #     qp.qubit_control.xy.play("x90")
                        # with if_(tomo_axis_target == 0): #X axis
                        #     qp.qubit_target.xy.play("y90")
                        # with if_(tomo_axis_target == 1): #Y axis
                        #     qp.qubit_target.xy.play("x90")
                        if tomo_axis_control == 0:
                            qp.qubit_control.xy.play("y90")
                        if tomo_axis_control == 1:
                            qp.qubit_control.xy.play("x90")
                        if tomo_axis_target == 0:
                            qp.qubit_target.xy.play("y90")
                        if tomo_axis_target == 1:
                            qp.qubit_target.xy.play("x90")
                        qp.align()            
                        # readout
                        readout_state(qp.qubit_control, state_control[i])
                        readout_state(qp.qubit_target, state_target[i])
                        assign(state[i], state_control[i]*2 + state_target[i])
                        save(state_control[i], state_st_control[i])
                        save(state_target[i], state_st_target[i])
                        save(state[i], state_st[i])
                    align()
            
        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                state_st_control[i].buffer(3).buffer(3).buffer(n_shots).save(f"state_control{i + 1}")
                state_st_target[i].buffer(3).buffer(3).buffer(n_shots).save(f"state_target{i + 1}")
                state_st[i].buffer(3).buffer(3).buffer(n_shots).save(f"state{i + 1}")

    # %% {Simulate_or_execute}
    if node.parameters.simulate:
        # Simulates the QUA program for the specified duration
        simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
        job = qmm.simulate(config, CPhase_Oscillations, simulation_config)
        job.get_simulated_samples().con1.plot()
        node.results = {"figure": plt.gcf()}
        node.machine = machine
        node.save()
    elif node.parameters.load_data_id is None:
        with qm_session(qmm, config, timeout=node.parameters.timeout ) as qm:
            job = qm.execute(CPhase_Oscillations)

            results = fetching_tool(job, ["n"], mode="live")
            while results.is_processing():
                # Fetch results
                n = results.fetch_all()[0]
                # Progress bar
                progress_counter(n, n_shots, start_time=results.start_time)

    # %% {Data_fetching_and_dataset_creation}
    if not node.parameters.simulate:
        if node.parameters.load_data_id is None:
            # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
            ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {"tomo_axis_target": [0,1,2], "tomo_axis_control": [0,1,2], "N": np.linspace(1, n_shots, n_shots)})
        else:
            ds, machine = load_dataset(node.parameters.load_data_id)
            
        node.results = {"ds": ds}
        
    # %%
    import xarray as xr
    if not node.parameters.simulate:
        states = [0,1,2,3]

        results = []
        for state in states:
            results.append((ds.state == state).sum(dim = "N") / node.parameters.num_shots)
            
    results_xr = xr.concat(results, dim=xr.DataArray(states, name="state"))
    results_xr = results_xr.rename({"dim_0": "state"})
    results_xr = results_xr.stack(
            tomo_axis=['tomo_axis_target', 'tomo_axis_control'])

    corrected_results = []
    fidelities = {}
    for qp in qubit_pairs:
        corrected_results_qp = [] 
        for tomo_axis_control in [0,1,2]:
            corrected_results_control = []
            for tomo_axis_target in [0,1,2]:
                results = results_xr.sel(tomo_axis_control = tomo_axis_control, tomo_axis_target = tomo_axis_target, 
                                        qubit = qp.name)
                results = np.linalg.inv(qp.confusion) @ results.data
                # results = np.linalg.inv(np.diag((1,1,1,1))) @ results.data

                results = results * (results > 0)
                results = results / results.sum()
                corrected_results_control.append(results)
            corrected_results_qp.append(corrected_results_control)
        corrected_results.append(corrected_results_qp)

        # %%

        # Convert corrected_results to an xarray DataArray
        corrected_results_xr = xr.DataArray(
            corrected_results,
            dims=['qubit', 'tomo_axis_control', 'tomo_axis_target', 'state'],
            coords={
                'qubit': [qp.name for qp in qubit_pairs],
                'tomo_axis_control': [0, 1, 2],
                'tomo_axis_target': [0, 1, 2],
                'state': ['00', '01', '10', '11']
            }
        )
        corrected_results_xr = corrected_results_xr.stack(
                tomo_axis=['tomo_axis_target', 'tomo_axis_control'])

        # Store the xarray in the node results

        # %%

        paulis_data = {}
        rhos = {}
        for qp in qubit_pairs:
            paulis_data[qp.name] = get_pauli_data(corrected_results_xr.sel(qubit = qp.name))
            rhos[qp.name] = get_density_matrix(paulis_data[qp.name])
            
        # %%
        from scipy.linalg import sqrtm
        ideal_dat = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])/2
        s_ideal = sqrtm(ideal_dat)

        for qp in qubit_pairs:
            fidelity = np.abs(np.trace(sqrtm(s_ideal @rhos[qp.name] @ s_ideal)))**2
            fidelities[qp.name] = fidelity
            print(f"Fidelity of {qp.name}: {fidelity:.3f}")
            purity = np.abs(np.trace(rhos[qp.name] @ rhos[qp.name]))
            print(f"Purity of {qp.name}: {purity:.3f}")
            print()
            node.results[f"{qp.name}_fidelity"] = fidelity
            node.results[f"{qp.name}_purity"] = purity



    # %%
    if not node.parameters.simulate:
        
        for qp in qubit_pairs:
            ideal_dat = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])/2
            fig = plot_3d_hist_with_frame(rhos[qp.name], ideal_dat, title = f"Fidelity of {qp.name}: {fidelity:.3f}")
            node.results[f"{qp.name}_figure_city"] = fig
        
        grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
        grid = QubitPairGrid(grid_names, qubit_pair_names)
        for ax, qubit_pair in grid_iter(grid):
            rho = np.real(rhos[qubit_pair['qubit']])
            ax.pcolormesh(rho, vmin = -0.5, vmax = 0.5, cmap = "RdBu")
            # plt.colorbar(ax.pcolormesh(rho), ax=ax)
            for i in range(4):
                for j in range(4):
                    if np.abs(rho[i][j]) < 0.1:
                        ax.text(i+0.5, j+0.5, f"{ rho[i][j]:.2f}", ha="center", va="center", color="k")
                    else:
                        ax.text(i+0.5, j+0.5, f"{ rho[i][j]:.2f}", ha="center", va="center", color="w")
            ax.set_title(qubit_pair['qubit'])
            ax.set_xlabel('Pauli Operators')
            ax.set_ylabel('Pauli Operators')
            ax.set_xticks(range(4), ['00', '01', '10', '11'])
            ax.set_yticks(range(4), ['00', '01', '10', '11'])
            ax.set_xticklabels(['00', '01', '10', '11'], rotation=45, ha='right')
            ax.set_yticklabels(['00', '01', '10', '11'])
        grid.fig.suptitle(f"Bell state tomography (real part)")
        node.results["figure_rho_real"] = grid.fig
            
        grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
        grid = QubitPairGrid(grid_names, qubit_pair_names)
        for ax, qubit_pair in grid_iter(grid):
            rho = np.imag(rhos[qubit_pair['qubit']])
            ax.pcolormesh(rho, vmin = -0.1, vmax = 0.1, cmap = "RdBu")
            # plt.colorbar(ax.pcolormesh(rho), ax=ax)
            for i in range(4):
                for j in range(4):
                    if np.abs(rho[i][j]) < 0.1:
                        ax.text(i+0.5, j+0.5, f"{ rho[i][j]:.2f}", ha="center", va="center", color="k")
                    else:
                        ax.text(i+0.5, j+0.5, f"{ rho[i][j]:.2f}", ha="center", va="center", color="w")
            ax.set_title(qubit_pair['qubit'])
            ax.set_xlabel('Pauli Operators')
            ax.set_ylabel('Pauli Operators')
            ax.set_xticks(range(4), ['00', '01', '10', '11'])
            ax.set_yticks(range(4), ['00', '01', '10', '11'])
            ax.set_xticklabels(['00', '01', '10', '11'], rotation=45, ha='right')
            ax.set_yticklabels(['00', '01', '10', '11'])
        grid.fig.suptitle(f"Bell state tomography (imaginary part)")
        node.results["figure_rho_imag"] = grid.fig

        grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
        grid = QubitPairGrid(grid_names, qubit_pair_names)
        for ax, qubit_pair in grid_iter(grid):
            # Extract the values and labels for plotting
            values = paulis_data[qubit_pair['qubit']].values
            labels = paulis_data[qubit_pair['qubit']].coords['pauli_op'].values

            # Create a bar plot
            bars = ax.bar(range(len(values)), values)

            # Customize the plot
            ax.set_xlabel('Pauli Operators')
            ax.set_ylabel('Value')
            ax.set_title(qubit_pair['qubit'])
            ax.set_xticks(range(len(labels)), labels, rotation=45, ha='right')

            # Add value labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom')

    # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()
        node.results["figure_paulis"] = grid.fig
    # %%

    # %% {Update_state}

    # %% {Save_results}
    if not node.parameters.simulate:
        node.outcomes = {qp.name: "successful" for qp in qubit_pairs}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()

    return list(fidelities.values()), list(fidelities.keys())





def to_float_clean(x):
    s = str(x).strip().lstrip("'")
    return float(s)

if __name__ == "__main__":
    num_runs = 30 # Change, 'inf' or an integer 
    track_offset_together:bool = True  # run ramsey vs flux after 2QRB but didn't update state.
    track_bell_together:bool = True
    cooldown_sec = 1 #Chnage
    target_operaion:Literal["cz", "idle_2q"] = 'cz'
    pairs_to_run:list[str] = ["coupler_q2_q3"]
    out_dir = Path('/home/ratiswu/Qualibrate_data/5Q4C_Qcage/2025-12-12/#0a0a_2QRB_Tracking') #Chnage, # folder path to save the file
    
    # ======================================================================================================================
    out_dir.mkdir(parents=True, exist_ok=True)
    # csv file name (contains the time to avoid overlap)
    session_ts = datetime.now().isoformat(timespec="seconds").replace(":", "-")
    wide_csv = out_dir / f"CZ_Fidelity_Tracking_{session_ts}.csv" 
    standard_csv = out_dir / f"Standard2Q_Fidelity_Tracking_{session_ts}.csv"
    header_qubits = None 


    if track_offset_together:
        offset_header = None 
        offset_csv = out_dir / f"Offset_Tracking_{session_ts}.csv" 
    
    if track_bell_together:
        bell_header = None 
        bell_csv = out_dir / f"BellState_Tracking_{session_ts}.csv"
    
    
    start = time.time()
    i = 0
    while True:
        i += 1
        ts = datetime.now().isoformat(timespec="seconds")
        print(f"[{i:02d}/{num_runs}] {ts} Starting run…")

        try:
            TQ_fidelity, CZ_fidelity, pair_names = [], [], []
            # Because 2QRB node can not run multiple pairs
            for pair in pairs_to_run:
                s_fidelity, fidelity, name = run_cz_tracking(pair,target_operation=target_operaion)
                CZ_fidelity.append(fidelity)
                TQ_fidelity.append(s_fidelity)
                pair_names.append(name)
            
            # create column names for qubits, determined by the first run
            if header_qubits is None:
                header_qubits = list(pair_names)
                with open(wide_csv, "w", newline="") as f:
                    csv.writer(f).writerow(["run_index", "timestamp", *header_qubits])
                f.close()
                with open(standard_csv, "w", newline="") as h:
                    csv.writer(h).writerow(["run_index", "timestamp", *header_qubits])
                h.close()

            with open(wide_csv, "a", newline="") as f:
                writer = csv.writer(f)
                name_to_val = {n: to_float_clean(v) for n, v in zip(pair_names, CZ_fidelity)}
                row_vals = [name_to_val[q] for q in header_qubits]  
                writer.writerow([i, ts, *row_vals])
            f.close()
            with open(standard_csv, "a", newline="") as h:
                writer = csv.writer(h)
                name_to_val = {n: to_float_clean(v) for n, v in zip(pair_names, TQ_fidelity)}
                row_vals = [name_to_val[q] for q in header_qubits]  
                writer.writerow([i, ts, *row_vals])
            h.close()

            print(f"CZ: [{i:02d}/{num_runs}] appended to {wide_csv.name}")
        except Exception as e:
            print(f"CZ: [{i:02d}/{num_runs}] error: {e!r}, continuing to the next run…")

        ## Optional, tracking bell state fidelity together.
        if track_bell_together:
            try:
                bell_fidelity, qubit_names = run_Bell_fidelity(pairs_to_run, shots=5000)
                if bell_header is None:
                    bell_header = list(qubit_names)
                    with open(bell_csv, "w", newline="") as g:
                        csv.writer(g).writerow(["run_index", "timestamp", *bell_header])
                    g.close()
                with open(bell_csv, "a", newline="") as g:
                    writer = csv.writer(g)
                    name_to_val = {n: to_float_clean(v) for n, v in zip(qubit_names, bell_fidelity)}
                    row_vals = [name_to_val[q] for q in bell_header]  
                    writer.writerow([i, ts, *row_vals])
                g.close()

                print(f"BELL-State: [{i:02d}/{num_runs}] appended to {bell_csv.name}")
            except Exception as e:
                print(f"BELL-State: [{i:02d}/{num_runs}] error: {e!r}, continuing to the next run…")
        
        ## Optional, tracking offset drift together without updating.
        if track_offset_together:
            try:
                fluctuations, qubit_names = run_offset_tracking(None, update_state = False)
                if offset_header is None:
                    offset_header = list(qubit_names)
                    with open(offset_csv, "w", newline="") as g:
                        csv.writer(g).writerow(["run_index", "timestamp", *offset_header])
                    g.close()
                with open(offset_csv, "a", newline="") as g:
                    writer = csv.writer(g)
                    name_to_val = {n: to_float_clean(v) for n, v in zip(qubit_names, fluctuations)}
                    row_vals = [name_to_val[q] for q in offset_header]  
                    writer.writerow([i, ts, *row_vals])
                g.close()

                print(f"OFFSET: [{i:02d}/{num_runs}] appended to {offset_csv.name}")
            except Exception as e:
                print(f"OFFSET: [{i:02d}/{num_runs}] error: {e!r}, continuing to the next run…")

        
        time.sleep(cooldown_sec)
        each_time = time.time()
        if each_time - start > 3*24*3600:
            break
        if not isinstance(num_runs, str):
            if isinstance(num_runs, int):
                if i == num_runs :
                    break

    print(f"All runs finished !")
    final_time = time.time()


    # Just make sure to close qms
    from quam_libs.components import QuAM
    machine = QuAM.load()
    qmm = machine.connect() # Use this line if you want to connect to the QOP defined in the state file
    qmm.close_all_qms()

    print(f"Total elapsed {final_time-start} secs for CD = {cooldown_sec} secs and {i} runs.")

