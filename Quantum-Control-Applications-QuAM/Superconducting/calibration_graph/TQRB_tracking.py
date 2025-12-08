import time
import csv
from datetime import datetime
from pathlib import Path
from quam_libs.experiments.rb_standard.data_utils import InterleavedRBResult
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
from quam_libs.experiments.rb_standard.rb_utils import InterleavedRB
import matplotlib.pyplot as plt
from quam_libs.lib.plot_utils import QubitGrid, grid_iter

## 2Q_RB
def run_cz_tracking(pair_name:str, depth_squences:tuple[int]|None=None, target_operation:str='cz'):
    ### The start of the copy ### copy the node to here

    if depth_squences is None:
        depth_squences = (0,1,2,3,4,5,6,7,8,11,14,19,25,35,40)

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
        reduce_to_1q_cliffords: bool = True
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

        

    return fidelity[0], pairs[0] # temporarily, 2QRB node can only run single pair


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



def to_float_clean(x):
    s = str(x).strip().lstrip("'")
    return float(s)

if __name__ == "__main__":
    num_runs = 1 # Change, 'inf' or an integer 
    track_offset_together:bool = True  # run ramsey vs flux after 2QRB but didn't update state.
    cooldown_sec = 1 #Chnage
    target_operaion:Literal["cz", "idle_2q"] = 'cz'
    pairs_to_run:list[str] = ["coupler_q2_q3"]
    out_dir = Path('/home/ratiswu/Qualibrate_data/5Q4C_Qcage/2025-12-08/#0a0a_2QRB_Tracking') #Chnage, # folder path to save the file
    
    # ======================================================================================================================
    out_dir.mkdir(parents=True, exist_ok=True)
    # csv file name (contains the time to avoid overlap)
    session_ts = datetime.now().isoformat(timespec="seconds").replace(":", "-")
    wide_csv = out_dir / f"CZ_Fidelity_Tracking_{session_ts}.csv" 
    header_qubits = None 


    if track_offset_together:
        offset_header = None 
        offset_csv = out_dir / f"Offset_Tracking_{session_ts}.csv" 
    
    
    start = time.time()
    i = 0
    while True:
        i += 1
        ts = datetime.now().isoformat(timespec="seconds")
        print(f"[{i:02d}/{num_runs}] {ts} Starting run…")

        try:
            CZ_fidelity, pair_names = [], []
            # Because 2QRB node can not run multiple pairs
            for pair in pairs_to_run:
                fidelity, name = run_cz_tracking(pair,target_operation=target_operaion)
                CZ_fidelity.append(fidelity)
                pair_names.append(name)
            
            # create column names for qubits, determined by the first run
            if header_qubits is None:
                header_qubits = list(pair_names)
                with open(wide_csv, "w", newline="") as f:
                    csv.writer(f).writerow(["run_index", "timestamp", *header_qubits])
                f.close()

            with open(wide_csv, "a", newline="") as f:
                writer = csv.writer(f)
                name_to_val = {n: to_float_clean(v) for n, v in zip(pair_names, CZ_fidelity)}
                row_vals = [name_to_val[q] for q in header_qubits]  
                writer.writerow([i, ts, *row_vals])
            f.close()

            print(f"CZ: [{i:02d}/{num_runs}] appended to {wide_csv.name}")
        except Exception as e:
            print(f"CZ: [{i:02d}/{num_runs}] error: {e!r}, continuing to the next run…")

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

