import time
import csv
from datetime import datetime
from pathlib import Path
from quam_libs.experiments.rb_standard.data_utils import InterleavedRBResult
from datetime import datetime, timezone, timedelta
from typing import List, Literal, Optional
import xarray as xr


from qm.qua import *
from qm import SimulationConfig
from qualang_tools.multi_user import qm_session

from qualang_tools.results import progress_counter, fetching_tool

from qualibrate import NodeParameters, QualibrationNode
from quam_libs.experiments.rb.circuit_utils import layerize_quantum_circuit, process_circuit_to_integers
from quam_libs.experiments.rb.qua_utils import QuaProgramHandler
from quam_libs.lib.plot_utils import plot_samples
from quam_libs.lib.save_utils import fetch_results_as_xarray

from quam_libs.components import QuAM
from quam_libs.experiments.rb.cloud_utils import write_sync_hook
from quam_libs.experiments.rb_standard.rb_utils import InterleavedRB



## 2Q_RB
def run_once(pair_name:str, depth_squences:tuple[int]|None=None, target_operation:str='cz'):
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

def to_float_clean(x):
    s = str(x).strip().lstrip("'")
    return float(s)

if __name__ == "__main__":
    num_runs = 'inf' # Change, 'inf' or an integer 
    cooldown_sec = 5 #Chnage
    target_operaion:Literal["cz", "idle_2q"] = 'cz'
    pairs_to_run:list[str] = ["coupler_q1_q2", "coupler_q4_q5","coupler_q2_q3", "coupler_q3_q4"]

    i = 0
    # folder path to save the file
    out_dir = Path('/home/ratiswu/Qualibrate_data/5Q4C_Qcage/2025-12-04/#9a9a_2QRB_Tracking') #Chnage
    out_dir.mkdir(parents=True, exist_ok=True)

    # csv file name (contains the time to avoid overlap)
    session_ts = datetime.now().isoformat(timespec="seconds").replace(":", "-")
    wide_csv = out_dir / f"CZ_Fidelity_Tracking_{session_ts}.csv" 

    header_qubits = None  

    start = time.time()
    # for i in range(1, num_runs + 1):
    while True:
        i += 1
        ts = datetime.now().isoformat(timespec="seconds")
        print(f"[{i:02d}/{num_runs}] {ts} Starting run…")

        try:
            CZ_fidelity, pair_names = [], []
            # Because 2QRB node can not run multiple pairs
            for pair in pairs_to_run:
                fidelity, name = run_once(pair,target_operation=target_operaion)
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

            print(f"[{i:02d}/{num_runs}] appended to {wide_csv.name}")

        except Exception as e:
            print(f"[{i:02d}/{num_runs}] error: {e!r}, continuing to the next run…")

        time.sleep(cooldown_sec)
        each_time = time.time()
        if each_time - start > 3*24*3600:
            break
        if not isinstance(num_runs, str):
            if isinstance(num_runs, int):
                if i == num_runs :
                    break

    print(f"All runs finished: {wide_csv}")
    final_time = time.time()


    # Just make sure to close qms
    from quam_libs.components import QuAM
    machine = QuAM.load()
    qmm = machine.connect() # Use this line if you want to connect to the QOP defined in the state file
    qmm.close_all_qms()

    print(f"Total elapsed {final_time-start} secs for CD = {cooldown_sec} secs and {i} runs.")

