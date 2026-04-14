"""
        TWO-QUBIT INTERLEAVED RANDOMIZED BENCHMARKING
The program consists in playing random sequences of Clifford gates interleaved with a target two-qubit gate and measuring 
the state of the resonators afterward. Each random sequence is generated for the maximum depth (specified as an input) 
and played for each depth asked by the user (the sequence is truncated to the desired depth). Each truncated sequence 
ends with the recovery gate that will bring the qubits back to their ground state.

The random circuits are generated offline and transpiled to a basis gate set (default is ['rz', 'sx', 'x', 'cz']). 
The circuits are executed per two-qubit layer using a switch_case block structure, allowing for efficient execution 
of the quantum circuits.

The program supports two types of target gates: 'idle_2q' and 'cz'. The 'idle_2q' gate is implemented as a hardcoded 
wait time of T1/50 for each qubit. The interleaved RB protocol allows for direct measurement of the fidelity of the 
target gate by comparing the decay rates of the interleaved sequences with reference sequences.

Key Features:
    - reduce_to_1q_cliffords: When enabled (default), the Clifford gates are sampled as 1q Cliffords per qubit 
      (this is of course a much smaller subset of the whole 2q Clifford group).
    - use_input_stream: When enabled (default), the circuit sequences are streamed to the OPX in using the 
      input stream feature. This allows for dynamic circuit execution and reduces memory usage on the OPX.

Each sequence is played multiple times for averaging, and multiple random sequences are generated for each depth to 
improve statistical significance. The data is then post-processed to extract both the two-qubit Clifford fidelity and 
the specific target gate fidelity.

Prerequisites:
    - Having calibrated both qubits' single-qubit gates (resonator_spectroscopy, qubit_spectroscopy, rabi_chevron, power_rabi).
    - Having calibrated the two-qubit gate (cz or idle_2q) that will be benchmarked.
    - Having calibrated the readout for both qubits (readout_frequency, amplitude, duration_optimization IQ_blobs).
    - Having set the appropriate flux bias points for the qubit pair.
    - Having calibrated the qubit frequencies and coupling strength.
"""

# %%

from datetime import datetime, timezone, timedelta
from typing import List, Literal, Optional
from more_itertools import flatten
from quam_libs.experiments.rb_standard.data_utils import RBResult, InterleavedRBResult
import xarray as xr
import numpy as np
from qm import generate_qua_script
from matplotlib.ticker import MaxNLocator
from qm.qua import *
from qm import SimulationConfig
from qualang_tools.multi_user import qm_session

from qualang_tools.results import progress_counter, fetching_tool

from qualibrate  import NodeParameters, QualibrationNode
from quam_libs.experiments.rb_standard.circuit_utils import layerize_quantum_circuit, process_circuit_to_integers
from quam_libs.experiments.rb_standard.qua_utils import QuaProgramHandler
from quam_libs.lib.plot_utils import plot_samples
from quam_libs.lib.save_utils import fetch_results_as_xarray

from quam_libs.components import QuAM
from quam_libs.experiments.rb_standard.cloud_utils import write_sync_hook
from quam_libs.experiments.rb_standard.rb_utils import InterleavedRB, StandardRB
from quam_libs.experiments.rb_standard.data_utils import plot_combined_rb
from quam_libs.experiments.rb_standard.plot_utils import gate_mapping

from time import sleep

def run_once(target_operation:Literal['idle_2q', 'cz'] = 'cz'):
    # %% {Node_parameters}
    class Parameters(NodeParameters):
        qubit_pairs: Optional[List[str]] = ["coupler_q4_q5"] #None
        circuit_lengths: tuple[int] = (0,1,3,5,9,16,20,25) # in number of cliffords
        num_circuits_per_length: int = 20
        num_averages: int = 300
        target_gate: str = target_operation # "idle_2q" or "cz" supported 
        basis_gates: list[str] = ['rz', 'sx', 'x', 'cz'] 
        flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
        reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
        reduce_to_1q_cliffords: Literal[False] = False
        use_input_stream: bool = False
        simulate: bool = False
        simulation_duration_ns: int = 10000
        load_data_id: Optional[int] = None
        timeout: int = 600
        seed: int|None = None
        targets_name = "qubit_pairs"

    node = QualibrationNode(name="70cx_SI_2QRB", parameters=Parameters())

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


    node.results = {}
    id, simulate = node.parameters.load_data_id, node.parameters.simulate
    length_incluse_inverse = [ n+1 if n!=0 else n for n in node.parameters.circuit_lengths]

    # %% {Random circuit generation}
    if node.parameters.load_data_id is None:
        standard_RB = StandardRB(
            amplification_lengths=node.parameters.circuit_lengths,
            num_circuits_per_length=node.parameters.num_circuits_per_length,
            basis_gates=node.parameters.basis_gates,
            reduce_to_1q_cliffords=False,
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

        circuits_as_ints_SRB = []
        for circuits_per_len in transpiled_circuits_as_ints.values():
            for circuit in circuits_per_len:
                circuit_with_measurement = circuit + [66] # readout
                circuits_as_ints_SRB.append(circuit_with_measurement)


        interleaved_RB = InterleavedRB(
            target_gate=node.parameters.target_gate,
            amplification_lengths=node.parameters.circuit_lengths,
            num_circuits_per_length=node.parameters.num_circuits_per_length,
            basis_gates=node.parameters.basis_gates,
            num_qubits=2,
            reduce_to_1q_cliffords=False,
            seed=node.parameters.seed
        )

        transpiled_circuits = interleaved_RB.transpiled_circuits
        transpiled_circuits_as_ints = {}
        for l, circuits in transpiled_circuits.items():
            transpiled_circuits_as_ints[l] = [process_circuit_to_integers(layerize_quantum_circuit(qc)) for qc in circuits]

        circuits_as_ints_IRB = []
        for circuits_per_len in transpiled_circuits_as_ints.values():
            for circuit in circuits_per_len:
                circuit_with_measurement = circuit + [66] # readout
                circuits_as_ints_IRB.append(circuit_with_measurement)
    else:
        print("Load data")
        node = node.load_from_id(node.parameters.load_data_id)
        
        ## SRB
        ds_SRB = node.results["SRB_ds"]
        circuit_depths_SRB = list(node.parameters.circuit_lengths)
        num_repeats_SRB = node.parameters.num_circuits_per_length
        num_averages_SRB = node.parameters.num_averages 

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
        circuits_as_ints_SRB = []


        ## IRB
        ds_IRB = node.results["IRB_ds"]
        target_gate = node.parameters.target_gate
        circuit_depths_IRB = list(node.parameters.circuit_lengths)
        num_repeats_IRB = node.parameters.num_circuits_per_length
        num_averages_IRB = node.parameters.num_averages 
        circuits_as_ints_IRB = []

        node.parameters.load_data_id = id
        node.parameters.simulate = simulate


    # %% {QUA_program}
    for RB_type, RB_circuit in {"SRB":circuits_as_ints_SRB, "IRB":circuits_as_ints_IRB}.items():

        num_pairs = len(qubit_pairs)
        qua_program_handler = QuaProgramHandler(node, num_pairs, RB_circuit, node.machine, qubit_pairs)
        rb = qua_program_handler.get_qua_program()

        if node.parameters.simulate:
            config = node.machine.generate_config()
            simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns//4)  # in clock cycles
            job = qmm.simulate(config, rb, simulation_config)
            samples = job.get_simulated_samples()
            qubit_names = [qubit_pair.qubit_control.name for qubit_pair in qubit_pairs] + [qubit_pair.qubit_target.name for qubit_pair in qubit_pairs]
            readout_lines = set([q[1] for q in qubit_names])
            fig = plot_samples(samples, qubit_names, readout_lines=list(readout_lines), xlim=(0,10000))

        else:
            if node.parameters.load_data_id is None:
                # Prepare data for saving
                
                date_time = datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S")
                config = node.machine.generate_config()
                

                # sourceFile = open(f'{RB_type}_debug.py', 'w')
                # print(generate_qua_script(rb, config), file=sourceFile) 
                # sourceFile.close()
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


                # for num in flatten(RB_circuit):
                #     print(gate_mapping[num])

                ds = fetch_results_as_xarray(
                job.result_handles,
                qubit_pairs,
                    { "sequence": range(node.parameters.num_circuits_per_length), "depths": length_incluse_inverse, "shots": range(node.parameters.num_averages)},
                )
                node.results[f"{RB_type}_ds"] = ds
                sleep(5)
            else:
                node.results[f"{RB_type}_ds"] = node.results[f"{RB_type}_ds"]


    # %% {Data_analysis and plotting}

    if not node.parameters.simulate:
        srb_result = {}
        irb_result = {}
        cz_fidelity = {}
        for RB_type, RB_circuit in {"SRB":circuits_as_ints_SRB, "IRB":circuits_as_ints_IRB}.items():
            ds = node.results[f"{RB_type}_ds"]
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

            for qp in qubit_pairs:
                if RB_type == 'SRB':
                    srb_result[qp.id] = RBResult(
                    circuit_depths=length_incluse_inverse,
                    num_repeats=node.parameters.num_circuits_per_length,
                    num_averages=node.parameters.num_averages,
                    state=ds_transposed.sel(qubit=qp.name).state.data
                    )
            
                    # Fit the data and calculate all error and fidelity metrics
                    srb_result[qp.id].fit(
                        average_layers_per_clifford=average_layers_per_clifford,
                        average_gates_per_2q_layer=None
                    )
                    
                    # Plot the results
                    fig = srb_result[qp.id].plot_with_fidelity()
                    ax = fig.axes[0]
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                    fig.suptitle(f"2Q_SRB - {qp.name}")
                    # node.add_node_info_subtitle(fig)
                    fig.show()
                    
                    node.results[f"{qp.id}_figure_2QSRB"] = fig
                    
                
                else:

                    irb_result[qp.id] = InterleavedRBResult(
                        # standard_rb_alpha=node.machine.qubit_pairs[qp.id].macros["cz"].fidelity.get("StandardRB", 1).get("alpha", 1),
                        standard_rb_alpha=srb_result[qp.id].alpha, # if "StandardRB" in qp.extras else 1,
                        circuit_depths=length_incluse_inverse,
                        num_repeats=node.parameters.num_circuits_per_length,
                        num_averages=node.parameters.num_averages,
                        state=ds_transposed.sel(qubit=qp.name).state.data
                    )

                    # Fit the data and calculate all error and fidelity metrics
                    irb_result[qp.id].fit()
                    
                    # Plot the results
                    fig = irb_result[qp.id].plot_with_fidelity()
                    fig.suptitle(f"2Q {node.parameters.target_gate.upper()} IRB - {qp.name}")
                    # node.add_node_info_subtitle(fig)
                    ax = fig.axes[0]
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


                    fig.show()
                    
                    node.results[f"{qp.id}_figure_IRB_decay"] = fig
        
        for qp in qubit_pairs:
            cz_fidelity[qp.id] = irb_result[qp.id].fidelity
            fig_combined = plot_combined_rb(
            qp.name,
            srb_result[qp.id],
            irb_result[qp.id],
            target_gate=node.parameters.target_gate.upper()
            )
            ax = fig_combined.axes[0]
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            fig_combined.show()

            node.results[f"{qp.id}_figure_combined_RB_decay"] = fig_combined

    # %% {Update_state}
    if not node.parameters.simulate:
        with node.record_state_updates():
            for qp in qubit_pairs:
                qp.extras["StandardRB"] = {
                    "error_per_clifford": 1 - srb_result[qp.id].fidelity, 
                    # "error_per_2q_layer": rb_result[qp.id].error_per_2q_layer,
                    # "error_per_gate": rb_result[qp.id].error_per_gate,
                    # "average_gate_fidelity": 1 - rb_result[qp.id].error_per_gate,
                    "alpha": srb_result[qp.id].alpha}
                qp.extras['Interleaved_RB'] = srb_result[qp.id].fidelity
        # %% {Save_results}
        node.save()
    return list(cz_fidelity.values()), list(cz_fidelity.keys())
