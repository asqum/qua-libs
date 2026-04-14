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
import math
from time import time, sleep


def run_batched_rb(depth_type:Literal["short", "standard", "long"], target_operation: Literal['idle_2q', 'cz'] = 'cz', total_circuits: int = 40, id_to_load:int|None=None, time_mark:bool=True, zero_removal_plot:bool=False):
    start = time()
    if total_circuits < 2:
        total_circuits = 2

    depth_mapping = {
        "short": (0,2), #(0, 1, 3, 5, 7, 11, 16),
        "standard": (0, 2, 4, 7, 11, 17, 24),
        "long": (0, 2, 4, 8, 16, 24, 32, 64)
    }

    tot_depth = depth_mapping[depth_type]
    depth_sum = sum(list(tot_depth))
    max_randomness = math.floor(16000/(depth_sum*12)) # 12 is an estimated gate number per Clifford 
    
    if max_randomness<2:
        maximum_depth_sum = 16000//(2*12)
        raise ValueError(f"random circuits per length must >=2, please reduce your depth by {math.ceil(depth_sum-maximum_depth_sum)}.")
   
   
    BATCH_SIZE = min(max_randomness, total_circuits)
    num_batches = math.ceil(total_circuits / BATCH_SIZE)
    total_circuits = num_batches * BATCH_SIZE # actually we ran
    
    class Parameters(NodeParameters):
        qubit_pairs: Optional[List[str]] = ["coupler_q4_q5"]
        circuit_lengths: tuple[int] = tot_depth
        num_circuits_per_length: int = BATCH_SIZE
        num_averages: int = 300
        target_gate: str = target_operation
        basis_gates: list[str] = ['rz', 'sx', 'x', 'cz']
        flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
        reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
        reduce_to_1q_cliffords: Literal[False] = False
        use_input_stream: bool = False
        simulate: bool = False
        simulation_duration_ns: int = 10000
        load_data_id: Optional[int] = id_to_load
        timeout: int = 600
        seed: int|None = None
        targets_name = "qubit_pairs"

    node = QualibrationNode(name="70cx_SI_2QRB_Batched", parameters=Parameters())

    
    node.machine = QuAM.load()
    
    if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
        qubit_pairs = node.machine.active_qubit_pairs
    else:
        qubit_pairs = [node.machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]

    if len(qubit_pairs) == 0:
        raise ValueError("No qubit pairs selected")

    # 🌟 修正點 1：開啟與 QOP 的連線
    qmm = node.machine.connect()
    
    node.results = {}
    length_incluse_inverse = list(node.parameters.circuit_lengths)

    # 用於存放結果的列表
    srb_ds_list = []
    irb_ds_list = []
    all_layers_per_clifford = []

    if node.parameters.load_data_id is None:
        for b in range(num_batches):
            print(f"\n>>> 正在執行批次 {b+1}/{num_batches} (電路索引: {b*BATCH_SIZE} 到 {(b+1)*BATCH_SIZE-1})")
            
            # 讓每一批次的 seed 不同，確保產生不同的隨機電路
            current_seed = node.parameters.seed + b if node.parameters.seed is not None else None

            # --- 生成 SRB 電路 ---
            standard_RB = StandardRB(
                amplification_lengths=node.parameters.circuit_lengths,
                num_circuits_per_length=BATCH_SIZE,
                basis_gates=node.parameters.basis_gates,
                reduce_to_1q_cliffords=False,
                num_qubits=2,
                seed=current_seed
            )

            transpiled_circuits_srb = standard_RB.transpiled_circuits
            transpiled_circuits_as_ints_srb = {}
            layerized_circuits_srb = {}
            for l, circuits in transpiled_circuits_srb.items():
                layerized_circuits_srb[l] = [layerize_quantum_circuit(qc) for qc in circuits]
                transpiled_circuits_as_ints_srb[l] = [process_circuit_to_integers(qc) for qc in layerized_circuits_srb[l]]

            batch_avg_layers = np.mean([
                np.mean([len(circ) for circ in circuits]) / np.array(length + 1)
                for length, circuits in transpiled_circuits_as_ints_srb.items() if length > 0
            ])
            all_layers_per_clifford.append(batch_avg_layers)

            circuits_as_ints_SRB = []
            for circuits_per_len in transpiled_circuits_as_ints_srb.values():
                for circuit in circuits_per_len:
                    circuits_as_ints_SRB.append(circuit + [66]) # readout

            # --- 生成 IRB 電路 ---
            interleaved_RB = InterleavedRB(
                target_gate=node.parameters.target_gate,
                amplification_lengths=node.parameters.circuit_lengths,
                num_circuits_per_length=BATCH_SIZE,
                basis_gates=node.parameters.basis_gates,
                num_qubits=2,
                reduce_to_1q_cliffords=False,
                seed=current_seed
            )

            transpiled_circuits_irb = interleaved_RB.transpiled_circuits
            transpiled_circuits_as_ints_irb = {}
            for l, circuits in transpiled_circuits_irb.items():
                transpiled_circuits_as_ints_irb[l] = [process_circuit_to_integers(layerize_quantum_circuit(qc)) for qc in circuits]

            circuits_as_ints_IRB = []
            for circuits_per_len in transpiled_circuits_as_ints_irb.values():
                for circuit in circuits_per_len:
                    circuits_as_ints_IRB.append(circuit + [66]) # readout

            # --- 執行 QUA Program ---
            batch_results = {}
            for RB_type, RB_circuit in {"SRB": circuits_as_ints_SRB, "IRB": circuits_as_ints_IRB}.items():
                switch:bool = True
                max_try = 3
                while switch:
                    try:
                        num_pairs = len(qubit_pairs)
                        qua_program_handler = QuaProgramHandler(node, num_pairs, RB_circuit, node.machine, qubit_pairs)
                        rb_prog = qua_program_handler.get_qua_program()

                        config = node.machine.generate_config()
                        
                        with qm_session(node.machine.qmm, config, timeout=node.parameters.timeout) as qm:
                            job = qm.execute(rb_prog)
                            results = fetching_tool(job, ["iteration"], mode="live")
                            while results.is_processing():
                                n = results.fetch_all()[0]
                                progress_counter(n, node.parameters.num_averages, start_time=results.start_time)

                        ds = fetch_results_as_xarray(
                            job.result_handles,
                            qubit_pairs,
                            {"sequence": range(BATCH_SIZE), "depths": length_incluse_inverse, "shots": range(node.parameters.num_averages)},
                        )

                        switch = False
                    
                    except Exception as e:
                        print(f"{e} while job is running, run it again")
                        max_try -= 1
                        if max_try > 0:
                            switch = True
                        else:
                            switch = False
                            raise RuntimeError(f"{e}")
                    
                # 修改 sequence 座標避免合併時衝突
                ds = ds.assign_coords(sequence=[i + b * BATCH_SIZE for i in range(BATCH_SIZE)])
                batch_results[RB_type] = ds
                sleep(2)

            srb_ds_list.append(batch_results["SRB"])
            irb_ds_list.append(batch_results["IRB"])

        
        print("\n>>> 所有批次執行完畢，正在合併數據...")
        node.results["SRB_ds"] = xr.concat(srb_ds_list, dim="sequence")
        node.results["IRB_ds"] = xr.concat(irb_ds_list, dim="sequence")
        final_avg_layers = np.mean(all_layers_per_clifford)
        node.results["final_avg_layers"] = final_avg_layers
        node.results["total_circuits"] = total_circuits
        shots= node.parameters.num_averages
        end = time()
        elapse_time_s = round(end-start,1)
        node.results["elapse_time_s"] = elapse_time_s
    
    else:
        a_node = node.load_from_id(node.parameters.load_data_id)
        final_avg_layers = a_node.results["final_avg_layers"]
        total_circuits = a_node.results["total_circuits"]
        length_incluse_inverse = list(a_node.parameters.circuit_lengths)
        shots = a_node.parameters.num_averages 
        elapse_time_s = a_node.results["elapse_time_s"]
        node.machine = QuAM.load()
        irb_ds = a_node.results["IRB_ds"]
        srb_ds = a_node.results["SRB_ds"]
        node.results = {}
        node.results[f"SRB_ds"], node.results[f"IRB_ds"] = srb_ds, irb_ds

    
    # 恢復參數以反映合併後的總電路數，確保繪圖的 repeat 數量正確
    node.parameters.num_circuits_per_length = total_circuits


    srb_result = {}
    irb_result = {}
    cz_fidelity = {}
        
    
    for RB_type in ["SRB", "IRB"]:
        ds = node.results[f"{RB_type}_ds"]

        if zero_removal_plot:
            if 0 in length_incluse_inverse:
                length_incluse_inverse.remove(0)
            if "circuit_depth" in ds.variables and "circuit_depth" not in ds.dims:
                ds = ds.drop_vars("circuit_depth")
        
        ds_transposed = ds.rename({"shots": "average", "sequence": "repeat", "depths": "circuit_depth"})
        ds_transposed = ds_transposed.transpose("qubit", "repeat", "circuit_depth", "average")


        if zero_removal_plot:
            ds_transposed = ds_transposed.isel(circuit_depth=slice(1, None))
        

        for qp in qubit_pairs:
            if RB_type == 'SRB':
                srb_result[qp.id] = RBResult(
                    circuit_depths=length_incluse_inverse,
                    num_repeats=total_circuits,
                    num_averages=shots,
                    state=ds_transposed.sel(qubit=qp.name).state.data
                )
                
                srb_result[qp.id].fit(
                    average_layers_per_clifford=final_avg_layers,
                    average_gates_per_2q_layer=None,
                    use_weights=True
                )
                
                fig = srb_result[qp.id].plot_with_fidelity()
                ax = fig.axes[0]
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                fig.suptitle(f"2Q_SRB - {qp.name}")
                fig.show()
                node.results[f"{qp.id}_figure_2QSRB"] = fig
                
            else:
                irb_result[qp.id] = InterleavedRBResult(
                    standard_rb_alpha=srb_result[qp.id].alpha,
                    standard_rb_alpha_err=srb_result[qp.id].alpha_err,
                    circuit_depths=length_incluse_inverse,
                    num_repeats=total_circuits,
                    num_averages=shots,
                    state=ds_transposed.sel(qubit=qp.name).state.data
                )

                irb_result[qp.id].fit(use_weights=True)
                
                fig = irb_result[qp.id].plot_with_fidelity()
                fig.suptitle(f"2Q {node.parameters.target_gate.upper()} IRB - {qp.name}")
                ax = fig.axes[0]
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                fig.show()
                node.results[f"{qp.id}_figure_IRB_decay"] = fig

    # 🌟 修正點 3：加回你截圖中的 Combined Plot
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
        if time_mark:
            fig_combined.suptitle(f"2Q Clifford RB\n elapsed time: {elapse_time_s}s")
        else:
            fig_combined.suptitle(f"2Q Clifford RB")
        fig_combined.tight_layout()
        fig_combined.show()
        node.results[f"{qp.id}_figure_combined_RB_decay"] = fig_combined

    
    with node.record_state_updates():
        for qp in qubit_pairs:
            qp.extras["StandardRB"] = {
                "error_per_clifford": 1 - srb_result[qp.id].fidelity, 
                "alpha": srb_result[qp.id].alpha
            }
            qp.extras['Interleaved_RB'] = srb_result[qp.id].fidelity
    
    node.save()
    return list(cz_fidelity.values()), list(cz_fidelity.keys())



if __name__ == "__main__":
    # # 20 random circuits takes 10 mins
    target_operation:Literal['cz', 'idle_2q'] = 'cz'
    random_gates_per_depth:int = 100
    job_id:int|None = None
    fidelities, names = run_batched_rb(target_operation, random_gates_per_depth, job_id, time_mark=True)