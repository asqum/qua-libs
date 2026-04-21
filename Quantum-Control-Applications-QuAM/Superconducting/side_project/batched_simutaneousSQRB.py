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

from quam_libs.lib.save_utils import fetch_results_as_xarray

from quam_libs.components import QuAM

from quam_libs.experiments.rb_standard.rb_utils import  StandardRB

import math
from time import time, sleep


def run_batched_simutanSQRB(couplers:list, total_circuits: int = 50, data_id_to_load:int|None = None, time_mark:bool=False):
    start = time()
    if total_circuits < 2:
        total_circuits = 2

    tot_depth = (0,10,30,60,90,120,170,250,350,550)
    depth_sum = sum(list(tot_depth))
    max_randomness = math.floor(16000/(depth_sum*3.5)) # 3.5 is an estimated gate number per Clifford when 'reduce_to_1q_cliffords' 
    
    if max_randomness<2:
        maximum_depth_sum = 16000//(2*3.5)
        raise ValueError(f"random circuits per length must >=2, please reduce your depth by {math.ceil(depth_sum-maximum_depth_sum)}.")
   
    BATCH_SIZE = min(max_randomness, total_circuits)
    num_batches = total_circuits // BATCH_SIZE 
    
    class Parameters(NodeParameters):
        qubit_pairs: Optional[List[str]] = couplers
        circuit_lengths: tuple[int] = tot_depth
        num_circuits_per_length: int = BATCH_SIZE
        num_averages: int = 300
        basis_gates: list[str] = ['rz', 'sx', 'x', 'cz']
        flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
        reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
        use_input_stream: bool = False
        simulate: bool = False
        simulation_duration_ns: int = 10000
        load_data_id: Optional[int] = data_id_to_load
        timeout: int = 600
        seed: int|None = None
        targets_name = "qubit_pairs"

    node = QualibrationNode(name="70cxx_Simutaneous_1QRB_Batched", parameters=Parameters())

    
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
                reduce_to_1q_cliffords=True,
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

            # --- 執行 QUA Program ---
            batch_results = {}
            for RB_type, RB_circuit in {"SQRB": circuits_as_ints_SRB}.items():
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
                    
                    # 修改 sequence 座標避免合併時衝突
                    ds = ds.assign_coords(sequence=[i + b * BATCH_SIZE for i in range(BATCH_SIZE)])
                    batch_results[RB_type] = ds
                    sleep(2)

            srb_ds_list.append(batch_results["SQRB"])
    
        print("\n>>> 所有批次執行完畢，正在合併數據...")
        final_avg_layers = np.mean(all_layers_per_clifford)
        node.results["final_avg_layers"] = final_avg_layers
        node.results["total_circuits"] = total_circuits
        node.results["SQRB_ds"] = xr.concat(srb_ds_list, dim="sequence")
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
        ds = a_node.results[f"SQRB_ds"]
        elapse_time_s = a_node.results["elapse_time_s"]
        node.machine = QuAM.load()
        node.results = {}
        node.results[f"SQRB_ds"] = ds

    
    node.parameters.num_circuits_per_length = total_circuits
    

    srb_result = {}
    
    fidelity = {}

    for RB_type in ["SQRB"]:
        ds = node.results[f"{RB_type}_ds"]
        
        ds_transposed = ds.rename({"shots": "average", "sequence": "repeat", "depths": "circuit_depth"})
        ds_transposed = ds_transposed.transpose("qubit", "repeat", "circuit_depth", "average")

        for qp in qubit_pairs:
            
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
            
            fig = srb_result[qp.id].plot_with_fidelity(conjugated_SQ_RB=True)
            fidelity[qp.id]=srb_result[qp.id].fidelity
            ax = fig.axes[0]
            
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if time_mark:
                fig.suptitle(f"Simultaneous 1Q RB - {qp.qubit_control.name}&{qp.qubit_target.name}\n elapsed time: {elapse_time_s}s")
            else:
                fig.suptitle(f"Simultaneous 1Q RB - {qp.qubit_control.name}&{qp.qubit_target.name}")
            fig.show()
            node.results[f"{qp.id}_figure_Simultaneous1QRB"] = fig


    if not node.parameters.simulate:
        node.save()
    
    return list(fidelity.values()), list(fidelity.keys())



if __name__ == '__main__':
    # 100 random circuits takes > 1 hrs
    couplers = ['coupler_q4_q5']
    random_gates_per_depth:int = 100
    job_id:int|None = None
    fidelities, names = run_batched_simutanSQRB(couplers, random_gates_per_depth, job_id, time_mark=True)