# %%
"""
Two-Qubit Readout Confusion Matrix Measurement (9-State GEF Basis)

1. Preparing qubits in 9 combinations (|00⟩, |01⟩, |02⟩, |10⟩, |11⟩, |12⟩, |20⟩, |21⟩, |22⟩)
2. Performing simultaneous readout_state_gef on both qubits
3. Outputting a 9x9 confusion matrix into coupler's extras "confusion_9x9".
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import active_reset_gef, readout_state_gef, active_reset, readout_state
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from quam_libs.lib.save_utils import (
    fetch_results_as_xarray,
    restore_load_data_id,
    resolve_qubit_pairs_from_node,
)
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np

# %% {Node_parameters}
class Parameters(NodeParameters):
    qubit_pairs: Optional[List[str]] = ["coupler_q4_q5"]
    num_shots: int = 10000
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type: Literal['active', 'thermal'] = "active"
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None
    plot_raw : bool = False


node = QualibrationNode(
    name="34_2Q_confusion_matrix_gef", parameters=Parameters()
)
assert not (node.parameters.simulate and node.parameters.load_data_id is not None), "If simulate is True, load_data_id must be None, and vice versa."

# %% {Initialize_QuAM_and_QOP}
u = unit(coerce_to_integer=True)
machine = QuAM.load()
node.machine = machine

if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs = machine.active_qubit_pairs
else:
    qubit_pairs = [machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]

num_qubit_pairs = len(qubit_pairs)

config = machine.generate_config()
octave_config = machine.get_octave_config()
if node.parameters.load_data_id is None:
    qmm = machine.connect()

# %% {QUA_program}
n_shots = node.parameters.num_shots
flux_point = node.parameters.flux_point_joint_or_independent

with program() as Confusion_Matrix_GEF:
    control_initial = declare(int)
    target_initial = declare(int)
    n = declare(int)
    n_st = declare_stream()
    state_control = [declare(int) for _ in range(num_qubit_pairs)]
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    state = [declare(int) for _ in range(num_qubit_pairs)]
    state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    state_st = [declare_stream() for _ in range(num_qubit_pairs)]
    
    for i, qp in enumerate(qubit_pairs):
        if not node.parameters.simulate:
            if flux_point == "independent":
                machine.apply_all_flux_to_min()
            elif flux_point == "joint":
                machine.apply_all_flux_to_joint_idle()
            else:
                machine.apply_all_flux_to_zero()
            wait(1000)

        with for_(n, 0, n < n_shots, n + 1):
            save(n, n_st)         
            # Loop 控制狀態: 0(|g>), 1(|e>), 2(|f>)
            with for_(*from_array(control_initial, [0, 1, 2])):
                with for_(*from_array(target_initial, [0, 1, 2])):
                    qp.align()
                    update_frequency(qp.qubit_control.xy.name, qp.qubit_control.xy.intermediate_frequency)
                    wait(4)
                    update_frequency(qp.qubit_target.xy.name, qp.qubit_target.xy.intermediate_frequency)
                    wait(4)

                    qp.align()

                    # Reset
                    if not node.parameters.simulate:
                        if node.parameters.reset_type == "active":
                            active_reset_gef(qp.qubit_control)
                            active_reset_gef(qp.qubit_target)                 
                        else:
                            wait(1 * qp.qubit_control.thermalization_time * u.ns)

                    qp.align()
                    
                    # 準備 Control Qubit 狀態
                    with if_(control_initial == 1):
                        qp.qubit_control.xy.play("x180")
                    with if_(control_initial == 2):
                        qp.qubit_control.xy.play("x180")
                        wait(4)
                        update_frequency(
                            qp.qubit_control.xy.name, 
                            int(qp.qubit_control.xy.intermediate_frequency - qp.qubit_control.anharmonicity)
                        )
                        wait(4)
                        # 3. 播 e -> f 脈衝
                        qp.qubit_control.xy.play("EF_x180")
                       
                        
                        
                        

                    # 準備 Target Qubit 狀態
                    with if_(target_initial == 1):
                        qp.qubit_target.xy.play("x180")
                    with if_(target_initial == 2):
                        qp.qubit_target.xy.play("x180")
                        wait(4)
                        update_frequency(
                            qp.qubit_target.xy.name, 
                            int(qp.qubit_target.xy.intermediate_frequency - qp.qubit_target.anharmonicity)
                        )
                        wait(4)
                        
                        # 3. 播 e -> f 脈衝
                        qp.qubit_target.xy.play("EF_x180")
                        
                        
                        # 4. 重要：切回原本 g -> e 頻率，避免影響下一輪實驗
                        
                        
                    qp.align()
                    
                    # GEF 3-state Readout (讀取輸出 0, 1, 2)
                    readout_state_gef(qp.qubit_control, state_control[i])
                    readout_state_gef(qp.qubit_target, state_target[i])

                    # 計算 9-State 基態索引 (公式: 3 * control + target)
                    assign(state[i], state_control[i] * 3 + state_target[i])

                    save(state_control[i], state_st_control[i])
                    save(state_target[i], state_st_target[i])
                    save(state[i], state_st[i])
        align()
        
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            # 準備了 3x3 種組合，因此內層 buffer 皆為 3
            state_st_control[i].buffer(3).buffer(3).buffer(n_shots).save(f"state_control{i + 1}")
            state_st_target[i].buffer(3).buffer(3).buffer(n_shots).save(f"state_target{i + 1}")
            state_st[i].buffer(3).buffer(3).buffer(n_shots).save(f"state{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    simulation_config = SimulationConfig(duration=10_000)
    job = qmm.simulate(config, Confusion_Matrix_GEF, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.save()
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(Confusion_Matrix_GEF)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            progress_counter(n, n_shots, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        ds = fetch_results_as_xarray(
            job.result_handles, 
            qubit_pairs, 
            {
                "init_state_target": [0, 1, 2], 
                "init_state_control": [0, 1, 2], 
                "N": np.linspace(1, n_shots, n_shots)
            }
        )
    else:
        load_data_id = node.parameters.load_data_id
        node = node.load_from_id(load_data_id)
        ds = node.results["ds"]
        restore_load_data_id(node, load_data_id)
        machine = node.machine
        qubit_pairs = resolve_qubit_pairs_from_node(machine, node)
    node.results = {"ds": ds}

# %% {Confusion_Matrix_Calculation}
if not node.parameters.simulate:
    states = list(range(9))  # 0 to 8
    labels = ['00', '01', '02', '10', '11', '12', '20', '21', '22']

    confusions = {}
    for qp in qubit_pairs:
        conf = []
        for state in states:
            row = []
            for q1 in [0, 1, 2]:
                for q0 in [0, 1, 2]:
                    count = (
                        ds.sel(qubit=qp.name).state.sel(
                            init_state_target=q0, init_state_control=q1
                        ) == state
                    ).sum().values
                    row.append(count)
            conf.append(row)
        confusions[qp.name] = np.array(conf) / node.parameters.num_shots

# %% {Plotting_9x9_Matrix}
if not node.parameters.simulate:
    states = list(range(9))  # 0 to 8
    labels = ['00', '01', '02', '10', '11', '12', '20', '21', '22']

    confusions = {}
    for qp in qubit_pairs:
        # 建立 9x9 矩陣: 列(Row)為 Prepared, 行(Col)為 Measured
        conf = np.zeros((9, 9))
        
        for p_idx, (q1, q0) in enumerate([(r, c) for r in [0,1,2] for c in [0,1,2]]):
            # 提取特定 Prepared 狀態下的所有 Measured 結果
            measured_states = ds.sel(qubit=qp.name).state.sel(
                init_state_control=q1, init_state_target=q0
            ).values.flatten()
            
            # 計算各 Measured 狀態出現的次數
            for m_idx in range(9):
                conf[p_idx, m_idx] = np.sum(measured_states == m_idx) / node.parameters.num_shots
                
        confusions[qp.name] = conf

    # --- 繪圖修正 ---
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    
    for ax, qubit_pair in grid_iter(grid):
        conf = confusions[qubit_pair['qubit']]
        
        # 轉置或正確映射: origin='lower' 配合 imshow/pcolormesh
        im = ax.imshow(conf, cmap='viridis', vmin=0, vmax=1, origin='lower')
        
        for i in range(9):        # Prepared (Y-axis)
            for j in range(9):    # Measured (X-axis)
                val = conf[i, j]
                color = "k" if val > 0.5 else "w"
                ax.text(j, i, f"{100 * val:.0f}%", ha="center", va="center", color=color, fontsize=7)
                
        ax.set_ylabel('Prepared')
        ax.set_xlabel('Measured')
        ax.set_xticks(range(9))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticks(range(9))
        ax.set_yticklabels(labels)
        ax.set_title(qubit_pair['qubit'])
        
    plt.show()
    node.results["figure_confusion"] = grid.fig

# %% {Update_state}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for qp in qubit_pairs:
                # 寫入 9x9 混淆矩陣至 QuAM 狀態
                qp.extras["confusion_9x9"] = confusions[qp.name].tolist()

# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {qp.name: "successful" for qp in qubit_pairs}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.save()
# %%
