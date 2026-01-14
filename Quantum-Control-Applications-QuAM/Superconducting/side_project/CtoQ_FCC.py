# %%
"""
C to Q Flux Crosstalk Compensation (FCC) measurement.
Measure the flux crosstalk from coupler to its neighboring qubits. like coupler_q1_q2 to q1 or q2, etc.
At this moment, only measure one coupler in a single time
The measurement process inherited from 64a

Noticed that if you want to check the target qubit defined in coupler, we need to bias it to make its transition frequeny get lower than the other. This bias voltage is named 'target_q_bias' in node.parameters and we need to manually search it at this moment.

* Once the flux crosstalk is fit, it will be saved into a dict in this coupler's extra with a name FCC and its key will be the target qubit.
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import active_reset, readout_state_gef, readout_state, active_reset_simple
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import gaussian_filter
from sklearn.linear_model import RANSACRegressor
def fit_crosstalk_slope(ds, qubit_label, y_limit=None, state:str='control', sigma=2.0, plot_result=True):
    """
    計算 Qubit Flux 對 Coupler Flux 的 Crosstalk 斜率。
    
    參數:
    - ds: 包含實驗數據的 xarray Dataset
    - qubit_label: 例如 'q1'
    - y_limit: 設定 Coupler Flux 的上限 (mV)，用來避開圖上方的交叉干擾區 (預設 50mV)
    - plot_result: 是否畫出擬合結果以供驗證
    
    回傳:
    - slope (dx/dy): 每 1 unit Coupler Flux 變化造成的 Qubit Flux 偏移量
    - intercept: 截距
    """
    
    # 1. 提取數據 (依照你原本的邏輯)
    # 注意：這裡假設你的 ds 結構與你的程式碼一致
    if state.lower() != 'control': 
        data = ds.state_target.sel(qubit=qubit_label)
    else:
        data = ds.state_control.sel(qubit=qubit_label)
    
    # 轉換座標單位 (依照你的繪圖邏輯)
    flux_qubit_mV = data.flux_qubit_full.values * 1e3
    flux_coupler_mV = data.flux_coupler_full.values * 1e3
    signal_values = data.values # 形狀通常是 (coupler, qubit)

    # --- 步驟 A: 自動判斷是要找最小值還是最大值 ---
    background_level = np.median(signal_values)
    max_val = np.max(signal_values)
    min_val = np.min(signal_values)
    
    # 比較特徵強度：是 "向下凹" 比較深，還是 "向上凸" 比較高？
    # dist_to_min: 背景到谷底的距離
    # dist_to_max: 背景到山頂的距離
    is_finding_min = (background_level - min_val) > (max_val - background_level)
    
    detect_mode = "Minima (Dark Line)" if is_finding_min else "Maxima (Bright Line)"
    print(f"[{qubit_label}] Auto-detection: Target feature is {detect_mode}")

    # --- 步驟 B: 高斯模糊前處理 ---
    smoothed_signal = gaussian_filter(signal_values, sigma=[0, sigma])

    valid_x = []
    valid_y = []

    # 2. 尋找特徵線
    for i, y_val in enumerate(flux_coupler_mV):
        if y_limit is not None:
            if y_val > y_limit:
                continue
            
        row_data = smoothed_signal[i, :]
        
        # --- 根據自動判斷的結果選擇方法 ---
        if is_finding_min:
            target_idx = np.argmin(row_data) # 找最暗
        else:
            target_idx = np.argmax(row_data) # 找最亮
        
        valid_x.append(flux_qubit_mV[target_idx])
        valid_y.append(y_val)

    # 轉為 numpy array，並 reshape 成 sklearn 需要的格式 (N, 1)
    X = np.array(valid_y).reshape(-1, 1) # Y軸當自變數 (Coupler)
    y = np.array(valid_x)                # X軸當應變數 (Qubit)

    # --- 改良點 2: 使用 RANSAC 取代一般線性回歸 ---
    # RANSAC 會自動剔除偏離太遠的點 (Outliers)
    ransac = RANSACRegressor(random_state=42, residual_threshold=2.0)
    ransac.fit(X, y)
    
    slope = ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_
    
    # 區分哪些點被認定為 Outliers (畫圖用)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    print(f"--- Robust Fitting Result for {qubit_label} ---")
    print(f"Slope (dx/dy): {slope:.5f}")

    # 3. 畫圖驗證
    fig = None
    if plot_result:
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # 改用 ax.pcolormesh
        ax.pcolormesh(flux_qubit_mV, flux_coupler_mV, signal_values, cmap='viridis', shading='auto')
        
        # 改用 ax.scatter
        ax.scatter(y[inlier_mask], X[inlier_mask].ravel(), color='lime', s=3, label='Inliers')
        ax.scatter(y[outlier_mask], X[outlier_mask].ravel(), color='red', marker='x', s=10, alpha=0.5, label='Outliers')
        
        # 畫擬合線
        line_y = np.linspace(min(flux_coupler_mV), max(flux_coupler_mV), 100).reshape(-1, 1)
        line_x = ransac.predict(line_y)
        ax.plot(line_x, line_y, 'r--', lw=2, label=f'Fit: slope={slope:.4f}')
        
        # 設定標籤
        ax.set_title(f"Flux source: {qubit_label}")
        ax.set_xlabel(f"{ds.attrs['target_q']} Flux [mV]")
        ax.set_xlim(min(flux_qubit_mV), max(flux_qubit_mV))
        ax.set_ylabel("Coupler Flux [mV]")
        ax.set_ylim(min(flux_coupler_mV), max(flux_coupler_mV))
        ax.legend()

    return slope, intercept, fig


# %% {Node_parameters}
class Parameters(NodeParameters):
    
    z_source_c:List[str] = ['coupler_q1_q2']
    target_q:str = 'q1'
    control_flux_min:float = -0.15
    control_flux_max:float = 0
    qubit_flux_step : float = 0.0015
    
    source_flux_max:float = 0
    source_flux_min:float = -0.6
    
    exam_FCC:bool = True
    num_averages: int = 200
    flux_point_joint_or_independent_or_pairwise: Literal["joint", "independent", "pairwise"] = "independent"
    reset_type: Literal['active', 'thermal'] = "active"
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None
    
    
    pulse_duration_ns: int = 100
    target_q_bias:float = 0.2      # if target_q freq is higher than control_q. Applied when coupler's control_q is the assigned target_q
    

node = QualibrationNode(
    name="SP_FCC", parameters=Parameters()
)
assert not (node.parameters.simulate and node.parameters.load_data_id is not None), "If simulate is True, load_data_id must be None, and vice versa."

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()





z_sources = [machine.qubit_pairs[c] for c in node.parameters.z_source_c]
z_source_flux_pt = 'joint'

if  node.parameters.target_q == z_sources[0].qubit_control.name:
    q_ctrl = z_sources[0].qubit_control
    q_target = z_sources[0].qubit_target
    q_bias_apply = False 
elif  node.parameters.target_q == z_sources[0].qubit_target.name:
    q_ctrl = z_sources[0].qubit_target
    q_target = z_sources[0].qubit_control
    q_bias_apply = True
# if you want to check the unconnected qubits
else:
    for c in machine.active_qubit_pairs:
        if node.parameters.target_q == c.qubit_control:
            q_ctrl = c.qubit_control
            q_target = c.qubit_target
            q_bias_apply = False 
            break # directly break this for loop
        else:
            q_ctrl = c.qubit_target
            q_target = c.qubit_control
            q_bias_apply = True

bias_wait_time = 800 // 4
print(f"Ctrl: {q_ctrl.name}, Target: {q_target.name}, CT-flip: {q_bias_apply}")

elements_to_reset = [q_ctrl, q_target]

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


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
target_q_bias = node.parameters.target_q_bias
flux_point = node.parameters.flux_point_joint_or_independent_or_pairwise  # 'independent' or 'joint' or 'pairwise'
# Loop parameters
fluxes_source = np.linspace(node.parameters.source_flux_min, node.parameters.source_flux_max+0.0001, 100)
fluxes_qubit = np.arange(node.parameters.control_flux_min, node.parameters.control_flux_max+0.0001, node.parameters.qubit_flux_step)

    
pulse_duration = node.parameters.pulse_duration_ns // 4
reset_coupler_bias = False

with program() as CPhase_Oscillations:
    n = declare(int)
    flux_source = declare(float)
    flux_qubit = declare(float)
    comp_flux_qubit = declare(float)
    n_st = declare_stream()
    qua_pulse_duration = declare(int, value = pulse_duration)
    
        
    
    state_target = [declare(int) for _ in range(len(z_sources))]
    state_control = [declare(int) for _ in range(len(z_sources))]
    state_st_target = [declare_stream() for _ in range(len(z_sources))]
    state_st_control = [declare_stream() for _ in range(len(z_sources))]

    for i, z_source in enumerate(z_sources):

        if not node.parameters.simulate:
            machine.apply_all_couplers_to_min()
            for q in elements_to_reset:
                machine.set_all_fluxes(flux_point, q)
            machine.set_all_fluxes(z_source_flux_pt, z_source)
            wait(1000)

    
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)         
            with for_(*from_array(flux_source, fluxes_source)):
                with for_(*from_array(flux_qubit, fluxes_qubit)):

                    if node.parameters.exam_FCC and "FCC" in z_source.extras:
                        try:
                            assign(comp_flux_qubit, flux_qubit + z_source.extras["FCC"][q_ctrl.name] * flux_source )
                        except:
                            print(f"Control qubit {q_ctrl} is not in {z_source}'s FCC, use 0 compensation instead.")
                            assign(comp_flux_qubit, flux_qubit) 
                    else:
                        assign(comp_flux_qubit, flux_qubit) 


                    # reset
                    for j, qubit in enumerate(elements_to_reset):
                        if node.parameters.reset_type == "active":
                            # active_reset(qubit, "readout")
                            active_reset_simple(qubit, "readout")
                        else:
                            if not node.parameters.simulate:
                                qubit.wait(qubit.thermalization_time * u.ns)
                            else:
                                qubit.wait(16 * u.ns)
                    
                    align()
                    # setting both qubits ot the initial state
                    q_ctrl.xy.play("x180")
                    q_target.xy.play("x180")
                    align()
                    if q_bias_apply:
                        q_target.z.play("const", amplitude_scale = target_q_bias / q_target.z.operations["const"].amplitude, duration = qua_pulse_duration+bias_wait_time)
                        q_ctrl.z.wait(bias_wait_time)
                        z_source.coupler.wait(bias_wait_time)
                    q_ctrl.z.play("const", amplitude_scale = comp_flux_qubit / q_ctrl.z.operations["const"].amplitude, duration = qua_pulse_duration)                
                    z_source.coupler.play("const", amplitude_scale = flux_source / z_source.coupler.operations["const"].amplitude, duration = qua_pulse_duration)
                    align()
                    wait(20)
                    # readout
                    readout_state_gef(q_ctrl, state_control[i])
                    wait(4)
                    z_source.align()
                    wait(4)
                    readout_state(q_target, state_target[i])
                    align()
                    save(state_control[i], state_st_control[i])
                    save(state_target[i], state_st_target[i])
                        
        align()
        
    with stream_processing():
        n_st.save("n")
        for i, z_source in enumerate(z_sources):
            state_st_control[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_source)).average().save(f"state_control{i + 1}")
            state_st_target[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_source)).average().save(f"state_target{i + 1}")
            
# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, CPhase_Oscillations, simulation_config)
    samples = job.get_simulated_samples()
    samples.con1.plot()
    node.results = {"figure": plt.gcf()}
    wf_report = job.get_simulated_waveform_report()
    wf_report.create_plot(samples, plot=True, save_path=None)
    node.machine = machine
    node.save()
elif node.parameters.load_data_id is None:
    
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(CPhase_Oscillations)

        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}

if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, z_sources, {  "flux_qubit": fluxes_qubit, "flux_coupler": fluxes_source})
        flux_qubit_full = np.array([fluxes_qubit for qp in z_sources])
        ds = ds.assign_coords({"flux_qubit_full": (["qubit", "flux_qubit"], flux_qubit_full)})
    else:
        ds, machine = load_dataset(node.parameters.load_data_id)
        
    node.results = {"ds": ds}
# %%
if not node.parameters.simulate:
    if reset_coupler_bias:
        flux_coupler_full = np.array([fluxes_source + qp.coupler.decouple_offset for qp in z_sources])
    else:
        flux_coupler_full = np.array([fluxes_source for qp in z_sources])
    ds = ds.assign_coords({"flux_coupler_full": (["qubit", "flux_coupler"], flux_coupler_full)})
    ds.attrs["target_q"] = node.parameters.target_q
    node.results = {"ds": ds}
  
# %%
node.results["results"] = {}

## HARD CODED FROM EXPERIMENT
    
    
# %% {Plotting}
plot_state:Literal['control', 'target'] = 'target'
if not node.parameters.simulate:
    for qp in z_sources:
        node.results["results"][qp.name] = {}
        slop, _, fig = fit_crosstalk_slope(ds, qp.name,state=plot_state, sigma=3.0, plot_result=True)
        node.results["results"][qp.name][node.parameters.target_q] = slop
        if fig is not None:
            node.results[f'figure_{plot_state}'] = fig


# %% {Update_state}
if not node.parameters.simulate:
    if not node.parameters.simulate:
        if not node.parameters.exam_FCC:
            with node.record_state_updates():
                for qp in z_sources:
                    if "FCC" not in qp.extras:
                        qp.extras["FCC"] = {}
                    qp.extras["FCC"][node.parameters.target_q] = node.results["results"][qp.name][node.parameters.target_q]
                    
# %% {Save_results}
if not node.parameters.simulate:    
    node.outcomes = {q.name: "successful" for q in z_sources}
    node.results['initial_parameters'] = node.parameters.model_dump()
    node.machine = machine
    node.save()
# %%
