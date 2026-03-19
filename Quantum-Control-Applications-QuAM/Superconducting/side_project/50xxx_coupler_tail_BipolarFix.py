"""
        RESONATOR SPECTROSCOPY VERSUS FLUX
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to
extract the 'I' and 'Q' quadratures. This is done across various readout intermediate dfs and flux biases.
The resonator frequency as a function of flux bias is then extracted and fitted so that the parameters can be stored in the state.

This information can then be used to adjust the readout frequency for the maximum and minimum frequency points.

Prerequisites:
    - Calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibration of the IQ mixer connected to the readout line (be it an external mixer or an Octave port).
    - Identification of the resonator's resonance frequency (referred to as "resonator_spectroscopy").
    - Configuration of the readout pulse amplitude and duration.
    - Specification of the expected resonator depletion time in the state.

Before proceeding to the next node:
    - Update the relevant flux biases in the state.
    - Save the current state
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset_simple, readout_state_coupler
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.save_utils import fetch_results_as_xarray
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import numpy as np
from dataclasses import asdict
from typing import Dict, List
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xarray as xr
from typing import List, Tuple, Optional, Any
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

from quam_libs.lib.pulses import FreeCosineBipolarPulse
# %% {Node_parameters}
class Parameters(NodeParameters):

    coupler: str = "coupler_q4_q5"
    """List of qubit pair names to be measured. If None or empty, all active qubit pairs are measured."""
    num_averages: int = 500
    """Number of averages for the measurement."""
    frequency_span_in_mhz: float = 60
    """Frequency span around the coupler RF frequency for the scan."""
    frequency_step_in_mhz: float = 2
    """Frequency step size for the scan."""
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    """Whether to set flux point jointly for all qubits or independently."""
    duration_in_ns: Optional[int] = 216
    """Total duration of the flux pulse in ns."""
    time_axis: Literal["linear", "log"] = "linear"
    """Type of time axis for the flux pulse duration sweep."""
    time_step_num: Optional[int] = 50
    """Number of time steps for logarithmic time axis."""
    min_wait_time_in_ns: Optional[int] = 16
    """Minimum wait time in ns for the flux pulse duration sweep."""
    coupler_flux: float = 0.1
    flux_druation_ns:int = 100
    """Coupler flux amplitude and duration set  """
    pi_pulse_duration_scale: int = 4
    """Duration of the control qubit pulse in ns."""
    bipolar_pole_ratio_pts:int = 12
    """Default starts from 0.5 to 1.0 for the length ratio about positive pole in bipolar waveform"""
    simulate: bool = False
    """Whether to simulate the QUA program instead of executing it."""
    simulation_duration_ns: int = 2500
    """Duration of the simulation in ns."""
    timeout: int = 100
    """Timeout for the QOP session in seconds."""
    load_data_id: Optional[int] = None
    """If provided, load data from the specified node ID instead of executing the program."""

    fitting_base_fractions: List[float] = [0.4, 0.15, 0.05]
    """Fitting coefs and can be editted later"""



node = QualibrationNode(name="50xxx_coupler_tail_BipolarFix", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

# Get the relevant QuAM components
coupler = [machine.qubit_pairs[node.parameters.coupler]] # currently supports 1 coupler a time only.
drive_q = [machine.qubits[coupler[0].extras["RD"]["driven_q"]]]
detector_q = [machine.qubits[coupler[0].extras["RD"]["readout_q"]]]

# Change driving LO
if not node.parameters.simulate and node.parameters.load_data_id is None:
    drive_LO_original = {drive_q[0].name: drive_q[0].xy.opx_output.upconverter_frequency}
    drive_q[0].xy.opx_output.upconverter_frequency = coupler[0].extras["RD"]["LO"]
    if "swap_direction" in coupler[0].extras["RD"]:
        detector_q[0].z.operations['aSWAP'].slope_direction = coupler[0].extras["RD"]["swap_direction"]

# add in Free_bipolar pulse into coupler.operation
for c in coupler:
    if not hasattr(c.coupler.operations, "free_bipolar"):
        c.coupler.operations['free_bipolar'] = FreeCosineBipolarPulse(length=100, amplitude=0.1, flat_length_ratio=0.9, neg_amp_scal=1.0, pos_len_ratio=0.5)


# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()

# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

# The frequency sweep around the resonator resonance frequency
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span / 2, +span / 2, step)
ratios = np.linspace(1.0, 0.5, node.parameters.bipolar_pole_ratio_pts)

# Flux bias sweep
if node.parameters.time_axis == "linear":
    times = np.linspace(
        node.parameters.min_wait_time_in_ns // 4,
        node.parameters.duration_in_ns // 4,
        node.parameters.time_step_num,
        dtype=np.int32,
    )
elif node.parameters.time_axis == "log":
    times = np.logspace(
        np.log10(node.parameters.min_wait_time_in_ns // 4),
        np.log10(node.parameters.duration_in_ns // 4),
        node.parameters.time_step_num,
        dtype=np.int32,
    )
    # Remove repetitions from times
times = np.unique(times)

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'


with program() as multi_res_spec_vs_flux:
    # Declare 'I' and 'Q' and the corresponding streams for the two resonators.
    # For instance, here 'I' is a python list containing two QUA fixed variables.
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=len(detector_q))
    state_target = [declare(int) for _ in range(len(detector_q))]
    state_stream_target = [declare_stream() for _ in range(len(detector_q))]
    df = declare(int)  # QUA variable for the readout frequency
    t_delay = declare(int)  # QUA variable for delay time scan
    duration = node.parameters.duration_in_ns * u.ns
    qp = coupler[0]
    
    if not node.parameters.simulate:
        machine.apply_all_couplers_to_min()

    align()
    for i, qubit in enumerate(drive_q):
        if not node.parameters.simulate:
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)
            if "c" in qubit.id: qubit.z.set_dc_offset(qubit.z.joint_offset) # for coupler-test case
            qubit.z.settle()
        new_du = node.parameters.pi_pulse_duration_scale*qubit.xy.operations['x180_cp'].length//4
        align()
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(df, dfs)):  # type: ignore
                with for_each_(t_delay, times):
                    
                    # update the frequency of the control qubit
                    qubit.xy.update_frequency(df + coupler[0].extras["RD"]["IF"])
                    
                    if not node.parameters.simulate:
                        if qubit.thermalization_time//5 > coupler[0].extras['T1']*1e9:
                            wait(qubit.thermalization_time * u.ns)
                        else:
                            wait(5*coupler[0].extras['T1']*1e9 * u.ns)
                    align()
                    
                    qp.coupler.play(
                        "free_bipolar",
                        amplitude_scale=node.parameters.coupler_flux / qp.coupler.operations["free_bipolar"].amplitude,
                        duration=node.parameters.flux_druation_ns//4,
                    )
                    align()
                    wait(t_delay)
                    align()
                    qubit.xy.play(
                        'x180_cp',
                        amplitude_scale=1/node.parameters.pi_pulse_duration_scale,
                        duration=new_du,
                    )
                    
                    readout_state_coupler(detector_q[0], state_target[i], method='aswap')
                    save(state_target[i], state_stream_target[i])
        

    with stream_processing():
        n_st.save("n")
        for i in range(len(drive_q)):
            state_stream_target[i].buffer(len(times)).buffer(len(dfs)).average().save(f"state_target{i + 1}")
            
        


# %% {Simulate_or_execute}
import time
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns // 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, multi_res_spec_vs_flux, simulation_config)
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

else:
    if node.parameters.load_data_id is None:
        dss = {}
        start = time.time()
        for bipolar_len_ratio in ratios:
            c.coupler.operations['free_bipolar'].pos_len_ratio = bipolar_len_ratio
            # Generate the OPX and Octave configurations
            config = machine.generate_config()
            with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
                job = qm.execute(multi_res_spec_vs_flux)
                results = fetching_tool(job, ["n"], mode="live")
                while results.is_processing():
                    # Fetch results
                    n = results.fetch_all()[0]
                    # Progress bar
                    # progress_counter(n, n_avg, start_time=results.start_time)
                    progress_counter(n, n_avg, start_time=results.start_time)

            # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
            ds = fetch_results_as_xarray(job.result_handles, coupler, {"time": times * 4, "detuning": dfs})
            dss[str(bipolar_len_ratio)] = ds
            
            # for _ in range(5):
            #     print(".", end='')
            #     time.sleep(1)
        end = time.time()
        print(f"Take: {round(end-start,1)} s")
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]

        

# %%
if  node.parameters.load_data_id is None and not node.parameters.simulate:
    import xarray as xr
    ratios = [float(k) for k in dss.keys()]
    datasets = list(dss.values())

    # 2. 沿著新維度 'bipoler_ratio' 合併
    combined_ds = xr.concat(datasets, dim='bipolar_ratio')
    combined_ds = combined_ds.assign_coords(bipolar_ratio=ratios)

    # 3. 排序座標確保繪圖正確
    ds = combined_ds.sortby('bipolar_ratio')

    RF_freq = np.array([dfs + c.extras["RD"]["LO"] + c.extras["RD"]["LO"] for c in coupler])
    ds = ds.assign_coords({"freq_full_control": (["qubit", "freq"], RF_freq)})
    detuned_freq = np.array([dfs for c in coupler]) * 1e-6
    ds = ds.assign_coords({"detunings": (["qubit", "freq"], detuned_freq)})
    ds.freq_full_control.attrs["long_name"] = "Frequency"
    ds.freq_full_control.attrs["units"] = "GHz"
    ds.detunings.attrs["long_name"] = "Detuning"
    ds.detunings.attrs["units"] = "MHz"




    # %% {Plot}
    


    # --- 1. 定義【左側強化版】Peak 偵測函式 ---
    def peaks_dips_adaptive_left(data_array, dim, sigma=(1.0, 0.8)):
        """
        針對 Flux Pulse 前段彎曲特別優化的偵測邏輯。
        """
        # 對數據進行初步平滑，sigma[0] 是時間軸，sigma[1] 是 detuning 軸
        smoothed_values = gaussian_filter(data_array.values, sigma=sigma)
        
        # 獲取時間軸索引的長度
        time_len = data_array.shape[data_array.get_axis_num('time')]
        peak_indices = []

        for t_idx in range(time_len):
            # 提取該時間點的 detuning 切片 (y-axis)
            # 根據維度順序動態提取
            if data_array.dims[0] == 'detuning':
                y_slice = smoothed_values[:, t_idx]
            else:
                y_slice = smoothed_values[t_idx, :]
                
            std_val = np.std(y_slice)
            
            # --- 核心邏輯：前段強制降低門檻 ---
            # 如果是前 15 個點 (時間早期)，門檻設得很低 (1.5x std) 以強制抓取彎曲
            current_prom = 1.5 if t_idx < 15 else 4.0
            
            peaks, props = find_peaks(y_slice, prominence=std_val * current_prom)
            
            # 如果還是抓不到，用最底線門檻再試一次
            if len(peaks) == 0:
                peaks, props = find_peaks(y_slice, prominence=std_val * 0.5)
                
            if len(peaks) > 0:
                # 選擇顯著性最高的 Peak
                peak_indices.append(peaks[np.argmax(props['prominences'])])
            else:
                peak_indices.append(np.nan)
        
        peak_indices = np.array(peak_indices)
        
        # 轉回實際座標值 (Detuning Hz)
        coords_values = data_array[dim].values
        def idx_to_val(idx):
            return coords_values[int(idx)] if np.isfinite(idx) else np.nan
        
        peak_positions = np.vectorize(idx_to_val)(peak_indices)
        
        # 回傳 Dataset
        return xr.Dataset(
            {"position": (("time"), peak_positions)},
            coords={"time": data_array.time}
        )

    # --- 2. 迭代擬合主程式 ---
    def exp_decay(t, a, tau, b):
        return a * np.exp(-t / tau) + b

    ratios = ds.bipolar_ratio.values
    qubit_name = coupler[0].name
    results = []

    for i, ratio in enumerate(ratios):
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        sub_ds = ds.sel(bipolar_ratio=ratio, qubit=qubit_name)
        try:
            # 執行強化版偵測
            peaks = peaks_dips_adaptive_left(sub_ds.state_target, dim="detuning")
            
            t_raw = peaks.time.values
            f_raw = peaks.position.values / 1e6 # MHz
            
            # 僅移除 NaN，保留所有有效點（包含前段）
            mask = np.isfinite(f_raw)
            t_fit = t_raw[mask]
            f_fit = f_raw[mask]
            
            if len(f_fit) < 10: raise ValueError("Points too few.")

            # --- 初始值優化 ---
            # 強制計算前段與末端之差作為振幅猜測
            amp_guess = f_fit[0] - f_fit[-1]
            p0 = [amp_guess, 50, f_fit[-1]]
            
            popt, _ = curve_fit(exp_decay, t_fit, f_fit, p0=p0, maxfev=5000)
            
            # 繪圖
            sub_ds.state_target.plot(ax=ax, x="time", y="detuning", cmap="viridis", add_colorbar=True, cbar_kwargs={'label': 'State'})
            ax.plot(t_fit, f_fit * 1e6, "w.", alpha=0.5, markersize=3) 
            
            t_plot = np.linspace(t_fit.min(), t_fit.max(), 100)
            ax.plot(t_plot, exp_decay(t_plot, *popt) * 1e6, "r-", label=f"Tau: {popt[1]:.1f} ns")
            ax.set_title(f"Ratio: {round(ratio,2)} (Left-Enhanced)")
            ax.legend()
            ax.set_xlabel("Time [ns]")
            results.append({"ratio": ratio, "tau": popt[1]})
            
        except Exception as e:
            print(f"Ratio {ratio} failed: {e}")
            sub_ds.state_target.plot(ax=ax, x="time", y="detuning", label="State Target", cmap="viridis", cbar_kwargs={'label': 'State'})
            ax.set_xlabel("Time [ns]")

        plt.tight_layout()
        node.results[f"every_heatmap_{i+1}"] = plt.gcf()
        plt.show()

    #%%
    # --- 1. 執行分析並收集數據 ---
    ratios = ds.bipolar_ratio.values
    qubit_name = coupler[0].name
    tau_results = []
    peak_matrix = [] 
    optimal_ratio = []
    # 設定物理合理的 Tau 上限 (ns)
    TAU_UPPER_LIMIT = 800 

    print(f"Analyzing {len(ratios)} ratios...")

    for ratio in ratios:
        try:
            sub_ds = ds.sel(bipolar_ratio=ratio, qubit=qubit_name)
            peaks = peaks_dips_adaptive_left(sub_ds.state_target, dim="detuning")
            
            mask = np.isfinite(peaks.position.values)
            t_fit = peaks.time.values[mask]
            f_fit = peaks.position.values[mask] / 1e6
            
            # 擬合
            amp_guess = f_fit[0] - f_fit[-1]
            p0 = [amp_guess, 50, f_fit[-1]]
            popt, _ = curve_fit(exp_decay, t_fit, f_fit, p0=p0, maxfev=5000)
            
            tau_val = popt[1]
            # 檢查是否為水平線或不合理的 Tau
            if tau_val > TAU_UPPER_LIMIT: # 振幅太小也視為平的
                tau_results.append(0) 
                optimal_ratio.append(ratio)
            else:
                tau_results.append(tau_val)
                
            peak_matrix.append(peaks.position.values)
        except:
            tau_results.append(np.nan)
            peak_matrix.append(np.full(len(ds.time), np.nan))

    # --- 2. 建立並排畫布 ---
    fig, (ax_heat, ax_tau) = plt.subplots(1, 2, figsize=(13, 6), sharey=True, 
                                        gridspec_kw={'width_ratios': [1.5, 1]})

    # --- 左邊: Heatmap (顯示所有 Ratio 的 Peak 位置) ---
    peak_da = xr.DataArray(peak_matrix, coords={"bipolar_ratio": ratios, "time": ds.time.values}, dims=["bipolar_ratio", "time"])
    peak_da.plot(ax=ax_heat, x="time", y="bipolar_ratio", cmap="RdBu_r",add_colorbar=True, cbar_kwargs={'label': 'Detuning [Hz]'})
    ax_heat.set_title("Peak Tracking (Left-Enhanced)")
    ax_heat.set_xlabel("Time [ns]")

    # --- 右邊: Scatter Plot (Tau 趨勢) ---
    tau_results = np.array(tau_results)
    valid_mask = (tau_results > 0)
    flat_mask = (tau_results == 0)

    # 畫正常的點 (紅點連線)
    ax_tau.plot(tau_results[valid_mask], ratios[valid_mask], "o-r", label="Fitted $\\tau$")

    # 畫水平線的點 (標註在 0，使用灰色 'x')
    ax_tau.scatter(np.zeros(np.sum(flat_mask)), ratios[flat_mask], 
                marker="x", color="gray", s=100, label="Flat Line ($\\tau \\approx \infty$)")

    ax_tau.axvline(0, color='black', linestyle=':', alpha=0.3)
    ax_tau.set_xlim(-50, max(tau_results[valid_mask])*1.2 if any(valid_mask) else 100)
    ax_tau.set_xlabel("Time Constant $\\tau$ [ns]")
    ax_tau.set_title("$\\tau$ Trend")
    ax_tau.grid(True, linestyle="--", alpha=0.5)
    ax_tau.legend()

    plt.tight_layout()
    node.results["tau_plot"] = plt.gcf()
    plt.show()
    if len(optimal_ratio) > 0:
        node.results["fit_results"] = {"Optimal_ratio": optimal_ratio}

    # %% {Update_state}

    if node.parameters.load_data_id is None and not node.parameters.simulate:
        if len(optimal_ratio) == 1:
            with node.record_state_updates():
                for c in coupler:
                    c.coupler.operations['free_bipolar'].pos_len_ratio = optimal_ratio[0]
        else:
            print("Too many ratio candidates, please check raw heatmaps and update then.")     

        # %% {Save_results}
        for q in drive_q:
            q.xy.opx_output.upconverter_frequency = drive_LO_original[q.name] # revert the driving LO
        for q in detector_q:
            q.z.operations['aSWAP'].slope_direction = -1 # always at -1
        node.outcomes = {q.name: "successful" for q in drive_q}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()

# %%
