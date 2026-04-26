# %%
"""
The transition frequency flux spectrum for the target coupler.

Prerequisites:
    - the driving frequency for the target coupler.

PS. load_data is working.
"""


# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, readout_state_coupler

from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray

from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter


# %% {Node_parameters}
class Parameters(NodeParameters):
    couplers: str = 'coupler_q5_q6' #'coupler_q3_q4'
    num_averages: int = 500
    operation: str = "saturation"
    operation_amplitude_factor: Optional[float] = 0.05 #0.004, 0.02 # q6:3e-3, q7:1e-2, q8:3e-3, q9:***,
    operation_len_in_ns: Optional[int] = None
    Driving_LO_GHz: float|None = 3.9 # 3.18
    frequency_span_in_mhz: float = 300 #12, 120
    frequency_step_in_mhz: float = 3 #0.1, 1
    frequency_shift_in_mhz: float = 0 #0  
    min_flux_offset_in_v: float = 0.05 ##-0.042
    max_flux_offset_in_v: float = 0.4 #0.042
    num_flux_points: int = 75
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    qubits_detune_flux_amp: float = 0.0 # once you see the spectrum split at sweet spot due to q-c ZZ, you can try this to detune its all neighboring qubits. 0.3 is recommanded. 
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False


node = QualibrationNode(name="03x_coupler_Spectroscopy_vs_Flux", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

assert abs(node.parameters.qubits_detune_flux_amp) <  0.7 , "WARNING: You are setting a flux amplitude for detuning the qubits that is quite large. Make sure this is intentional to avoid any damage to the qubits."


# Get the relevant QuAM components
coupler = [machine.qubit_pairs[node.parameters.couplers]] # currently supports 1 coupler a time only.
drive_q = [machine.qubits[coupler[0].extras["RD"]["driven_q"]]]
detector_q = [machine.qubits[coupler[0].extras["RD"]["readout_q"]]]
# Change driving LO
if node.parameters.load_data_id is None and not node.parameters.simulate:
    drive_LO_original = {drive_q[0].name: drive_q[0].xy.opx_output.upconverter_frequency}
    if node.parameters.Driving_LO_GHz is None:
        drive_q[0].xy.opx_output.upconverter_frequency = coupler[0].extras["RD"]["LO"]
        LO_to_plot = coupler[0].extras["RD"]["LO"]
    else:
        LO_to_plot = node.parameters.Driving_LO_GHz * 1e9
        drive_q[0].xy.opx_output.upconverter_frequency = node.parameters.Driving_LO_GHz * 1e9
    if "swap_direction" in coupler[0].extras["RD"]:
        detector_q[0].z.operations['aSWAP'].slope_direction = coupler[0].extras["RD"]["swap_direction"]

    if 'strategy' not in coupler[0].extras["RD"]:
        readout_strategy = 'aswap'
    else:
        readout_strategy = coupler[0].extras["RD"]["strategy"]


# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()



# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
operation = node.parameters.operation  # The qubit operation to play
# Adjust the pulse duration and amplitude to drive the qubit into a mixed state - can be None
operation_len = node.parameters.operation_len_in_ns
if node.parameters.operation_amplitude_factor:
    # pre-factor to the value defined in the config - restricted to [-2; 2)
    operation_amp = node.parameters.operation_amplitude_factor
else:
    operation_amp = 1.0
# Qubit detuning sweep with respect to their resonance frequencies
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
shift = int(node.parameters.frequency_shift_in_mhz * u.MHz)
dfs = np.arange(-span//2, span//2, step, dtype=np.int32)
# Flux bias sweep
dcs = np.linspace(
    node.parameters.min_flux_offset_in_v,
    node.parameters.max_flux_offset_in_v,
    node.parameters.num_flux_points,
)
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

with program() as multi_qubit_spec_vs_flux:
    # Macro to declare I, Q, n and their respective streams for a given number of qubit (defined in macros.py)
    _, _, _, _, n, n_st = qua_declaration(num_qubits=len(detector_q))
    state = [declare(int) for _ in range(len(detector_q))]
    state_st = [declare_stream() for _ in range(len(detector_q))]
    df = declare(int)  # QUA variable for the qubit frequency
    dc = declare(fixed)  # QUA variable for the flux dc level

    machine.apply_all_couplers_to_min()
    for i, qubit in enumerate(drive_q):

        # Fixed qubit for debugging unknown flux-dependency: 
        fixed_qubit = machine.qubits[qubit.name]
        c = coupler[0].coupler
        
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)
        if "c" in qubit.id: qubit.z.set_dc_offset(qubit.z.joint_offset) # for coupler-test case
        qubit.z.settle()
        qubit.align()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)

            with for_(*from_array(df, dfs)):
                # Update the qubit frequency
                fixed_qubit.xy.update_frequency(df + coupler[0].extras["RD"]["IF"] + shift, keep_phase=True)
                with for_(*from_array(dc, dcs)):
                    # Flux sweeping for a qubit
                    align()
                    duration = operation_len * u.ns if operation_len is not None else qubit.xy.operations[operation].length * u.ns
                    # Bring the qubit to the desired point during the saturation pulse
                    # qubit.z.play("const", amplitude_scale=dc / qubit.z.operations["const"].amplitude, duration=duration)
                    drive_q[0].z.play("const", amplitude_scale= node.parameters.qubits_detune_flux_amp / drive_q[0].z.operations["const"].amplitude, duration=duration)
                    detector_q[0].z.play("const", amplitude_scale= node.parameters.qubits_detune_flux_amp / detector_q[0].z.operations["const"].amplitude, duration=duration)
                    c.play("const", amplitude_scale=dc / qubit.z.operations["const"].amplitude, duration=duration)
                    # qp.coupler.play("const", amplitude_scale=dc / qubit.z.operations["const"].amplitude, duration=duration)
                    # Apply saturation pulse to all qubits
                    fixed_qubit.xy.play(
                        operation,
                        amplitude_scale=operation_amp,
                        duration=duration,
                    )
                    align()
                    # QUA macro to read the state of the active resonators
                    readout_state_coupler(detector_q[i], state[i], method=readout_strategy)
                    save(state[i], state_st[i])
                    # Wait for the qubit to decay to the ground state
                    # Wait for the qubits to decay to the ground state
                    qubit.resonator.wait(machine.depletion_time * u.ns)

        # Measure sequentially
        if not node.parameters.multiplexed:
            align()

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(drive_q):
            state_st[i].buffer(len(dcs)).buffer(len(dfs)).average().save(f"state{i + 1}")



# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, multi_qubit_spec_vs_flux, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()),1,i+1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    # Save the figure
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

else:
    if node.parameters.load_data_id is None:
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(multi_qubit_spec_vs_flux)
            results = fetching_tool(job, ["n"], mode="live")
            while results.is_processing():
                # Fetch results
                n = results.fetch_all()[0]
                # Progress bar
                progress_counter(n, n_avg, start_time=results.start_time)

        ds = fetch_results_as_xarray(job.result_handles, coupler, {"flux": dcs, "freq": dfs})
        ds = ds.assign_coords(
            {
                "freq_full": (
                    ["qubit", "freq"],
                    np.array([shift + dfs + c.extras["RD"]["IF"] + LO_to_plot for c in coupler]),
                )
            }
        )
        ds.freq_full.attrs["long_name"] = "Frequency"
        ds.freq_full.attrs["units"] = "GHz"
    
        node.results = {"ds": ds}
        reload_qbs = False
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"] 
        machine = node.machine
        reload_qbs = True


    # %% {Data_analysis}

    def analyze_coupler_flux_robust(ds, qubit_name):
        da = ds.state.sel(qubit=qubit_name)
        
        # 1. 強力平滑，消除顆粒感
        smoothed = gaussian_filter(da.values, sigma=1.5)
        
        # 2. 找到全局最強信號點 (x, y)
        max_idx = np.unravel_index(np.argmax(smoothed, axis=None), smoothed.shape)
        peak_freq_idx, peak_flux_idx = max_idx
        center_flux = da.flux.values[peak_flux_idx]
        
        # 3. 建立「信號遮罩」：只取數值最高的 5% 點來進行擬合
        # 這比 peaks_dips 穩定得多，因為它只看最強的信號區域
        threshold = np.percentile(smoothed, 95)
        
        # 提取所有高於門檻的座標點 (Flux, Freq)
        mask = smoothed > threshold
        freq_indices, flux_indices = np.where(mask)
        
        fit_x = da.flux.values[flux_indices]
        fit_y = da.freq_full.values[freq_indices] # 這是原始 IF 頻率

        # 4. 限制 Flux 範圍，只取中心點左右 0.1V，徹底排除邊緣干擾
        roi_mask = (fit_x > center_flux - 0.1) & (fit_x < center_flux + 0.1)
        fit_x = fit_x[roi_mask]
        fit_y = fit_y[roi_mask]

        if len(fit_x) < 5:
            return None, center_flux

        # 5. 直接使用 numpy 進行二次方擬合 (y = ax^2 + bx + c)
        # 比 xarray 的 polyfit 在處理點集時更直觀
        p_coeffs = np.polyfit(fit_x, fit_y, 2) # 回傳 [a, b, c]
        
        return p_coeffs, center_flux
    def smooth_data(data, sigma=1):
        # 使用高斯濾波平滑數據，sigma 越大越平滑，但也越模糊
        denoised_data = median_filter(data, size=2)
        return gaussian_filter(denoised_data, sigma=sigma)

    node.results["fit_results"] = {}
    if reload_qbs:
        coupler = [machine.qubit_pairs[c_name] for c_name in ds.qubit.values]
    for q in coupler:
        q_name = q.name
        da = ds.state.sel(qubit=q_name)
        
        # --- 自適應 ROI 計算 ---
        flux_values = da.flux.values
        flux_span = np.ptp(flux_values) # 計算目前 Flux 的總掃描寬度
        adaptive_width = flux_span * 0.25 # 動態取總寬度的 25% 作為 ROI
        
        # 影像平滑以精確定位中心
        smoothed = gaussian_filter(da.values, sigma=1.5)
        max_idx = np.unravel_index(np.argmax(smoothed), smoothed.shape)
        center_flux = flux_values[max_idx[1]]
        
        # 動態切片：不受固定數值限制
        roi_mask = (da.flux >= center_flux - adaptive_width) & (da.flux <= center_flux + adaptive_width)
        roi_da = da.where(roi_mask, drop=True)
        
        # --- 加權擬合邏輯 ---
        # 提高門檻，確保只抓取黃色條紋最亮的部分
        threshold = np.percentile(roi_da.values, 95)
        f_idx, x_idx = np.where(roi_da.values > threshold)
        
        fit_x = roi_da.flux.values[x_idx]
        fit_y = roi_da.freq_full.values[f_idx]
        weights = roi_da.values[f_idx, x_idx] # 使用數值強度作為權重

        if len(fit_x) >= 6:
            # 加權二次擬合：y = ax^2 + bx + c
            p = np.polyfit(fit_x, fit_y, 2, w=weights)
            
            # 物理限制：Coupler 頂點向上，a (二次項) 必須小於 0
            if p[0] > 0: 
                # 如果算出來開口向上，代表被雜訊干擾，嘗試排除邊緣點再算一次
                p = np.polyfit(fit_x, fit_y, 2, w=weights**2)

            f_shift = -p[1] / (2 * p[0])
            d_freq = p[0] * f_shift**2 + p[1] * f_shift + p[2]
            
            node.results["fit_results"][q_name] = {
                "flux_shift": float(f_shift),
                "drive_freq": float(d_freq),
                "quad_term": float(p[0]),
                "coeff":[p[2], p[1], p[0]],
                "center_detected": float(center_flux)
            }
        else:
            print(f"❌ {q_name} Fitting Failed !")

    
    # %% {Plotting}
    grid = QubitGrid(ds, [q.grid_location for q in drive_q])
    info_to_keep = {}
    for ax, qubit in grid_iter(grid):
        qubit['qubit'] = coupler[0].name
        qubit_ds = ds.loc[qubit].state
        q_fit = node.results.get("fit_results", {}).get(coupler[0].name)
        
        # 繪圖
        qubit_ds.assign_coords(freq_GHz=qubit_ds.freq_full / 1e9).plot(
            ax=ax, 
            add_colorbar=True, 
            x="flux", 
            y="freq_GHz", 
            robust=True,
            cmap="viridis"
        )
        
        # 疊加擬合曲線與點 (維持原樣)
        if "coeff" in q_fit:
            info_to_keep["sweet_freq"] = q_fit["drive_freq"]    
            info_to_keep["quad_term"] = q_fit["quad_term"]
            info_to_keep["bias_to_sweet"] = q_fit["flux_shift"]
            info_to_keep["neighboring_qubit_detune_flux_amp"] = node.parameters.qubits_detune_flux_amp

            c = q_fit["coeff"]
            # 直接使用絕對頻率計算
            f_vals = (c[2]*ds.flux**2 + c[1]*ds.flux + c[0])
            
            # 繪圖時直接除以 1e9 對齊座標系
            (f_vals / 1e9).plot(ax=ax, ls="--", color="r")
            ax.axhline(q_fit["drive_freq"] / 1e9, color="cyan", ls=":")

        else:
            print(f"Warning: No valid fit found for {qubit['qubit']}")
        
        # 標籤與格式
        ax.set_ylabel("Freq (GHz)")
        ax.set_xlabel("Flux (V)")
        ax.set_title(f"{coupler[0].name}") # 明確標示當前 Qubit
        ax.grid(True, alpha=0.3)

    grid.fig.suptitle("coupler spectroscopy vs flux ")
    
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig


    # %% {Update_state}
    if not node.parameters.simulate:
        with node.record_state_updates():
            for c in coupler:
                if "Fx" not in c.extras:
                    c.extras["Fx"] = info_to_keep
                else:
                    for item in info_to_keep:
                        c.extras["Fx"][item] = info_to_keep[item]

        # %% {Save_results}
        if node.parameters.load_data_id is None:
            for q in drive_q:
                q.xy.opx_output.upconverter_frequency = drive_LO_original[q.name] # revert the driving LO
            for q in detector_q:
                q.z.operations['aSWAP'].slope_direction = -1
        node.results["ds"] = ds
        node.outcomes = {q.name: "successful" for q in drive_q}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()

# %%
