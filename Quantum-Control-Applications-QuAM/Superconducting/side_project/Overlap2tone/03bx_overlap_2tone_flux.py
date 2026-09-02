# %%
"""
        QUBIT SPECTROSCOPY
This sequence involves sending a saturation pulse to the qubit, placing it in a mixed state,
and then measuring the state of the resonator across various qubit drive intermediate frequencies dfs.
In order to facilitate the qubit search, the qubit pulse duration and amplitude can be changed manually in the QUA
program directly from the node parameters.

The data is post-processed to determine the qubit resonance frequency and the width of the peak.

Note that it can happen that the qubit is excited by the image sideband or LO leakage instead of the desired sideband.
This is why calibrating the qubit mixer is highly recommended.

Prerequisites:
    - Identification of the resonator's resonance frequency when coupled to the qubit in question (referred to as "resonator_spectroscopy").
    - Calibration of the IQ mixer connected to the qubit drive line (whether it's an external mixer or an Octave port).
    - Set the flux bias to the desired working point, independent, joint or arbitrary, in the state.
    - Configuration of the saturation pulse amplitude and duration to transition the qubit into a mixed state.

Before proceeding to the next node:
    - Update the qubit frequency in the state, as well as the expected x180 amplitude and IQ rotation angle.
    - Save the current state
"""


# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters

from quam_libs.components import QuAM
from quam_libs.lib.instrument_limits import instrument_limits
from quam_libs.macros import qua_declaration
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import (
    fetch_results_as_xarray,
    restore_load_data_id,
    resolve_qubits_from_node,
)
from quam_libs.lib.fit import peaks_dips
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

    qubits: Optional[List[str]] = ['q4']
    num_averages: int = 100
    operation_amplitude_factor: Optional[float] = 1.5  #0.004, 0.0004
    frequency_span_in_mhz: float = 100 #200, 4, 800
    frequency_step_in_mhz: float = 1 #0.25, 0.01
    flux_range_V: float = 0.05 
    flux_pts:int = 20
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    target_peak_width: Optional[float] = 30e6 #1e6
    arbitrary_flux_bias: Optional[float] = None
    arbitrary_qubit_frequency_in_ghz: Optional[float] = None
    simulate: bool = False
    simulation_duration_ns: int = 10_000
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False
    xy_ro_overlap: bool = True


node = QualibrationNode(name="03bx_Overlap2tone_flux", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
node.machine = machine

# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)

operation = 'saturation'

original_paras = {}
if node.parameters.xy_ro_overlap:
    for q in qubits:
        original_paras[q.name] = {"saturation_length":q.xy.operations[operation].length, "readout_length":q.resonator.operations['readout'].length}
        q.xy.operations[operation].length = 100_000
        q.resonator.operations['readout'].length = 100_000

        if q.xy.thread == q.resonator.thread:
            original_paras[q.name]['readout_thread'] = q.resonator.thread
            q.resonator.thread = q.xy.thread*4

# Generate the OPX and Octave configurations
config = machine.generate_config()


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
# Adjust the pulse duration and amplitude to drive the qubit into a mixed state - can be None
if node.parameters.operation_amplitude_factor:
    # pre-factor to the value defined in the config - restricted to [-2; 2)
    operation_amp = node.parameters.operation_amplitude_factor
else:
    operation_amp = 1.0
# Qubit detuning sweep with respect to their resonance frequencies
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span // 2, +span // 2, step, dtype=np.int32)
flux_point = node.parameters.flux_point_joint_or_independent
qubit_freqs = {q.name: q.xy.RF_frequency for q in qubits}  # for opx

dcs = np.linspace(
    -node.parameters.flux_range_V/2,
    node.parameters.flux_range_V/2,
    node.parameters.flux_pts,
)



# Set the qubit frequency for a given flux point
if node.parameters.arbitrary_flux_bias is not None:
    arb_flux_bias_offset = {q.name: node.parameters.arbitrary_flux_bias for q in qubits}
    detunings = {q.name: q.freq_vs_flux_01_quad_term * arb_flux_bias_offset[q.name] ** 2 for q in qubits}
elif node.parameters.arbitrary_qubit_frequency_in_ghz is not None:
    detunings = {
        q.name: 1e9 * node.parameters.arbitrary_qubit_frequency_in_ghz - qubit_freqs[q.name] for q in qubits
    }
    arb_flux_bias_offset = {q.name: np.sqrt(detunings[q.name] / q.freq_vs_flux_01_quad_term) for q in qubits}

else:
    arb_flux_bias_offset = {q.name: 0.0 for q in qubits}
    detunings = {q.name: 0.0 for q in qubits}


target_peak_width = node.parameters.target_peak_width
if target_peak_width is None:
    target_peak_width = (
        3e6  # the desired width of the response to the saturation pulse (including saturation amp), in Hz
    )

with program() as qubit_spec:
    # Macro to declare I, Q, n and their respective streams for a given number of qubit (defined in macros.py)
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    df = declare(int)  # QUA variable for the qubit frequency
    dc = declare(fixed)
    machine.apply_all_couplers_to_min()
    for i, qubit in enumerate(qubits):
        max_freq = dfs[-1] + qubit.xy.intermediate_frequency
        min_freq = dfs[0] + qubit.xy.intermediate_frequency
        assert max_freq <= 400e6 and min_freq >= -400e6, (
            f"{qubit.name} IF span out of range: min={min_freq/1e6:.2f} MHz, "
            f"max={max_freq/1e6:.2f} MHz (limit ±400 MHz), please adjust the frequency span.")
        
        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)
        if "c" in qubit.id: qubit.z.set_dc_offset(qubit.z.joint_offset) # for coupler-test case
        qubit.align()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(df, dfs)):
                # Update the qubit frequency
                qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency + detunings[qubit.name])
                with for_(*from_array(dc, dcs)):
                    qubit.align()
                    qubit.z.play("const", amplitude_scale=dc / qubit.z.operations["const"].amplitude, duration=qubit.xy.operations["saturation"].length)
                    qubit.xy.play(operation,amplitude_scale=operation_amp)
                    # readout the resonator
                    qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                    # Wait for the qubit to decay to the ground state
                    qubit.resonator.wait(10*machine.depletion_time * u.ns)
                    # save data
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])

        # Measure sequentially
        if not node.parameters.multiplexed:
            align()

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(qubits):
            I_st[i].buffer(len(dcs)).buffer(len(dfs)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(dcs)).buffer(len(dfs)).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns)  # In clock cycles = 4ns
    job = qmm.simulate(config, qubit_spec, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    samples.con1.plot()
    node.results = {"figure": plt.gcf()}
    wf_report = job.get_simulated_waveform_report()
    wf_report.create_plot(samples, plot=True, save_path=None)
    

elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(qubit_spec)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:

    if node.parameters.load_data_id is not None:
        load_data_id = node.parameters.load_data_id
        node = node.load_from_id(load_data_id)
        ds = node.results["ds"]
        restore_load_data_id(node, load_data_id)
        machine = node.machine
        qubits = resolve_qubits_from_node(machine, node)
    else:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"flux": dcs, "freq": dfs})
        # Convert IQ data into volts
        ds = convert_IQ_to_V(ds, qubits)
        # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2) and phase
        ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
        
        # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
        ds = ds.assign_coords(
            {
                "freq_full": (
                    ["qubit", "freq"],
                    np.array([dfs + qubit_freqs[q.name] + detunings[q.name] for q in qubits]),
                )
            }
        )
        ds.freq_full.attrs["long_name"] = "Frequency"
        ds.freq_full.attrs["units"] = "GHz"
    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}
    from scipy.signal import find_peaks

    def extract_prominent_peaks(da_norm, prominence=1.0):
        """
        使用顯著度 (Prominence) 沿 freq 軸搜尋真正的 Qubit 訊號峰值，剔除邊界噪聲。
        """
        freq_dim = "freq" if "freq" in da_norm.dims else "freq_GHz"
        
        # 確保頻率單位為 GHz
        if "freq_GHz" in da_norm.coords:
            freq_vals = da_norm.coords["freq_GHz"].values
        else:
            freq_vals = da_norm.coords[freq_dim].values / 1e9
            
        flux_vals = da_norm.coords["flux"].values
        
        x_list, y_list = [], []
        
        # 逐一對每一個 Flux 垂直截面尋找 Peak
        for i, flux in enumerate(flux_vals):
            slice_data = da_norm.isel(flux=i).values
            
            # 尋找顯著峰值
            peaks, props = find_peaks(slice_data, prominence=prominence)
            
            if len(peaks) > 0:
                # 若同一 Flux 下有多個 Peak，選取顯著度 (Prominence) 最高者
                best_idx = peaks[np.argmax(props["prominences"])]
                x_list.append(flux)
                y_list.append(freq_vals[best_idx])
                
        return np.array(x_list), np.array(y_list)


    def ransac_polyfit(x, y, deg, max_trials=100, residual_threshold=0.003):
        """
        RANSAC 隨機抽樣一致性多項式擬合，能極端有效抵禦邊界離群點。
        """
        n_samples = len(x)
        min_samples = deg + 2
        if n_samples < min_samples:
            return np.polyfit(x, y, deg), np.ones(n_samples, dtype=bool)

        best_inliers = None
        best_p = None
        max_inlier_count = -1

        for _ in range(max_trials):
            # 隨機抽取子集
            sample_idx = np.random.choice(n_samples, min_samples, replace=False)
            try:
                p = np.polyfit(x[sample_idx], y[sample_idx], deg)
            except np.linalg.LinAlgError:
                continue

            # 計算殘差並計算內點 (Inliers)
            residuals = np.abs(y - np.polyval(p, x))
            inliers = residuals < residual_threshold
            inlier_count = np.sum(inliers)

            if inlier_count > max_inlier_count:
                max_inlier_count = inlier_count
                best_inliers = inliers
                best_p = p

        if best_inliers is not None and np.sum(best_inliers) >= min_samples:
            # 使用所有內點重新進行精確擬合
            best_p = np.polyfit(x[best_inliers], y[best_inliers], deg)
            return best_p, best_inliers
        else:
            return np.polyfit(x, y, deg), np.ones(n_samples, dtype=bool)


    def auto_fit_spectroscopy_slice(da_norm, ax=None, aic_threshold=6.0, prominence=1.2):
        """
        自動判定並擬合 Qubit Spectrogram 訊號 (具備 RANSAC 與 Prominence 防護)。
        """
        # 1. 抓取具備顯著度的訊號峰值
        x, y = extract_prominent_peaks(da_norm, prominence=prominence)
        n = len(x)

        if n < 4:
            return None  # 有效訊號點過少，放棄擬合

        # 2. 直線與二次曲線的 RANSAC 頑健擬合
        p_lin, inliers_lin = ransac_polyfit(x, y, deg=1, residual_threshold=0.003)
        p_quad, inliers_quad = ransac_polyfit(x, y, deg=2, residual_threshold=0.003)

        # 3. 計算內點的殘差與 AIC 指標
        n_lin = np.sum(inliers_lin)
        rss_lin = np.sum((y[inliers_lin] - np.polyval(p_lin, x[inliers_lin]))**2)
        aic_lin = n_lin * np.log(max(rss_lin / n_lin, 1e-12)) + 2 * 2

        n_quad = np.sum(inliers_quad)
        rss_quad = np.sum((y[inliers_quad] - np.polyval(p_quad, x[inliers_quad]))**2)
        aic_quad = n_quad * np.log(max(rss_quad / n_quad, 1e-12)) + 2 * 3

        # 4. 模型決策 (AIC 判斷 + 邊界保護機制)
        if (aic_lin - aic_quad) > aic_threshold:
            best_model = "Quadratic"
            best_coeffs = p_quad
            valid_inliers = inliers_quad
        else:
            best_model = "Linear"
            best_coeffs = p_lin
            valid_inliers = inliers_lin

        # 5. 計算頂點座標 (極值點)
        vertex = None
        if best_model == "Quadratic":
            A, B, C = best_coeffs
            if abs(A) > 1e-12:
                x_v = -B / (2 * A)
                y_v = np.polyval(best_coeffs, x_v)
                # 若頂點超出採樣 X 軸區間太遠，代表僅為微小弧度，修正回線性
                if x_v < x.min() - 0.002 or x_v > x.max() + 0.002:
                    best_model = "Linear"
                    best_coeffs = p_lin
                    valid_inliers = inliers_lin
                else:
                    vertex = (x_v, y_v)

        # 6. 繪圖與呈現結果
        if ax is not None:
            # 標註真實採納的 Peak 點
            ax.plot(x[valid_inliers], y[valid_inliers], 'r.', markersize=3, alpha=0.7, label='Inlier Peaks')

            x_smooth = np.linspace(x.min(), x.max(), 200)
            y_smooth = np.polyval(best_coeffs, x_smooth)

            line_color = 'cyan' if best_model == 'Linear' else 'magenta'
            ax.plot(x_smooth, y_smooth, color=line_color, linestyle='--', linewidth=1.5, label=f'Fit: {best_model}')

            # 標註頂點 (僅限於 Quadratic 且頂點在圖表區域內)
            if best_model == "Quadratic" and vertex is not None:
                x_v, y_v = vertex
                ax.plot(x_v, y_v, marker='*', markersize=9, color='yellow', markeredgecolor='red')
                ax.annotate(
                    f"({x_v:.4f} V, {y_v:.4f} GHz)",
                    xy=(x_v, y_v),
                    xytext=(0, 15),
                    textcoords="offset points",
                    ha='center',
                    fontsize=7,
                    color='black',
                    weight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", lw=0.8, alpha=0.8),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="red", lw=0.8)
                )

            ax.legend(loc='lower right', fontsize=7)

        return {
            "model_type": best_model,
            "coefficients": best_coeffs,
            "vertex": vertex
        }
    

    # %% {Plotting}
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    
    for ax, qubit in grid_iter(grid):
        # 建立 freq_GHz 座標
        da = ds.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit].I
        da_norm = (da - da.mean(dim="freq")) / da.std(dim="freq")
        
        # 1. 繪製熱圖 (務必指定 y="freq_GHz")
        da_norm.plot(ax=ax, add_colorbar=False, x="flux", y="freq_GHz", robust=True)
        
        # 2. 自動擬合 (傳入已含有 freq_GHz 座標的 da_norm)
        fit_res = auto_fit_spectroscopy_slice(da_norm, ax=ax)
        
        ax.set_ylabel("Freq (GHz)")
        ax.set_xlabel("Flux (V)")
        ax.set_title(qubit["qubit"])
    grid.fig.suptitle("Qubit spectroscopy VS Flux (Overlap)")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig

    # %% {Save_results}
    for q in qubits:
        q.resonator.operations['readout'].length = original_paras[q.name]['readout_length']
        q.resonator.thread = original_paras[q.name]['readout_thread']

    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    
    node.save()


# %%
