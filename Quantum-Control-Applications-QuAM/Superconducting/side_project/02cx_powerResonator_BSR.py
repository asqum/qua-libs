"""
        RESONATOR SPECTROSCOPY VERSUS READOUT AMPLITUDE
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to
extract the 'I' and 'Q' quadratures for all resonators simultaneously.
This is done across various readout intermediate dfs and amplitudes.
Based on the results, one can determine if a qubit is coupled to the resonator by noting the resonator frequency
splitting. This information can then be used to adjust the readout amplitude, choosing a readout amplitude value
just before the observed frequency splitting.

Prerequisites:
    - Calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibration of the IQ mixer connected to the readout line (be it an external mixer or an Octave port).
    - Identification of the resonator's resonance frequency (referred to as "resonator_spectroscopy").
    - Configuration of the readout pulse amplitude (the pulse processor will sweep up to twice this value) and duration.
    - Specification of the expected resonator depletion time in the state.

Before proceeding to the next node:
    - Update the readout frequency, in the state.
    - Adjust the readout amplitude, in the state.
    - Save the current state
"""


# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.lib.mw_power_utils import optimal_mw_power_settings
from quam_libs.lib.fit_utils import fit_resonator
from quam_libs.macros import qua_declaration
from quam_libs.lib.qua_datasets import convert_IQ_to_V, subtract_slope, apply_angle
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import (
    fetch_results_as_xarray,
    restore_load_data_id,
    resolve_qubits_from_node,
)
from quam_libs.trackable_object import tracked_updates
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

# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = ['q4']
    num_averages: int = 100
    frequency_span_in_mhz: float = 5 #15
    frequency_step_in_mhz: float = 0.05
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    max_power_dbm: int = 5 #-30, -10
    min_power_dbm: int = -10 # -40
    num_power_points: int = 20
    max_amp: float = 0.9999 #0.1
    operation: Literal['x180', 'saturation'] = 'x180'
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    ro_line_attenuation_dB: float = 0
    derivative_crossing_threshold_in_hz_per_dbm: int = int(-50e3)
    derivative_smoothing_window_num_points: int = 5
    moving_average_filter_window_num_points: int = 5
    multiplexed: bool = False
    load_data_id: Optional[int] = None

node = QualibrationNode(name="02cx_PowerResonator_BSR", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
node.machine = machine
op = node.parameters.operation if node.parameters.operation in ['x180', 'saturation'] else 'saturation'
# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]

if len(qubits) != 1:
    raise ValueError("This node only supports you measure a qubit in each run.")

resonators = [qubit.resonator for qubit in qubits]
prev_amps = [rr.operations["readout"].amplitude for rr in resonators]
num_qubits = len(qubits)

# Update the readout power to match the desired range, this change will be reverted at the end of the node.
tracked_resonators = []
original_ROthread = {}
for i, qubit in enumerate(qubits):
    with tracked_updates(qubit.resonator, auto_revert=False, dont_assign_to_none=True) as resonator:
        resonator.set_output_power(
            power_in_dbm=node.parameters.max_power_dbm,
            max_amplitude=node.parameters.max_amp
        )
        tracked_resonators.append(resonator)

for q in qubits:
    if q.xy.thread == q.resonator.thread:
        original_ROthread[q.name] = q.resonator.thread
        q.resonator.thread = q.xy.thread*4

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
    


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

# The readout amplitude sweep (as a pre-factor of the readout amplitude) - must be within [-2; 2)
amp_min = resonators[0].calculate_voltage_scaling_factor(
    fixed_power_dBm=node.parameters.max_power_dbm,
    target_power_dBm=node.parameters.min_power_dbm
)
amp_max = 1

amps = np.geomspace(amp_min, amp_max, node.parameters.num_power_points)

# The frequency sweep around the resonator resonance frequencies
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span / 2, +span / 2, step)  # The frequency sweep around the resonator resonance frequencies
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

with program() as multi_res_spec_vs_amp:
    # Declare 'I' and 'Q' and the corresponding streams for the two resonators.
    # For instance, here 'I' is a python list containing two QUA fixed variables.
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    a = declare(fixed)  # QUA variable for the readout amplitude pre-factor
    df = declare(int)  # QUA variable for the readout frequency
    prepare_state = declare(fixed)

    machine.apply_all_couplers_to_min()
    for i, qubit in enumerate(qubits):

        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)
        # qubit.z.set_dc_offset(-0.384) # for coupler special case
        qubit.align()
        
        # resonator of this qubit
        rr = qubit.resonator

        with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
            save(n, n_st)

            with for_(*from_array(df, dfs)):  # QUA for_ loop for sweeping the frequency
                # Update the resonator frequencies for all resonators
                update_frequency(rr.name, df + rr.intermediate_frequency)
                rr.wait(machine.depletion_time * u.ns)
                # QUA for_ loop for sweeping the readout amplitude
                with for_(*from_array(prepare_state, np.array([0.0,1.0]))):
                    with for_(*from_array(a, amps)):
                        # readout the resonator
                        qubit.xy.play(op, amplitude_scale=prepare_state)
                        qubit.align()
                        rr.measure("readout", qua_vars=(I[i], Q[i]), amplitude_scale=a)
                        # wait for the resonator to relax
                        rr.wait(10 * machine.depletion_time * u.ns)
                        # save data
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
        if not node.parameters.multiplexed:
            align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_st[i].buffer(len(amps)).buffer(2).buffer(len(dfs)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(amps)).buffer(2).buffer(len(dfs)).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, multi_res_spec_vs_amp, simulation_config)
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
    node.save()

elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(multi_res_spec_vs_amp)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    if node.parameters.load_data_id is not None:
        load_data_id = node.parameters.load_data_id
        node = node.load_from_id(load_data_id)
        ds = node.results["ds"]
        restore_load_data_id(node, load_data_id)
        machine = node.machine
        qubits = resolve_qubits_from_node(machine, node)
    else:
        power_dbm = np.linspace(
            node.parameters.min_power_dbm,
            node.parameters.max_power_dbm,
            node.parameters.num_power_points
        ) - node.parameters.ro_line_attenuation_dB
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"power_dbm": power_dbm, "prepared":np.array([0.0,1.0]), "freq": dfs})
        # Convert IQ data into volts
        ds = convert_IQ_to_V(ds, qubits)
        # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2)
        ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
        ds = ds.assign({"phase": subtract_slope(apply_angle(ds.I + 1j * ds.Q, dim="freq"), dim="freq")})
        # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
        RF_freq = np.array([dfs + q.resonator.RF_frequency for q in qubits])
        ds = ds.assign_coords({"freq_full": (["qubit", "freq"], RF_freq)})
        ds.freq_full.attrs["long_name"] = "Frequency"
        ds.freq_full.attrs["units"] = "GHz"
        ds.power_dbm.attrs["long_name"] = "Power"
        ds.power_dbm.attrs["units"] = "dBm"

        # Normalize the IQ_abs with respect to the amplitude axis
        ds = ds.assign({"IQ_abs_norm": ds["IQ_abs"] / ds.IQ_abs.mean(dim=["freq"])})

    # Add the dataset to the node
    node.results = {"ds": ds}
    
    # %% {Data_analysis}
    # node.parameters.derivative_smoothing_window_num_points = 6
    # node.parameters.moving_average_filter_window_num_points = 6


    # Generate 1D dataset tracking the minimum IQ value, as a proxy for resonator frequency
    # ds["rr_min_response"] = ds.IQ_abs_norm.idxmin(dim="freq")

    ### Skip temporarily
    # rr_min_response = ds.IQ_abs_norm.idxmin(dim="freq")
    # # Calculate the derivative along the power_dbm axis
    # ds["rr_min_response_diff"] = ds.rr_min_response.differentiate(coord="power_dbm").dropna("power_dbm")
    # # Calculate the moving average of the derivative
    # ds["rr_min_response_diff_avg"] = ds.rr_min_response_diff.rolling(
    #     power_dbm=node.parameters.derivative_smoothing_window_num_points,  # window size in points
    #     center=True
    # ).mean().dropna("power_dbm")
    # # Apply a filter to scale down the initial noisy values in the moving average if needed
    # for j in range(node.parameters.moving_average_filter_window_num_points):
    #     ds.rr_min_response_diff_avg.isel(power_dbm=j).data /= (node.parameters.moving_average_filter_window_num_points - j)
    # # Find the first position where the moving average crosses below the threshold
    # below_threshold = ds.rr_min_response_diff_avg < node.parameters.derivative_crossing_threshold_in_hz_per_dbm
    # # Get the first occurrence below the derivative threshold
    # rr_optimal_power_dbm = {}
    # rr_optimal_frequencies = {}
    # for qubit in qubits:
    #     if below_threshold.sel(qubit=qubit.name).any():
    #         rr_optimal_power_dbm[qubit.name] = below_threshold.sel(qubit=qubit.name).idxmax(dim="power_dbm")  # Get the first occurrence
    #     else:
    #         rr_optimal_power_dbm[qubit.name] = np.nan

    #     if not np.isnan(rr_optimal_power_dbm[qubit.name]):
    #         fit, fit_eval = fit_resonator(
    #             s21_data=ds.sel(power_dbm=rr_optimal_power_dbm[qubit.name].data).sel(qubit=qubit.name),
    #             frequency_LO_IF=qubit.resonator.RF_frequency,
    #             print_report=True
    #         )
    #         rr_optimal_frequencies[qubit.name] = int(fit.params["omega_r"].value)
    #     else:
    #         rr_optimal_frequencies[qubit.name] = np.nan


    # %% {Plotting}
    

    # 1. 計算 prepared=1 減去 prepared=0 的差值 (DataArray)
    diff_i = ds.I.sel(prepared=1.0) - ds.I.sel(prepared=0.0)
    diff_q = ds.Q.sel(prepared=1.0) - ds.Q.sel(prepared=0.0)

    # 除以 Prepared=0 狀態下的頻率平均振幅進行歸一化
    mean_amp = ds.IQ_abs.sel(prepared=0.0).mean(dim=["freq"])
    diff_data = np.hypot(diff_i, diff_q) / mean_amp

    # 建立網格畫布
    grid_diff = QubitGrid(ds, [q.grid_location for q in qubits])
    grid_diff.fig.suptitle(
        "Resonator Spectroscopy Difference & Region Max Location", fontsize=12
    )

    # 用於儲存每個 Qubit 找到的局部區域最大特徵點資訊
    detected_max_points = {}

    # 2. 走訪每個 Qubit 子圖
    for ax, qubit in grid_iter(grid_diff):
        # 取得單一 Qubit 的差異資料 (維度為 [freq, power_dbm])
        sub_diff = diff_data.loc[qubit]

        # --- 關鍵步驟：2D 局部區域平滑濾波 ---
        # 取絕對值，確保響應劇烈（不論正向或負向變化）的區域都能被偵測
        diff_matrix = np.abs(sub_diff.values)
        diff_matrix = np.nan_to_num(diff_matrix, nan=0.0)

        # 套用 2D 高斯濾波器 (sigma 代表特徵區域的半徑範圍，建議設 2~5)
        # sigma=(3, 3) 代表會在頻率與功率維度同時對周圍點進行區域加權平均
        smoothed_matrix = gaussian_filter(diff_matrix, sigma=(3, 3))

        # 尋找平滑後最大值的 2D 矩陣索引 (freq_index, power_index)
        freq_idx, power_idx = np.unravel_index(
            np.argmax(smoothed_matrix), smoothed_matrix.shape
        )

        # 映射回實際物理座標與原始差異數值
        if sub_diff.freq_full.ndim == 2:
            target_freq = sub_diff.freq_full.values[freq_idx, power_idx]
        else:
            target_freq = sub_diff.freq_full.values[freq_idx]

        target_power = sub_diff.power_dbm.values[power_idx]
        original_max_val = sub_diff.values[freq_idx, power_idx]

        # 紀錄偵測結果
        detected_max_points[qubit['qubit']] = {
            "freq": float(target_freq),
            "power": float(target_power),
            "max_diff": float(original_max_val),
        }

        # 3. 繪製差異熱圖
        sub_diff.plot(
            ax=ax,
            add_colorbar=True,
            x="freq_full",
            y="power_dbm",
            robust=True,
            cmap="RdBu_r",
            center=0,
        )

        # 4. 在熱圖上標示出『局部區域最大值』位置
        ax.scatter(
            target_freq,
            target_power,
            color="lime",
            marker="x",
            s=120,
            linewidths=2.5,
            zorder=5,
            label="Region Max",
        )

        ax.set_ylabel("Power (dBm)")
        ax.set_title(f"qubit = {qubit}")

    plt.tight_layout()


    # 將畫布與找到的特徵座標點輸出儲存
    node.results["figure_diff"] = grid_diff.fig
    # node.results["detected_max_points"] = detected_max_points

    for prep in [0.0, 1.0]:
        # 提取特定 prepared 狀態的 DataArray
        data_prep = ds.IQ_abs_norm.sel(prepared=prep)

        # 為當前狀態建立新的 QubitGrid 畫布
        grid = QubitGrid(ds, [q.grid_location for q in qubits])
        grid.fig.suptitle(
            f"Resonator spectroscopy VS. power at base (prepared = {prep})",
            fontsize=12,
        )

        # 走訪每個 Qubit 子圖並繪製
        for ax, qubit in grid_iter(grid):
            data_prep.loc[qubit].plot(
                ax=ax,
                add_colorbar=True,
                x="freq_full",
                y="power_dbm",
                robust=True,
            )
            ax.set_ylabel("Power (dBm)")
            ax.set_title(f"\nqubit = {qubit}")

            ax.scatter(
                target_freq,
                target_power,
                color="red",
                marker="x",
                s=120,
                linewidths=2.5,
                zorder=5,
                label="Region Max",
            )

        plt.tight_layout()
        node.results[f"figure_prepared_{str(prep)}"] = grid.fig
    

    # 螢幕列印出尋找到的物理座標
    for q_name, info in detected_max_points.items():
        print(
            f"[{q_name}] 偵測到最大差異區域中心：=====================\n"
            f"Freq = {info['freq']/1e9:.6f} GHz, "
            f"Power = {info['power']:.2f} dBm, "
            f"原始差值 = {info['max_diff']:.4f}"
        )
        qubit = machine.qubits[q_name]
        print(f"RO-IF: ",int(info['freq'] - qubit.resonator.opx_output.upconverter_frequency))
        print("===========================================\n\n")


    # %% {Update_state}
    # Revert the change done at the beginning of the node
    for tracked_resonator in tracked_resonators:
        tracked_resonator.revert_changes()
    with node.record_state_updates():
        for q in qubits:
            if q.name in original_ROthread:
                q.resonator.thread = original_ROthread[q.name]
                qubit = machine.qubits[q_name]
                qubit.resonator.intermediate_frequency = info['freq'] - qubit.resonator.opx_output.upconverter_frequency
                settings = optimal_mw_power_settings(info['power'], 0.9)
                qubit.resonator.opx_output.full_scale_power_dbm = float(settings.full_scale_power_dbm)
                qubit.resonator.operations["readout"].amplitude = float(settings.amplitude)

    # %% {Save_results}
    if node.parameters.load_data_id is not None:
        if node.storage_manager is not None:
            node.storage_manager.active_machine_path = None
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.save() 


# %%
