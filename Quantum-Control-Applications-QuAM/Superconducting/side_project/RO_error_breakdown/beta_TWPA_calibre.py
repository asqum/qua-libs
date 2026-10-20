"""
Doing TWPA calibration by observing IQ-blobs with 2 Gaussian Mixtrue fitting to get the best RO SNR. 
Treat TWPA as a Transmon and use its XY operations to be the pumping source and charging it with 10 us square pulse.

- Prerequisites:
    1. A transmon qubit for TWPA and wired. 
    2. 'TWPA_pump' xy operation in TWPA transmon with square waveform
    3.
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
from quam_libs.experiments.simulation import simulate_and_plot
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
from quam_libs.lib.mw_power_utils import optimal_mw_power_settings
from quam_libs.lib.qubit_thermometer import repetition_data, StateDiscrimination, prepare_dataset_for_qcat

from scipy.interpolate import RectBivariateSpline


class Parameters(NodeParameters):

    qubits: Optional[List[str]] = ['q4', 'q5']
    twpa_q:str = 'qTWPA'
    num_averages: int = 1200
    pump_power_min_dbm: float = 4
    pump_power_max_dbm: float = 8    # must less than or equals 16
    power_pts:int = 20
    pump_frequency_max_ghz: float = 6.75
    frequency_total_range_mhz: int = 200 # less than 700 MHz just in case
    frequency_resolution_mhz: int = 10 
    flux_point_joint_or_independent: Literal["joint"] = "joint"
    multiplexed:bool = True
    simulate: bool = False
    simulation_duration_ns: int = 25_000
    timeout: int = 1000
    load_data_id: Optional[int] = None
    use_waveform_report: bool = True

    # sequence settings
    pump_type:Literal["overlap","separate"] = 'overlap'
    overlap_charging_time_us:int = 60
    seperate_charging_time_ns:int = 400
    


node = QualibrationNode(name="beta_TWPA_calibre", parameters=Parameters())

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# node.machine = machine# Open Communication with the QOP

def get_band(freq):
    """Determine the MW fem DAC band corresponding to a given frequency.

    Args:
        freq (float): The frequency in Hz.

    Returns:
        int: The Nyquist band number.
            - 1 if 50 MHz <= freq < 5.5 GHz
            - 2 if 4.5 GHz <= freq < 7.5 GHz
            - 3 if 6.5 GHz <= freq <= 10.5 GHz

    Raises:
        ValueError: If the frequency is outside the MW fem bandwidth [50 MHz, 10.5 GHz].
    """
    if 50e6 <= freq < 5.5e9:
        return 1
    elif 4.5e9 <= freq < 7.5e9:
        return 2
    elif 6.5e9 <= freq <= 10.5e9:
        return 3
    else:
        raise ValueError(f"The specified frequency {freq} Hz is outside of the MW fem bandwidth [50 MHz, 10.5 GHz]")

if node.parameters.load_data_id is None:
    qmm = machine.connect(timeout=300)

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits_item = machine.active_qubits
else:
    qubits_item = [machine.qubits[q] for q in node.parameters.qubits]

qubits = machine.get_qubits_used_in_node(node.parameters)
num_qubits = len(qubits)

twpa = machine.qubits[node.parameters.twpa_q]

pump_operation = 'TWPA_pump'
max_power_dbm = 16
initial_IF_MHz = 45 * u.MHz
charging_time = node.parameters.seperate_charging_time_ns * u.ns

# Pump time settings
if node.parameters.pump_type == 'separate':
    max_ro_time = 0

    for q in qubits_item:
        ro_len = q.resonator.operations.readout.length * u.ns
        if ro_len > max_ro_time:
            max_ro_time = ro_len

    twpa.xy.operations[pump_operation].length = max_ro_time + 2 * charging_time
else:
    twpa.xy.operations[pump_operation].length = node.parameters.overlap_charging_time_us * u.us


# Pump frequency settings
LO_at = node.parameters.pump_frequency_max_ghz * u.GHz - (node.parameters.frequency_total_range_mhz * u.MHz)/2
settings = optimal_mw_power_settings(max_power_dbm, max_amplitude=1.0)
twpa.xy.thread = 'twpa'
print(f"TWPA Pumping power setting: amplitude={round(settings.amplitude,3)}, Full_Scale_Power={settings.full_scale_power_dbm}")
band = get_band(LO_at)
twpa.xy.opx_output.band = band
twpa.xy.opx_output.upconverter_frequency = LO_at
twpa.xy.opx_output.full_scale_power_dbm = settings.full_scale_power_dbm
twpa.xy.operations[pump_operation].amplitude = settings.amplitude
twpa.xy.intermediate_frequency = initial_IF_MHz

# Generate the OPX and Octave configurations
config = machine.generate_config()

align_elements = qubits_item
align_elements.append(twpa)

# %% {QUA_program}
n_avg = node.parameters.num_averages 
flux_point = node.parameters.flux_point_joint_or_independent
span = node.parameters.frequency_total_range_mhz * u.MHz
dfs = np.arange(-span // 2, +span // 2, node.parameters.frequency_resolution_mhz*u.MHz, dtype=np.int32)

def get_voltage_ratio_from_db(db:float)-> float:
    return 10 ** (db / 20.0)

power_span_amplitude_ratio = np.linspace(get_voltage_ratio_from_db(node.parameters.pump_power_min_dbm-max_power_dbm), get_voltage_ratio_from_db(node.parameters.pump_power_max_dbm-max_power_dbm), node.parameters.power_pts)

with program() as separate_pump:
    reset_global_phase()
    I_g, I_g_st, Q_g, Q_g_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    I_e, I_e_st, Q_e, Q_e_st, _, _ = qua_declaration(num_qubits=num_qubits)
    df = declare(int)
    damp = declare(fixed)
    for multiplexed_qubits in qubits.batch():
        if not node.parameters.simulate:
            machine.set_all_fluxes(flux_point=flux_point, target=list(multiplexed_qubits.values())[0])
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(df, dfs)):
                # pumping frequency update
                twpa.xy.update_frequency(df + twpa.xy.intermediate_frequency)
                with for_(*from_array(damp, power_span_amplitude_ratio)):

                    # measure |0> IQ blob for all qubits
                    if not node.parameters.simulate:
                        for i, qubit in multiplexed_qubits.items():
                            qubit.wait(qubit.thermalization_time * u.ns)

                    align(*[q.xy.name for q in align_elements] +
                           [q.resonator.name for q in align_elements] +
                           [q.z.name for q in align_elements])

                    # pumping ON
                    twpa.xy.play(pump_operation, amplitude_scale=damp)
                    for i, qubit in multiplexed_qubits.items():
                        qubit.resonator.wait(charging_time)
                        qubit.resonator.measure("readout", qua_vars=(I_g[i], Q_g[i]))
                        qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                        save(I_g[i], I_g_st[i])
                        save(Q_g[i], Q_g_st[i])

                    align(*[q.xy.name for q in align_elements] +
                           [q.resonator.name for q in align_elements] +
                           [q.z.name for q in align_elements])


                    # measure |1> IQ blob for all qubits
                    if not node.parameters.simulate:
                        for i, qubit in multiplexed_qubits.items():
                            qubit.wait(qubit.thermalization_time * u.ns)

                    align(*[q.xy.name for q in align_elements] +
                            [q.resonator.name for q in align_elements] +
                            [q.z.name for q in align_elements])

                    for i, qubit in multiplexed_qubits.items():
                        qubit.xy.play("x180")

                    align(*[q.xy.name for q in align_elements] +
                           [q.resonator.name for q in align_elements] +
                           [q.z.name for q in align_elements])

                    # pumping ON
                    twpa.xy.play(pump_operation, amplitude_scale=damp)
                    for i, qubit in multiplexed_qubits.items():
                        qubit.resonator.wait(charging_time)
                        qubit.resonator.measure("readout", qua_vars=(I_e[i], Q_e[i]))
                        qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                        save(I_e[i], I_e_st[i])
                        save(Q_e[i], Q_e_st[i])

                    if node.parameters.multiplexed:
                        align(*[q.xy.name for q in align_elements] +
                                [q.resonator.name for q in align_elements] +
                                [q.z.name for q in align_elements])
                    else:
                        align()
                
                twpa.xy.update_frequency(initial_IF_MHz)

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_g_st[i].buffer(len(power_span_amplitude_ratio)).buffer(len(dfs)).buffer(n_avg).save(f"I_g{i + 1}")
            Q_g_st[i].buffer(len(power_span_amplitude_ratio)).buffer(len(dfs)).buffer(n_avg).save(f"Q_g{i + 1}")
            I_e_st[i].buffer(len(power_span_amplitude_ratio)).buffer(len(dfs)).buffer(n_avg).save(f"I_e{i + 1}")
            Q_e_st[i].buffer(len(power_span_amplitude_ratio)).buffer(len(dfs)).buffer(n_avg).save(f"Q_e{i + 1}")

with program() as overlap_pump:
    reset_global_phase()
    I_g, I_g_st, Q_g, Q_g_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    I_e, I_e_st, Q_e, Q_e_st, _, _ = qua_declaration(num_qubits=num_qubits)
    df = declare(int)
    damp = declare(fixed)
    for multiplexed_qubits in qubits.batch():
        if not node.parameters.simulate:
            machine.set_all_fluxes(flux_point=flux_point, target=list(multiplexed_qubits.values())[0])
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(df, dfs)):
                # pumping frequency update
                twpa.xy.update_frequency(df + twpa.xy.intermediate_frequency)
                with for_(*from_array(damp, power_span_amplitude_ratio)):

                    # measure |0> IQ blob for all qubits
                    if not node.parameters.simulate:
                        for i, qubit in multiplexed_qubits.items():
                            qubit.wait(qubit.thermalization_time * u.ns)

                    align(*[q.xy.name for q in align_elements] +
                           [q.resonator.name for q in align_elements] +
                           [q.z.name for q in align_elements])

                    # pumping ON
                    twpa.xy.play(pump_operation, amplitude_scale=damp)
                    for i, qubit in multiplexed_qubits.items():
                        qubit.resonator.wait(twpa.xy.operations[pump_operation].length * u.ns - 2*qubit.resonator.operations.readout.length * u.ns)
                        qubit.resonator.measure("readout", qua_vars=(I_g[i], Q_g[i]))
                        qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                        save(I_g[i], I_g_st[i])
                        save(Q_g[i], Q_g_st[i])

                    align(*[q.xy.name for q in align_elements] +
                           [q.resonator.name for q in align_elements] +
                           [q.z.name for q in align_elements])

                    wait(1 * u.us)

                    # measure |1> IQ blob for all qubits
                    if not node.parameters.simulate:
                        for i, qubit in multiplexed_qubits.items():
                            qubit.wait(qubit.thermalization_time * u.ns)

                    align(*[q.xy.name for q in align_elements] +
                            [q.resonator.name for q in align_elements] +
                            [q.z.name for q in align_elements])

                    # pumping ON
                    twpa.xy.play(pump_operation, amplitude_scale=damp)

                    for i, qubit in multiplexed_qubits.items():
                        qubit.xy.wait(twpa.xy.operations[pump_operation].length * u.ns - 2*qubit.resonator.operations.readout.length * u.ns - qubit.xy.operations['x180'].length * u.ns)
                        qubit.xy.play('x180')
                        qubit.resonator.wait(qubit.xy.operations['x180'].length * u.ns)
                        qubit.resonator.measure("readout", qua_vars=(I_e[i], Q_e[i]))
                        qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                        save(I_e[i], I_e_st[i])
                        save(Q_e[i], Q_e_st[i])

                    if node.parameters.multiplexed:
                        align(*[q.xy.name for q in align_elements] +
                                [q.resonator.name for q in align_elements] +
                                [q.z.name for q in align_elements])
                    else:
                        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_g_st[i].buffer(len(power_span_amplitude_ratio)).buffer(len(dfs)).buffer(n_avg).save(f"I_g{i + 1}")
            Q_g_st[i].buffer(len(power_span_amplitude_ratio)).buffer(len(dfs)).buffer(n_avg).save(f"Q_g{i + 1}")
            I_e_st[i].buffer(len(power_span_amplitude_ratio)).buffer(len(dfs)).buffer(n_avg).save(f"I_e{i + 1}")
            Q_e_st[i].buffer(len(power_span_amplitude_ratio)).buffer(len(dfs)).buffer(n_avg).save(f"Q_e{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.pump_type == 'separate':
    qua_prog = separate_pump
else:
    qua_prog = overlap_pump



if node.parameters.simulate:
    samples, fig = simulate_and_plot(qmm, config, qua_prog, node.parameters)
    node.results = {"figure": fig}
    node.machine = machine
    node.save()

else:
    if node.parameters.load_data_id is not  None:
        load_data_id = node.parameters.load_data_id
        node = node.load_from_id(load_data_id)
        ds = node.results["ds"]
        restore_load_data_id(node, load_data_id)
        machine = node.machine
        qubits = resolve_qubits_from_node(machine, node)
    else:
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(qua_prog)
            
            results = fetching_tool(job, ["n"], mode="live")
            while results.is_processing():
                n = results.fetch_all()[0]
                progress_counter(n, n_avg, start_time=results.start_time)

        ds = fetch_results_as_xarray(job.result_handles, qubits, {"amp": power_span_amplitude_ratio, "freq": dfs, "N":np.linspace(1, node.parameters.num_averages, node.parameters.num_averages)})
        # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
        ds = ds.assign_coords(
            {
                "power_dBm": (
                    ["power_dBm"],
                    np.linspace(node.parameters.pump_power_min_dbm, node.parameters.pump_power_max_dbm, node.parameters.power_pts),
                ),
                "pump_freq": (
                    ["pump_freq"],
                    dfs + LO_at + 45*u.MHz
                ),
            }
        )
        ds.power_dBm.attrs["long_name"] = "Pumping_Power"
        ds.power_dBm.attrs["units"] = "dBm"
        ds.pump_freq.attrs["long_name"] = "Pumping_Frequency"
        ds.pump_freq.attrs["units"] = "Hz"
    node.results = {"ds": ds}

# %% {Data_analysis}


if not node.parameters.simulate:
    models = {q.name:[] for q in qubits}
    trained_params = {}
    blob_sigma = {q.name:[] for q in qubits}
    blob_dist = {q.name:[] for q in qubits}
    snr = {q.name:[] for q in qubits}
    ds_for_qcat = prepare_dataset_for_qcat(ds)

    for f in ds_for_qcat.freq.values:
        for a in ds_for_qcat.amp.values:
            # 切片提取特定 freq 與 amp 的資料
            sub_ds = ds_for_qcat.sel(freq=f, amp=a)
            sep_data = repetition_data(sub_ds, repetition_dim="qubit")
            for sq_data in sep_data:
                qubit_name = sq_data["qubit"].values.item()

                analysis = StateDiscrimination(sq_data)
                analysis._start_analysis()
                models[qubit_name].append(analysis)
                trained_params[qubit_name] = analysis.analysis_result['trained_paras']    # save trained parameters for each qubit
                means = np.array(trained_params[qubit_name]['mean'])

                # Noise
                std = trained_params[qubit_name]['std']
                blob_sigma[qubit_name].append(std)
                # Signal
                d = np.linalg.norm(means[1] - means[0])
                blob_dist[qubit_name].append(d)
                # SNR
                try:
                    snr[qubit_name].append(d / std)
                except:
                    snr[qubit_name].append()


# %% {Plot}

x_vals = ds.power_dBm.values  # Power (10 points)
y_vals = ds.pump_freq.values / 1e9  # Pump Freq in GHz (10 points)

# 1. 將每個 Qubit 的 SNR 與 Outlier List 轉為 2D 矩陣 (shape: 10 x 10)
shape_2d = (len(ds.freq), len(ds.amp))
Z_dict = {}
Z_outlier_dict = {}

for q_name, snr_list in snr.items():
    Z_dict[q_name] = np.array(snr_list).reshape(shape_2d)

for q_name, ms in models.items():
    outlier_list = [
        100 * max(m.analysis_result["outlier_probability"]) for m in ms
    ]
    Z_outlier_dict[q_name] = np.array(outlier_list).reshape(shape_2d)

# 2. 評估「大家都好」的共同點與 Outlier 統計過濾
all_snr_stack = np.array(list(Z_dict.values()))
joint_snr = np.min(all_snr_stack, axis=0)  # 找出每個點上最差 Qubit 的 SNR

all_outliers = np.array(list(Z_outlier_dict.values()))
mu = np.mean(all_outliers)
sigma = np.std(all_outliers)
thresh = max(0.5, mu)  # 採納 µ - σ 門檻（含 0.5% 下限保護）

# 建立遮罩：要求「所有 Qubit」的 Outlier 都不超過門檻
valid_mask = np.ones(shape_2d, dtype=bool)
for q_name, z_outlier in Z_outlier_dict.items():
    valid_mask &= z_outlier <= thresh

constrained_snr = np.where(valid_mask, joint_snr, -np.inf)

# 若條件過嚴無可用點，自動回退至前 25% 低 Outlier 門檻 (Q1)
if np.all(np.isinf(constrained_snr)):
    thresh = np.percentile(all_outliers, 25)
    valid_mask = np.ones(shape_2d, dtype=bool)
    for q_name, z_outlier in Z_outlier_dict.items():
        valid_mask &= z_outlier <= thresh
    constrained_snr = np.where(valid_mask, joint_snr, -np.inf)

# 3. 找出符合 Outlier 門檻且共同 SNR 最高的索引 (f_idx, a_idx)
best_f_idx, best_a_idx = np.unravel_index(
    np.argmax(constrained_snr), constrained_snr.shape
)

opt_power = x_vals[best_a_idx]
opt_pump_freq = y_vals[best_f_idx]

print("=== 共同最佳操作點 (結合 Outlier 統計過濾) ===")
print(f"Outlier 篩選門檻 : ≤ {thresh:.2f}% (µ={mu:.2f}%, σ={sigma:.2f}%)")
print(f"Power            : {opt_power:.2f} dBm")
print(f"Pump Freq        : {opt_pump_freq:.4f} GHz")
print(f"最差 Qubit Amp SNR: {joint_snr[best_f_idx, best_a_idx]:.2f}\n")

node.results["Optimal_Pumping_Frequency_MHz"] = round(opt_pump_freq * 1000, 1)
node.results["Optimal_Pumping_Power_dBm"] = round(opt_power, 2)

# 4. 繪製各 Qubit 的 Heatmap 並加註資訊
for q_name, Z in Z_dict.items():
    # 取得該 Qubit 在共同最佳點上的數值
    amp_snr = Z[best_f_idx, best_a_idx]
    outlier_p = Z_outlier_dict[q_name][best_f_idx, best_a_idx]

    # 計算該 Qubit 個別獨立的 Power SNR (dB)
    power_snr_db = 20 * np.log10(amp_snr)

    plt.figure(figsize=(8, 6))

    # 繪製 Heatmap
    mesh = plt.pcolormesh(
        x_vals, y_vals, Z, shading="auto", cmap="viridis", alpha=0.9
    )
    cbar = plt.colorbar(mesh)
    cbar.set_label("Amplitude SNR ($d/\\sigma$)")

    # 標示共同最佳點（紅星標記）
    plt.plot(
        opt_power,
        opt_pump_freq,
        "r*",
        markersize=16,
        markeredgecolor="white",
        label=f"Joint Best ({opt_power:.2f} dBm, {opt_pump_freq:.3f} GHz)",
    )

    # Title 呈現各 Qubit 個別 Amp SNR、Power SNR (dB) 與 Outlier Prob
    plt.title(
        f"SNR Heatmap - Qubit {q_name}\n"
        f"Joint Best Point: Power = {opt_power:.2f} dBm, Pump Freq = {opt_pump_freq:.3f} GHz\n"
        f"Amp SNR = {amp_snr:.2f} | Power SNR = {power_snr_db:.2f} dB | Outlier = {outlier_p:.2f}%",
        fontsize=10,
    )

    plt.xlabel("Power (dBm)")
    plt.ylabel("Pump Frequency (GHz)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    node.results[f"{q_name}_SNR_plot"] = plt.gcf()
    plt.show()

for qubit_name, snr_list in blob_sigma.items():
    Z = 1000*np.array(snr_list).reshape(len(ds.freq), len(ds.amp))

    # 1. 建立 2D Spline 插值物件
    spline = RectBivariateSpline(y_vals, x_vals, Z)

    # 2. 建立高密度網格 (例如 200x200)
    x_dense = np.linspace(x_vals.min(), x_vals.max(), 200)
    y_dense = np.linspace(y_vals.min(), y_vals.max(), 200)
    Z_dense = spline(y_dense, x_dense)

    # 3. 找出高密度網格中的最高點
    max_y_idx, max_x_idx = np.unravel_index(np.argmin(Z_dense), Z_dense.shape)
    best_power = x_dense[max_x_idx]
    best_pump_freq = y_dense[max_y_idx]
    peak_snr = Z_dense[max_y_idx, max_x_idx]

    # 4. 繪圖
    plt.figure(figsize=(8, 6))

    # 熱圖底圖
    mesh = plt.pcolormesh(
        x_vals, y_vals, Z, shading="auto", cmap="viridis", alpha=0.85
    )
    plt.colorbar(mesh, label="Noise (mV)")

    plt.title(f"Noise Heatmap for {qubit_name}")
    plt.xlabel("Pumping Power (dBm)")
    plt.ylabel("Pumping Frequency (GHz)")
    plt.tight_layout()
    node.results[f"{q_name}_Noise_plot"] = plt.gcf()
    plt.show()

for qubit_name, snr_list in blob_dist.items():
    Z = 1000*np.array(snr_list).reshape(len(ds.freq), len(ds.amp))

    # 1. 建立 2D Spline 插值物件
    spline = RectBivariateSpline(y_vals, x_vals, Z)

    # 2. 建立高密度網格 (例如 200x200)
    x_dense = np.linspace(x_vals.min(), x_vals.max(), 200)
    y_dense = np.linspace(y_vals.min(), y_vals.max(), 200)
    Z_dense = spline(y_dense, x_dense)

    # 3. 找出高密度網格中的最高點
    max_y_idx, max_x_idx = np.unravel_index(np.argmin(Z_dense), Z_dense.shape)
    best_power = x_dense[max_x_idx]
    best_pump_freq = y_dense[max_y_idx]
    peak_snr = Z_dense[max_y_idx, max_x_idx]

    # 4. 繪圖
    plt.figure(figsize=(8, 6))

    # 熱圖底圖
    mesh = plt.pcolormesh(
        x_vals, y_vals, Z, shading="auto", cmap="viridis", alpha=0.85
    )
    plt.colorbar(mesh, label="Signal (mV)")

    plt.title(f"Signal Heatmap for {qubit_name}")
    plt.xlabel("Pumping Power (dBm)")
    plt.ylabel("Pumping Frequency (GHz)")
    plt.tight_layout()
    node.results[f"{q_name}_Signal_plot"] = plt.gcf()
    plt.show()

for qubit_name, ms in models.items():
    outlier = []
    for m in ms:
        outlier.append(100*max(m.analysis_result['outlier_probability']))

    Z = np.array(outlier).reshape(len(ds.freq), len(ds.amp))

    # 4. 繪圖
    plt.figure(figsize=(8, 6))

    # 熱圖底圖
    mesh = plt.pcolormesh(
        x_vals, y_vals, Z, shading="auto", cmap="viridis", alpha=0.85
    )
    plt.colorbar(mesh, label="Maxima Outlier Probability [%]")

    plt.title(f"Outlier Heatmap for {qubit_name}")
    plt.xlabel("Pumping Power (dBm)")
    plt.ylabel("Pumping Frequency (GHz)")
    plt.tight_layout()
    node.results[f"{q_name}_outlier_plot"] = plt.gcf()
    plt.show()
# %%
# 1. 找出 opt_power 在 power_dBm 中的索引位置
a_idx = np.where(ds.power_dBm.values == opt_power)[0][0]

# 2. 找出 opt_pump_freq 在 pump_freq 中的索引位置 (注意 pump_freq 原始單位是 Hz)
f_idx = np.where(np.isclose(ds.pump_freq.values / 1e9, opt_pump_freq))[0][0]

opt_freq = ds.freq.values[f_idx]
opt_amp = ds.amp.values[a_idx]

num_amp = len(ds.amp)
flat_idx = best_f_idx * num_amp + best_a_idx

grid_2 = QubitGrid(ds, [q.grid_location for q in qubits])
for ax, qubit in grid_iter(grid_2): 
    models[qubit['qubit']][flat_idx].show_scatter_analysis(qubit['qubit'], ax)
plt.suptitle("IQ Blobs")
plt.tight_layout()
plt.show()
node.results["Colored_Scatter"] = grid_2.fig

grid_4 = QubitGrid(ds, [q.grid_location for q in qubits])
for ax, qubit in grid_iter(grid_4): 
    models[qubit['qubit']][flat_idx].show_error_analysis(qubit['qubit'], ax)

plt.show()
node.results["Error_source"] = grid_4.fig

# %% {Checking parameters to update}


print(f"pumping freq = {round((opt_freq * u.Hz + twpa.xy.intermediate_frequency * u.Hz + LO_at)*1e-6, 1)} MHz, pumping power = {opt_power} dBm")


# %% {Update state}
if node.parameters.load_data_id is None:
    with node.record_state_updates():
        # freq
        twpa.xy.intermediate_frequency += opt_freq * u.Hz
        # power
        twpa.xy.operations[pump_operation].amplitude *= opt_amp
        # additional
        twpa.extras = {}
        twpa.extras['charging_time'] = charging_time
        twpa.extras['pumping_mode'] = node.parameters.pump_type

# %% {Save_results}
node.machine = machine
node.results["initial_parameters"] = node.parameters.model_dump()
node.save()  
# %%
