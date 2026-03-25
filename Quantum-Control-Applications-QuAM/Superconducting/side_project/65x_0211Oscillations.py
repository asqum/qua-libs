# %%
"""
By tuning qubit's flux amplitude and pulse duration, observed the oscillation between |11> and |02> states.

Prerequisites:
- node 61 done
- Cz_coupler_flux and Cz_qubit_flux calibrated in TransmonPair's extras (updates in node 61).


Outcomes:
- Extracted J2 coupling strength.
- Optimal flux pulse amplitude and duration for the CPhase gate.

"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import active_reset, readout_state, readout_state_gef, active_reset_gef
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from quam_libs.lib.fit import fit_oscillation
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import warnings
from qualang_tools.bakery import baking
from quam_libs.lib.fit import fit_oscillation_decay_exp, oscillation_decay_exp
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from scipy.optimize import curve_fit
from scipy.fft import rfft, rfftfreq
from quam_libs.components.gates.two_qubit_gates import CZGate
from quam_libs.lib.pulses import FluxPulse

# %% {Node_parameters}
qubit_pair_indexes = [4]  # The indexes of the qubit pair in the QuAM
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = ["coupler_q%s_q%s"%(i,i+1) for i in qubit_pair_indexes]
    num_averages: int = 300
    max_time_in_ns: int = 416 #200
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type: Literal['active', 'thermal'] = "active"
    simulate: bool = False
    timeout: int = 100
    amp_range : float = 0.1 #0.1
    amp_step : float = 0.001
    load_data_id: Optional[int] = None 
    readout_flip:bool = True

node = QualibrationNode(
    name="65x_0211_oscillations", parameters=Parameters()
)
assert not (node.parameters.simulate and node.parameters.load_data_id is not None), "If simulate is True, load_data_id must be None, and vice versa."

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

# Get the relevant QuAM components
if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs = machine.active_qubit_pairs
else:
    qubit_pairs = [machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]
# if any([qp.q1.z is None or qp.q2.z is None for qp in qubit_pairs]):
#     warnings.warn("Found qubit pairs without a flux line. Skipping")

num_qubit_pairs = len(qubit_pairs)

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

def rabi_chevron_model(ft, J, f0, a, offset,tau):
    f,t = ft
    J = J
    w = f
    w0 = f0
    g = offset+a * np.sin(2*np.pi*np.sqrt(4*J**2 + (w-w0)**2) * t)**2*np.exp(-tau*np.abs((w-w0))) 
    return g.ravel()

def fit_rabi_chevron(ds_qp, init_length, init_detuning):
    da_target = ds_qp.state_target
    exp_data = da_target.values
    detuning = da_target.detuning
    time = da_target.time*1e-9
    t,f  = np.meshgrid(time,detuning)
    initial_guess = (1e9/init_length/2,
            init_detuning,
            -1,
            1.0,
            100e-9)
    fdata = np.vstack((f.ravel(),t.ravel()))
    tdata = exp_data.ravel()
    popt, pcov = curve_fit(rabi_chevron_model, fdata, tdata, p0=initial_guess)
    J = popt[0]
    f0 = popt[1]
    a = popt[2]
    offset = popt[3]
    tau = popt[4]

    return J, f0, a, offset, tau

# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

# define the amplitudes for the flux pulses
pulse_amplitudes = {}
for qp in qubit_pairs:
    detuning = qp.qubit_control.xy.RF_frequency - qp.qubit_target.xy.RF_frequency - qp.qubit_target.anharmonicity
    pulse_amplitudes[qp.name] = float(np.sqrt(-detuning/qp.qubit_control.freq_vs_flux_01_quad_term))

# Loop parameters
amplitudes = np.arange(1-node.parameters.amp_range, 1+node.parameters.amp_range, node.parameters.amp_step)
times_cycles = np.arange(4, node.parameters.max_time_in_ns // 4)

with program() as CPhase_Oscillations:
    t = declare(int)  # QUA variable for the flux pulse segment index
    idx = declare(int)
    amp = declare(fixed)    
    n = declare(int)
    n_st = declare_stream()
    comp_flux_qubit = declare(fixed)
    comp_flux_coupler = declare(fixed)
    state_control = [declare(int) for _ in range(num_qubit_pairs)]
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    
    for i, qp in enumerate(qubit_pairs):
        # Bring the active qubits to the minimum frequency point
        machine.set_all_fluxes(flux_point, qp)
        assign(comp_flux_coupler, qp.extras["Cz_coupler_flux"])
        if "coupler_qubit_crosstalk" in qp.extras:
            assign(comp_flux_qubit, qp.extras["Cz_qubit_flux"]  +  qp.extras["coupler_qubit_crosstalk"] * qp.extras["Cz_coupler_flux"] )
        else:
            assign(comp_flux_qubit, qp.extras["Cz_qubit_flux"])        
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            
            with for_(*from_array(amp, amplitudes)):                                       
                # rest of the pulse
                with for_(*from_array(t, times_cycles)):
                    # reset                    
                    if node.parameters.reset_type == "active":
                        active_reset(qp.qubit_control)
                        active_reset(qp.qubit_target)
                        qp.align()
                    else:
                        wait(qp.qubit_control.thermalization_time * u.ns)

                    # set both qubits to the excited state
                    for state,qubit in zip([state_control, state_target], [qp.qubit_control, qp.qubit_target]):
                        qubit.xy.play("x180")
                        qubit.xy.wait(5)
                    qp.align()

                    # play the flux pulse
                    qp.qubit_control.z.play("const", amplitude_scale = comp_flux_qubit / qp.qubit_control.z.operations["const"].amplitude* amp, duration = t)                
                    qp.coupler.play("const", amplitude_scale = comp_flux_coupler / qp.coupler.operations["const"].amplitude, duration = t)
                    # wait for the flux pulse to end and some extra time
                    for qubit in [qp.qubit_control, qp.qubit_target]:
                        qubit.xy.wait(node.parameters.max_time_in_ns // 4 + 10)
                    qp.align()         
                                
                    # measure both qubits
                    if not node.parameters.readout_flip:
                        readout_state_gef(qp.qubit_control, state_control[i])
                        readout_state(qp.qubit_target, state_target[i])
                    else:
                        readout_state(qp.qubit_control, state_control[i])
                        readout_state_gef(qp.qubit_target, state_target[i])

                    save(state_control[i], state_st_control[i])
                    save(state_target[i], state_st_target[i])

        align()
        
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            state_st_control[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"state_control{i + 1}")
            state_st_target[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"state_target{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, CPhase_Oscillations, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
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
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {"time": 4*times_cycles, "amp": amplitudes})
    else:
        ds, loaded_machine = load_dataset(node.parameters.load_data_id)
        if loaded_machine is not None:
            machine = loaded_machine

    node.results = {"ds": ds}


# %% {Data_analysis}

# 1. 首先定義 Guess Function (這是你缺少的)
def guess_initial_parameters(t, y):
    # Offset: 平均值
    offset_guess = np.mean(y)
    y_detrend = y - offset_guess
    
    # Amplitude: 振幅一半
    a_guess = (np.max(y) - np.min(y)) / 2
    
    # Frequency: 使用 FFT 找出最強的頻率成分
    N = len(t)
    dt = t[1] - t[0]
    yf = rfft(y_detrend)
    xf = rfftfreq(N, dt)
    # 找到頻譜中能量最大的索引，並對應到頻率
    f_guess = xf[np.argmax(np.abs(yf))]
    
    # Decay: 假設 1/e 發生在時間的中間
    decay_guess = 1 / (t[-1] * 0.5)
    
    return [a_guess, f_guess, 0.0, offset_guess, decay_guess]

t_vals = ds.time.values
amp_vals = ds.amp.values
data_2d = ds.state_control.values[0] # (amp_size, time_size)

# 3. 執行 2D 迴圈擬合
fitted_freqs = []
simulated_2d = np.zeros_like(data_2d)

for i in range(len(amp_vals)):
    y_vals = data_2d[i, :]
    p0 = guess_initial_parameters(t_vals, y_vals)
    
    try:
        # 進行擬合
        popt, _ = curve_fit(oscillation_decay_exp, t_vals, y_vals, p0=p0)
        fitted_freqs.append(popt[1])
        # 用擬合結果產生成一條完美的線，存入模擬矩陣
        simulated_2d[i, :] = oscillation_decay_exp(t_vals, *popt)
    except:
        fitted_freqs.append(np.nan)
        simulated_2d[i, :] = np.nan

# 1. 模型定義
def avoided_crossing(v, g, k, v_offset):
    return np.sqrt(g**2 + (k * (v - v_offset))**2)

# 2. 數據清洗 (Data Cleaning)
v_raw = ds.amp.values
f_raw = np.array(fitted_freqs) * 1000  # 轉為 MHz

# 過濾條件：
# a) 排除 NaN
# b) 根據你的圖，頻率應該在 10 ~ 60 MHz 之間比較合理，排除掉上方那些 75+ 的離群點
mask = (~np.isnan(f_raw)) & (f_raw < 65) & (f_raw > 5)
v_clean = v_raw[mask]
f_clean = f_raw[mask]

# 3. 執行擬合，加入 Bounds (邊界限制)
# 根據圖觀察：g 約在 10 左右, V_offset 約在 0.97 左右
# bounds 格式: ([下限], [上限])
try:
    popt_g, _ = curve_fit(
        avoided_crossing, 
        v_clean, 
        f_clean, 
        p0=[12, 400, 0.97],
        bounds=([0, 10, 0.9], [30, 2000, 1.1])
    )
    g_val, k_val, vo_val = popt_g
    print(f"修正後的耦合強度 g = {g_val:.2f} MHz")
    print(f"修正後的中心電壓 V_offset = {vo_val:.4f} V")
except Exception as e:
    print(f"二次擬合失敗: {e}")
    popt_g = [12, 400, 0.97] # 失敗時的回退值



# 4. 繪圖：左圖為擬合出的模擬 Heatmap，右圖為頻率曲線
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# 左圖：顯示擬合後的平滑結果，檢查紋理是否跟原圖一致
im = ax[0].pcolormesh(t_vals, amp_vals, simulated_2d, shading='auto', cmap='viridis')
ax[0].set_title("Simulated Heatmap (Fitted)")
ax[0].set_xlabel("Time (ns)")
ax[0].set_ylabel("Flux Amplitude")
plt.colorbar(im, ax=ax[0])

# 右圖：頻率 vs Amplitude
ax[1].plot(v_raw, f_raw, 'r.', alpha=0.2, label='Filtered outliers')
ax[1].plot(v_clean, f_clean, 'r.', markersize=8, label='Cleaned data')
v_plot = np.linspace(v_raw.min(), v_raw.max(), 100)
ax[1].plot(v_plot, avoided_crossing(v_plot, *popt_g), 'b-', linewidth=2, 
    label=f'Hyperbola Fit (g={popt_g[0]:.2f}MHz)')
ax[1].set_title("Detuning Frequency vs. Flux Amplitude")
ax[1].set_ylim(0, 100)
ax[1].set_xlabel("Flux Amplitude")
ax[1].set_ylabel("Frequency (MHz)")
ax[1].grid(True)
ax[1].legend()

plt.tight_layout()
node.results["figure_osci"] = fig # 存入結果
plt.show()

# %%

target_v = vo_val

# 2. 找到 ds.amp 中最接近 target_v 的索引
amp_array = ds.amp.values
idx = np.abs(amp_array - target_v).argmin()
actual_v = amp_array[idx]

# 3. 取得該索引下的時間與狀態數據
t_data = ds.time.values
y_data = ds.state_control.values[0, idx, :]

# 4. 取得該行當初擬合的參數 (假設你剛才跑過 2D 迴圈)
# 如果你手邊沒有 popt，我們直接用剛剛得到的 g 重新在現場擬合一次
from scipy.optimize import curve_fit

def oscillation_decay_exp(t, a, f, phi, offset, decay):
    return a * np.exp(-t * decay) * np.cos(2 * np.pi * f * t + phi) + offset

p0 = guess_initial_parameters(t_data, y_data)
popt_slice, _ = curve_fit(oscillation_decay_exp, t_data, y_data, p0=p0)

# 5. 繪圖
plt.figure(figsize=(10, 5))
plt.scatter(t_data, y_data, color='black', s=15)
t_ext = np.linspace(min(t_data), max(t_data), 10000)
plt.plot(t_ext, oscillation_decay_exp(t_ext, *popt_slice), 'r-', label='Fitted', linewidth=2)

# 標註紅點 (第一個波谷)
f_fit = popt_slice[1]
phi_fit = popt_slice[2]
t_red_dot = round((2*np.pi - phi_fit) / (2 * np.pi * f_fit))
y_red_dot = oscillation_decay_exp(t_red_dot, *popt_slice)

plt.scatter(t_red_dot, y_red_dot, color='red', s=100, zorder=5, label=f'Gate time {t_red_dot:.1f} ns')

plt.title(f"|11>-|02> Oscillation at qubit flux amplitude factor=({actual_v:.4f}), g = {popt_slice[1]*1000:.2f} MHz")
plt.xlabel("CZ gate time (ns)")
plt.ylabel("Measured state")
plt.legend()
plt.grid(True, alpha=0.3)

# 存入結果
node.results["figure_1Dslice"] = plt.gcf()
plt.show()

actual_v = 0.95
grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
grid = QubitPairGrid(grid_names, qubit_pair_names)
for ax, qubit_pair in grid_iter(grid):
    plot = ds.state_control.sel(qubit=qubit_pair['qubit']).plot(ax = ax, x= 'time', y= 'amp', add_colorbar=False)        
    plt.colorbar(plot, ax=ax, orientation='horizontal', pad=0.2, aspect=30, label='Amplitude')
    ax.hlines(actual_v, np.min(ds.time.values), np.max(ds.time.values), color='red')
    ax.set_title(qubit_pair["qubit"])
    ax.set_ylabel('Detuning [MHz]')
    ax.set_xlabel('time [nS]')
    

    quad = machine.qubit_pairs[qubit_pair["qubit"]].qubit_control.freq_vs_flux_01_quad_term
    print(f"qubit_pair: {qubit_pair['qubit']}, quad: {quad}")
    
    def detuning_to_flux(det, quad = quad):
        return 1e3 * np.sqrt(-1e6 * det / quad)

    def flux_to_detuning(flux, quad = quad):
        return -1e-6 * (flux/1e3)**2 * quad
    
    ax2 = ax.secondary_yaxis('right', functions=(detuning_to_flux, flux_to_detuning))
    ax2.set_ylabel('Flux amplitude [V]')
    ax.set_ylabel('Detuning [MHz]')
        
    plt.suptitle('control qubit state')
    plt.show()
    node.results["figure_control_raw"] = grid.fig


# %% {Update_state}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            op_names = ['Cz_unipolar', 'Cz_flattop', 'Cz_bipolar']
            for qp in qubit_pairs:
                for operation in op_names:
                    qp.gates[operation].flux_pulse_control.amplitude *= actual_v
                qp.J2 = popt_slice[1]*1e9
                
                
# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {qp.name: "successful" for qp in qubit_pairs}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()
        

# %%
