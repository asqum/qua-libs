# %%
"""
Two-Qubit Readout Confusion Matrix Measurement

This sequence measures the readout error when simultaneously measuring the state of two qubits. The process involves:

1. Preparing the two qubits in all possible combinations of computational basis states (|00⟩, |01⟩, |10⟩, |11⟩)
2. Performing simultaneous readout on both qubits
3. Calculating the confusion matrix based on the measurement results

For each prepared state, we measure:
1. The readout result of the first qubit
2. The readout result of the second qubit

The measurement process involves:
1. Initializing both qubits to the ground state
2. Applying single-qubit gates to prepare the desired input state
3. Performing simultaneous readout on both qubits
4. Repeating the process multiple times to gather statistics

The outcome of this measurement will be used to:
1. Quantify the readout fidelity for two-qubit states
2. Identify and characterize crosstalk effects in the readout process
3. Provide data for readout error mitigation in two-qubit experiments

Prerequisites:
- Calibrated single-qubit gates for both qubits in the pair
- Calibrated readout for both qubits

Outcomes:
- 4x4 confusion matrix representing the probabilities of measuring each two-qubit state given a prepared input state
- Readout fidelity metrics for simultaneous two-qubit measurement
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import active_reset, readout_state, readout_state_gef, active_reset_gef, active_reset_simple
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
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
import warnings
from qualang_tools.bakery import baking
from quam_libs.lib.fit import fit_oscillation, oscillation, fix_oscillation_phi_2pi
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from scipy.optimize import curve_fit
from quam_libs.components.gates.two_qubit_gates import CZGate
import xarray as xr

# %% {Node_parameters}
qubit_pair_indexes = [3]  # The indexes of the qubit pairs to measure
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = ["coupler_q%s_q%s"%(i,i+1) for i in qubit_pair_indexes]
    num_averages: int = 250
    flux_point_joint_or_independent_or_pairwise: Literal["joint", "independent", "pairwise"] = "joint"
    reset_type: Literal['active', 'thermal'] = "active"
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None
    qubit_amp_range : float = 0.6
    qubit_amp_step : float = 0.6/50
    coupler_amp_range : float = 0.6
    coupler_amp_step : float = 0.6/50
    use_state_discrimination: bool = True
    con_tar_flip:bool = True
    operation: Literal["Cz"] = "Cz"
    
    
    

node = QualibrationNode(
    name="62c_11_leakage_vs_qubit_coupler_flux", parameters=Parameters()
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


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

flux_point = node.parameters.flux_point_joint_or_independent_or_pairwise  # 'independent' or 'joint' or 'pairwise'
# Loop parameters
flux_qubit_amplitudes = np.arange(1-node.parameters.qubit_amp_range, 1+node.parameters.qubit_amp_range, node.parameters.qubit_amp_step)
flux_coupler_amplitudes = np.arange(1-node.parameters.coupler_amp_range, 1+node.parameters.coupler_amp_range, node.parameters.coupler_amp_step)
    
reset_coupler_bias = False
operation_name = node.parameters.operation

with program() as CPhase_Oscillations:
    n = declare(int)
    flux_coupler_amp = declare(float)
    flux_qubit_amp = declare(float)
    n_st = declare_stream()
    
    state_control = [declare(int) for _ in range(num_qubit_pairs)]
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    I_control = [declare(float) for _ in range(num_qubit_pairs)]
    Q_control = [declare(float) for _ in range(num_qubit_pairs)]
    I_target = [declare(float) for _ in range(num_qubit_pairs)]
    Q_target = [declare(float) for _ in range(num_qubit_pairs)]
    I_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    Q_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    I_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    Q_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    
    
    for i, qp in enumerate(qubit_pairs):
        qp.gates[operation_name].phase_shift_control = 0.0
        qp.gates[operation_name].phase_shift_target = 0.0
        # Bring the active qubits to the minimum frequency point
        machine.set_all_fluxes(flux_point, qp)
        if reset_coupler_bias:
            qp.coupler.set_dc_offset(0.0)
        wait(1000)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)         
            with for_(*from_array(flux_coupler_amp, flux_coupler_amplitudes)):
                with for_(*from_array(flux_qubit_amp, flux_qubit_amplitudes)):
                        # reset
                        if node.parameters.reset_type == "active":
                            # active_reset(qp.qubit_control)
                            # active_reset(qp.qubit_target)
                            active_reset_gef(qp.qubit_control)
                            active_reset_gef(qp.qubit_target)
                        else:
                            wait(qp.qubit_control.thermalization_time * u.ns)
                            wait(qp.qubit_target.thermalization_time * u.ns)

                        # state preparation
                        qp.qubit_control.xy.play("x180")
                        qp.qubit_target.xy.play("x180")
                        align()
                        qp.gates[operation_name].execute(amplitude_scale = flux_qubit_amp, coupler_amplitude_scale= flux_coupler_amp)
                        align()
                        wait(20)
                        # readout
                        if node.parameters.use_state_discrimination:
                            readout_state_gef(qp.qubit_control, state_control[i])
                            readout_state_gef(qp.qubit_target, state_target[i])
                            # readout_state(qp.qubit_control, state_control[i])
                            # readout_state(qp.qubit_target, state_target[i])
                            save(state_control[i], state_st_control[i])
                            save(state_target[i], state_st_target[i])

                        else:
                            qp.qubit_control.resonator.measure("readout", qua_vars=(I_control[i], Q_control[i]))
                            qp.qubit_target.resonator.measure("readout", qua_vars=(I_target[i], Q_target[i]))
                            save(I_control[i], I_st_control[i])
                            save(Q_control[i], Q_st_control[i])
                            save(I_target[i], I_st_target[i])
                            save(Q_target[i], Q_st_target[i])
        # align()
        
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            if node.parameters.use_state_discrimination:
                state_st_control[i].buffer(len(flux_qubit_amplitudes)).buffer(len(flux_coupler_amplitudes)).buffer(n_avg).save(f"state_control{i + 1}")
                state_st_target[i].buffer(len(flux_qubit_amplitudes)).buffer(len(flux_coupler_amplitudes)).buffer(n_avg).save(f"state_target{i + 1}")
            else:
                I_st_control[i].buffer(len(flux_qubit_amplitudes)).buffer(len(flux_coupler_amplitudes)).buffer(n_avg).save(f"I_control{i + 1}")
                Q_st_control[i].buffer(len(flux_qubit_amplitudes)).buffer(len(flux_coupler_amplitudes)).buffer(n_avg).save(f"Q_control{i + 1}")
                I_st_target[i].buffer(len(flux_qubit_amplitudes)).buffer(len(flux_coupler_amplitudes)).buffer(n_avg).save(f"I_target{i + 1}")
                Q_st_target[i].buffer(len(flux_qubit_amplitudes)).buffer(len(flux_coupler_amplitudes)).buffer(n_avg).save(f"Q_target{i + 1}")

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
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {"qubit_amp": flux_qubit_amplitudes, "coupler_amp": flux_coupler_amplitudes, "N": np.linspace(1, n_avg, n_avg)})
    else:
        ds, machine = load_dataset(node.parameters.load_data_id)
        
    node.results = {"ds": ds}
    node.results["results"] = {}

    
# %% data processing
if not node.parameters.simulate:
    def qubit_flux_shift(qp, amp):
        return amp * qp.gates[operation_name].flux_pulse_control.amplitude
    def coupler_flux_shift(qp, amp):
        return amp * qp.gates[operation_name].coupler_flux_pulse.amplitude
    def abs_coupler_amp(qp, amp):
        return amp * qp.gates[operation_name].coupler_flux_pulse.amplitude + qp.coupler.decouple_offset
    def detuning(qp, amp):
        return -(amp * qp.gates[operation_name].flux_pulse_control.amplitude)**2 * qp.qubit_control.freq_vs_flux_01_quad_term
    ds = ds.assign_coords(
        {"flux_qubit": (["qubit", "qubit_amp"], np.array([qubit_flux_shift(qp, ds.qubit_amp) for qp in qubit_pairs]))}
    )
    ds = ds.assign_coords(
        {"detuning": (["qubit", "qubit_amp"], np.array([detuning(qp, ds.qubit_amp) for qp in qubit_pairs]))}
    )
    ds = ds.assign_coords(
        {"flux_coupler_full": (["qubit", "coupler_amp"], np.array([abs_coupler_amp(qp, ds.coupler_amp) for qp in qubit_pairs]))}
    )
    ds = ds.assign_coords(
        {"flux_coupler": (["qubit", "coupler_amp"], np.array([coupler_flux_shift(qp, ds.coupler_amp) for qp in qubit_pairs]))}
    )

# %%Data analysis
import scipy.ndimage as nd

if node.parameters.use_state_discrimination:
    sc = ds.state_control
    st = ds.state_target

    # compute populations averaged over N (shots)
    P11 = ((sc == 1) & (st == 1)).mean("N").rename("P11")
    P02 = ((sc == 0) & (st == 2)).mean("N").rename("P02")
    P20 = ((sc == 2) & (st == 0)).mean("N").rename("P20")

    # merge into dataset
    ds = xr.merge([ds, P11, P02, P20])

for qp in qubit_pairs:
    qpname = qp.name
    # --- Select data ---
    P11 = ds.P11.sel(qubit=qpname)
    flux_qb = ds.flux_qubit.sel(qubit=qpname)
    flux_cpl_full = ds.flux_coupler_full.sel(qubit=qpname)

    # Smooth to suppress noise ---
    P11_smooth = P11.copy(data=nd.gaussian_filter(P11.values, sigma=2))

    # Find blob minimum (global)
    i_cpl_min, i_qb_min = np.unravel_index(np.nanargmin(P11_smooth), P11.shape)

    P11_min_value = float(P11.values[i_cpl_min, i_qb_min])
    coupler_amp_min = float(P11.coupler_amp.values[i_cpl_min])
    qubit_amp_min   = float(P11.qubit_amp.values[i_qb_min])
    flux_coupler_full_min = float(flux_cpl_full.interp(coupler_amp=coupler_amp_min))
    flux_qubit_min  = float(flux_qb.interp(qubit_amp=qubit_amp_min))

    # Search ALONG the same qubit column (fixed qubit_amp) for max
    col_data = P11_smooth[:, i_qb_min]        # all coupler points for fixed qubit_amp
    flux_col = flux_cpl_full.data

    # Find maximum of P11 along this vertical column
    i_cpl_max = np.nanargmax(col_data)

    P11_max_value = float(col_data[i_cpl_max])
    coupler_amp_max = float(P11.coupler_amp.values[i_cpl_max])
    flux_coupler_full_max = float(flux_cpl_full.interp(coupler_amp=coupler_amp_max))
    flux_qubit_max = float(flux_qubit_min)  # same column
    flux_coupler_max = float(ds.flux_coupler.sel(qubit=qpname).interp(coupler_amp=coupler_amp_max))

    print(f"\n Optimal values:")
    print(f" Coupler flux shift = {flux_coupler_max:.4f}, Qubit flux shift = {flux_qubit_max:.4f} V")
    node.results["results"][qpname] = {
        "flux_coupler_full_max": flux_coupler_full_max,
        "flux_qubit_max": flux_qubit_max,
        "flux_coupler_max": flux_coupler_max,   
        }

# %% {Plotting}
DGXQ_tuneup_bound = {"Aq":0.075, "Ac":0.1} # Do NOT edit.
import matplotlib.patches as patches
## Hardcoded:
HCd:bool = 1
if HCd:
    node.results["results"][qubit_pairs[0].name]["flux_qubit_max"] = 106*1e-3
    node.results["results"][qubit_pairs[0].name]["flux_coupler_max"] = -118*1e-3

if not node.parameters.simulate:
    # 這裡我們調整繪圖邏輯：為每個 Qubit Pair 建立一個含有 1 列 2 欄 (P11 & P20) 的 Figure
    for qp in qubit_pairs:
        qubit_name = qp.name
        qubit_pair = machine.qubit_pairs[qubit_name]
        
        # 建立一個新的 Figure，包含兩個子圖 (ax1 為 P11, ax2 為 P20)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        control_freq_place:str = qp.extras.get("Cz_control", 'high')
        # 定義要處理的數據對應與標題
        plot_configs = [
            {"data_key": "P11", "ax": ax1, "title": "P11 Leakage"},
            {"data_key": "P20" if control_freq_place!='low' else "P02", "ax": ax2, "title": "|02> Population" if control_freq_place!='low' else "|20> Population"}
        ]

        for config in plot_configs:
            ax = config["ax"]
            key = config["data_key"]
            
            try:
                if node.parameters.use_state_discrimination:
                    # 1. 提取並轉換座標
                    values_to_plot = ds[key].sel(qubit=qubit_name)
                    values_to_plot = values_to_plot.assign_coords({
                        "flux_qubit_mV": 1e3 * values_to_plot.flux_qubit,
                        "flux_coupler_mV": 1e3 * values_to_plot.flux_coupler,
                    })
                    
                    # 2. 繪製熱圖
                    values_to_plot.plot(ax=ax, cmap="viridis", x="flux_qubit_mV", y="flux_coupler_mV")
                    ax.set_title(config["title"])
                    
                    # 3. 標註分析結果 (Optimal Point) - 兩張圖都標上以便對照
                    res = node.results["results"].get(qubit_name, {})
                    f_cpl_max_mV = 1e3 * res.get("flux_coupler_max", np.nan)
                    f_qb_max_mV = 1e3 * res.get("flux_qubit_max", np.nan)

                    if np.isfinite(f_cpl_max_mV):
                        ax.axhline(f_cpl_max_mV, color="white", lw=1.0, ls="--", alpha=0.5)
                    if np.isfinite(f_qb_max_mV):
                        ax.axvline(f_qb_max_mV, color="white", lw=1.0, ls="--", alpha=0.5)
                    if np.isfinite(f_qb_max_mV) and np.isfinite(f_cpl_max_mV):
                        # 1. 計算矩形的邊界
                        qb_low = f_qb_max_mV * (1 - DGXQ_tuneup_bound["Aq"])
                        qb_high = f_qb_max_mV * (1 + DGXQ_tuneup_bound["Aq"])
                        cpl_low = f_cpl_max_mV * (1 - DGXQ_tuneup_bound["Ac"])
                        cpl_high = f_cpl_max_mV * (1 + DGXQ_tuneup_bound["Ac"])

                        # 2. 建立矩形物件 (左下角座標, 寬, 高)
                        # 注意：如果你的值是負的，需確保 width 和 height 計算邏輯正確
                        width = qb_high - qb_low
                        height = cpl_high - cpl_low
                        
                        rect = patches.Rectangle(
                            (qb_low, cpl_low), width, height,
                            linewidth=1.5, edgecolor='red', facecolor='none', 
                            linestyle='--', alpha=0.8, label="DGXQ optimize bound"
                        )

                        # 3. 將矩形加入畫布
                        ax.add_patch(rect)
                        ax.plot(f_qb_max_mV, f_cpl_max_mV, marker="+", color="red", markersize=10, label="Optimal")

            except Exception as e:
                print(f"[WARN] Plot {key} failed for {qubit_name}: {e}")

        # 4. 裝飾性設定 (僅在左圖設定 Y 軸標籤)
        ax1.set_ylabel("Coupler flux shift [mV]")
        ax1.set_xlabel("Qubit flux shift [mV]")
        ax2.set_xlabel("Qubit flux shift [mV]")
        ax2.set_ylabel("") # 因為 sharey=True

        # 5. 加入上方 Detuning 軸 (以 ax1 為基準)
        try:
            sel = ds.sel(qubit=qubit_name)
            fq_data = (sel.flux_qubit.values * 1e3).ravel()
            det_data = (sel.detuning.values * 1e-6).ravel()
            order = np.argsort(fq_data)
            x_u, idx = np.unique(fq_data[order], return_index=True)
            y_u = det_data[order][idx]
            
            if x_u.size >= 2:
                sec_ax = ax1.secondary_xaxis("top", functions=(
                    lambda x: np.interp(x, x_u, y_u),
                    lambda y: np.interp(y, y_u, x_u)
                ))
                sec_ax.set_xlabel("Detuning [MHz]")
        except Exception as e:
            print(f"[WARN] Secondary axis failed: {e}")

        fig.suptitle(f"{qubit_name}: P11 vs P20 | Decouple: {qubit_pair.coupler.decouple_offset*1e3:.0f} mV", y=1.02)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 儲存圖片
        node.results[f"figure_{qubit_name}_P11_P20"] = fig




# %% {Update_state}
if not node.parameters.simulate:
    if not node.parameters.simulate:
        with node.record_state_updates():
            for qp in qubit_pairs:
                    qp.extras["CZ_coupler_flux"] = node.results["results"][qp.name]["flux_coupler_max"]
                    qp.gates[operation_name].coupler_flux_pulse.amplitude = node.results["results"][qp.name]["flux_coupler_max"]
# %% {Save_results}
if not node.parameters.simulate:    
    node.outcomes = {q.name: "successful" for q in qubit_pairs}
    node.results['initial_parameters'] = node.parameters.model_dump()
    node.machine = machine
    node.save()
# %%