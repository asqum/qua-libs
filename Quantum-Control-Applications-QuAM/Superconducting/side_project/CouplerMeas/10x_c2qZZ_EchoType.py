# %%
"""
Coupler to qubit JAZZ
Measures the ZZ from coupler to its readout_q.

Prerequisites:
- π-pulse calibrated for the coupler.

Outcomes:
- ZZ strength statistics

PS. load_data is working
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import active_reset, readout_state, active_reset_coupler
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
from scipy.stats import norm
from quam_libs.lib.fit import fit_oscillation_decay_exp, oscillation_decay_exp

# %% {Node_parameters}
qubit_pair_indexes = [4]  # The indexes of the qubit pairs to measure
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = ["coupler_q%s_q%s"%(i,i+1) for i in qubit_pair_indexes]
    num_averages: int = 1000
    flux_point_joint_or_independent_or_pairwise: Literal["joint", "independent", "pairwise"] = "joint"
    reset_type: Literal['active', 'thermal'] = "active"
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None
    frequency_detuning_in_mhz: float = 2.0
    """Frequency detuning in MHz. Default is 1.0 MHz. Determined by the T2"""
    min_wait_time_in_ns: int = 16
    """Minimum wait time in nanoseconds. Default is 16."""
    max_wait_time_in_ns: int = 5016 
    """Maximum wait time in nanoseconds. Default is 5000."""
    wait_time_step_in_ns: int = 50
    """Number of time points to sample. Default is 50."""
    histo_num:int = 101
    """Number of statistics. Default is 100."""
    use_state_discrimination: bool = True

    

node = QualibrationNode(
    name="67bxx_ZZstrength_Histogram", parameters=Parameters()
)
assert not (node.parameters.simulate and node.parameters.load_data_id is not None), "If simulate is True, load_data_id must be None, and vice versa."

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

## Get the relevant QuAM components
coupler = [machine.qubit_pairs[node.parameters.qubit_pairs[0]]] # currently supports 1 coupler a time only.
drive_q = [machine.qubits[coupler[0].extras["RD"]["driven_q"]]]
detector_q = [machine.qubits[coupler[0].extras["RD"]["readout_q"]]]

# Change driving LO
if not node.parameters.simulate and node.parameters.load_data_id is None:
    aswap_dir_update_is_q = True
    drive_LO_original = {drive_q[0].name: drive_q[0].xy.opx_output.upconverter_frequency}
    drive_q[0].xy.opx_output.upconverter_frequency = coupler[0].extras["RD"]["LO"]
    if "swap_direction" in coupler[0].extras["RD"]:
        detector_q[0].z.operations['aSWAP'].slope_direction = coupler[0].extras["RD"]["swap_direction"]
    if 'strategy' not in coupler[0].extras["RD"]:
        readout_strategy = 'aswap'
    else:
        readout_strategy = coupler[0].extras["RD"]["strategy"]
    if coupler[0].extras["RD"]["aswap_supplier"].lower() == 'c':
        print("*** aSWAP is applied on coupler itself !")
        if not hasattr(coupler[0].coupler.operations, "aSWAP"):
            raise  LookupError(f"aSWAP operation now is not in {coupler[0].name}.coupler.operation, please add it to unlock the ability for coupler's measurement!")
        aswaper = coupler[0]
        coupler[0].coupler.operations['aSWAP'].slope_direction = coupler[0].extras["RD"]["swap_direction"]
        aswap_dir_update_is_q = False
    else:
        aswaper = None

num_qubit_pairs = 1

# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
detuning = int(1e6 * node.parameters.frequency_detuning_in_mhz)
flux_point = node.parameters.flux_point_joint_or_independent_or_pairwise  # 'independent' or 'joint' or 'pairwise'

# Loop parameters
fluxes_coupler = np.arange(node.parameters.histo_num)

idle_times = np.arange(
    node.parameters.min_wait_time_in_ns // 4,
    node.parameters.max_wait_time_in_ns // 4,
    node.parameters.wait_time_step_in_ns // 4,
)
reset_coupler_bias = False

with program() as Ramsey_ZZ_coupling:
    n = declare(int)
    flux_coupler = declare(float)
    flux_qubit = declare(float)
    n_st = declare_stream()
    control_initial = declare(int)  # initial state of the control qubit
    t = declare(int)  # QUA variable for the idle time
    t_half = declare(int)
    phi = declare(fixed)  # QUA variable for dephasing the second pi/2 pulse (virtual Z-rotation)
    init_state = [declare(int) for _ in range(num_qubit_pairs)]
    current_state = [declare(int) for _ in range(num_qubit_pairs)]
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    I_target = [declare(float) for _ in range(num_qubit_pairs)]
    Q_target = [declare(float) for _ in range(num_qubit_pairs)]

    state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    I_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    Q_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    
    
    for i, qubit in enumerate(detector_q):
        
       
        # Bring the active qubits to the minimum frequency point
        if not node.parameters.simulate:
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)
            if "c" in qubit.id: qubit.z.set_dc_offset(qubit.z.joint_offset) # for coupler-test case
            qubit.z.settle()
        
        wait(1000)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(flux_coupler, fluxes_coupler)):
                with for_(*from_array(t, idle_times)):
                    # Rotate the frame of the second x90 gate to implement a virtual Z-rotation
                    # 4*tau because tau was in clock cycles and 1e-9 because tau is ns                    
                    assign(phi, Cast.mul_fixed_by_int(detuning * 1e-9, 4 * t))
                    
                    assign(t_half, t/2)
                    
                    if node.parameters.reset_type == "active":
                        active_reset_coupler(drive_q[0], qubit, f"x180_{coupler[0].name}", flux_applied_target=aswaper, method='aswap')
                        
                    else:
                        if not node.parameters.simulate:
                            if qubit.thermalization_time//5 > coupler[0].extras['T1']*1e9:
                                wait(qubit.thermalization_time * u.ns)
                            else:
                                wait(10*coupler[0].extras['T1']*1e9 * u.ns)
                    align()
                    drive_q[0].xy.update_frequency(coupler[0].extras["RD"]["IF"])
                    wait(4)
                    
                    
                    # pi pulse on target qubit
                    qubit.xy.play("x90")
                    wait(t_half)
                    align()

                    # Echo pulse
                    drive_q[0].xy.play(f"x180_{coupler[0].name}")
                    qubit.xy.play("x180")
                    align()
                    
                    
                    wait(t_half)
                    align()
                    # rotate the frame
                    qubit.xy.frame_rotation_2pi(phi)
                    # Tomographic rotation on the target qubit
                    qubit.xy.play("x90")
                    align() 
                    
                    # target qubit readout
                    if node.parameters.use_state_discrimination:
                        readout_state(qubit, current_state[i])
                        assign(state_target[i], init_state[i] ^ current_state[i])
                        save(state_target[i], state_st_target[i])
                        
                    else:
                        qubit.resonator.measure("readout", qua_vars=(I_target[i], Q_target[i]))
                        save(I_target[i], I_st_target[i])
                        save(Q_target[i], Q_st_target[i])
                    
                    reset_frame(qubit.xy.name)
                    drive_q[0].xy.update_frequency(drive_q[0].xy.intermediate_frequency)
                    wait(4)
                    align()
        
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            if node.parameters.use_state_discrimination:
                state_st_target[i].buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"state_target{i + 1}")
            else:
                I_st_target[i].buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"I_target{i + 1}")
                Q_st_target[i].buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"Q_target{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000 //4)  # In clock cycles = 4ns
    job = qmm.simulate(config, Ramsey_ZZ_coupling, simulation_config)
    samples = job.get_simulated_samples()
    samples.con1.plot()
    node.results = {"figure": plt.gcf()}
    wf_report = job.get_simulated_waveform_report()
    wf_report.create_plot(samples, plot=True, save_path=None)
    node.machine = machine
    node.save()
else:
    if node.parameters.load_data_id is None:
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(Ramsey_ZZ_coupling)

            results = fetching_tool(job, ["n"], mode="live")
            while results.is_processing():
                # Fetch results
                n = results.fetch_all()[0]
                # Progress bar
                progress_counter(n, n_avg, start_time=results.start_time)

        ds = fetch_results_as_xarray(job.result_handles, coupler, {"idle_time": idle_times, "flux_coupler": fluxes_coupler})
        node.results = {"ds": ds}
        reload_qbs = False
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"] 
        machine = node.machine
        reload_qbs = True
    

# %% {Data_analysis}
if reload_qbs:
    coupler = [machine.qubit_pairs[c_name] for c_name in ds.qubit.values]
node.results["fit_results"] = {}   # optional if you also store raw fits
node.results["analysis_success"] = {}  # for success flags per pair

if not node.parameters.simulate:

    try:
        print("Starting analysis...")
        
        flux_coupler_full = np.array([
            fluxes_coupler for _ in coupler
        ])
        
        ds = ds.assign_coords({
            "flux_coupler_full": (["qubit", "flux_coupler"], flux_coupler_full)
        })
        ds = ds.assign_coords(idle_time=4 * idle_times / 1e3)

        for i, c in enumerate(coupler):
            print(f"Analyzing qubit pair {i}: {c.name}")

            # ---------------- FIT ----------------
            fit_data = fit_oscillation_decay_exp(ds.state_target, "idle_time")

            # Save fit data for reference

            # Determine if fit succeeded (e.g. presence of valid freq)
            fit_ok = (
                np.isfinite(fit_data.sel(fit_vals="f")).any()
                and not np.isnan(fit_data.sel(fit_vals="f")).all()
            )

            # ---------------- χZZ COMPUTATION ----------------
            fitted = oscillation_decay_exp(
                ds.state_target.idle_time,
                fit_data.sel(fit_vals="a"),
                fit_data.sel(fit_vals="f"),
                fit_data.sel(fit_vals="phi"),
                fit_data.sel(fit_vals="offset"),
                fit_data.sel(fit_vals="decay"),
            )
            ds["state_target_fit"] = fitted

            chiZZ = 2 * (fit_data.sel(fit_vals="f") * 1e3 - node.parameters.frequency_detuning_in_mhz * 1e3)  # MHz → kHz
            ds["chiZZ"] = chiZZ

            chiZZ_std = 1e3 * np.sqrt(fit_data.sel(fit_vals="f_f"))
            ds["chiZZ_std"] = chiZZ_std

            chiZZ_std_filt = chiZZ_std.where(chiZZ_std < chiZZ_std.median() * 2)
            chiZZ_filt = chiZZ.where(~np.isnan(chiZZ_std_filt))

            max_idx = abs(chiZZ_filt).argmax(dim="flux_coupler").item()
            min_idx = abs(chiZZ_filt).argmin(dim="flux_coupler").item()

            flux_full = ds["flux_coupler_full"].isel(qubit=i)
            flux_val_max = float(flux_full.isel(flux_coupler=max_idx))
            flux_val_min = float(flux_full.isel(flux_coupler=min_idx))
            chi_max = float(chiZZ.isel(flux_coupler=max_idx))
            chi_min = float(chiZZ.isel(flux_coupler=min_idx))

            

            # ---------------- SAVE FIT SUCCESS FLAG ----------------
            node.results["analysis_success"][c.name] = "successful" if fit_ok else "failed"

        print("Analysis complete ")

    except Exception as e:
        node.results = {"ds": ds}
        node.results["analysis_error"] = str(e)
        analysis_successful = False
        print(f"Analysis failed : {e}")

        

# %% {plotting}

if not node.parameters.simulate:
    for i, qp in enumerate(coupler):
        qubit_name = qp.name
        node.results["fit_results"][qubit_name] = {}
        fit_status = node.results.get("analysis_success", {}).get(qubit_name, "failed")

        # =====================================================
        # --- Always plot raw Ramsey / state_target heatmaps ---
        # =====================================================
        print(f"Plotting raw data for {qubit_name}")
        ds_pair = ds.isel(qubit=i)
        flux_full = ds_pair["flux_coupler_full"]
        flux_mV = flux_full.squeeze()
        idle_time_us = ds_pair["idle_time"]

        fig, ax = plt.subplots(figsize=(6, 4))
        data = ds_pair["state_target"]
        im = ax.pcolormesh(
            flux_mV,
            idle_time_us,
            data.transpose(),
            shading="auto",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        fig.colorbar(im, ax=ax, label="State probability")
        ax.set_xlabel("Coupler flux [mV]")
        ax.set_ylabel("Idle time [µs]")
        ax.set_title(f"{qubit_name} – Raw Ramsey", fontsize=10)
        plt.tight_layout()
        node.results[f"figure_raw_{qubit_name}"] = fig
        plt.show()

        if fit_status == "successful" and "chiZZ" in ds:
            print(f"Plotting χZZ analysis for {qubit_name}")
            ds_pair = ds.isel(qubit=i)
            chi = ds_pair["chiZZ"]
            chi_std = ds_pair["chiZZ_std"]
            flux_full = ds_pair["flux_coupler_full"]
            flux_mV = flux_full.squeeze()

            res = node.results["fit_results"][qubit_name]

            # --- χZZ vs flux plot ---
            # --- χZZ vs flux: 2 subplots (linear + log-log) ---
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=False)

            # Common data
            chi_val = chi.squeeze()
            chi_std_val = chi_std.squeeze()
            chi_abs = np.abs(chi_val)

            # ---------------- (1) Linear plot ----------------
            ax = axes[0]
            ax.errorbar(
                flux_mV,
                chi_val,
                yerr=chi_std_val,
                fmt="-o",
                color="tab:red",
                lw=1.8,
                elinewidth=1,
                capsize=3,
                markersize=4,
                alpha=0.9,
                label="χZZ ± fit std",
            )

            # ax.axhline(0, color="k", lw=1, alpha=0.4)
            ax.set_title("Linear scale")
            ax.set_xlabel("The index")
            ax.set_ylabel("χZZ [kHz]")
            ax.grid(True)
            ax.legend(fontsize=8)

            # ---------------- (2) Log-log plot ----------------
            ax = axes[1]   # axes[1]
            ax.errorbar(
                flux_mV,     # must be positive for log
                chi_abs,
                yerr=chi_std_val,
                fmt="-o",
                color="tab:blue",
                lw=1.8,
                elinewidth=1,
                capsize=3,
                markersize=4,
                alpha=0.9,
                label="|χZZ| ± std",
            )
            ax.set_yscale("log")
            ax.set_title("Log-log scale")
            ax.set_xlabel("The index")
            ax.set_ylabel("|χZZ| [kHz]")
            ax.grid(True, which="both", ls="--", alpha=0.5)
            ax.legend(fontsize=8)

            fig.suptitle(f"{qubit_name}: χZZ vs Coupler Flux", fontsize=12)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            node.results[f"figure_zz_curve_{qubit_name}"] = fig
            plt.show()


        else:
            print(f"No χZZ analysis or failed fit for {qubit_name} — only raw data plotted.")
        
        # --- 繪製 χZZ 直方圖 (Histogram) ---
        # 1. 建立新的圖表視窗
        fig_hist, ax_h = plt.subplots(figsize=(7, 5))

        # 2. 準備資料（排除 NaN 值以確保繪圖正常）
        chi_raw = ds.isel(qubit=i)["chiZZ"].squeeze().values.flatten()
        chi_raw = chi_raw[~np.isnan(chi_raw)]

        lower_bound = np.percentile(chi_raw, 1)   # 下界
        upper_bound = np.percentile(chi_raw, 99)  # 上界
        chi_data = chi_raw[(chi_raw >= lower_bound) & (chi_raw <= upper_bound)]

        # 2. 繪製直方圖 (維持 Counts，不使用 density=True)
        
        n, bins, patches = ax_h.hist(
            chi_data, 
            bins='auto', 
            color="#B19CD9", # 淡淡的紫色
            edgecolor="#00F8D3", 
            alpha=0.99, 
            label="χZZ distribution"
        )

        # 3. 計算統計參數
        mu, std = norm.fit(chi_data)

        # 4. 生成高斯曲線並「縮放」以匹配次數
        xmin, xmax = ax_h.get_xlim()
        x = np.linspace(xmin, xmax, 10*chi_data.shape[0])
        p = norm.pdf(x, mu, std)

        # 重點：將機率密度縮放成次數
        # 縮放因子 = 資料總筆數 * 組距寬度
        bin_width = bins[1] - bins[0]
        p_scaled = p * len(chi_data) * bin_width

        # 5. 繪製縮放後的高斯曲線
        ax_h.plot(x, p_scaled, 'r', linewidth=2, label=f'$\mu$={mu:.1f}, $\sigma$={std:.1f}')

        # 6. 加入平均值虛線
        ax_h.axvline(mu, color="red", linestyle="dashed", linewidth=1.5)

        # 7. 圖表美化
        ax_h.set_title(f"#={node.parameters.histo_num}", fontsize=12)
        ax_h.set_xlabel("χZZ [kHz]")
        ax_h.set_ylabel("Counts") # 回到次數單位
        ax_h.grid(axis='y', alpha=0.3)
        ax_h.legend()
        plt.suptitle(f"{qubit_name} to {detector_q[0].name} χZZ Histogram")
        plt.tight_layout()
        node.results[f"figure_zz_histogram_{qubit_name}"] = fig_hist
        
        plt.show()
        node.results["fit_results"][qubit_name] = {
                "chiZZ_kHz": mu,
                "chiZZ_kHz_dev": std,
            }



#  %% {Save_results}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        for q in drive_q:
            q.xy.opx_output.upconverter_frequency = drive_LO_original[q.name] # revert the driving LO
        if aswap_dir_update_is_q:
            for q in detector_q:
                q.z.operations['aSWAP'].slope_direction = -1
        else:
            for c in coupler:
                c.coupler.operations['aSWAP'].slope_direction = -1
    node.outcomes = {q.name: "successful" for q in coupler}
    node.results['initial_parameters'] = node.parameters.model_dump()
    node.machine = machine
    node.save()
# %%
