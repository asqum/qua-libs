# %%
"""
Two-Qubit ZZ Coupling Measurement (JAZZ, differential control-state readout)

Corrected variant of 67b_JAZZ_ZZ_coupling: sweeps control qubit prepared in |0⟩ and |1⟩,
then extracts χZZ from the difference of fitted oscillation frequencies divided by 2.
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import active_reset, readout_state, readout_state_gef, active_reset_gef, active_reset_simple
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
import warnings
from qualang_tools.bakery import baking
from quam_libs.lib.fit import fit_oscillation, oscillation, fix_oscillation_phi_2pi
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from scipy.optimize import curve_fit
from quam_libs.components.gates.two_qubit_gates import CZGate
from quam_libs.lib.pulses import FluxPulse
from quam_libs.lib.fit import fit_oscillation_decay_exp, oscillation_decay_exp
import xarray as xr

# %% {Node_parameters}
qubit_pair_indexes = [4]  # The indexes of the qubit pairs to measure
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = ["coupler_q%s_q%s"%(i,i+1) for i in qubit_pair_indexes]
    num_averages: int = 200
    flux_point_joint_or_independent_or_pairwise: Literal["joint", "independent", "pairwise"] = "joint"
    reset_type: Literal['active', 'thermal'] = 'active'
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None
    frequency_detuning_in_mhz: float = 4.0
    """Frequency detuning in MHz. Default is 1.0 MHz."""
    min_wait_time_in_ns: int = 16
    """Minimum wait time in nanoseconds. Default is 16."""
    max_wait_time_in_ns: int = 816
    """Maximum wait time in nanoseconds. Default is 5000."""
    wait_time_step_in_ns: int = 8
    """Step size for the wait time scan in nanoseconds. Default is 60."""
    flux_span: float = 0.6
    """Span of flux values to sweep in volts. Default is 0.01 V."""
    flux_num: int = 501
    """Number of flux points to sample. Default is 21."""
    use_state_discrimination: bool = True

    

node = QualibrationNode(
    name="67bx_JAZZ_ZZ_coupling_correct", parameters=Parameters()
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

num_qubit_pairs = len(qubit_pairs)

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
fluxes_coupler = np.linspace(
        -node.parameters.flux_span / 2,
        node.parameters.flux_span / 2,
        node.parameters.flux_num,
    )
idle_times = np.arange(
    node.parameters.min_wait_time_in_ns // 4,
    node.parameters.max_wait_time_in_ns // 4,
    node.parameters.wait_time_step_in_ns // 4,
)

with program() as Ramsey_ZZ_coupling:
    n = declare(int)
    flux_coupler = declare(float)
    flux_qubit = declare(float)
    n_st = declare_stream()
    control_initial = declare(int)  # initial state of the control qubit
    t = declare(int)  # QUA variable for the idle time
    t_half = declare(int)
    phi = declare(fixed)  # QUA variable for dephasing the second pi/2 pulse (virtual Z-rotation)
    current_state = [declare(int) for _ in range(num_qubit_pairs)]
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    I_target = [declare(float) for _ in range(num_qubit_pairs)]
    Q_target = [declare(float) for _ in range(num_qubit_pairs)]
    state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    I_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    Q_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    
    
    for i, qp in enumerate(qubit_pairs):
        q_control = qp.qubit_control
        q_target = qp.qubit_target
        # Bring the active qubits to the minimum frequency point
        if not node.parameters.simulate:
            machine.set_all_fluxes(flux_point, qp)
            wait(1000)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(flux_coupler, fluxes_coupler)):
                with for_(*from_array(t, idle_times)):
                    with for_(*from_array(control_initial, [0, 1])):
                        # Rotate the frame of the second x90 gate to implement a virtual Z-rotation
                        # 4*tau because tau was in clock cycles and 1e-9 because tau is ns                    
                        assign(phi, Cast.mul_fixed_by_int(detuning * 1e-9, 4 * t))
                        
                        assign(t_half, t/2)
                        
                        if not node.parameters.simulate:
                            if node.parameters.reset_type == "active":
                                active_reset(qp.qubit_control)
                                active_reset(qp.qubit_target)
                            else:
                                wait(qp.qubit_control.thermalization_time * u.ns)
                        qp.align()
                        
                        # Reset the frames of both qubits
                        reset_frame(qp.qubit_target.xy.name)
                        reset_frame(qp.qubit_control.xy.name)

                        # Prepare control qubit in |1⟩ when control_initial == 1
                        qp.qubit_control.xy.play("x180", condition=control_initial == 1)
                        qp.align()
                        
                        # pi pulse on target qubit
                        qp.qubit_target.xy.play("x90")
                        qp.align()
                        
                        # Coupler flux pulse
                        qp.coupler.play(
                            "const", amplitude_scale=flux_coupler / qp.coupler.operations["const"].amplitude, duration=t_half
                        )
                        qp.qubit_target.xy.wait(t_half)
                        qp.qubit_control.xy.wait(t_half)
                        
                        # Echo pulse
                        qp.qubit_control.xy.play("x180")
                        qp.qubit_target.xy.play("x180")
                        qp.coupler.wait(qp.qubit_target.xy.operations["x180"].length//4)
                        
                        # Coupler flux pulse
                        qp.coupler.play(
                            "const", amplitude_scale=flux_coupler / qp.coupler.operations["const"].amplitude, duration=t_half
                        )
                        qp.qubit_target.xy.wait(t_half)
                        qp.qubit_control.xy.wait(t_half)
                        
                        # rotate the frame
                        qp.qubit_target.xy.frame_rotation_2pi(phi)
                        # Tomographic rotation on the target qubit
                        qp.qubit_target.xy.play("x90")
                        qp.align() 
                        
                        # target qubit readout
                        if node.parameters.use_state_discrimination:
                            readout_state(q_target, current_state[i])
                            save(current_state[i], state_st_target[i])
                            reset_frame(q_target.xy.name)
                        else:
                            q_target.resonator.measure("readout", qua_vars=(I_target[i], Q_target[i]))
                            save(I_target[i], I_st_target[i])
                            save(Q_target[i], Q_st_target[i])
                        align()
        
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            if node.parameters.use_state_discrimination:
                state_st_target[i].buffer(2).buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"state_target{i + 1}")
            else:
                I_st_target[i].buffer(2).buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"I_target{i + 1}")
                Q_st_target[i].buffer(2).buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"Q_target{i + 1}")

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
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(Ramsey_ZZ_coupling)

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
        ds = fetch_results_as_xarray(
            job.result_handles,
            qubit_pairs,
            {"control_state": [0, 1], "idle_time": idle_times, "flux_coupler": fluxes_coupler},
        )
    else:
        load_data_id = node.parameters.load_data_id
        node = node.load_from_id(load_data_id)
        ds = node.results["ds"]
        restore_load_data_id(node, load_data_id)
        machine = node.machine
        qubit_pairs = resolve_qubit_pairs_from_node(machine, node)
# %% {Data_analysis}
node.results = {"ds": ds}
node.results["fit_results"] = {}
node.results["analysis_success"] = {}

if not node.parameters.simulate:
    try:
        print("Starting analysis...")
        target_signal_name = "state_target" if "state_target" in ds.data_vars else "I_target"
        print(f"Using '{target_signal_name}' for ZZ analysis.")

        if node.parameters.load_data_id is None:
            flux_coupler_full = np.array([
                fluxes_coupler + qp.coupler.decouple_offset for qp in qubit_pairs
            ])
            ds = ds.assign_coords({
                "flux_coupler_full": (["qubit", "flux_coupler"], flux_coupler_full)
            })
            ds = ds.assign_coords(idle_time=4 * idle_times / 1e3)
        else:
            if "flux_coupler_full" not in ds.coords:
                fluxes_coupler = np.linspace(
                    -node.parameters.flux_span / 2,
                    node.parameters.flux_span / 2,
                    node.parameters.flux_num,
                )
                flux_coupler_full = np.array([
                    fluxes_coupler + qp.coupler.decouple_offset for qp in qubit_pairs
                ])
                ds = ds.assign_coords({
                    "flux_coupler_full": (["qubit", "flux_coupler"], flux_coupler_full)
                })
            if "idle_time" not in ds.coords:
                idle_times = np.arange(
                    node.parameters.min_wait_time_in_ns // 4,
                    node.parameters.max_wait_time_in_ns // 4,
                    node.parameters.wait_time_step_in_ns // 4,
                )
                ds = ds.assign_coords(idle_time=4 * idle_times / 1e3)

        fit_curve_list = []
        chiZZ_list = []
        chiZZ_std_list = []

        for i, qp in enumerate(qubit_pairs):
            print(f"Analyzing qubit pair {i}: {qp.id if hasattr(qp, 'id') else qp}")
            target_signal_pair = ds[target_signal_name].sel(qubit=qp.name)

            fit_data = fit_oscillation_decay_exp(target_signal_pair, "idle_time")

            fit_ok = (
                np.isfinite(fit_data.sel(fit_vals="f")).any()
                and not np.isnan(fit_data.sel(fit_vals="f")).all()
            )

            fitted_pair = oscillation_decay_exp(
                target_signal_pair.idle_time,
                fit_data.sel(fit_vals="a"),
                fit_data.sel(fit_vals="f"),
                fit_data.sel(fit_vals="phi"),
                fit_data.sel(fit_vals="offset"),
                fit_data.sel(fit_vals="decay"),
            )
            fit_curve_list.append(fitted_pair.expand_dims(qubit=[qp.name]))

            f0 = fit_data.sel(fit_vals="f", control_state=0)
            f1 = fit_data.sel(fit_vals="f", control_state=1)
            chiZZ_pair = (f1 - f0) * 1e3  # MHz → kHz, differential χZZ
            chiZZ_list.append(chiZZ_pair.expand_dims(qubit=[qp.name]))

            chiZZ_std_pair = 1e3 / 2 * np.sqrt(
                fit_data.sel(fit_vals="f_f", control_state=0)
                + fit_data.sel(fit_vals="f_f", control_state=1)
            )
            chiZZ_std_list.append(chiZZ_std_pair.expand_dims(qubit=[qp.name]))

            chiZZ_std_filt = chiZZ_std_pair.where(chiZZ_std_pair < chiZZ_std_pair.median() * 2)
            chiZZ_filt = chiZZ_pair.where(~np.isnan(chiZZ_std_filt))

            max_idx = abs(chiZZ_filt).argmax(dim="flux_coupler").item()
            min_idx = abs(chiZZ_filt).argmin(dim="flux_coupler").item()

            flux_full = ds["flux_coupler_full"].sel(qubit=qp.name)
            flux_val_max = float(flux_full.isel(flux_coupler=max_idx))
            flux_val_min = float(flux_full.isel(flux_coupler=min_idx))
            chi_max = float(chiZZ_pair.isel(flux_coupler=max_idx))
            chi_min = float(chiZZ_pair.isel(flux_coupler=min_idx))

            node.results["fit_results"][qp.name] = {
                "chiZZ_max_flux": flux_val_max,
                "chiZZ_min_flux": flux_val_min,
                "chiZZ_max_value": chi_max,
                "chiZZ_min_value": chi_min,
            }
            node.results["analysis_success"][qp.name] = "successful" if fit_ok else "failed"

            print(f"  → max |χZZ|={chi_max:.2f} kHz @ {flux_val_max:.3f}")
            print(f"  → χZZ≈0={chi_min:.2f} kHz @ {flux_val_min:.3f}")
            print(f"  → fit status: {'success' if fit_ok else 'failed'}")

        ds["target_signal_fit"] = xr.concat(fit_curve_list, dim="qubit")
        ds["chiZZ"] = xr.concat(chiZZ_list, dim="qubit")
        ds["chiZZ_std"] = xr.concat(chiZZ_std_list, dim="qubit")
        node.results["ds"] = ds
        print("Analysis complete ")

    except Exception as e:
        node.results["analysis_error"] = str(e)
        node.results["ds"] = ds
        print(f"Analysis failed : {e}")

# %% {plotting}

if not node.parameters.simulate:
    for i, qp in enumerate(qubit_pairs):
        qubit_name = qp.name
        fit_status = node.results.get("analysis_success", {}).get(qubit_name, "failed")

        # =====================================================
        # --- Always plot raw Ramsey / state_target heatmaps ---
        # =====================================================
        print(f"Plotting raw data for {qubit_name}")
        ds_pair = ds.isel(qubit=i)
        flux_full = ds_pair["flux_coupler_full"]
        flux_mV = 1e3 * flux_full.squeeze()
        idle_time_us = ds_pair["idle_time"]
        target_signal_name = "state_target" if "state_target" in ds_pair.data_vars else "I_target"
        raw_data_label = "State probability" if target_signal_name == "state_target" else "I quadrature [a.u.]"

        for ctrl in [0, 1]:
            fig, ax = plt.subplots(figsize=(6, 4))
            data = ds_pair[target_signal_name].sel(control_state=ctrl)
            im = ax.pcolormesh(
                flux_mV,
                idle_time_us,
                data.transpose(),
                shading="auto",
                cmap="viridis",
                vmin=0 if target_signal_name == "state_target" else None,
                vmax=1 if target_signal_name == "state_target" else None,
            )
            fig.colorbar(im, ax=ax, label=raw_data_label)
            ax.set_xlabel("Coupler flux [mV]")
            ax.set_ylabel("Idle time [µs]")
            ax.set_title(f"{qubit_name} – Raw JAZZ (control={ctrl})", fontsize=10)
            plt.tight_layout()
            node.results[f"figure_raw_{qubit_name}_ctrl{ctrl}"] = fig
            plt.show()

        if fit_status == "successful" and "chiZZ" in ds:
            print(f"Plotting χZZ analysis for {qubit_name}")
            ds_pair = ds.isel(qubit=i)
            chi = ds_pair["chiZZ"]
            chi_std = ds_pair["chiZZ_std"]
            flux_full = ds_pair["flux_coupler_full"]
            flux_mV = 1e3 * flux_full.squeeze()

            res = node.results["fit_results"][qubit_name]
            flux_val_max = res["chiZZ_max_flux"]
            flux_val_min = res["chiZZ_min_flux"]
            chi_max = res["chiZZ_max_value"]
            chi_min = res["chiZZ_min_value"]

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
            ax.axvline(1e3 * flux_val_max, color="k", linestyle="--", lw=1.2, alpha=0.8, label="max |χZZ|")
            ax.axvline(1e3 * flux_val_min, color="gray", linestyle="--", lw=1.2, alpha=0.6, label="min |χZZ|")
            ax.axhline(0, color="k", lw=1, alpha=0.4)
            ax.set_title("Linear scale")
            ax.set_xlabel("Coupler flux [mV]")
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
            ax.set_xlabel("|Coupler flux| [mV]")
            ax.set_ylabel("|χZZ| [kHz]")
            ax.grid(True, which="both", ls="--", alpha=0.5)
            ax.legend(fontsize=8)

            coupler_offset_mV = 1e3 * qp.coupler.decouple_offset
            fig.suptitle(
                f"{qubit_name}: χZZ vs Coupler Flux (Current coupler offset = {coupler_offset_mV:.3f} mV)",
                fontsize=12,
            )

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            node.results[f"figure_zz_curve_{qubit_name}"] = fig
            plt.show()


            # --- Time-domain fits at χZZ max and min ---
            for label, flux_val, chi_val in [
                ("max", flux_val_max, chi_max),
                ("min", flux_val_min, chi_min),
            ]:
                flux_idx = int((abs(flux_full - flux_val)).argmin(dim="flux_coupler").item())
                idle_time = ds_pair["idle_time"]

                for ctrl in [0, 1]:
                    meas = ds_pair[target_signal_name].isel(flux_coupler=flux_idx).sel(control_state=ctrl)
                    fit = ds_pair["target_signal_fit"].isel(flux_coupler=flux_idx).sel(control_state=ctrl)

                    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True)
                    axes[0].plot(
                        idle_time,
                        meas,
                        "-",
                        alpha=0.8,
                    )
                    axes[1].plot(
                        idle_time,
                        fit,
                        "-",
                        lw=2,
                    )
                    axes[0].set_title(f"Measured ({qubit_name}, ctrl={ctrl})")
                    axes[1].set_title("Fitted")
                    for ax_ in axes:
                        ax_.set_xlabel("Idle time [µs]")
                        ax_.set_ylabel("State")
                    fig.suptitle(
                        f"{qubit_name} ({label} |χZZ|), ctrl={ctrl}: Flux={flux_val:.3f}, χZZ={chi_val:.2f} kHz",
                        fontsize=12,
                    )
                    plt.tight_layout(rect=[0, 0, 1, 0.96])
                    node.results[f"figure_fit_{label}_chiZZ_{qubit_name}_ctrl{ctrl}"] = fig
                    plt.show()

        else:
            print(f"No χZZ analysis or failed fit for {qubit_name} — only raw data plotted.")

#%% {Update state}
if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for i, qp in enumerate(qubit_pairs):
                qp.extras["ZZ_zero_flux"] = node.results['fit_results'][qp.name]["chiZZ_min_flux"]
                qp.coupler.decouple_offset = node.results['fit_results'][qp.name]["chiZZ_min_flux"]
#  %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {q.name: "successful" for q in qubit_pairs}
    node.results['initial_parameters'] = node.parameters.model_dump()
    node.machine = machine
    node.save()
# %%