# %%
"""
Two-Qubit ZZ Coupling Measurement

This experiment measures the static ZZ interaction (cross-Kerr coupling) between two qubits connected via a tunable coupler. 
The ZZ coupling quantifies the conditional frequency shift experienced by one qubit depending on the state of the other qubit. 
For each coupler flux bias, we measure, population of the target qubit as a function of idle time in the JAZZ sequence. 
The ZZ interaction introduces a relative phase ϕ accumulated on the superposition state of target qubit, |0⟩ + e^iϕ |1⟩, which can be read out, 
P|0⟩ = (1 − cos ϕ) /2. 
Therefore, the population is oscillating with the duration t, where t/2 is defined as the pulse duration of the applied Z-pulse.
The phase can ebe simply expressed as ϕ =  ζ_ZZ t/2 + ϕ0, where the first term arises from the ZZ interaction and second term ϕ0 is due to any detuning between 
the qubit drive frequency and the qubit frequency (f0). 
With the frequency f extracted from fitting the the oscillating signal P|0⟩, the ZZ interaction is calculated as ζ = 2(f − f0).

The process involves:
1. Performing a JAZZ (Joint Amplification of ZZ interaction) pulse sequence (PHYS. REV. X 14, 041050 (2024)).
2. Repeating the measurement while varying the coupler flux bias.


The outcome of this measurement will be used to:
1. Identify the coupler flux bias point where ζ_ZZ ≈ 0 (the “zero-ZZ” operating point).
2. Quantify unwanted static interactions affecting single- and two-qubit gate fidelities.

Prerequisites:
- Calibrated single-qubit X/Y gates for both qubits.
- Known qubit frequencies.
- Qubit frq vs coupler flux trend rougnly known.

Outcomes:
- ζ_ZZ as a function of coupler flux bias.
- Identification of the zero-ZZ operating point.
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
from quam_libs.lib.pulses import FluxPulse
from quam_libs.lib.fit import fit_oscillation_decay_exp, oscillation_decay_exp

# %% {Node_parameters}
qubit_pair_indexes = [3]  # The indexes of the qubit pairs to measure
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = ["coupler_q%s_q%s"%(i,i+1) for i in qubit_pair_indexes]
    num_averages: int = 20
    flux_point_joint_or_independent_or_pairwise: Literal["joint", "independent", "pairwise"] = "joint"
    reset_type: Literal['active', 'thermal'] = "active"
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None
    frequency_detuning_in_mhz: float = 1.0
    """Frequency detuning in MHz. Default is 1.0 MHz."""
    min_wait_time_in_ns: int = 16
    """Minimum wait time in nanoseconds. Default is 16."""
    max_wait_time_in_ns: int = 5016
    """Maximum wait time in nanoseconds. Default is 5000."""
    wait_time_step_in_ns: int = 50
    """Step size for the wait time scan in nanoseconds. Default is 60."""
    flux_span: float = 0.25
    """Span of flux values to sweep in volts. Default is 0.01 V."""
    flux_num: int = 101
    """Number of flux points to sample. Default is 21."""
    use_state_discrimination: bool = False

    

node = QualibrationNode(
    name="67b_JAZZ_ZZ_coupling", parameters=Parameters()
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


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
detuning = int(1e6 * node.parameters.frequency_detuning_in_mhz)
flux_point = node.parameters.flux_point_joint_or_independent_or_pairwise  # 'independent' or 'joint' or 'pairwise'

# Loop parameters
fluxes_coupler = np.linspace(
        -0.05,
        0.3,
        node.parameters.flux_num,
    )
idle_times = np.arange(
    node.parameters.min_wait_time_in_ns // 4,
    node.parameters.max_wait_time_in_ns // 4,
    node.parameters.wait_time_step_in_ns // 4,
)
reset_coupler_bias = True

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
    
    
    for i, qp in enumerate(qubit_pairs):
        q_control = qp.qubit_control
        q_target = qp.qubit_target
        XY_delay = q_target.xy.opx_output.delay + 4  # Delay to account delayed pulses in the XY channel
        # Bring the active qubits to the minimum frequency point
        machine.set_all_fluxes(flux_point, qp)
        if reset_coupler_bias:
            qp.coupler.set_dc_offset(0.0)
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
                        active_reset(qp.qubit_control)
                        active_reset(qp.qubit_target)
                    else:
                        wait(qp.qubit_control.thermalization_time * u.ns)
                    qp.align()
                    
                    # Reset the frames of both qubits
                    reset_frame(qp.qubit_target.xy.name)
                    reset_frame(qp.qubit_control.xy.name)
                    
                    # pi pulse on target qubit
                    qp.qubit_target.xy.play("x90")
                    qp.align()
                    
                    # Coupler flux pulse
                    qp.coupler.play(
                        "const", amplitude_scale=flux_coupler / qp.coupler.operations["const"].amplitude, duration=t_half
                    )
                    qp.align()
                    
                    # Echo pulse
                    qp.qubit_control.xy.play("x180")
                    qp.qubit_target.xy.play("x180")
                    qp.align()
                    
                    # Coupler flux pulse
                    qp.coupler.play(
                        "const", amplitude_scale=flux_coupler / qp.coupler.operations["const"].amplitude, duration=t_half
                    )
                    qp.align()
                    
                    # rotate the frame
                    qp.qubit_target.xy.frame_rotation_2pi(phi)
                    # Tomographic rotation on the target qubit
                    qp.qubit_target.xy.play("x90")
                    qp.align() 
                    
                    # target qubit readout
                    if node.parameters.use_state_discrimination:
                        readout_state(q_target, current_state[i])
                        assign(state_target[i], init_state[i] ^ current_state[i])
                        save(state_target[i], state_st_target[i])
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
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {"idle_time": idle_times, "flux_coupler": fluxes_coupler})
    else:
        ds, machine = load_dataset(node.parameters.load_data_id)


# %% {Data_analysis}
node.results = {"ds": ds}
node.results["fit_results"] = {}   # optional if you also store raw fits
node.results["analysis_success"] = {}  # for success flags per pair

if not node.parameters.simulate:
    try:
        print("Starting analysis...")

        flux_coupler_full = np.array([
            fluxes_coupler + qp.coupler.decouple_offset for qp in qubit_pairs
        ])
        ds = ds.assign_coords({
            "flux_coupler_full": (["qubit", "flux_coupler"], flux_coupler_full)
        })
        ds = ds.assign_coords(idle_time=4 * idle_times / 1e3)

        for i, qp in enumerate(qubit_pairs):
            print(f"Analyzing qubit pair {i}: {qp.id if hasattr(qp, 'id') else qp}")

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

            # ---------------- SAVE PER-PAIR RESULTS ----------------
            node.results["fit_results"][qp.name] = {
                "chiZZ_max_flux": flux_val_max,
                "chiZZ_min_flux": flux_val_min,
                "chiZZ_max_value": chi_max,
                "chiZZ_min_value": chi_min,
            }

            # ---------------- SAVE FIT SUCCESS FLAG ----------------
            node.results["analysis_success"][qp.name] = "successful" if fit_ok else "failed"

            print(f"  → max |χZZ|={chi_max:.2f} kHz @ {flux_val_max:.3f}")
            print(f"  → χZZ≈0={chi_min:.2f} kHz @ {flux_val_min:.3f}")
            print(f"  → fit status: {'success' if fit_ok else '❌ failed'}")

        print("Analysis complete ")

    except Exception as e:
        node.results = {"ds": ds}
        node.results["analysis_error"] = str(e)
        analysis_successful = False
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

            fig.suptitle(f"{qubit_name}: χZZ vs Coupler Flux", fontsize=12)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            node.results[f"figure_zz_curve_{qubit_name}"] = fig
            plt.show()


            # --- Time-domain fits at χZZ max and min ---
            for label, flux_val, chi_val in [
                ("max", flux_val_max, chi_max),
                ("min", flux_val_min, chi_min),
            ]:
                flux_idx = int((abs(flux_full - flux_val)).argmin(dim="flux_coupler").item())
                meas = ds_pair["state_target"].isel(flux_coupler=flux_idx)
                fit = ds_pair["state_target_fit"].isel(flux_coupler=flux_idx)
                idle_time = ds_pair["idle_time"]

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
                axes[0].set_title(f"Measured ({qubit_name})")
                axes[1].set_title("Fitted")
                for ax_ in axes:
                    ax_.legend()
                    ax_.set_xlabel("Idle time [µs]")
                    ax_.set_ylabel("State")
                fig.suptitle(
                    f"{qubit_name} ({label} |χZZ|): Flux={flux_val:.3f}, χZZ={chi_val:.2f} kHz",
                    fontsize=12,
                )
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                node.results[f"figure_fit_{label}_chiZZ_{qubit_name}"] = fig
                plt.show()

        else:
            print(f"No χZZ analysis or failed fit for {qubit_name} — only raw data plotted.")

#%% {Update state}
if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for i, qp in enumerate(qubit_pairs):
                qp.extras["ZZ_zero_flux"] = node.results['fit_results'][qp.name]["chiZZ_min_flux"]
                qp.coupler.decouple_offset = node.results['results'][qp.name]["chiZZ_min_flux"]
#  %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {q.name: "successful" for q in qubit_pairs}
    node.results['initial_parameters'] = node.parameters.model_dump()
    node.machine = machine
    node.save()
# %%
