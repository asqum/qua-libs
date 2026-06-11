# %%
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
from quam_libs.macros import qua_declaration
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import (
    fetch_results_as_xarray,
    restore_load_data_id,
    resolve_qubits_from_node,
)
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
import numpy as np
import pandas as pd
# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = ["q1", "q2", "c1"] #["q3"] #["q1"]
    target_qubits: Optional[List[str]] = ["q1"]
    aggressor_qubits: Optional[List[str]] = ["q1"]
    num_averages: int = 10 
    min_flux_offset_in_v: float = -0.5
    max_flux_offset_in_v: float = 0.5
    num_flux_points: int = 51
    frequency_span_in_mhz: float = 7.5 #15
    frequency_step_in_mhz: float = 0.025 #0.1
    flux_point_joint_or_independent: Literal["joint", "independent", ""] = "independent"
    input_line_impedance_in_ohm: float = 50
    line_attenuation_in_db: float = 0
    update_flux_min: bool = False
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    wait_duration:int = 500_000

node = QualibrationNode(name="02b_Resonator_Spectroscopy_vs_Flux_crosstalk", parameters=Parameters())

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
node.machine = machine

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
if any([q.z is None for q in qubits]):
    warnings.warn("Found qubits without a flux line. Skipping")

qubits = [q for q in qubits if q.z is not None]
node.target_qubits = target_qubits = [machine.qubits[q] for q in node.parameters.target_qubits]
node.aggressor_qubits = aggressor_qubits = [machine.qubits[q] for q in node.parameters.aggressor_qubits]
resonators = [target_qubit.resonator for target_qubit in target_qubits]
num_target_qubits = len(target_qubits)

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()

# selected coupler to drive flux from: 
# qp = machine.qubit_pairs["coupler_q1_q2"]


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
# Flux bias sweep in V
dcs = np.linspace(
    node.parameters.min_flux_offset_in_v,
    node.parameters.max_flux_offset_in_v,
    node.parameters.num_flux_points,
)
# The frequency sweep around the resonator resonance frequency
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span / 2, +span / 2, step)

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
update_flux_min = node.parameters.update_flux_min  # Update the min flux point

with program() as multi_res_spec_vs_flux:
    # Declare 'I' and 'Q' and the corresponding streams for the two resonators.
    # For instance, here 'I' is a python list containing two QUA fixed variables.
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_target_qubits)
    dc = declare(fixed)  # QUA variable for the flux bias
    df = declare(int)  # QUA variable for the readout frequency

    for i, target_qubit in enumerate(target_qubits):
        rr = resonators[i]
        # get the readout duration
        readout_duration = rr.operations["readout"].length
        
        for j, aggressor_qubit in enumerate(aggressor_qubits):
            
            # for qubit in qubits:
            # machine.set_all_fluxes(flux_point=flux_point, target=target_qubit)
            # target_qubit.z.settle()
            # target_qubit.align()
                
            # wait(node.parameters.wait_duration //4 ) 
        
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(dc, dcs)):
                    with for_(*from_array(df, dfs)):
                        aggressor_qubit.z.play(
                            "const",
                            amplitude_scale=dc / aggressor_qubit.z.operations["const"].amplitude,
                            duration=(node.parameters.wait_duration + readout_duration + 10000) // 4
                        )
                        # wait on the resonator for flux to settle after the flux pulse has started
                        rr.wait(node.parameters.wait_duration //4 ) 
                        # Update the resonator frequencies for resonator
                        rr.update_frequency(df + rr.intermediate_frequency)
                        # readout the resonator
                        rr.measure("readout", qua_vars=(I[i], Q[i]))
                        # wait for flus to settle after the flux pulse has ended
                        wait(node.parameters.wait_duration //4 )
                        # save data
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
        # Measure sequentially
        align(*[rr.name for rr in resonators])

    with stream_processing():
        n_st.save("n")
        for i, target_qubit in enumerate(target_qubits):
            I_st[i].buffer(len(dfs)).buffer(len(dcs)).buffer(n_avg).map(FUNCTIONS.average()).buffer(len(aggressor_qubits)).save(f"I{i + 1}")
            Q_st[i].buffer(len(dfs)).buffer(len(dcs)).buffer(n_avg).map(FUNCTIONS.average()).buffer(len(aggressor_qubits)).save(f"Q{i + 1}")



# %% {Simulate_or_execute}


if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, multi_res_spec_vs_flux, simulation_config)
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
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(multi_res_spec_vs_flux)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
aggressor_qubit_names = []
for q in aggressor_qubits:
            aggressor_qubit_names.append(q.name)
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
        ds = fetch_results_as_xarray(job.result_handles, target_qubits, {"freq": dfs, "flux": dcs, "aggressor": aggressor_qubit_names})
        # Convert IQ data into volts
        ds = convert_IQ_to_V(ds, target_qubits)
        # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2)
        ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
        # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
        RF_freq = np.array([dfs + q.resonator.RF_frequency for q in target_qubits])
        ds = ds.assign_coords({"freq_full": (["qubit", "freq"], RF_freq * 1e-9)})
        ds.freq_full.attrs["long_name"] = "Frequency"
        ds.freq_full.attrs["units"] = "GHz"
    # Add the dataset to the node
    node.results = {"ds": ds}

# %% {Data_analysis}
# Find the minimum of each frequency line to follow the resonance vs flux
peak_freq = ds.IQ_abs.idxmin(dim="freq")
# Fit to a cosine using the qiskit function: a * np.cos(2 * np.pi * f * t + phi) + offset
fit_osc = fit_oscillation(peak_freq.dropna(dim="flux"), "flux")


import numpy as np
import pandas as pd

def build_fit_results(fit_osc):
    """
    Build a nested dict of fit results including fit coefficients, period, and crosstalk ratios.
    Automatically computes the crosstalk matrix from fitted periods.

    Parameters
    ----------
    fit_osc : xarray.DataArray
        Fit results with dims (qubit, aggressor, fit_vals).

    Returns
    -------
    fit_results : dict
        Nested dict structured as fit_results[target_qubit][aggressor_qubit] = {...}
    crosstalk_df : pandas.DataFrame
        Crosstalk ratio matrix C_ij = - sign * (T_self / T_cross)
    crosstalk_db_df : pandas.DataFrame
        Crosstalk magnitude in decibels: 20*log10(|C_ij|)
    """
    qubits = fit_osc.qubit.values
    aggressors = fit_osc.aggressor.values

    # Step 1: Extract periods T = 1/f
    periods = {}
    for qb in qubits:
        periods[qb] = {}
        for aggr in aggressors:
            try:
                f = float(fit_osc.sel(qubit=qb, aggressor=aggr, fit_vals="f"))
                T = 1.0 / f
            except Exception:
                T = np.nan
            periods[qb][aggr] = T

    # Step 2: Compute crosstalk ratios C_ij = -sign * (T_self / T_cross)
    crosstalk_matrix = np.full((len(qubits), len(aggressors)), np.nan)

    for i, qb in enumerate(qubits):
        T_self = periods[qb].get(qb, np.nan)

        for j, aggr in enumerate(aggressors):
            T_cross = periods[qb].get(aggr, np.nan)

            if np.isfinite(T_self) and np.isfinite(T_cross):
                # --- determine sign from phase alignment ---
                try:
                    phi_self = float(fit_osc.sel(qubit=qb, aggressor=qb, fit_vals="phi"))
                    phi_cross = float(fit_osc.sel(qubit=qb, aggressor=aggr, fit_vals="phi"))
                    sign_ij = np.sign(np.cos(phi_self - phi_cross))
                except Exception:
                    sign_ij = np.nan

                # --- compute signed crosstalk ratio ---
                crosstalk_matrix[i, j] = - sign_ij * (T_self / T_cross)

    # Step 3: Crosstalk matrices (linear + dB)
    crosstalk_df = pd.DataFrame(crosstalk_matrix, index=qubits, columns=aggressors)
    crosstalk_db_df = 20 * np.log10(np.abs(crosstalk_df))
    crosstalk_db_df.replace(-np.inf, np.nan, inplace=True)

    # Step 4: Build nested fit_results dict
    fit_results = {}
    for qb in qubits:
        fit_results[qb] = {}
        for aggr in aggressors:
            try:
                params = fit_osc.sel(qubit=qb, aggressor=aggr)
                a = float(params.sel(fit_vals="a"))
                f = float(params.sel(fit_vals="f"))
                phi = float(params.sel(fit_vals="phi"))
                offset = float(params.sel(fit_vals="offset"))
                T = periods[qb][aggr]
                Cij = crosstalk_df.loc[qb, aggr]
                Cij_dB = crosstalk_db_df.loc[qb, aggr]

                fit_results[qb][aggr] = {
                    "fit_coefficients": {"a": a, "f": f, "phi": phi, "offset": offset},
                    "period": T,
                    "crosstalk_coefficient": Cij,
                    "crosstalk_dB": Cij_dB,
                }
            except Exception:
                fit_results[qb][aggr] = {
                    "fit_coefficients": None,
                    "period": np.nan,
                    "crosstalk_coefficient": np.nan,
                    "crosstalk_dB": np.nan,
                }

    return fit_results, crosstalk_df, crosstalk_db_df

fit_results, crosstalk_df, crosstalk_db_df = build_fit_results(fit_osc)

print("\n==============================")
print(" Crosstalk Matrix (Linear Ratios)")
print("==============================")
print(crosstalk_df.round(4).to_string(index=True, header=True))

print("\n==============================")
print(" Crosstalk Matrix (Magnitude in dB)")
print("==============================")
print(crosstalk_db_df.round(2).to_string(index=True, header=True))

node.results["fit_results"] = fit_results

# %% {Plotting}
import matplotlib.pyplot as plt
import numpy as np

data_var = "IQ_abs"
tqs = target_qubits
aqs = aggressor_qubits


fig, axes = plt.subplots(
    nrows=len(tqs), ncols=len(aqs),
    figsize=(4 * len(aqs), 3.5 * len(tqs)),
    squeeze=False
)

vmin = ds[data_var].min().item()
vmax = ds[data_var].max().item()

for i, qb in enumerate(tqs):
    for j, aggr in enumerate(aqs):
        ax = axes[i, j]
        ds_sel = ds.sel(qubit=qb.id, aggressor=aggr.id)

        # Choose offset based on flux_point
        if flux_point == "independent":
            offset = qb.z.independent_offset
        elif flux_point == "joint":
            offset = qb.z.joint_offset
        else:
            offset = 0.0

        # Plot IQ_abs heatmap
        pcm = ds_sel[data_var].plot(
            x="flux",
            y="freq_full",
            ax=ax,
            add_colorbar=False,
            cmap="viridis",
            robust=True,
        )

        # Mark the idle offset
        ax.axvline(
            offset,
            linestyle="--",
            linewidth=2,
            color="r",
        )
        try:
            # Extract measured minima
            peak = peak_freq.sel(qubit=qb.id, aggressor=aggr.id).dropna(dim="flux")

            # Plot the measured resonance (minima)
            # ax.plot(peak.flux, (peak + qb.resonator.RF_frequency) / 1e9, "o", color="white", markersize=3, label="Measured min")

            # Extract fit parameters from fit_osc
            params = fit_osc.sel(qubit=qb.id, aggressor=aggr.id)
            a = float(params.sel(fit_vals="a"))
            f = float(params.sel(fit_vals="f"))
            phi = float(params.sel(fit_vals="phi"))
            offset_fit = float(params.sel(fit_vals="offset"))
            T = 1/f
                            # Crosstalk ratio
            Cij = crosstalk_df.loc[qb.id, aggr.id] if qb.id in crosstalk_df.index and aggr.id in crosstalk_df.columns else np.nan
            Cij_db = crosstalk_db_df.loc[qb.id, aggr.id] if qb.id in crosstalk_df.index and aggr.id in crosstalk_df.columns else np.nan


            # Generate smooth fitted curve
            flux_fit = np.linspace(peak.flux.min().item(), peak.flux.max().item(), 400)
            fit_curve = a * np.cos(2 * np.pi * f * flux_fit + phi) + offset_fit

            # Overlay fit line
            ax.plot(flux_fit, (fit_curve + qb.resonator.RF_frequency) / 1e9, "-", color="red", linewidth=2, label="Fit")

            # Add fit frequency info to title
            ax.set_title(
                f"{aggr.id} → {qb.id}\n"
                f"Period ={T:.3f}V, C={Cij * 100:.1f}% ({np.abs(Cij_db):.1f} dB)",
                fontsize=10
            )

        except Exception:
            # If fit or data not found
            ax.set_title(f"{aggr.id} → {qb.id}\n(no fit data)")

        # Labels and title
        ax.set_xlabel("Absolute flux [V]")
        ax.set_ylabel("Frequency [GHz]")


plt.tight_layout()
plt.show()

node.results["figure"] = fig

# %% Update state
"""Update the relevant parameters if the qubit data analysis was successful."""
with node.record_state_updates():
    for target_qubit_name, target_qubit_results in node.results["fit_results"].items():
        for aggressor_qubit_name, fit_result in target_qubit_results.items():
            if target_qubit_name == aggressor_qubit_name:
                continue

            # Find the qubit objects
            target_qubit = node.machine.qubits[target_qubit_name]
            aggressor_qubit = node.machine.qubits[aggressor_qubit_name]

            target_output = target_qubit.z.opx_output
            aggressor_output = aggressor_qubit.z.opx_output

            # Update crosstalk coefficients
            if target_output.fem_id == aggressor_output.fem_id and target_output.controller_id == aggressor_output.controller_id:
                if not target_output.crosstalk:
                    target_output.crosstalk = {}
                if not aggressor_output.port_id in target_output.crosstalk or np.isnan(target_output.crosstalk[aggressor_output.port_id]):
                    target_output.crosstalk[aggressor_output.port_id] = 0
                target_output.crosstalk[aggressor_output.port_id] += fit_result["crosstalk_coefficient"]

            else:
                node.log(f"Couldn't compensate crosstalk between {target_qubit.name} and {aggressor_qubit.name}, "
                            f"since they are on different fems ({target_output.controller_id, target_output.fem_id} and "
                            f"{aggressor_output.controller_id, aggressor_output.fem_id}) respectively.")

    
# %% {Save_results}
node.outcomes = {q.name: "successful" for q in qubits}
node.results["initial_parameters"] = node.parameters.model_dump()
node.machine = machine
node.save()

# %%
