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
qubit_pair_indexes = [1]  # The indexes of the qubit pairs to measure
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = ["coupler_q%s_q%s"%(i,i+1) for i in qubit_pair_indexes]
    num_averages: int = 50
    flux_point_joint_or_independent_or_pairwise: Literal["joint", "independent", "pairwise"] = "joint"
    reset_type: Literal['active', 'thermal'] = "thermal"
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None
    coupler_flux_min: float = -0.02  # relative to the coupler set point
    coupler_flux_max: float = 0.01 # relative to the coupler set point

    coupler_flux_step: float = 0.001
    qubit_flux_span: float = 0.03  # relative to the known/calculated detuning between the qubits
    qubit_flux_step: float = 0.001
    use_state_discrimination: bool = True
    pulse_duration_ns: int = 88
    operation: Literal["Cz_flattop", "Cz_unipolar", "Cz_bipolar"] = "Cz_unipolar"
    """Type of CZ operation to perform. Options are 'cz_flattop', 'cz_unipolar', or 'cz_bipolar'. Default is 'cz_unipolar'."""
    
    

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
fluxes_coupler = np.arange(
    node.parameters.coupler_flux_min, node.parameters.coupler_flux_max + 0.0001, node.parameters.coupler_flux_step
)
fluxes_qubit = np.arange(
    -node.parameters.qubit_flux_span / 2, node.parameters.qubit_flux_span / 2 + 0.0001, node.parameters.qubit_flux_step
)
fluxes_qp = {}
for qp in qubit_pairs:
    # estimate the flux shift to get the control qubit to the target qubit frequency
    fluxes_qp[qp.name] = fluxes_qubit + qp.detuning

reset_coupler_bias = False
operation_name = node.parameters.operation

with program() as CPhase_Oscillations:
    n = declare(int)
    flux_coupler = declare(float)
    flux_qubit = declare(float)
    comp_flux_qubit = declare(float)
    n_st = declare_stream()
    qua_pulse_duration = declare(int, value=node.parameters.pulse_duration_ns // 4)
    
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
            with for_(*from_array(flux_coupler, fluxes_coupler)):
                with for_(*from_array(flux_qubit,fluxes_qp[qp.name])):
                        # reset
                        if node.parameters.reset_type == "active":
                            # active_reset(qp.qubit_control)
                            # active_reset(qp.qubit_target)
                            active_reset_gef(qp.qubit_control)
                            active_reset_gef(qp.qubit_target)
                        else:
                            wait(qp.qubit_control.thermalization_time * u.ns)
                            wait(qp.qubit_target.thermalization_time * u.ns)
                        
                        if "coupler_qubit_crosstalk" in qp.extras:
                            assign(comp_flux_qubit, flux_qubit + qp.extras["coupler_qubit_crosstalk"] * flux_coupler)
                        else:
                            print("No crosstalk compensated")
                            assign(comp_flux_qubit, flux_qubit)

                        # state preparation
                        qp.qubit_control.xy.play("x180")
                        qp.qubit_target.xy.play("x180")
                        align()
                        qp.qubit_control.z.play(
                            "const",
                            amplitude_scale=comp_flux_qubit / qp.qubit_control.z.operations["const"].amplitude,
                            duration=qua_pulse_duration,
                        )
                        qp.coupler.play(
                            "const",
                            amplitude_scale=flux_coupler / qp.coupler.operations["const"].amplitude,
                            duration=qua_pulse_duration,
                        )
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
                state_st_control[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).buffer(n_avg).save(f"state_control{i + 1}")
                state_st_target[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).buffer(n_avg).save(f"state_target{i + 1}")
            else:
                I_st_control[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).buffer(n_avg).save(f"I_control{i + 1}")
                Q_st_control[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).buffer(n_avg).save(f"Q_control{i + 1}")
                I_st_target[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).buffer(n_avg).save(f"I_target{i + 1}")
                Q_st_target[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).buffer(n_avg).save(f"Q_target{i + 1}")

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
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {"flux_qubit": fluxes_qubit, "flux_coupler": fluxes_coupler, "N": np.linspace(1, n_avg, n_avg)})
    else:
        ds, machine = load_dataset(node.parameters.load_data_id)
        
    node.results = {"ds": ds}
    node.results["results"] = {}

    
# %% data processing
if not node.parameters.simulate:
    detuning = np.array([-fluxes_qp[qp.name] ** 2 * qp.qubit_control.freq_vs_flux_01_quad_term  for qp in qubit_pairs])
    ds = ds.assign_coords({"detuning": (["qubit", "flux_qubit"], detuning)})
    flux_coupler_full = np.array([fluxes_coupler + qp.coupler.decouple_offset for qp in qubit_pairs])
    ds = ds.assign_coords({"flux_coupler_full": (["qubit", "flux_coupler"], flux_coupler_full)})
    flux_qubit_full = np.array([fluxes_qp[qp.name] for qp in qubit_pairs])
    ds = ds.assign_coords({"flux_qubit_full": (["qubit", "flux_qubit"], flux_qubit_full)})# %%Data analysis

#%%
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
#%%

for qp in qubit_pairs:
    qpname = qp.name
    # --- Select data ---
    P11 = ds.P11.sel(qubit=qpname)
    flux_qb = ds.flux_qubit
    flux_cpl_full = ds.flux_coupler_full

    # Smooth to suppress noise ---
    P11_smooth = P11.copy(data=nd.gaussian_filter(P11.values, sigma=2))

    # Find blob minimum (global)
    i_cpl_min, i_qb_min = np.unravel_index(np.nanargmin(P11_smooth), P11.shape)

    P11_min_value = float(P11.values[i_cpl_min, i_qb_min])
    coupler_amp_min = float(P11.flux_coupler.values[i_cpl_min])
    qubit_amp_min   = float(P11.flux_qubit.values[i_qb_min])
    flux_coupler_full_min = float(flux_cpl_full.interp(flux_coupler=coupler_amp_min))
    flux_qubit_min  = float(flux_qb.interp(flux_qubit=qubit_amp_min))

    # Search ALONG the same qubit column (fixed qubit_amp) for max
    col_data = P11_smooth[:, i_qb_min]        # all coupler points for fixed qubit_amp
    flux_col = flux_cpl_full.data

    # Find maximum of P11 along this vertical column
    i_cpl_max = np.nanargmax(col_data)

    P11_max_value = float(col_data[i_cpl_max])
    coupler_amp_max = float(P11.flux_coupler.values[i_cpl_max])
    flux_coupler_full_max = float(flux_cpl_full.interp(flux_coupler=coupler_amp_max))
    flux_qubit_max = float(flux_qubit_min)  # same column
    flux_coupler_max = float(ds.flux_coupler.interp(flux_coupler=coupler_amp_max))

    print(f"\n Optimal values:")
    print(f" Coupler flux shift = {flux_coupler_max:.4f}, Qubit flux shift = {flux_qubit_max:.4f} V")
    node.results["results"][qpname] = {
        "flux_coupler_full_max": flux_coupler_full_max,
        "flux_qubit_max": flux_qubit_max,
        "flux_coupler_max": flux_coupler_max,   
        }

# %% {Plotting}
if not node.parameters.simulate:
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qp in grid_iter(grid):
        qubit_name = qp["qubit"]
        qubit_pair = machine.qubit_pairs[qubit_name]
        # --- Select data (raw heatmap will always be plotted) ---
        try:
            if node.parameters.use_state_discrimination:
                values_to_plot = ds["P11"].sel(qubit=qubit_name)
                # Coordinates in mV
                values_to_plot = values_to_plot.assign_coords({
                    "flux_qubit_mV": 1e3 * values_to_plot.flux_qubit,
                    "flux_coupler_mV": 1e3 * values_to_plot.flux_coupler,
                })
                # Plot raw data (always)
                values_to_plot.plot(ax=ax, cmap="viridis", x="flux_qubit_mV", y="flux_coupler_mV")
        except Exception as e:
            print(f"[WARN] Plot data failed for {qubit_name}: {e}")
            ax.set_title(f"{qubit_name} (raw plot failed)")
            continue  # nothing else to do for this panel
        
        # --- Optional analysis plotted if it exists ---
        legend_entries = []
        try:
            res = node.results["results"].get(qubit_name, {})
            # Extract (may be missing or NaN)
            flux_coupler_min_mV      = 1e3 * res.get("flux_coupler_min_full", np.nan)
            flux_coupler_max_mV = 1e3 * res.get("flux_coupler_max", np.nan)
            flux_qubit_max_mV        = 1e3 * res.get("flux_qubit_max", np.nan)

            # add crosshair + marker 
            if np.isfinite(flux_coupler_max_mV):
                ax.axhline(flux_coupler_max_mV, color="black", lw=1.0, ls=":")
            if np.isfinite(flux_qubit_max_mV):
                    ax.axvline(flux_qubit_max_mV, color="black", lw=1.0, ls=":")
            if np.isfinite(flux_qubit_max_mV) and np.isfinite(flux_coupler_max_mV):
                    ax.plot(
                        flux_qubit_max_mV, flux_coupler_max_mV,
                        marker="+", color="blue", markersize=10, mew=2.0,
                        label="Optimal"
                    )
            legend_entries.append("Optimal")
        except Exception as e:
                print(f"[WARN] Annotations failed for {qubit_name}: {e}")
        
        # --- Secondary x-axis for detuning (only if mapping is sane) ---
        try:
            sel = ds.sel(qubit=qubit_name)
            flux_qubit_data = (sel.flux_qubit.values * 1e3).ravel()
            detuning_data   = (sel.detuning.values * 1e-6).ravel()  # MHz

            # Ensure strictly increasing x for interpolation
            order = np.argsort(flux_qubit_data)
            x_sorted = flux_qubit_data[order]
            y_sorted = detuning_data[order]

            # Remove duplicates
            x_unique, unique_idx = np.unique(x_sorted, return_index=True)
            y_unique = y_sorted[unique_idx]

            if x_unique.size >= 2:
                def flux_to_detuning(x):
                    return np.interp(np.asarray(x), x_unique, y_unique)

                def detuning_to_flux(y):
                    return np.interp(np.asarray(y), y_unique, x_unique)

                sec_ax = ax.secondary_xaxis("top", functions=(flux_to_detuning, detuning_to_flux))
                sec_ax.set_xlabel("Detuning [MHz]")
        except Exception as e:
            print(f"[WARN] Secondary axis failed for {qubit_name}: {e}")

        
        ax.set_xlabel("Qubit flux shift [mV]")
        ax.set_ylabel("Coupler flux shift [mV]")
        ax.set_title(f"{qubit_name}, Decoupling offset = {qubit_pair.coupler.decouple_offset * 1e3 :.0f} mV ", fontsize=9)
        if legend_entries:
                ax.legend(fontsize=7, loc="upper right", frameon=True)

        # overall title per qubit pair
        grid.fig.suptitle(f'Lekage out of 11 state', y=0.97, fontsize=12)
        plt.tight_layout()
        plt.show()

        # store figure
        node.results[f"figure_11_leakage"] = grid.fig

# %% {Update_state}
if not node.parameters.simulate:
    if not node.parameters.simulate:
        with node.record_state_updates():
            for qp in qubit_pairs:
                    qp.extras["Cz_coupler_flux"] = node.results["results"][qp.name]["flux_coupler_max"]
                    qp.gates[operation_name].coupler_flux_pulse.amplitude = node.results["results"][qp.name]["flux_coupler_max"]
# %% {Save_results}
if not node.parameters.simulate:    
    node.outcomes = {q.name: "successful" for q in qubit_pairs}
    node.results['initial_parameters'] = node.parameters.model_dump()
    node.machine = machine
    node.save()
# %%