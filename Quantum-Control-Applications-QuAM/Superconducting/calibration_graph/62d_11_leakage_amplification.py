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
from quam_libs.macros import active_reset, readout_state_gef
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
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
import xarray as xr
from quam_libs.lib.fit import fit_oscillation


# %% {Node_parameters}
qubit_pair_indexes = [2]  # The indexes of the qubit pairs to measure
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = ["coupler_q%s_q%s"%(i,i+1) for i in qubit_pair_indexes]
    """List of qubit pair names to calibrate. If None or empty, all active qubit pairs will be used."""
    num_averages: int = 50
    """Number of averages to perform for each amplitude. Default is 100."""
    flux_point_joint_or_independent_or_pairwise: Literal["joint", "independent", "pairwise"] = "joint"
    """Flux point setting strategy: 'joint', 'independent', or 'pairwise'. Default is 'joint'."""
    reset_type: Literal['active', 'thermal'] = "active"
    """Type of reset to use between experiments. Options are 'active' or 'thermal'. Default is 'active'."""
    simulate: bool = False
    """If True, simulates the QUA program instead of executing it on hardware. Default is False."""
    timeout: int = 100
    """Timeout for the QOP session in seconds. Default is 100 seconds."""
    load_data_id: Optional[int] = None
    """If provided, loads data from a previous calibration with this ID instead of executing the experiment."""
    number_of_operations: int = 50
    """Number of operations to perform for each amplitude. Default is 10."""
    operation: Literal["Cz_unipolar", "Cz_flattop", "Cz_bipolar", "Cz_slepian", "Cz_slepian_flattop"] = "Cz_flattop"
    """Type of CZ operation to perform."""
    coupler_amp_range : float = 0.06
    """ Range around 1 for coupler amplitude scaling."""
    coupler_amp_step : float = 0.0005
    """ Step for coupler amplitude scaling."""
    use_state_discrimination: bool = True
    """Whether to use state discrimination for readout or raw IQ data."""


node = QualibrationNode(
    name="62d_11_eakage_amplification", parameters=Parameters()
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
flux_coupler_amplitudes = np.arange(1-node.parameters.coupler_amp_range, 1+node.parameters.coupler_amp_range, node.parameters.coupler_amp_step)
num_operations = node.parameters.number_of_operations

reset_coupler_bias = False
operation_name = node.parameters.operation

with program() as leakage_amplification:
    n = declare(int)
    flux_coupler_amp = declare(float)
    flux_qubit_amp = declare(float)
    n_st = declare_stream()
    n_op = declare(int)  # number of CZ operations
    count = declare(int)  # loop counter

    
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
            with for_(n_op, 1, n_op <= num_operations, n_op + 1):         
                with for_(*from_array(flux_coupler_amp, flux_coupler_amplitudes)):
                            # reset
                            if node.parameters.reset_type == "active":
                                active_reset(qp.qubit_control)
                                active_reset(qp.qubit_target)
                            else:
                                wait(qp.qubit_control.thermalization_time * u.ns)
                                wait(qp.qubit_target.thermalization_time * u.ns)

                            # state preparation
                            qp.qubit_control.xy.play("x180")
                            qp.qubit_target.xy.play("x180")
                            align()
                            with for_(count, 0, count < n_op, count + 1):
                                # play the CZ gate
                                qp.gates[operation_name].execute(coupler_amplitude_scale=flux_coupler_amp)
                                qp.align()  # wait for flux to settle
                            wait(20)
                            # readout
                            if node.parameters.use_state_discrimination:
                                readout_state_gef(qp.qubit_control, state_control[i])
                                readout_state_gef(qp.qubit_target, state_target[i])
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
                state_st_control[i].buffer(len(flux_coupler_amplitudes)).buffer(num_operations).buffer(n_avg).save(f"state_control{i + 1}")
                state_st_target[i].buffer(len(flux_coupler_amplitudes)).buffer(num_operations).buffer(n_avg).save(f"state_target{i + 1}")
            else:
                I_st_control[i].buffer(len(flux_coupler_amplitudes)).buffer(num_operations).buffer(n_avg).save(f"I_control{i + 1}")
                Q_st_control[i].buffer(len(flux_coupler_amplitudes)).buffer(num_operations).buffer(n_avg).save(f"Q_control{i + 1}")
                I_st_target[i].buffer(len(flux_coupler_amplitudes)).buffer(num_operations).buffer(n_avg).save(f"I_target{i + 1}")
                Q_st_target[i].buffer(len(flux_coupler_amplitudes)).buffer(num_operations).buffer(n_avg).save(f"Q_target{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=20_000 // 4 )  # In clock cycles = 4ns
    job = qmm.simulate(config, leakage_amplification, simulation_config)
    samples = job.get_simulated_samples()
    samples.con1.plot()
    node.results = {"figure": plt.gcf()}
    wf_report = job.get_simulated_waveform_report()
    wf_report.create_plot(samples, plot=True, save_path=None)
    node.machine = machine
    node.save()
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(leakage_amplification)

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
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {"coupler_amp": flux_coupler_amplitudes, "number_of_operations": np.arange(1, num_operations + 1), "N": np.linspace(1, n_avg, n_avg)})
    else:
        ds, machine = load_dataset(node.parameters.load_data_id)
        
    node.results = {"ds": ds}
    node.results["results"] = {}

    
# %% data processing
if not node.parameters.simulate:
    def coupler_flux_shift(qp, amp):
        return amp * qp.gates[operation_name].coupler_flux_pulse.amplitude
    def abs_coupler_amp(qp, amp):
        return amp * qp.gates[operation_name].coupler_flux_pulse.amplitude + qp.coupler.decouple_offset
    
    ds = ds.assign_coords(
        {"flux_coupler_full": (["qubit", "coupler_amp"], np.array([abs_coupler_amp(qp, ds.coupler_amp) for qp in qubit_pairs]))}
    )
    ds = ds.assign_coords(
        {"flux_coupler": (["qubit", "coupler_amp"], np.array([coupler_flux_shift(qp, ds.coupler_amp) for qp in qubit_pairs]))}
    )

# %%Data analysis

if node.parameters.use_state_discrimination:
    sc = ds.state_control
    st = ds.state_target

    P11 = ((sc == 1) & (st == 1)).mean("N").rename("P11")
    P02 = ((sc == 0) & (st == 2)).mean("N").rename("P02")
    P20 = ((sc == 2) & (st == 0)).mean("N").rename("P20")

    ds = xr.merge([ds, P11, P02, P20])
    
for qp in qubit_pairs:
    qpname = qp.name
    P11 = ds.P11.sel(qubit=qpname)        
    data_min_idx = P11.mean("number_of_operations").argmax()
    optimal_coupler_flux_shift = ds.flux_coupler.sel(qubit=qpname)[data_min_idx]  # corresponding flux value
    print(f"\n Optimal values:")
    print(f" optimal coupler flux shift = {optimal_coupler_flux_shift:.4f} V")
    node.results["results"][qpname] = {
        "flux_coupler_max": float(optimal_coupler_flux_shift),   
        }


# %% {Plotting}
if not node.parameters.simulate:
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qp in grid_iter(grid):
        qubit_name = qp["qubit"]
        qubit_pair = machine.qubit_pairs[qubit_name]
        try:
            if node.parameters.use_state_discrimination:
                values_to_plot = ds["P11"].sel(qubit=qubit_name)
                # Coordinates in mV
                values_to_plot = values_to_plot.assign_coords({
                    "flux_coupler_mV": 1e3 * values_to_plot.flux_coupler,
                })
                # Plot raw data (always)
                values_to_plot.plot(ax=ax, cmap="viridis", x="number_of_operations", y="flux_coupler_mV")
        except Exception as e:
            print(f"[WARN] Plot data failed for {qubit_name}: {e}")
            ax.set_title(f"{qubit_name} (raw plot failed)")
            continue  # nothing else to do for this panel
        
        # --- Optional analysis plotted if it exists ---
        legend_entries = []
        try:
            res = node.results["results"].get(qubit_name, {})
            # Extract (may be missing or NaN)
            flux_coupler_max_mV = 1e3 * res.get("flux_coupler_max", np.nan)

            # add crosshair + marker 
            if np.isfinite(flux_coupler_max_mV):
                ax.axhline(flux_coupler_max_mV, color="red", lw=2.0, ls="--", label="Optimal")
            legend_entries.append("Optimal")
        except Exception as e:
                print(f"[WARN] Annotations failed for {qubit_name}: {e}")
        
        
        ax.set_xlabel("# CZ operations")
        ax.set_ylabel("Coupler flux shift [mV]")
        ax.set_title(f"{qubit_name}, Decoupling offset = {qubit_pair.coupler.decouple_offset * 1e3 :.0f} mV ", fontsize=9)
        if legend_entries:
                ax.legend(fontsize=7, loc="upper right", frameon=True)
        # overall title per qubit pair
        grid.fig.suptitle(f'Amplification of leakage out of 11 state \n {operation_name} pulse', y=0.97, fontsize=12)
        plt.tight_layout()
        plt.show()
        # store figure
        node.results[f"figure_11_leakage_amplification"] = grid.fig
        
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