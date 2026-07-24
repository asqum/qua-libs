# %%
"""
Calibration of the Controlled-Phase (CPhase) of the CZ Gate

This sequence calibrates the CPhase of the CZ gate by scanning the pulse amplitude and measuring the resulting phase of the target qubit. The calibration compares two scenarios:

1. Control qubit in the ground state
2. Control qubit in the excited state

For each amplitude, we measure:
1. The phase difference of the target qubit between the two scenarios
2. The amount of leakage to the |f> state when the control qubit is in the excited state

The calibration process involves:
1. Applying a CZ gate with varying amplitudes
2. Measuring the phase of the target qubit for both control qubit states
3. Calculating the phase difference
4. Measuring the population in the |f> state to quantify leakage

The optimal CZ gate amplitude is determined by finding the point where:
1. The phase difference is closest to π (0.5 in normalized units)
2. The leakage to the |f> state is minimized

Prerequisites:
- Calibrated single-qubit gates for both qubits in the pair
- Calibrated readout for both qubits
- Initial estimate of the CZ gate amplitude

Outcomes:
- Optimal CZ gate amplitude for achieving a π phase shift
- Leakage characteristics across the amplitude range
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import readout_state, active_reset_simple
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

# %% {Node_parameters}
qubit_pair_indexes = [4]  # The indexes of the qubit pair to calibrate
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = ["coupler_q%s_q%s"%(i,i+1) for i in qubit_pair_indexes]
    num_averages: int = 200
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type: Literal['active', 'thermal'] = "active"
    simulate: bool = True
    timeout: int = 100
    amp_range : float = 0.1#0.12
    amp_step : float = 0.004
    num_frames: int = 20
    num_repeats: int = 20 #12
    load_data_id: Optional[int] = None
    measure_leak : bool = True


node = QualibrationNode(
    name="32bx_Adiabatic_cz_phase_coupler_error_amp", parameters=Parameters()
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

def tanh_fit(x, a, b, c, d):
    return a * np.tanh(b * x + c) + d


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

repeats = np.arange(1, node.parameters.num_repeats, 2)

# Loop parameters
amplitudes = np.arange(1-node.parameters.amp_range, 1+node.parameters.amp_range, node.parameters.amp_step)
frames = np.arange(0, 1, 1/node.parameters.num_frames)

with program() as CPhase_Oscillations:
    amp = declare(fixed)   
    frame = declare(fixed)
    control_initial = declare(int)
    n = declare(int)
    n_st = declare_stream()
    state_control = [declare(int) for _ in range(num_qubit_pairs)]
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    n_repeats = declare(int)
    count = declare(int)
    
    for i, qp in enumerate(qubit_pairs):
        # Bring the active qubits to the minimum frequency point
        if flux_point == "independent":
            machine.apply_all_flux_to_min()
            # qp.apply_mutual_flux_point()
        elif flux_point == "joint":
            machine.apply_all_flux_to_joint_idle()
        else:
            machine.apply_all_flux_to_zero()
        wait(1000)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)         
            with for_(*from_array(amp, amplitudes)):
                with for_(*from_array(frame, frames)):
                    with for_(*from_array(control_initial, [0,1])):
                        with for_(*from_array(n_repeats, repeats)):
                            # reset
                            if not node.parameters.simulate:
                                if node.parameters.reset_type == "active":
                                    active_reset_simple(qp.qubit_control)
                                    qp.align()
                                    active_reset_simple(qp.qubit_target)
                                    qp.align()
                                else:
                                    wait(qp.qubit_control.thermalization_time * u.ns)
                            qp.align()
                            reset_frame(qp.qubit_target.xy.name)
                            reset_frame(qp.qubit_control.xy.name)                   
                            # setting both qubits ot the initial state
                            # with if_(control_initial == 1):
                            #     qp.qubit_control.xy.play("x180")
                            qp.qubit_control.xy.play("x180", condition = control_initial == 1)                    
                            qp.qubit_target.xy.play("x90")
                            qp.align()

                            with for_(count, 0, count < n_repeats, count + 1):
                                #play the CZ gate
                                qp.gates['Cz'].execute(coupler_amplitude_scale = amp)
                                qp.align()
                                qp.qubit_control.z.wait(50)
                                qp.align()
                                
                                #rotate the frame (NOTE: should this be in the loop?)
                                frame_rotation_2pi(frame, qp.qubit_target.xy.name)
                                qp.align()
                            
                            # return the target qubit before measurement
                            qp.qubit_target.xy.play("x90")                        
                                
                            # measure both qubits (g/e only)
                            readout_state(qp.qubit_control, state_control[i])
                            readout_state(qp.qubit_target, state_target[i])
                            save(state_control[i], state_st_control[i])
                            save(state_target[i], state_st_target[i])
        align()
        
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            state_st_control[i].buffer(len(repeats)).buffer(2).buffer(len(frames)).buffer(len(amplitudes)).average().save(f"state_control{i + 1}")
            state_st_target[i].buffer(len(repeats)).buffer(2).buffer(len(frames)).buffer(len(amplitudes)).average().save(f"state_target{i + 1}")

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
    with qm_session(qmm, config, timeout=node.parameters.timeout ) as qm:
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
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {"repeats": repeats, "control_axis": [0,1], "frame": frames, "amp": amplitudes})
    else:
        ds, machine = load_dataset(node.parameters.load_data_id)

        
    node.results = {"ds": ds}

# %% {Data_analysis}
if not node.parameters.simulate:
    def abs_amp(qp, amp):
        return amp * qp.gates['Cz'].coupler_flux_pulse.amplitude

    ds = ds.assign_coords(
        {"amp_full": (["qubit", "amp"], np.array([abs_amp(qp, ds.amp) for qp in qubit_pairs]))}
    )
# %% Analysis
if not node.parameters.simulate:

    phase_diffs = {}
    optimal_amps = {}
    for qp in qubit_pairs:
        ds_qp = ds.sel(qubit=qp.name)
        fit_data = fit_oscillation(ds_qp.state_target, "frame")
        
        ds_qp = ds_qp.assign({'fitted': oscillation(ds_qp.frame,
                                                    fit_data.sel(fit_vals="a"),
                                                    fit_data.sel(fit_vals="f"),
                                                    fit_data.sel(
                                                        fit_vals="phi"),
                                                    fit_data.sel(fit_vals="offset"))})
        phase = fix_oscillation_phi_2pi(fit_data)    
        phase_diff = (phase.sel(control_axis=0)-phase.sel(control_axis=1)) % 1 
        optimal_amps[qp.name] = phase_diff.amp_full[np.abs(phase_diff-0.5).mean(dim = 'repeats').argmin(dim = 'amp')]
        phase_diffs[qp.name] = phase_diff

    # %%
    (phase_diff-0.5).plot(x = "repeats", y = "amp_full")
    phase_diff.amp_full[np.abs(phase_diff-0.5).mean(dim = 'repeats').argmin(dim = 'amp')]
# %%

# %%
if not node.parameters.simulate:
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qubit_pair in grid_iter(grid):
        
        data_to_plot = phase_diffs[qubit_pair['qubit']].assign_coords(coupler_amp_V=phase_diffs[qubit_pair['qubit']].amp_full) - 0.5
        plot = data_to_plot.plot(x="repeats", y="coupler_amp_V", add_colorbar=False)
        plt.colorbar(plot, ax=ax, orientation='horizontal', pad=0.2, aspect=30, label='Phase')

        cz_gate = machine.qubit_pairs[qubit_pair["qubit"]].gates['Cz']
        ax.axhline(y=float(optimal_amps[qubit_pair['qubit']]), color='k', linestyle='--', lw=0.62)
        ax.axhline(y=cz_gate.coupler_flux_pulse.amplitude, color='b', linestyle='--', lw=0.57)
        ax.set_ylabel('Coupler amplitude [V]')

        
    plt.suptitle('Cz phase calibration (coupler amp)', y=0.95)
    plt.tight_layout()
    plt.show()
    node.results["figure_phase"] = grid.fig
    
    if node.parameters.measure_leak:
        _, qubit_pair_names = grid_pair_names(qubit_pairs)
        n_pairs = len(qubit_pair_names)
        fig, axes = plt.subplots(n_pairs, 2, figsize=(10, 4 * n_pairs), squeeze=False)
        for row, qubit_pair in enumerate(qubit_pair_names):
            ds_qp = ds.sel(qubit=qubit_pair)
            amp_coords = {"coupler_amp_V": ds_qp.amp_full}
            state_0 = ds_qp.state_control.sel(control_axis=0).mean(dim="frame").assign_coords(amp_coords)
            state_1 = ds_qp.state_control.sel(control_axis=1).mean(dim="frame").assign_coords(amp_coords)

            for col, (data_to_plot, label) in enumerate([(state_0, "|0>"), (state_1, "|1>")]):
                ax = axes[row, col]
                plot = data_to_plot.plot(x="repeats", y="coupler_amp_V", ax=ax, add_colorbar=False)
                plt.colorbar(plot, ax=ax, orientation='horizontal', pad=0.2, aspect=30, label=label)

                cz_gate = machine.qubit_pairs[qubit_pair].gates['Cz']
                ax.axhline(y=float(optimal_amps[qubit_pair]), color='r', linestyle='--', lw=0.62)
                ax.axhline(y=cz_gate.coupler_flux_pulse.amplitude, color='b', linestyle='--', lw=0.57)
                ax.set_ylabel('Coupler amplitude [V]')
                ax.set_title(f"{qubit_pair} {label}")

        plt.suptitle('Cz phase calibration state (coupler amp)', y=0.98)
        plt.tight_layout()
        plt.show()
        node.results['figure_leak'] = fig

# %% {Update_state}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for qp in qubit_pairs:
                qp.gates['Cz'].coupler_flux_pulse.amplitude = float(optimal_amps[qp.name].values)

                
# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {qp.name: "successful" for qp in qubit_pairs}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()
        
# %%