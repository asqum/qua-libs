# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.cz_conditional_phase_error_amp import (
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
)
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from qm import SimulationConfig
from quam_libs.components import QuAM
from quam_libs.components.gates.two_qubit_gates import CZGate
from qualibrate import QualibrationNode, NodeParameters
from typing import Literal, Optional, List
from quam_libs.macros import (
    active_reset,
    readout_state,
    readout_state_gef,
    active_reset_gef,
    active_reset_simple,
    qua_declaration,
)
from qualang_tools.results import progress_counter, fetching_tool

# %% {Initialisation}
description = """
CALIBRATION OF THE CONTROLLED-PHASE (CPHASE) OF THE CZ GATE with error amplification

This sequence calibrates the CPhase of the CZ gate by scanning the pulse amplitude and measuring the
resulting phase of the target qubit. The calibration compares two scenarios:

1. Control qubit in the ground state
2. Control qubit in the excited state

For each amplitude, we measure:
1. The phase difference of the target qubit between the two scenarios
2. The average population in the |g>, |e>, and |f> states of the control qubit when the control qubit is in the excited state.

**Error amplification:**
To improve sensitivity to small phase errors, the CZ gate is applied repeatedly (multiple times in sequence) for each measurement. This introduces an extra dimension to the experiment: the number of repeated CZ operations. By increasing the number of repetitions, small phase errors accumulate, making them easier to detect and fit.

The calibration process involves:
1. Applying a CZ gate with varying amplitudes
2. Repeating the CZ operation a variable number of times (error amplification dimension)
3. Measuring the phase of the target qubit for both control qubit states
4. Calculating the phase difference
5. Measuring the population fractions of the |g>, |e>, and |f> states on the control qubit to quantify leakage

The optimal CZ gate amplitude is determined by finding the point where:
1. The phase difference (after error amplification) is closest to π (0.5 in normalized units)
2. The leakage to the |f> state is minimized

Prerequisites:
- Calibrated single-qubit gates for both qubits in the pair
- Calibrated readout for both qubits
- Initial estimate of the CZ gate amplitude

State update:
- The optimal CZ gate amplitude: qubit_pair.gates["Cz"].flux_pulse_control.amplitude
"""
qubit_pair_indexes = [2]  # The indexes of the qubit pair to calibrate


class Parameters(NodeParameters):
    qubit_pairs: Optional[List[str]] = ["coupler_q%s_q%s" % (i, i + 1) for i in qubit_pair_indexes]
    num_averages: int = 50
    """Number of averages to perform. Default is 100."""
    amp_range: float = 0.02
    """Range of amplitude variation around the nominal value, will scan between center - range and center + range. Default is 0.010."""
    amp_step: float = 0.001
    """Step size for amplitude scanning. Default is 0.001."""
    num_frame_rotations: int = 17
    """Number of frame rotation points for phase measurement. Default is 10."""
    operation: Literal["Cz_unipolar", "Cz_flattop", "Cz_bipolar", "Cz_slepian", "Cz_slepian_flattop"] = "Cz_flattop"
    """Type of CZ operation to perform"""
    number_of_operations: int = 50
    """Number of operations to perform for each amplitude. Default is 10."""
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    load_data_id: Optional[int] = None  # 92417
    """If provided, loads data from a previous calibration with this ID instead of executing the experiment."""
    reset_type: Literal["thermal", "active"] = "active"
    """Type of reset to use between experiments. Options are 'thermal' or 'active'. Default is 'active'."""
    use_state_discrimination: bool = True
    """Whether to use state discrimination for readout. Default is True."""
    simulate: bool = False
    """If True, simulates the QUA program instead of executing it on hardware. Default is False."""
    simulation_duration_ns: int = 1500
    """Duration of the simulation in nanoseconds. Default is 1500 ns."""
    timeout: int = 100
    """Timeout for the QOP session in seconds. Default is 100 seconds."""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode(name="32x_cz_conditional_phase_error_amp", parameters=Parameters())


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

node.namespace["qubit_pairs"] = qubit_pairs
n_avg = node.parameters.num_averages
amplitudes = np.arange(1 - node.parameters.amp_range, 1 + node.parameters.amp_range, node.parameters.amp_step)
frames = np.arange(0, 1, 1 / node.parameters.num_frame_rotations)

operation_name = node.parameters.operation
num_operations = node.parameters.number_of_operations
# Register the sweep axes to be added to the dataset when fetching data
node.namespace["sweep_axes"] = {
    "qubit_pair": xr.DataArray([qp.id for qp in qubit_pairs], attrs={"long_name": "qubit pair index"}),
    "number_of_operations": xr.DataArray(
        np.arange(1, num_operations + 1),
        attrs={"long_name": "number of operations"},
    ),
    "amp": xr.DataArray(amplitudes, attrs={"long_name": "amplitude scale", "units": "a.u."}),
    "frame": xr.DataArray(frames, attrs={"long_name": "frame rotation", "units": "2π"}),
    "control_axis": xr.DataArray([0, 1], attrs={"long_name": "control qubit state"}),
}
flux_point = node.parameters.flux_point_joint_or_independent


# Extract the sweep parameters and axes from the node parameters
with program() as CZ_phase_calibration_error_amp:
    amp = declare(fixed)  # amplitude scaling factor for the CZ gate
    frame = declare(fixed)  # frame rotation of the target qubit (even number of operations)
    frame_odd = declare(fixed)  # frame rotation of the target qubit (odd number of operations)
    control_initial = declare(int)  # initial state of the control qubit
    n = declare(int)
    n_op = declare(int)  # number of CZ operations
    count = declare(int)  # loop counter
    n_st = declare_stream()
    I_c = [declare(fixed) for _ in range(num_qubit_pairs)]
    Q_c = [declare(fixed) for _ in range(num_qubit_pairs)]
    I_c_st = [declare_stream() for _ in range(num_qubit_pairs)]
    Q_c_st = [declare_stream() for _ in range(num_qubit_pairs)]
    I_t = [declare(fixed) for _ in range(num_qubit_pairs)]
    Q_t = [declare(fixed) for _ in range(num_qubit_pairs)]
    I_t_st = [declare_stream() for _ in range(num_qubit_pairs)]
    Q_t_st = [declare_stream() for _ in range(num_qubit_pairs)]
    if node.parameters.use_state_discrimination:
        state_c = [declare(int) for _ in range(num_qubit_pairs)]
        state_t = [declare(int) for _ in range(num_qubit_pairs)]
        state_c_st = [declare_stream() for _ in range(num_qubit_pairs)]
        state_t_st = [declare_stream() for _ in range(num_qubit_pairs)]
    for i, qp in enumerate(qubit_pairs):
        qp.gates[operation_name].phase_shift_control = 0.0
        qp.gates[operation_name].phase_shift_target = 0.0
        machine.set_all_fluxes(flux_point, qp)
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(n_op, 1, n_op <= num_operations, n_op + 1):
                with for_(*from_array(amp, amplitudes)):
                    with for_(*from_array(frame, frames)):
                        with for_(*from_array(control_initial, [0, 1])):
                            # Reset and align both qubits
                            if node.parameters.reset_type == "active":
                                active_reset_gef(qp.qubit_control)
                                active_reset(qp.qubit_target)
                            else:
                                wait(qp.qubit_control.thermalization_time * u.ns)
                            qp.align()
                            reset_frame(qp.qubit_target.xy.name)
                            reset_frame(qp.qubit_control.xy.name)
                            # setting both qubits to the initial state
                            qp.qubit_control.xy.play("x180", condition=control_initial == 1)
                            qp.qubit_target.xy.play("x90")
                            qp.align()
                            # Loop over the number of CZ operations
                            with for_(count, 0, count < n_op, count + 1):
                                # play the CZ gate
                                qp.gates[operation_name].execute(amplitude_scale=amp)
                                qp.align()  # wait for flux to settle
                            # rotate the frame by 𝜋/2 for odd number of operations
                            with if_(((n_op & 1) == 0) & (control_initial == 1)):
                                assign(frame_odd, frame - 0.5)
                                qp.qubit_target.xy.frame_rotation_2pi(frame_odd)
                            with else_():
                                qp.qubit_target.xy.frame_rotation_2pi(frame)
                            # return the target qubit before measurement
                            qp.qubit_target.xy.play("x90")
                            qp.align()

                            if node.parameters.use_state_discrimination:
                                # measure both qubits
                                readout_state_gef(qp.qubit_control, state_c[i])
                                readout_state_gef(qp.qubit_target, state_t[i])
                                # save each state outcome to its corresponding stream
                                save(state_c[i], state_c_st[i])
                                save(state_t[i], state_t_st[i])
                            else:
                                qp.qubit_control.resonator.measure("readout", qua_vars=(I_c[i], Q_c[i]))
                                qp.qubit_target.resonator.measure("readout", qua_vars=(I_t[i], Q_t[i]))
                                save(I_c[i], I_c_st[i])
                                save(Q_c[i], Q_c_st[i])
                                save(I_t[i], I_t_st[i])
                                save(Q_t[i], Q_t_st[i])
        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            if node.parameters.use_state_discrimination:
                state_c_st[i].buffer(2).buffer(len(frames)).buffer(len(amplitudes)).buffer(
                    num_operations
                ).average().save(f"state_control{i + 1}")
                state_t_st[i].buffer(2).buffer(len(frames)).buffer(len(amplitudes)).buffer(
                    num_operations
                ).average().save(f"state_target{i + 1}")
            else:
                I_c_st[i].buffer(2).buffer(len(frames)).buffer(len(amplitudes)).buffer(num_operations).average().save(
                    f"I_control{i + 1}"
                )
                Q_c_st[i].buffer(2).buffer(len(frames)).buffer(len(amplitudes)).buffer(num_operations).average().save(
                    f"Q_control{i + 1}"
                )
                I_t_st[i].buffer(2).buffer(len(frames)).buffer(len(amplitudes)).buffer(num_operations).average().save(
                    f"I_target{i + 1}"
                )
                Q_t_st[i].buffer(2).buffer(len(frames)).buffer(len(amplitudes)).buffer(num_operations).average().save(
                    f"Q_target{i + 1}"
                )
# %% {Simulate}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, CZ_phase_calibration_error_amp, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(CZ_phase_calibration_error_amp)

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
            {
                "control_axis": [0, 1],
                "frame": frames,
                "amp": amplitudes,
                "number_of_operations": np.arange(1, num_operations + 1),
            },
        )
    else:
        ds, machine = load_dataset(node.parameters.load_data_id)

    ds = ds.rename({"qubit": "qubit_pair"})
    node.results = {"ds_raw": ds}


# %% {Analyse_data}

node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
log_fitted_results(fit_results, log_callable=node.log)
node.outcomes = {
    qubit_pair_name: ("successful" if fit_result.success else "failed")
    for qubit_pair_name, fit_result in fit_results.items()
}


# %% {Plot_data}
qubit_pairs = node.namespace["qubit_pairs"]
# Plot phase calibration data
fig_phase = plot_raw_data_with_fit(
    node.results["ds_fit"],
    qubit_pairs,
)
plt.show()

node.results["phase_figure"] = fig_phase


# %% {Update_state}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            fit_results = node.results["fit_results"]
            for qp in node.namespace["qubit_pairs"]:
                if node.outcomes[qp.name] == "failed":
                    continue
                qp.gates[operation_name].flux_pulse_control.amplitude = fit_results[qp.name]["optimal_amplitude"]


# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {qp.name: "successful" for qp in qubit_pairs}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()


# %%
