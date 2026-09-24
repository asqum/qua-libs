# %% {Imports}
from dataclasses import asdict
from typing import Literal, Optional, List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qm import SimulationConfig
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.units import unit
from qualibrate import QualibrationNode, NodeParameters

from calibration_utils.cz_jazz2_n import (
    coerce_to_even,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
)
from quam_libs.components import QuAM
from quam_libs.lib.save_utils import (
    fetch_results_as_xarray,
    restore_load_data_id,
    resolve_qubit_pairs_from_node,
)
from quam_libs.macros import active_reset, readout_state, active_reset_gef

# %% {Initialisation}
description = """
        JAZZ2-N CZ AMPLITUDE CALIBRATION
This node calibrates the CZ-pulse amplitude using the JAZZ2-N protocol
(arXiv:2402.18926v3, Appendix I.1, Fig. 13(b)). The pulse sequence is

    x90(control) & x90(target)               (X_{pi/2} X_{pi/2})
    CZ                                       (initial Z)
    [X_pi(control) & X_pi(target) -- CZ] x (2N + 1)
    x90(control) & x90(target)               (X_{pi/2} X_{pi/2})
    measure both qubits -> p00 = (1 - state_control) * (1 - state_target)

where N = 2k (k = 0, 1, 2, ...). With the X_pi refocusing pulses on both
qubits, the joint ground-state probability evolves as

    P_|00>(amp, N) = (1 - cos((N + 1) * theta_CZ(amp))) / 2,

independently of any virtual-Z (single-qubit) phase shifts inside the CZ
gate. The optimal CZ amplitude is the value where theta_CZ = pi, i.e. where
P_|00> is maximal. Compared to JAZZ-N, the principal-peak fringe is denser in
amplitude for a given total pulse count, so this node is a sharper follow-up
amplitude calibration.

Measuring both qubits in superposition together, rather than reading out one
qubit alone, makes the extracted phase more robust to single-qubit gate
errors: an imperfect x90/X_pi on either qubit is folded symmetrically into
the joint correlator instead of being dumped entirely onto one qubit's
readout.

Prerequisites:
    - Calibrated single-qubit gates (x90, x180) for both qubits in the pair.
    - Calibrated, state-discriminating readout for BOTH qubits.
    - An initial estimate of the CZ amplitude (e.g. from 32a or 32x).

State update:
    - qubit_pair.gates[operation].flux_pulse_control.amplitude
"""

qubit_pair_indexes = [4]  # The indexes of the qubit pair to calibrate


class Parameters(NodeParameters):
    qubit_pairs: Optional[List[str]] = ["coupler_q%s_q%s" % (i, i + 1) for i in qubit_pair_indexes]
    num_averages: int = 100
    """Number of averages to perform. Default is 50."""
    amp_range: float = 0.02
    """Half-width of the amplitude-scale sweep around the stored CZ amplitude (center = 1.0). Default is 0.010."""
    amp_step: float = 0.0001
    """Step of the amplitude-scale sweep. Default is 0.001."""
    N_min: int = 0
    """Minimum repetition count. Required form: N = 2k; auto-coerced if not. Default is 0."""
    N_max: int = 30
    """Maximum repetition count. Required form: N = 2k; auto-coerced if not. Default is 50."""
    operation: Literal["Cz_flattop", "Cz_unipolar", "Cz_bipolar"] = "Cz"
    """Type of CZ operation to perform. Default is 'Cz'."""
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    load_data_id: Optional[int] = None
    reset_type: Literal["thermal", "active"] = "active"
    use_state_discrimination: bool = True
    simulate: bool = False
    timeout: int = 100


node = QualibrationNode(
    name="32y_cz_conditional_phase_error_amp_Jazz_2N", parameters=Parameters()
)
assert not (node.parameters.simulate and node.parameters.load_data_id is not None), (
    "If simulate is True, load_data_id must be None, and vice versa."
)

u = unit(coerce_to_integer=True)
machine = QuAM.load()
node.machine = machine

if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs = machine.active_qubit_pairs
else:
    qubit_pairs = [machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]

num_qubit_pairs = len(qubit_pairs)

config = machine.generate_config()
octave_config = machine.get_octave_config()
if node.parameters.load_data_id is None:
    qmm = machine.connect()

if not node.parameters.use_state_discrimination:
    raise RuntimeError(
        "JAZZ2-N reads the joint P_|00> of the qubit pair and therefore requires "
        "use_state_discrimination = True."
    )

node.namespace["qubit_pairs"] = qubit_pairs
n_avg = node.parameters.num_averages
amplitudes = np.arange(1 - node.parameters.amp_range, 1 + node.parameters.amp_range, node.parameters.amp_step)

n_min_req = int(node.parameters.N_min)
n_max_req = int(node.parameters.N_max)
n_min = coerce_to_even(n_min_req)
n_max = coerce_to_even(n_max_req)
if n_min > n_max:
    n_min, n_max = n_max, n_min
if n_min != n_min_req:
    print(f"N_min {n_min_req} coerced to nearest even value: {n_min}.")
if n_max != n_max_req:
    print(f"N_max {n_max_req} coerced to nearest even value: {n_max}.")
n_values = np.arange(n_min, n_max + 1, 2, dtype=int)

operation_name = node.parameters.operation
flux_point = node.parameters.flux_point_joint_or_independent

gate_refs = {}
for qp in qubit_pairs:
    gate_refs[qp.name] = {
        "qubit_amplitude": qp.gates[operation_name].flux_pulse_control.amplitude,
    }
node.namespace["gate_refs"] = gate_refs
node.namespace["sweep_axes"] = {
    "qubit_pair": xr.DataArray([qp.id for qp in qubit_pairs], attrs={"long_name": "qubit pair index"}),
    "N": xr.DataArray(n_values, attrs={"long_name": "repetition count N = 2k"}),
    "amp": xr.DataArray(amplitudes, attrs={"long_name": "amplitude scale", "units": "a.u."}),
}

# %% {QUA_program}
with program() as jazz2_n:
    amp = declare(fixed)
    n = declare(int)
    n_op = declare(int)
    count = declare(int)
    n_st = declare_stream()
    state_c = [declare(int) for _ in range(num_qubit_pairs)]
    state_t = [declare(int) for _ in range(num_qubit_pairs)]
    p00 = [declare(int) for _ in range(num_qubit_pairs)]
    p00_st = [declare_stream() for _ in range(num_qubit_pairs)]
    state_c_st = [declare_stream() for _ in range(num_qubit_pairs)]
    state_t_st = [declare_stream() for _ in range(num_qubit_pairs)]

    for i, qp in enumerate(qubit_pairs):
        qp.gates[operation_name].phase_shift_control = 0.0
        qp.gates[operation_name].phase_shift_target = 0.0
        if not node.parameters.simulate:
            machine.set_all_fluxes(flux_point, qp)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(n_op, n_min, n_op <= n_max, n_op + 2):
                with for_(*from_array(amp, amplitudes)):
                    if not node.parameters.simulate:
                        if node.parameters.reset_type == "active":
                            active_reset_gef(qp.qubit_control)
                            active_reset_gef(qp.qubit_target)
                        else:
                            wait(qp.qubit_control.thermalization_time * u.ns)
                            wait(qp.qubit_target.thermalization_time * u.ns)
                    qp.align()
                    reset_frame(qp.qubit_control.xy.name)
                    reset_frame(qp.qubit_target.xy.name)

                    # Boundary X_{pi/2} X_{pi/2} (both qubits).
                    qp.qubit_control.xy.play("x90")
                    qp.qubit_target.xy.play("x90")
                    qp.align()

                    # First CZ (the "Z" preceding the (pi-Z)^(2N+1) pattern).
                    qp.gates[operation_name].execute(amplitude_scale=amp)
                    qp.qubit_control.xy.play("x180")
                    qp.qubit_target.xy.play("x180")
                    qp.align()
                    qp.gates[operation_name].execute(amplitude_scale=amp)

                    # (X_pi X_pi, CZ) x N, with virtual-Z echo on both qubits.
                    with for_(count, 1, count <= n_op, count + 1):
                        qp.qubit_control.xy.play("x180")
                        qp.qubit_target.xy.play("x180")
                        qp.align()
                        qp.gates[operation_name].execute(amplitude_scale=amp)
                        qp.qubit_control.xy.frame_rotation_2pi(0.5)
                        qp.qubit_target.xy.frame_rotation_2pi(0.5)
                        qp.qubit_control.xy.play("x180")
                        qp.qubit_target.xy.play("x180")
                        qp.align()
                        qp.gates[operation_name].execute(amplitude_scale=amp)
                        qp.qubit_control.xy.frame_rotation_2pi(-0.5)
                        qp.qubit_target.xy.frame_rotation_2pi(-0.5)

                    qp.align()
                    # Boundary X_{pi/2} then X_pi (matches 32d JAZZ2-N closing layer).
                    qp.qubit_control.xy.play("x90")
                    qp.qubit_control.xy.play("x180")
                    qp.qubit_target.xy.play("x90")
                    qp.qubit_target.xy.play("x180")
                    qp.align()

                    readout_state(qp.qubit_control, state_c[i])
                    readout_state(qp.qubit_target, state_t[i])
                    assign(p00[i], (1 - state_c[i]) * (1 - state_t[i]))
                    save(p00[i], p00_st[i])
                    save(state_c[i], state_c_st[i])
                    save(state_t[i], state_t_st[i])
        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            p00_st[i].buffer(len(amplitudes)).buffer(len(n_values)).average().save(f"p{i + 1}")
            state_c_st[i].buffer(len(amplitudes)).buffer(len(n_values)).average().save(f"state_control{i + 1}")
            state_t_st[i].buffer(len(amplitudes)).buffer(len(n_values)).average().save(f"state_target{i + 1}")

# %% {Simulate}
if node.parameters.simulate:
    simulation_config = SimulationConfig(duration=10_000)
    job = qmm.simulate(config, jazz2_n, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.save()
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(jazz2_n)

        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        ds = fetch_results_as_xarray(
            job.result_handles,
            qubit_pairs,
            {"amp": amplitudes, "N": n_values},
        )
    else:
        load_data_id = node.parameters.load_data_id
        node = node.load_from_id(load_data_id)
        ds = node.results["ds_raw"]
        restore_load_data_id(node, load_data_id)
        machine = node.machine
        qubit_pairs = resolve_qubit_pairs_from_node(machine, node)
        node.namespace["qubit_pairs"] = qubit_pairs
        operation_name = node.parameters.operation
        gate_refs = {}
        for qp in qubit_pairs:
            gate_refs[qp.name] = {
                "qubit_amplitude": qp.gates[operation_name].flux_pulse_control.amplitude,
            }
        node.namespace["gate_refs"] = gate_refs

    if "qubit" in ds.dims:
        ds = ds.rename({"qubit": "qubit_pair"})
    node.results = {"ds_raw": ds}

# %% {Analyse_data}
if not node.parameters.simulate:
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(fit_results, log_callable=node.log)
    node.outcomes = {
        qubit_pair_name: ("successful" if fit_result.success else "failed")
        for qubit_pair_name, fit_result in fit_results.items()
    }

# %% {Plot_data}
if not node.parameters.simulate:
    qubit_pairs = node.namespace["qubit_pairs"]
    figures = plot_raw_data_with_fit(node.results["ds_fit"], qubit_pairs)
    for fig in figures.values():
        plt.show()
    node.results["phase_figure"] = figures["map"]
    node.results["jazz2_n_map"] = figures["map"]
    node.results["jazz2_n_avg"] = figures["avg"]

# %% {Update_state}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            fit_results = node.results["fit_results"]
            for qp in node.namespace["qubit_pairs"]:
                if node.outcomes[qp.name] == "failed":
                    print(f"Skipping state update for {qp.name}: fit failed.")
                    continue
                qp.gates[operation_name].flux_pulse_control.amplitude = fit_results[qp.name]["optimal_amplitude"]

# %% {Save_results}
if not node.parameters.simulate:
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.save()

# %%
