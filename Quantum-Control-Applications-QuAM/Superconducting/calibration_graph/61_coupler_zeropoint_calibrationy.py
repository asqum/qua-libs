# %% {Imports}
import math

from qualibrate import QualibrationNode, NodeParameters

from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import active_reset, readout_state, readout_state_gef, active_reset_gef, active_reset_simple
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
from quam_libs.lib.fit import fit_oscillation, oscillation, fix_oscillation_phi_2pi
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from quam_libs.components.gates.two_qubit_gates import CZGate, SWAP_Coupler_Gate
from quam.components.pulses import FlatTopGaussianPulse, SquarePulse
from quam_libs.lib.pulses import CosineBipolarPulse, CosineFlatTopPulse, FluxPulse, SlepianPulse, SlepianFlatTopPulse

# %% {Description}
description = """
        COUPLER ZERO-INTERACTION CALIBRATION
This calibration program determines the flux bias point for tunable couplers that
results in zero effective coupling (g ≈ 0) between pairs of flux-tunable qubits.
This is a crucial step for architectures relying on dynamically tunable coupling
to implement high-fidelity two-qubit gates and isolate qubits during single-qubit operations.

The method performs a 2D sweep of:
    - The coupler flux bias (around its idle point).
    - The qubit control flux (to bring qubit frequencies closer to resonance).

Each point in this sweep involves initializing the control qubit in the excited state and applying
concurrent flux pulses to both the control qubit and the coupler. The resulting excitation in the
target qubit is measured either using state discrimination or IQ integration, depending on the
configuration. The aim is to identify the coupler bias point at which the residual interaction vanishes.

From the data, the optimal coupler flux (yielding minimal excitation transfer) and corresponding
control qubit flux (yielding maximal excitation retention) are extracted. These values are used
to update the coupler’s `decouple_offset` and the estimated qubit `detuning`.

This procedure ensures precise decoupling between qubits during idle or single-qubit operations, helping
mitigate unwanted crosstalk and residual ZZ interactions.

Prerequisites:
    - Coupler hardware model with known calibration structure.
    - Qubit frequencies, flux tuning models (quadratic term at least).
    - Active reset routines for fast initialization (optional).
    - Calibrated readout and XY pulses on the control and target qubits.
    - Initial coupler `decouple_offset` set near its expected g ≈ 0 point.

State update:
    - Coupler zero-point flux: `coupler.decouple_offset`
    - Control qubit detuning: `qubit_pair.detuning`

Additional notes:
    - Supports both simulation and hardware execution.
    - Results are visualized in a 2D map with overlays for idle and calibrated zero-g coupler flux points.
    - If enabled, detuning is also plotted on a secondary axis for interpretation.

This calibration is essential for optimizing gate scheduling, minimizing idling errors,
and preparing the system for entangling gate calibration.
"""

# %% {Node_parameters}
qubit_pair_indexes = [2]  # [1, 2]


class Parameters(NodeParameters):
    qubit_pairs: Optional[List[str]] = ["coupler_q%s_q%s" % (i, i + 1) for i in qubit_pair_indexes]  # ["coupler_q1_q2"]
    num_averages: int = 100
    flux_point_joint_or_independent_or_pairwise: Literal["joint", "independent", "pairwise"] = "joint"
    reset_type: Literal["active", "thermal"] = "active"
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None

    coupler_flux_min: float = -0.1  # relative to the coupler set point
    coupler_flux_max: float = -0.01  # relative to the coupler set point

    coupler_flux_step: float = 0.0005
    qubit_flux_span: float = 0.025  # relative to the known/calculated detuning between the qubits
    qubit_flux_step: float = 0.0001
    use_state_discrimination: bool = True
    pulse_duration_ns: int = 128
    cz_or_iswap: Literal["cz", "iswap"] = "cz"
    use_saved_detuning: bool = False


node = QualibrationNode(name="18a_coupler_zero_point_calibration", parameters=Parameters())
assert not (
    node.parameters.simulate and node.parameters.load_data_id is not None
), "If simulate is True, load_data_id must be None, and vice versa."

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
    if node.parameters.use_saved_detuning:
        est_flux_shift = qp.detuning
    elif node.parameters.cz_or_iswap == "iswap":
        est_flux_shift = np.sqrt(
            -(qp.qubit_control.xy.RF_frequency - qp.qubit_target.xy.RF_frequency)
            / qp.qubit_control.freq_vs_flux_01_quad_term
        )  # TODO: figure out how to make this run properly after filters
    elif node.parameters.cz_or_iswap == "cz":
        est_flux_shift = np.sqrt(
            -(qp.qubit_control.xy.RF_frequency - qp.qubit_target.xy.RF_frequency - qp.qubit_target.anharmonicity)
            / qp.qubit_control.freq_vs_flux_01_quad_term
        )  # TODO: figure out how to make this run properly after filters
    est_flux_shift = 0.085
    fluxes_qp[qp.name] = fluxes_qubit + est_flux_shift

assert (
    node.parameters.pulse_duration_ns % 4 == 0
), f"Expected pulse duration to be divisible by 4, got {node.parameters.pulse_duration_ns} ns"
pulse_duration_ns = node.parameters.pulse_duration_ns
reset_coupler_bias = False

with program() as coupler_zero_point_calibration:
    n = declare(int)
    flux_coupler = declare(float)
    flux_qubit = declare(float)
    comp_flux_qubit = declare(float)
    n_st = declare_stream()
    qua_pulse_duration = declare(int, value=pulse_duration_ns // 4)
    if node.parameters.use_state_discrimination:
        state_control = [declare(int) for _ in range(num_qubit_pairs)]
        state_target = [declare(int) for _ in range(num_qubit_pairs)]
        state = [declare(int) for _ in range(num_qubit_pairs)]
        state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
        state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
        state_st = [declare_stream() for _ in range(num_qubit_pairs)]
    else:
        I_control = [declare(float) for _ in range(num_qubit_pairs)]
        Q_control = [declare(float) for _ in range(num_qubit_pairs)]
        I_target = [declare(float) for _ in range(num_qubit_pairs)]
        Q_target = [declare(float) for _ in range(num_qubit_pairs)]
        I_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
        Q_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
        I_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
        Q_st_target = [declare_stream() for _ in range(num_qubit_pairs)]

    for i, qp in enumerate(qubit_pairs):
        print("qubit control: %s, qubit target: %s" % (qp.qubit_control.name, qp.qubit_target.name))
        # Bring the active qubits to the minimum frequency point
        machine.set_all_fluxes(flux_point, qp)
        if reset_coupler_bias:
            qp.coupler.set_dc_offset(0.0)
        else:
            qp.coupler.to_decouple_idle()
        wait(1000)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(flux_coupler, fluxes_coupler)):
                with for_(*from_array(flux_qubit, fluxes_qp[qp.name])):
                    # reset
                    if node.parameters.reset_type == "active":
                        active_reset_simple(qp.qubit_control)
                        active_reset_simple(qp.qubit_target)
                        qp.align()
                    else:
                        wait(qp.qubit_control.thermalization_time * u.ns)
                        wait(qp.qubit_target.thermalization_time * u.ns)
                    align()
                    if "coupler_qubit_crosstalk" in qp.extras:
                        assign(comp_flux_qubit, flux_qubit + qp.extras["coupler_qubit_crosstalk"] * flux_coupler)
                    else:
                        print("No crosstalk compensated")
                        assign(comp_flux_qubit, flux_qubit)
                    # setting both qubits ot the initial state
                    qp.qubit_control.xy.play("x180")
                    if node.parameters.cz_or_iswap == "cz":
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
                        if node.parameters.cz_or_iswap == "cz":
                            readout_state_gef(qp.qubit_control, state_control[i])
                            readout_state_gef(qp.qubit_target, state_target[i])
                        else:
                            readout_state(qp.qubit_control, state_control[i])
                            readout_state(qp.qubit_target, state_target[i])
                        assign(state[i], state_control[i] * 2 + state_target[i])
                        save(state_control[i], state_st_control[i])
                        save(state_target[i], state_st_target[i])
                        save(state[i], state_st[i])
                    else:
                        qp.qubit_control.resonator.measure("readout", qua_vars=(I_control[i], Q_control[i]))
                        qp.qubit_target.resonator.measure("readout", qua_vars=(I_target[i], Q_target[i]))
                        save(I_control[i], I_st_control[i])
                        save(Q_control[i], Q_st_control[i])
                        save(I_target[i], I_st_target[i])
                        save(Q_target[i], Q_st_target[i])
        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            if node.parameters.use_state_discrimination:
                state_st_control[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(
                    f"state_control{i + 1}"
                )
                state_st_target[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(
                    f"state_target{i + 1}"
                )
                state_st[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"state{i + 1}")
            else:
                I_st_control[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(
                    f"I_control{i + 1}"
                )
                Q_st_control[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(
                    f"Q_control{i + 1}"
                )
                I_st_target[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"I_target{i + 1}")
                Q_st_target[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"Q_target{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, coupler_zero_point_calibration, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        from qm import generate_qua_script

        with open("debug.py", "w+") as f:
            f.write(generate_qua_script(coupler_zero_point_calibration, config))
        job = qm.execute(coupler_zero_point_calibration)

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
            job.result_handles, qubit_pairs, {"flux_qubit": fluxes_qubit, "flux_coupler": fluxes_coupler}
        )
        flux_qubit_full = np.array([fluxes_qp[qp.name] for qp in qubit_pairs])
        ds = ds.assign_coords({"flux_qubit_full": (["qubit", "flux_qubit"], flux_qubit_full)})
    else:
        ds, machine, _, qubit_pairs = load_dataset(node.parameters.load_data_id)

    node.results = {"ds": ds}

# %% Data processing
detuning_mode = "quadratic"  # "cosine" or "quadratic"
if not node.parameters.simulate:
    flux_coupler_full = np.array([fluxes_coupler + qp.coupler.decouple_offset for qp in qubit_pairs])
    if detuning_mode == "quadratic":
        detuning = np.array(
            [-fluxes_qp[qp.name] ** 2 * qp.qubit_control.freq_vs_flux_01_quad_term for qp in qubit_pairs]
        )
    elif detuning_mode == "cosine":
        detuning = np.array(
            [
                oscillation(
                    fluxes_qubit,
                    qp.qubit_control.extras["a"],
                    qp.qubit_control.extras["f"],
                    qp.qubit_control.extras["phi"],
                    qp.qubit_control.extras["offset"],
                )
                for qp in qubit_pairs
            ]
        )
    ds = ds.assign_coords({"flux_coupler_full": (["qubit", "flux_coupler"], flux_coupler_full)})
    ds = ds.assign_coords({"detuning": (["qubit", "flux_qubit"], detuning)})
    node.results = {"ds": ds}

node.results["results"] = {}

# %% Data Analysis
if not node.parameters.simulate:
    for i, qp in enumerate(qubit_pairs):
        try:
            # --- Select and compute contrast for this qubit ---
            if node.parameters.use_state_discrimination:
                sc = ds.state_control.sel(qubit=qp.id)
                st = ds.state_target.sel(qubit=qp.id)
            else:
                sc = ds.I_control.sel(qubit=qp.id)
                st = ds.I_target.sel(qubit=qp.id)
            contrast = sc - st

            # 1. qubit flux where average contrast is maximum
            i_qb_max = int(contrast.mean(dim="flux_coupler").argmax())
            flux_qubit_max = fluxes_qp[qp.name][i_qb_max]

            # 2. coupler index where average contrast is minimum
            i_cpl_min = int(contrast.mean(dim="flux_qubit").argmin())
            flux_coupler_min = float(contrast.flux_coupler[i_cpl_min])
            flux_coupler_min_full = ds.flux_coupler_full.sel(qubit=qp.id)[i_cpl_min]

            # 3. Extract the vertical slice at qubit index *i_qb_max*
            col_data = contrast[:, i_qb_max]

            # 4. Look *below* that coupler index for the next minimum
            col_below = col_data[:i_cpl_min]
            i_rel = int(np.nanargmin(np.abs(col_below).values))  # index inside the sliced array
            i_cpl_next = int(i_rel)  # absolute index (same as relative here)
            flux_coupler_max = float(contrast.flux_coupler[i_cpl_next])
            flux_coupler_max_full = float(contrast.flux_coupler_full[i_cpl_next])

            node.results["results"][qp.name] = {
                "flux_coupler_min": float(flux_coupler_min),
                "flux_coupler_min_full": float(flux_coupler_min_full),
                "flux_qubit_max": float(flux_qubit_max),
                "flux_coupler_max": float(flux_coupler_max),
                "flux_coupler_max_full": float(flux_coupler_max_full),
            }

            print(
                f"{qp.name}: Decoupling offset={flux_coupler_min:.5f} V, "
                f"Qubit and coupler flux at Coupler ON point =({flux_qubit_max:.5f}, {flux_coupler_max:.5f}) V"
            )

        except Exception as e:
            print(f"[WARN] Analysis failed for {qp.name}: {e}")
            node.results["results"][qp.name] = {
                "flux_coupler_min": np.nan,
                "flux_coupler_min_full": np.nan,
                "flux_qubit_max": np.nan,
                "flux_coupler_max": np.nan,
                "flux_coupler_max_full": np.nan,
            }
            continue

# %% {Plotting}
if not node.parameters.simulate:
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)

    for state_type in ["control", "target"]:  # plot both maps
        grid = QubitPairGrid(grid_names, qubit_pair_names)

        for ax, qp in grid_iter(grid):
            qubit_name = qp["qubit"]
            qubit_pair = machine.qubit_pairs[qubit_name]

            # --- Select data (raw heatmap will always be plotted) ---
            try:
                if node.parameters.use_state_discrimination:
                    values_to_plot = ds[f"state_{state_type}"].sel(qubit=qubit_name)
                else:
                    values_to_plot = ds[f"I_{state_type}"].sel(qubit=qubit_name)

                # Coordinates in mV
                values_to_plot = values_to_plot.assign_coords(
                    {
                        "flux_qubit_mV": 1e3 * values_to_plot.flux_qubit_full,
                        "flux_coupler_mV": 1e3 * values_to_plot.flux_coupler_full,
                    }
                )

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
                flux_coupler_min_mV = 1e3 * res.get("flux_coupler_min_full", np.nan)
                flux_coupler_max_full_mV = 1e3 * res.get("flux_coupler_max_full", np.nan)
                flux_qubit_max_mV = 1e3 * res.get("flux_qubit_max", np.nan)
                idle_offset_mV = 1e3 * getattr(qubit_pair.coupler, "decouple_offset", np.nan)

                # Horizontal lines
                if np.isfinite(flux_coupler_min_mV):
                    ax.axhline(flux_coupler_min_mV, color="red", lw=2.0, ls="--", label="Decoupling offset")
                    legend_entries.append("Decoupling offset")
                if np.isfinite(idle_offset_mV):
                    ax.axhline(idle_offset_mV, color="blue", lw=0.5, ls="--", label="Current Decoupling offset")
                    legend_entries.append("Idle offset")
                if np.isfinite(flux_coupler_max_full_mV):
                    ax.axhline(flux_coupler_max_full_mV, color="black", lw=1.0, ls=":")
                    # legend_entries.append("Coupler @ max")

                # Vertical line
                if np.isfinite(flux_qubit_max_mV):
                    ax.axvline(flux_qubit_max_mV, color="black", lw=1.0, ls=":")

                # Crosshair marker (only if both are valid)
                if np.isfinite(flux_qubit_max_mV) and np.isfinite(flux_coupler_max_full_mV):
                    ax.plot(
                        flux_qubit_max_mV,
                        flux_coupler_max_full_mV,
                        marker="+",
                        color="black",
                        markersize=10,
                        mew=2.0,
                        label="Gate starting point",
                    )
                    legend_entries.append("Gate starting point")
            except Exception as e:
                print(f"[WARN] Annotations failed for {qubit_name}: {e}")

            # --- Secondary x-axis for detuning (only if mapping is sane) ---
            try:
                sel = ds.sel(qubit=qubit_name)
                flux_qubit_data = (sel.flux_qubit_full.values * 1e3).ravel()
                detuning_data = (sel.detuning.values * 1e-6).ravel()  # MHz

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

            # --- Labels & legend ---
            ax.set_xlabel("Qubit flux shift [mV]")
            ax.set_ylabel("Coupler flux [mV]")
            ax.set_title(f"{qubit_name}", fontsize=9)
            if legend_entries:
                ax.legend(fontsize=7, loc="upper right", frameon=True)

        # --- Layout / title ---
        grid.fig.suptitle(f"{state_type.capitalize()} Qubit", y=0.97, fontsize=12, weight="bold")
        plt.tight_layout()
        plt.show()

        node.results[f"figure_{state_type}"] = grid.fig


# %% {Update_state}
if not node.parameters.simulate:
    for qp in qubit_pairs:
        # --- Select gate type and naming ---
        if node.parameters.cz_or_iswap == "cz":
            GateClass = CZGate
            gate_label = "Cz"
            coupler_arg_name = "coupler_flux_pulse"
            coupler_attr = "coupler_flux_pulse"
        elif node.parameters.cz_or_iswap == "iswap":
            GateClass = SWAP_Coupler_Gate
            gate_label = "SWAP"
            coupler_arg_name = "coupler_pulse_control"
            coupler_attr = "coupler_pulse_control"
        else:
            raise ValueError(f"Unknown gate type: {node.parameters.cz_or_iswap}")

        # ---------- Unipolar ----------
        if not hasattr(qp.gates, f"{gate_label}_unipolar"):
            print(f"Creating {gate_label} Unipolar gate macro for {qp.name}")

            qubit_pulse = FluxPulse(
                length=100,
                amplitude=0.5,
                zero_padding=16,
                id=f"flux_pulse_control_{qp.qubit_target.name}_{qp.qubit_control.name}",
            )
            coupler_pulse = FluxPulse(
                length=100,
                amplitude=0.5,
                zero_padding=16,
                id=(
                    f"coupler_pulse_control_{qp.qubit_target.name}_{qp.qubit_control.name}"
                    if gate_label == "SWAP"
                    else f"coupler_flux_pulse_{qp.qubit_target.name}_{qp.qubit_control.name}"
                ),
            )

            gate_kwargs = {"flux_pulse_control": qubit_pulse, coupler_arg_name: coupler_pulse}
            gate = GateClass(**gate_kwargs)
            qp.gates[f"{gate_label}_unipolar"] = gate

            # Bind Z operation to the control qubit
            control_qb = qp.qubit_control
            control_qb.z.operations[f"{gate_label}_unipolar"] = SquarePulse(length=100, amplitude=0.25)
            control_qb.z.operations[f"{gate_label}_unipolar"].length = (
                gate.flux_pulse_control.get_reference() + "/length"
            )
            control_qb.z.operations[f"{gate_label}_unipolar"].amplitude = (
                gate.flux_pulse_control.get_reference() + "/amplitude"
            )
            control_qb.z.operations[f"{gate_label}_unipolar"].zero_padding = (
                gate.flux_pulse_control.get_reference() + "/zero_padding"
            )

        # ---------- Flattop ----------
        if not hasattr(qp.gates, f"{gate_label}_flattop"):
            print(f"Creating {gate_label} Flattop gate macro for {qp.name}")

            qubit_pulse = CosineFlatTopPulse(
                amplitude=0.5,
                flat_length=64,
                smoothing_time=24,
                post_zero_padding_time=0,
                id=f"flux_pulse_control_{qp.qubit_target.name}_{qp.qubit_control.name}",
            )
            coupler_pulse = CosineFlatTopPulse(
                amplitude=0.5,
                flat_length=64,
                smoothing_time=24,
                post_zero_padding_time=0,
                id=(
                    f"coupler_pulse_control_{qp.qubit_target.name}_{qp.qubit_control.name}"
                    if gate_label == "SWAP"
                    else f"coupler_flux_pulse_{qp.qubit_target.name}_{qp.qubit_control.name}"
                ),
            )

            gate_kwargs = {"flux_pulse_control": qubit_pulse, coupler_arg_name: coupler_pulse}
            gate = GateClass(**gate_kwargs)
            qp.gates[f"{gate_label}_flattop"] = gate

            # Bind Z operation to the control qubit
            control_qb = qp.qubit_control
            control_qb.z.operations[f"{gate_label}_flattop"] = CosineFlatTopPulse(amplitude=0.25, flat_length=50)
            control_qb.z.operations[f"{gate_label}_flattop"].flat_length = (
                gate.flux_pulse_control.get_reference() + "/flat_length"
            )
            control_qb.z.operations[f"{gate_label}_flattop"].amplitude = (
                gate.flux_pulse_control.get_reference() + "/amplitude"
            )
            control_qb.z.operations[f"{gate_label}_flattop"].smoothing_time = (
                gate.flux_pulse_control.get_reference() + "/smoothing_time"
            )
            control_qb.z.operations[f"{gate_label}_flattop"].post_zero_padding_time = (
                gate.flux_pulse_control.get_reference() + "/post_zero_padding_time"
            )

        # ---------- Bipolar ----------
        if not hasattr(qp.gates, f"{gate_label}_bipolar"):
            print(f"Creating {gate_label} Bipolar gate macro for {qp.name}")

            qubit_pulse = CosineBipolarPulse(
                amplitude=0.5,
                flat_length=64,
                smoothing_time=24,
                post_zero_padding_time=0,
                id=f"flux_pulse_control_{qp.qubit_target.name}_{qp.qubit_control.name}",
            )
            coupler_pulse = CosineBipolarPulse(
                amplitude=0.5,
                flat_length=64,
                smoothing_time=24,
                post_zero_padding_time=0,
                id=(
                    f"coupler_pulse_control_{qp.qubit_target.name}_{qp.qubit_control.name}"
                    if gate_label == "SWAP"
                    else f"coupler_flux_pulse_{qp.qubit_target.name}_{qp.qubit_control.name}"
                ),
            )

            gate_kwargs = {"flux_pulse_control": qubit_pulse, coupler_arg_name: coupler_pulse}
            gate = GateClass(**gate_kwargs)
            qp.gates[f"{gate_label}_bipolar"] = gate

            control_qb = qp.qubit_control
            control_qb.z.operations[f"{gate_label}_bipolar"] = CosineBipolarPulse(
                amplitude=0.25,
                flat_length=50,
            )
            control_qb.z.operations[f"{gate_label}_bipolar"].flat_length = (
                gate.flux_pulse_control.get_reference() + "/flat_length"
            )
            control_qb.z.operations[f"{gate_label}_bipolar"].amplitude = (
                gate.flux_pulse_control.get_reference() + "/amplitude"
            )
            control_qb.z.operations[f"{gate_label}_bipolar"].smoothing_time = (
                gate.flux_pulse_control.get_reference() + "/smoothing_time"
            )
            control_qb.z.operations[f"{gate_label}_bipolar"].post_zero_padding_time = (
                gate.flux_pulse_control.get_reference() + "/post_zero_padding_time"
            )
        
        # ---------- Slepian ----------
        if not hasattr(qp.gates, f"{gate_label}_slepian"):
            print(f"Creating {gate_label} Slepian gate macro for {qp.name}")

            qubit_pulse = SlepianPulse(
                amplitude=0.5,
                length=64,
                slepian_NW=2.5,
                id=f"flux_pulse_control_{qp.qubit_target.name}_{qp.qubit_control.name}",
            )
            coupler_pulse = SlepianPulse(
                amplitude=0.5,
                length=64,
                slepian_NW=2.5,
                id=(
                    f"coupler_pulse_control_{qp.qubit_target.name}_{qp.qubit_control.name}"
                    if gate_label == "SWAP"
                    else f"coupler_flux_pulse_{qp.qubit_target.name}_{qp.qubit_control.name}"
                ),
            )

            gate_kwargs = {"flux_pulse_control": qubit_pulse, coupler_arg_name: coupler_pulse}
            gate = GateClass(**gate_kwargs)
            qp.gates[f"{gate_label}_slepian"] = gate

            control_qb = qp.qubit_control
            control_qb.z.operations[f"{gate_label}_slepian"] = SlepianPulse(
                amplitude=0.25,
                length=50,
                slepian_NW=2.5,
            )
            control_qb.z.operations[f"{gate_label}_slepian"].length = (
                gate.flux_pulse_control.get_reference() + "/length"
            )
            control_qb.z.operations[f"{gate_label}_slepian"].amplitude = (
                gate.flux_pulse_control.get_reference() + "/amplitude"
            )
            control_qb.z.operations[f"{gate_label}_slepian"].length = (
                gate.flux_pulse_control.get_reference() + "/length"
            )
        
       # ---------- Slepian Flattop  ----------
        if not hasattr(qp.gates, f"{gate_label}_slepian_flattop"):
            print(f"Creating {gate_label} Slepian flattop gate macro for {qp.name}")

            qubit_pulse = SlepianFlatTopPulse(
                amplitude=0.5,
                flat_length=64,
                smoothing_time=24,
                post_zero_padding_time=0,
                slepian_NW=2.5,
                id=f"flux_pulse_control_{qp.qubit_target.name}_{qp.qubit_control.name}",
            )
            coupler_pulse = SlepianFlatTopPulse(
                amplitude=0.5,
                flat_length=64,
                smoothing_time=24,
                post_zero_padding_time=0,
                slepian_NW=2.5,
                id=(
                    f"coupler_pulse_control_{qp.qubit_target.name}_{qp.qubit_control.name}"
                    if gate_label == "SWAP"
                    else f"coupler_flux_pulse_{qp.qubit_target.name}_{qp.qubit_control.name}"
                ),
            )

            gate_kwargs = {"flux_pulse_control": qubit_pulse, coupler_arg_name: coupler_pulse}
            gate = GateClass(**gate_kwargs)
            qp.gates[f"{gate_label}_slepian_flattop"] = gate

            # Bind Z operation to the control qubit
            control_qb = qp.qubit_control
            control_qb.z.operations[f"{gate_label}_slepian_flattop"] = SlepianFlatTopPulse(amplitude=0.25, flat_length=50)
            control_qb.z.operations[f"{gate_label}_slepian_flattop"].flat_length = (
                gate.flux_pulse_control.get_reference() + "/flat_length"
            )
            control_qb.z.operations[f"{gate_label}_slepian_flattop"].amplitude = (
                gate.flux_pulse_control.get_reference() + "/amplitude"
            )
            control_qb.z.operations[f"{gate_label}_slepian_flattop"].smoothing_time = (
                gate.flux_pulse_control.get_reference() + "/smoothing_time"
            )
            control_qb.z.operations[f"{gate_label}_slepian_flattop"].post_zero_padding_time = (
                gate.flux_pulse_control.get_reference() + "/post_zero_padding_time"
            )
            control_qb.z.operations[f"{gate_label}_slepian_flattop"].slepian_NW = (
                gate.flux_pulse_control.get_reference() + "/slepian_NW"
            )

            

    # --- Record updated gate parameters after analysis ---
    with node.record_state_updates():
        for qp in qubit_pairs:
            res = node.results["results"][qp.name]

            # qp.coupler.decouple_offset = res["flux_coupler_min_full"]
            qp.detuning = res["flux_qubit_max"]

            # Keys for extras
            time_key = f"{gate_label}_time"
            qubit_flux_key = f"{gate_label}_qubit_flux"
            coupler_flux_key = f"{gate_label}_coupler_flux"

            # Store in extras
            qp.extras[time_key] = int(np.ceil(node.parameters.pulse_duration_ns / 4) * 4)
            qp.extras[qubit_flux_key] = res["flux_qubit_max"]
            qp.extras[coupler_flux_key] = res["flux_coupler_max"]

            # Convenience refs
            t = qp.extras[time_key]
            q_flux = qp.extras[qubit_flux_key]
            c_flux = qp.extras[coupler_flux_key]

            # --- Update Unipolar ---
            gate = qp.gates[f"{gate_label}_unipolar"]
            gate.flux_pulse_control.amplitude = q_flux
            getattr(gate, coupler_attr).amplitude = c_flux
            gate.flux_pulse_control.length = t
            getattr(gate, coupler_attr).length = t
            

            # --- Update Flattop ---
            gate = qp.gates[f"{gate_label}_flattop"]
            gate.flux_pulse_control.amplitude = q_flux
            getattr(gate, coupler_attr).amplitude = c_flux
            gate.flux_pulse_control.smoothing_time = 24
            gate.flux_pulse_control.flat_length = t - gate.flux_pulse_control.smoothing_time

            # --- Update Bipolar ---
            gate = qp.gates[f"{gate_label}_bipolar"]
            gate.flux_pulse_control.amplitude = q_flux
            getattr(gate, coupler_attr).amplitude = c_flux
            gate.flux_pulse_control.smoothing_time = 24
            gate.flux_pulse_control.flat_length = t - gate.flux_pulse_control.smoothing_time
            
            # --- Update Slepian ---
            gate = qp.gates[f"{gate_label}_slepian"]
            gate.flux_pulse_control.amplitude = q_flux  
            getattr(gate, coupler_attr).amplitude = c_flux
            gate.flux_pulse_control.length = t
            getattr(gate, coupler_attr).length = t
        

# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {q.name: "successful" for q in qubit_pairs}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()
# %%
