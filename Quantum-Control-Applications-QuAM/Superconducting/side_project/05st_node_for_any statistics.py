# MUST THESE # Do not remove
import matplotlib
matplotlib.use('Agg')

# %% {cope the Import package}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.experiments.iq_blobs.fetch_dataset import fetch_dataset
from quam_libs.experiments.simulation import simulate_and_plot
from quam_libs.macros import qua_declaration, readout_state, active_reset, active_reset_simple
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from quam_libs.lib.fit import fit_oscillation_decay_exp, oscillation_decay_exp, fit_decay_exp, decay_exp
from qualang_tools.analysis import two_state_discriminator
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit


from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np


def run_flux_offset(params: dict | None = None):
    ### The start of the copy ### copy the node to here
    # %% {Node_parameters}
    class Parameters(NodeParameters):
        qubits: Optional[List[str]] = None #['q4','q5']
        num_averages: int = 500
        frequency_detuning_in_mhz: float = 8.0
        min_wait_time_in_ns: int = 16
        max_wait_time_in_ns: int = 500
        wait_time_step_in_ns: int = 10
        flux_span: float = 0.04
        flux_step: float = 0.001
        flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
        simulate: bool = False
        simulation_duration_ns: int = 2500
        timeout: int = 100
        load_data_id: Optional[int] = None
        multiplexed: bool = False
    p = Parameters()
    if params is not None:
        for k, v in params.items():
            if hasattr(p, k) and v is not None:
                setattr(p, k, v)
    node = QualibrationNode(name="06a_Ramsey_vs_Flux_Calibration", parameters=p)


    # %% {Initialize_QuAM_and_QOP}
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Instantiate the QuAM class from the state file
    machine = QuAM.load()
    # Generate the OPX and Octave configurations
    config = machine.generate_config()
    # Open Communication with the QOP
    if node.parameters.load_data_id is None:
        qmm = machine.connect()

    # Get the relevant QuAM components
    if node.parameters.qubits is None or node.parameters.qubits == "":
        qubits = machine.active_qubits
    else:
        qubits = [machine.qubits[q] for q in node.parameters.qubits]
    num_qubits = len(qubits)


    # %% {QUA_program}
    n_avg = node.parameters.num_averages  # The number of averages

    # Dephasing time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
    idle_times = np.arange(
        node.parameters.min_wait_time_in_ns // 4,
        node.parameters.max_wait_time_in_ns // 4,
        node.parameters.wait_time_step_in_ns // 4,
    )

    # Detuning converted into virtual Z-rotations to observe Ramsey oscillation and get the qubit frequency
    detuning = int(1e6 * node.parameters.frequency_detuning_in_mhz)
    flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
    fluxes = np.arange(
        -node.parameters.flux_span / 2, node.parameters.flux_span / 2 + 0.001, step=node.parameters.flux_step
    )

    with program() as ramsey:
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
        init_state = declare(int)
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]
        t = declare(int)  # QUA variable for the idle time
        phi = declare(fixed)  # QUA variable for dephasing the second pi/2 pulse (virtual Z-rotation)
        flux = declare(fixed)  # QUA variable for the flux dc level
        reset_global_phase()

        machine.apply_all_couplers_to_min()
        for i, qubit in enumerate(qubits):
            # Bring the active qubits to the desired frequency point
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)
            if "c" in qubit.id: qubit.z.set_dc_offset(qubit.z.joint_offset) # for coupler-test case
            qubit.z.settle()
            qubit.align()   

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(flux, fluxes)):
                    with for_(*from_array(t, idle_times)):
                        # Read the state of the qubit before Ramsey starts
                        readout_state(qubit, init_state)
                        qubit.align()
                        # Rotate the frame of the second x90 gate to implement a virtual Z-rotation
                        # 4*tau because tau was in clock cycles and 1e-9 because tau is ns
                        assign(phi, Cast.mul_fixed_by_int(detuning * 1e-9, 4 * t))
                        # TODO: this has gaps and the Z rotation is not derived properly, is it okay still?
                        # Ramsey sequence
                        qubit.xy.play("x180", amplitude_scale=0.5)
                        qubit.align()
                        wait(20, qubit.z.name)
                        qubit.z.play("const", amplitude_scale=flux / qubit.z.operations["const"].amplitude, duration=t)
                        wait(20, qubit.z.name)
                        qubit.xy.frame_rotation_2pi(phi)
                        qubit.align()
                        qubit.xy.play("x180", amplitude_scale=0.5)

                        # Align the elements to measure after playing the qubit pulse.
                        align()
                        # Measure the state of the resonators
                        readout_state(qubit, state[i])
                        assign(state[i], init_state ^ state[i])
                        save(state[i], state_st[i])

                        # Reset the frame of the qubits in order not to accumulate rotations
                        reset_frame(qubit.xy.name)
                        qubit.align()

            if not node.parameters.multiplexed:
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                state_st[i].buffer(len(idle_times)).buffer(len(fluxes)).average().save(f"state{i + 1}")


    # %% {Simulate_or_execute}
    if node.parameters.simulate:
        # Simulates the QUA program for the specified duration
        simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
        job = qmm.simulate(config, ramsey, simulation_config)
        # Get the simulated samples and plot them for all controllers
        samples = job.get_simulated_samples()
        fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
        for i, con in enumerate(samples.keys()):
            plt.subplot(len(samples.keys()), 1, i + 1)
            samples[con].plot()
            plt.title(con)
        plt.tight_layout()
        # Save the figure
        node.results = {"figure": plt.gcf()}
        node.machine = machine
        node.save()

    elif node.parameters.load_data_id is None:
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(ramsey)
            results = fetching_tool(job, ["n"], mode="live")
            while results.is_processing():
                # Fetch results
                n = results.fetch_all()[0]
                # Progress bar
                progress_counter(n, n_avg, start_time=results.start_time)

    # %% {Data_fetching_and_dataset_creation}
    if not node.parameters.simulate:
        if node.parameters.load_data_id is None:
            ds = fetch_results_as_xarray(job.result_handles, qubits, {"idle_time": idle_times, "flux": fluxes})
            # Add the absolute time in µs to the dataset
            ds = ds.assign_coords(idle_time=4 * ds.idle_time / 1e3)
            ds.flux.attrs = {"long_name": "flux", "units": "V"}
            ds.idle_time.attrs = {"long_name": "idle time", "units": "µs"}
        else:
            ds, machine, json_data, qubits, node.parameters = load_dataset(node.parameters.load_data_id, parameters = node.parameters)
        # Add the dataset to the node
        node.results = {"ds": ds}

        # %% {Data_analysis}
        # TODO: explain the data analysis
        fit_data = fit_oscillation_decay_exp(ds.state, "idle_time")
        fit_data.attrs = {"long_name": "time", "units": "µs"}
        fitted = oscillation_decay_exp(
            ds.state.idle_time,
            fit_data.sel(fit_vals="a"),
            fit_data.sel(fit_vals="f"),
            fit_data.sel(fit_vals="phi"),
            fit_data.sel(fit_vals="offset"),
            fit_data.sel(fit_vals="decay"),
        )

        frequency = fit_data.sel(fit_vals="f")
        frequency.attrs = {"long_name": "frequency", "units": "MHz"}

        decay = fit_data.sel(fit_vals="decay")
        decay.attrs = {"long_name": "decay", "units": "nSec"}

        tau = 1 / fit_data.sel(fit_vals="decay")
        tau.attrs = {"long_name": "T2*", "units": "uSec"}

        frequency = frequency.where(frequency > 0, drop=True)

        fitvals = frequency.polyfit(dim="flux", deg=2)
        flux = frequency.flux
        a = {}
        flux_offset = {}
        freq_offset = {}
        for q in qubits:
            a[q.name] = float(-1e6 * fitvals.sel(qubit=q.name, degree=2).polyfit_coefficients.values)
            flux_offset[q.name] = float(
                (
                    -0.5
                    * fitvals.sel(qubit=q.name, degree=1).polyfit_coefficients
                    / fitvals.sel(qubit=q.name, degree=2).polyfit_coefficients
                ).values
            )
            freq_offset[q.name] = 1e6 * (flux_offset[q.name]**2 * float(fitvals.sel(qubit=q.name, degree=2).polyfit_coefficients.values) +
                                        flux_offset[q.name] * float(fitvals.sel(qubit=q.name, degree=1).polyfit_coefficients.values) + 
                                        float(fitvals.sel(qubit=q.name, degree=0).polyfit_coefficients.values)) - detuning

        # Save fitting results
        node.results["fit_results"] = {}
        for q in qubits:
            node.results["fit_results"][q.name] = {}
            node.results["fit_results"][q.name]["flux_offset"] = flux_offset[q.name]
            node.results["fit_results"][q.name]["freq_offset"] = freq_offset[q.name]
            node.results["fit_results"][q.name]["quad_term"] = a[q.name]

        # %% {Plotting}
        grid_names = [q.grid_location for q in qubits]
        grid = QubitGrid(ds, grid_names)
        for ax, qubit in grid_iter(grid):
            ds.sel(qubit=qubit["qubit"]).state.plot(ax=ax)
            ax.set_title(qubit["qubit"])
            ax.set_xlabel("Idle_time (uS)")
            ax.set_ylabel(" Flux (V)")
        grid.fig.suptitle("Ramsey freq. Vs. flux")
        plt.tight_layout()
        plt.show()
        node.results["figure_raw"] = grid.fig

        grid = QubitGrid(ds, grid_names)
        for ax, qubit in grid_iter(grid):
            fitted_freq = (
                fitvals.sel(qubit=qubit["qubit"], degree=2).polyfit_coefficients * flux**2
                + fitvals.sel(qubit=qubit["qubit"], degree=1).polyfit_coefficients * flux
                + fitvals.sel(qubit=qubit["qubit"], degree=0).polyfit_coefficients
            )
            frequency.sel(qubit=qubit["qubit"]).plot(marker=".", linewidth=0, ax=ax)
            ax.plot(flux, fitted_freq)
            ax.set_title(qubit["qubit"])
            ax.set_xlabel(" Flux (V)")
            print(f"The quad term for {qubit['qubit']} is {a[qubit['qubit']]/1e9:.3f} GHz/V^2")
            print(f"Flux offset for {qubit['qubit']} is {flux_offset[qubit['qubit']]*1e3:.1f} mV")
            print(f"Freq offset for {qubit['qubit']} is {freq_offset[qubit['qubit']]/1e6:.3f} MHz")
            print()
        grid.fig.suptitle("Ramsey freq. Vs. flux")
        plt.tight_layout()
        plt.show()
        node.results["figure"] = grid.fig

        # %% {Update_state}
        if node.parameters.load_data_id is None:
            with node.record_state_updates():
                for qubit in qubits:
                    qubit.xy.intermediate_frequency -= freq_offset[qubit.name]
                    if flux_point == "independent":
                        qubit.z.independent_offset += flux_offset[qubit.name]
                        if "c" in qubit.id: # for coupler-test case
                            qubit.z.joint_offset += flux_offset[qubit.name]
                            qubit.z.independent_offset = qubit.z.joint_offset - qubit.phi0_voltage / 2 
                    elif flux_point == "joint":
                        qubit.z.joint_offset += flux_offset[qubit.name]
                    else:
                        raise RuntimeError(f"unknown flux_point")
                    qubit.freq_vs_flux_01_quad_term = float(a[qubit.name])

            # %% {Save_results}
            # node.outcomes = {q.name: "successful" for q in qubits}
            # node.results["initial_parameters"] = node.parameters.model_dump()
            # node.machine = machine
            # node.save()

            ### The end of the copy ### 
            #modify the parameters you want to save
            fluctuation = []
            freq = []
            qubits_name = []
            for qubit in qubits:
                fluctuation.append(qubit.z.independent_offset)
                freq.append(freq_offset[qubit.name])
                qubits_name.append(qubit.name)
    print(freq)
    return (fluctuation, "fluctuation"), (freq, "freq"), qubits_name

def run_T1(params: dict | None = None):
    """
    Run T1 measurement and return extracted T1, T1_err (in µs), and qubit names.
    """

    # %% {Node_parameters}
    class Parameters(NodeParameters):
        qubits: Optional[List[str]] = None
        num_averages: int = 300
        min_wait_time_in_ns: int = 16
        max_wait_time_in_ns: int = 90000
        wait_time_step_in_ns: int = 300
        flux_point_joint_or_independent_or_arbitrary: Literal["joint", "independent", "arbitrary"] = "independent"
        reset_type: Literal["active", "thermal"] = "thermal"
        use_state_discrimination: bool = False
        simulate: bool = False
        simulation_duration_ns: int = 2500
        timeout: int = 100
        load_data_id: Optional[int] = None
        multiplexed: bool = True
        
    p = Parameters()

    if params is not None:
        for k, v in params.items():
            if hasattr(p, k) and v is not None:
                setattr(p, k, v)
    node = QualibrationNode(name="05_T1", parameters=p)

    # %% {Initialize_QuAM_and_QOP}
    u = unit(coerce_to_integer=True)
    machine = QuAM.load()
    config = machine.generate_config()
    if node.parameters.load_data_id is None:
        qmm = machine.connect()

    if node.parameters.qubits is None or node.parameters.qubits == "":
        qubits = machine.active_qubits
    else:
        qubits = [machine.qubits[q] for q in node.parameters.qubits]
    num_qubits = len(qubits)

    # %% {QUA_program}
    n_avg = node.parameters.num_averages
    idle_times = np.arange(
        node.parameters.min_wait_time_in_ns // 4,
        node.parameters.max_wait_time_in_ns // 4,
        node.parameters.wait_time_step_in_ns // 4,
    )

    flux_point = node.parameters.flux_point_joint_or_independent_or_arbitrary
    if flux_point == "arbitrary":
        detunings = {q.name: q.arbitrary_intermediate_frequency for q in qubits}
        arb_flux_bias_offset = {q.name: q.z.arbitrary_offset for q in qubits}
    else:
        arb_flux_bias_offset = {q.name: 0.0 for q in qubits}
        detunings = {q.name: 0.0 for q in qubits}

    with program() as t1:
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
        t = declare(int)
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        machine.apply_all_couplers_to_min()
        for i, qubit in enumerate(qubits):
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)
            if "c" in qubit.id:
                qubit.z.set_dc_offset(qubit.z.joint_offset)
            qubit.z.settle()
            qubit.align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(t, idle_times)):
                    if node.parameters.reset_type == "active":
                        active_reset_simple(qubit, "readout")
                    else:
                        qubit.resonator.wait(qubit.thermalization_time * u.ns)
                        qubit.align()

                    qubit.xy.play("x180")
                    qubit.align()
                    qubit.z.wait(20)
                    qubit.z.play(
                        "const",
                        amplitude_scale=arb_flux_bias_offset[qubit.name] / qubit.z.operations["const"].amplitude,
                        duration=t,
                    )
                    qubit.z.wait(20)
                    qubit.align()

                    if node.parameters.use_state_discrimination:
                        readout_state(qubit, state[i])
                        save(state[i], state_st[i])
                    else:
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])

            if not node.parameters.multiplexed:
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(idle_times)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(idle_times)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(idle_times)).average().save(f"Q{i + 1}")

    # %% {Simulate_or_execute}
    if node.parameters.simulate:
        simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)
        job = qmm.simulate(config, t1, simulation_config)
        samples = job.get_simulated_samples()
        fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
        for i, con in enumerate(samples.keys()):
            plt.subplot(len(samples.keys()), 1, i + 1)
            samples[con].plot()
            plt.title(con)
        plt.tight_layout()
        node.results = {"figure": plt.gcf()}
        node.machine = machine
        node.save()
    elif node.parameters.load_data_id is None:
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(t1)
            results = fetching_tool(job, ["n"], mode="live")
            while results.is_processing():
                n = results.fetch_all()[0]
                progress_counter(n, n_avg, start_time=results.start_time)

    # %% {Data_fetching_and_dataset_creation}
    if not node.parameters.simulate:
        if node.parameters.load_data_id is None:
            ds = fetch_results_as_xarray(job.result_handles, qubits, {"idle_time": idle_times})
            ds = convert_IQ_to_V(ds, qubits)
            ds = ds.assign_coords(idle_time=4 * ds.idle_time / u.us)
            ds.idle_time.attrs = {"long_name": "idle time", "units": "µs"}
        else:
            ds, machine, json_data, qubits, node.parameters = load_dataset(
                node.parameters.load_data_id, parameters=node.parameters
            )
        node.results = {"ds": ds}

        # %% {Data_analysis}
        if node.parameters.use_state_discrimination:
            fit_data = fit_decay_exp(ds.state, "idle_time")
        else:
            fit_data = fit_decay_exp(ds.I, "idle_time")
        fit_data.attrs = {"long_name": "time", "units": "µs"}

        fitted = decay_exp(
            ds.idle_time,
            fit_data.sel(fit_vals="a"),
            fit_data.sel(fit_vals="offset"),
            fit_data.sel(fit_vals="decay"),
        )

        decay = fit_data.sel(fit_vals="decay")
        decay_res = fit_data.sel(fit_vals="decay_decay")
        tau = -1 / fit_data.sel(fit_vals="decay")
        tau.attrs = {"long_name": "T1", "units": "µs"}
        tau_error = -tau * (np.sqrt(decay_res) / decay)
        tau_error.attrs = {"long_name": "T1 error", "units": "µs"}

        # %% {Plotting}
        grid = QubitGrid(ds, [q.grid_location for q in qubits])
        for ax, qubit in grid_iter(grid):
            if node.parameters.use_state_discrimination:
                ds.sel(qubit=qubit["qubit"]).state.plot(ax=ax)
                ax.set_ylabel("State")
            else:
                ds.sel(qubit=qubit["qubit"]).I.plot(ax=ax)
                ax.set_ylabel("I (V)")
            ax.plot(ds.idle_time, fitted.loc[qubit], "r--")
            ax.set_title(qubit["qubit"])
            ax.set_xlabel("Idle_time (uS)")
            ax.text(
                0.1,
                0.9,
                f'T1 = {tau.sel(qubit=qubit["qubit"]).values:.1f} ± {tau_error.sel(qubit=qubit["qubit"]).values:.1f} µs',
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.5),
            )
        grid.fig.suptitle("T1")
        plt.tight_layout()
        plt.show()
        node.results["figure_raw"] = grid.fig

        # %% {Update_state}
        if node.parameters.load_data_id is None:
            with node.record_state_updates():
                for index, q in enumerate(qubits):
                    if (
                        float(tau.sel(qubit=q.name).values) > 0
                        and tau_error.sel(qubit=q.name).values / float(tau.sel(qubit=q.name).values) < 1
                    ):
                        q.T1 = float(tau.sel(qubit=q.name).values) * 1e-6

            # node.results["initial_parameters"] = node.parameters.model_dump()
            # node.machine = machine
            # node.save()

    # --- return only T1, T1_err, and qubit names ---
    T1 = [float(tau.sel(qubit=q.name).values) for q in qubits]
    T1_err = [float(tau_error.sel(qubit=q.name).values) for q in qubits]
    qubits_name = [q.name for q in qubits]


    return (T1, "T1"), qubits_name

def run_T2_echo(params: dict | None = None):
    """
    Run T2-echo measurement and return extracted T2, T2_error (in µs), and qubit names.
    """

    # %% {Node_parameters}
    class Parameters(NodeParameters):
        qubits: Optional[List[str]] = None
        num_averages: int = 1000
        min_wait_time_in_ns: int = 16
        max_wait_time_in_ns: int = 10000
        wait_time_step_in_ns: int = 200
        flux_point_joint_or_independent_or_arbitrary: Literal["joint", "independent", "arbitrary"] = "independent"
        reset_type: Literal["active", "thermal"] = "thermal"
        use_state_discrimination: bool = True
        simulate: bool = False
        simulation_duration_ns: int = 2500
        timeout: int = 100
        load_data_id: Optional[int] = None
        multiplexed: bool = True

    p = Parameters()
    if params is not None:
        for k, v in params.items():
            if hasattr(p, k) and v is not None:
                setattr(p, k, v)
    node = QualibrationNode(name="06b_T2_echo", parameters=p)

    # %% {Initialize_QuAM_and_QOP}
    u = unit(coerce_to_integer=True)
    machine = QuAM.load()
    config = machine.generate_config()
    if node.parameters.load_data_id is None:
        qmm = machine.connect()

    if node.parameters.qubits is None or node.parameters.qubits == "":
        qubits = machine.active_qubits
    else:
        qubits = [machine.qubits[q] for q in node.parameters.qubits]
    num_qubits = len(qubits)

    # %% {QUA_program}
    n_avg = node.parameters.num_averages
    idle_times = np.arange(
        node.parameters.min_wait_time_in_ns // 4,
        node.parameters.max_wait_time_in_ns // 4,
        node.parameters.wait_time_step_in_ns // 4,
    )

    flux_point = node.parameters.flux_point_joint_or_independent_or_arbitrary
    if flux_point == "arbitrary":
        arb_flux_bias_offset = {q.name: q.z.arbitrary_offset for q in qubits}
    else:
        arb_flux_bias_offset = {q.name: 0.0 for q in qubits}

    with program() as t2_echo:
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
        t = declare(int)
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        machine.apply_all_couplers_to_min()
        for i, qubit in enumerate(qubits):
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)
            if "c" in qubit.id:
                qubit.z.set_dc_offset(qubit.z.joint_offset)
            qubit.z.settle()
            qubit.align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(t, idle_times)):
                    if node.parameters.reset_type == "active":
                        active_reset(qubit, "readout")
                    else:
                        qubit.resonator.wait(qubit.thermalization_time * u.ns)
                        qubit.align()

                    # T2-echo sequence
                    qubit.xy.play("x90")
                    qubit.align()
                    qubit.z.wait(20)
                    qubit.z.play(
                        "const",
                        amplitude_scale=arb_flux_bias_offset[qubit.name] / qubit.z.operations["const"].amplitude,
                        duration=t,
                    )
                    qubit.z.wait(20)
                    qubit.align()
                    qubit.xy.play("x180")
                    qubit.align()
                    qubit.z.wait(20)
                    qubit.z.play(
                        "const",
                        amplitude_scale=arb_flux_bias_offset[qubit.name] / qubit.z.operations["const"].amplitude,
                        duration=t,
                    )
                    qubit.z.wait(20)
                    qubit.align()
                    qubit.xy.play("-x90")
                    qubit.align()

                    if node.parameters.use_state_discrimination:
                        readout_state(qubit, state[i])
                        save(state[i], state_st[i])
                    else:
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])

            if not node.parameters.multiplexed:
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(idle_times)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(idle_times)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(idle_times)).average().save(f"Q{i + 1}")

    # %% {Simulate_or_execute}
    if node.parameters.simulate:
        simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)
        job = qmm.simulate(config, t2_echo, simulation_config)
        samples = job.get_simulated_samples()
        fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
        for i, con in enumerate(samples.keys()):
            plt.subplot(len(samples.keys()), 1, i + 1)
            samples[con].plot()
            plt.title(con)
        plt.tight_layout()
        node.results = {"figure": plt.gcf()}
        node.machine = machine
        node.save()
    elif node.parameters.load_data_id is None:
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(t2_echo)
            results = fetching_tool(job, ["n"], mode="live")
            while results.is_processing():
                n = results.fetch_all()[0]
                progress_counter(n, n_avg, start_time=results.start_time)

    # %% {Data_fetching_and_dataset_creation}
    if not node.parameters.simulate:
        if node.parameters.load_data_id is None:
            ds = fetch_results_as_xarray(job.result_handles, qubits, {"idle_time": idle_times})
            ds = ds.assign_coords(idle_time=8 * ds.idle_time / 1e3)
            ds.idle_time.attrs = {"long_name": "idle time", "units": "µs"}
        else:
            ds, machine, json_data, qubits, node.parameters = load_dataset(node.parameters.load_data_id, parameters=node.parameters)
        node.results = {"ds": ds}

        # %% {Data_analysis}
        if node.parameters.use_state_discrimination:
            fit_data = fit_decay_exp(ds.state, "idle_time")
        else:
            fit_data = fit_decay_exp(ds.I, "idle_time")
        fit_data.attrs = {"long_name": "time", "units": "µs"}

        fitted = decay_exp(
            ds.idle_time,
            fit_data.sel(fit_vals="a"),
            fit_data.sel(fit_vals="offset"),
            fit_data.sel(fit_vals="decay"),
        )

        decay = fit_data.sel(fit_vals="decay")
        decay_res = fit_data.sel(fit_vals="decay_decay")
        tau = -1 / fit_data.sel(fit_vals="decay")
        tau.attrs = {"long_name": "T2", "units": "µs"}
        tau_error = -tau * (np.sqrt(decay_res) / decay)
        tau_error.attrs = {"long_name": "T2 error", "units": "µs"}

        # %% {Plotting}
        grid = QubitGrid(ds, [q.grid_location for q in qubits])
        for ax, qubit in grid_iter(grid):
            if node.parameters.use_state_discrimination:
                ds.sel(qubit=qubit["qubit"]).state.plot(ax=ax)
                ax.set_ylabel("State")
            else:
                ds.sel(qubit=qubit["qubit"]).I.plot(ax=ax)
                ax.set_ylabel("I (V)")
            ax.plot(ds.idle_time, fitted.loc[qubit], "r--")
            ax.set_title(qubit["qubit"])
            ax.set_xlabel("Idle_time (µs)")
            ax.text(
                0.1,
                0.9,
                f'T2 = {tau.sel(qubit=qubit["qubit"]).values:.1f} ± {tau_error.sel(qubit=qubit["qubit"]).values:.1f} µs',
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.5),
            )
        grid.fig.suptitle("T2 echo")
        plt.tight_layout()
        plt.show()
        node.results["figure_raw"] = grid.fig

        # %% {Update_state}
        if node.parameters.load_data_id is None:
            with node.record_state_updates():
                for index, q in enumerate(qubits):
                    if float(tau.sel(qubit=q.name).values) > 0 and tau_error.sel(qubit=q.name).values / float(tau.sel(qubit=q.name).values) < 1:
                        q.T2_echo = float(tau.sel(qubit=q.name).values) * 1e-6

            # node.results["initial_parameters"] = node.parameters.model_dump()
            # node.machine = machine
            # node.save()

    # --- return only T2, T2_error, and qubit names ---
    T2 = [float(tau.sel(qubit=q.name).values) for q in qubits]
    T2_err = [float(tau_error.sel(qubit=q.name).values) for q in qubits]
    qubits_name = [q.name for q in qubits]

    return (T2, "T2"), qubits_name

def run_IQ_blobs(params: dict | None = None):
    """
    Run IQ blobs measurement on selected qubits and return results including:
        - rotation angle for integration weights
        - threshold for g->e discrimination
        - readout fidelity and confusion matrices
    """

    # %% {Node_parameters}
    class Parameters(NodeParameters):
        qubits: Optional[List[str]] = None
        num_runs: int = 3000
        flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
        reset_type: Literal["active", "thermal"] = "thermal"
        operation_name: str = "readout"
        simulate: bool = False
        simulation_duration_ns: int = 1000
        timeout: int = 100
        load_data_id: Optional[int] = None
        multiplexed: bool = False

    p = Parameters()
    if params is not None:
        for k, v in params.items():
            if hasattr(p, k) and v is not None:
                setattr(p, k, v)

    node = QualibrationNode(name="07b_IQ_Blobs", parameters=p)

    # %% {Initialize_QuAM_and_QOP}
    u = unit(coerce_to_integer=True)
    machine = QuAM.load()
    config = machine.generate_config()
    if node.parameters.load_data_id is None:
        qmm = machine.connect()

    qubits = machine.get_qubits_used_in_node(node.parameters)
    num_qubits = len(qubits)
    print(num_qubits)

    config = machine.generate_config()

    # %% {QUA_program}
    n_runs = node.parameters.num_runs
    flux_point = node.parameters.flux_point_joint_or_independent
    reset_type = node.parameters.reset_type
    operation_name = node.parameters.operation_name

    with program() as iq_blobs:
        reset_global_phase()
        I_g, I_g_st, Q_g, Q_g_st, n, n_st = qua_declaration(num_qubits=num_qubits)
        I_e, I_e_st, Q_e, Q_e_st, _, _ = qua_declaration(num_qubits=num_qubits)

        for multiplexed_qubits in qubits.batch():
            machine.set_all_fluxes(flux_point=flux_point, target=list(multiplexed_qubits.values())[0])

            with for_(n, 0, n < n_runs, n + 1):
                save(n, n_st)

                # measure ground-state IQ blob
                for i, qubit in multiplexed_qubits.items():
                    if reset_type == "active":
                        active_reset(qubit)
                    elif reset_type == "thermal":
                        qubit.wait(4 * qubit.thermalization_time * u.ns)
                    else:
                        raise ValueError(f"Unrecognized reset type {reset_type}.")

                align(*[q.xy.name for q in multiplexed_qubits.values()] +
                       [q.resonator.name for q in multiplexed_qubits.values()] +
                       [q.z.name for q in multiplexed_qubits.values()])

                for i, qubit in multiplexed_qubits.items():
                    qubit.resonator.measure(operation_name, qua_vars=(I_g[i], Q_g[i]))
                    qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                    save(I_g[i], I_g_st[i])
                    save(Q_g[i], Q_g_st[i])

                # measure excited-state IQ blob
                align(*[q.xy.name for q in multiplexed_qubits.values()] +
                       [q.resonator.name for q in multiplexed_qubits.values()] +
                       [q.z.name for q in multiplexed_qubits.values()])

                for i, qubit in multiplexed_qubits.items():
                    qubit.xy.play("x180")
                    qubit.resonator.wait(qubit.xy.operations["x180"].length * u.ns)
                    qubit.resonator.measure(operation_name, qua_vars=(I_e[i], Q_e[i]))
                    qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                    save(I_e[i], I_e_st[i])
                    save(Q_e[i], Q_e_st[i])

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_g_st[i].save_all(f"I_g{i + 1}")
                Q_g_st[i].save_all(f"Q_g{i + 1}")
                I_e_st[i].save_all(f"I_e{i + 1}")
                Q_e_st[i].save_all(f"Q_e{i + 1}")

    # %% {Simulate_or_execute}
    if node.parameters.simulate:
        samples, fig = simulate_and_plot(qmm, config, iq_blobs, node.parameters)
        node.results = {"figure": fig}
        node.machine = machine
        node.save()

    elif node.parameters.load_data_id is None:
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(iq_blobs)
            results = fetching_tool(job, ["n"], mode="live")
            while results.is_processing():
                n = results.fetch_all()[0]
                progress_counter(n, n_runs, start_time=results.start_time)

    # %% {Data_fetching_and_analysis}
    if not node.parameters.simulate:
        if node.parameters.load_data_id is None:
            ds = fetch_dataset(job, qubits, node.parameters)
        else:
            node = node.load_from_id(node.parameters.load_data_id)
            ds = node.results["ds"]

        node.results = {"ds": ds, "figs": {}, "results": {}}
        plot_individual = False
        for q in qubits:
            angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(
                ds.I_g.sel(qubit=q.name),
                ds.Q_g.sel(qubit=q.name),
                ds.I_e.sel(qubit=q.name),
                ds.Q_e.sel(qubit=q.name),
                True,
                b_plot=plot_individual,
            )
            I_rot = ds.I_g.sel(qubit=q.name) * np.cos(angle) - ds.Q_g.sel(qubit=q.name) * np.sin(angle)
            hist = np.histogram(I_rot, bins=100)
            RUS_threshold = hist[1][1:][np.argmax(hist[0])]
            temperature = 1.054571817e-34*q.xy.RF_frequency/(1.380649e-23*np.log(1/ge-1))*1e3
            node.results["results"][q.name] = {
                "angle": float(angle),
                "threshold": float(threshold),
                "fidelity": float(fidelity),
                "confusion_matrix": np.array([[gg, ge], [eg, ee]]),
                "rus_threshold": float(RUS_threshold),
                "temperature": float(temperature),
            }

        # %% {Plotting}
        grid = QubitGrid(ds, [q.grid_location for q in qubits])
        for ax, qubit in grid_iter(grid):
            qn = qubit["qubit"]
            ax.plot(
                ds.I_g.sel(qubit=qn) * np.cos(node.results["results"][qn]["angle"])
                - ds.Q_g.sel(qubit=qn) * np.sin(node.results["results"][qn]["angle"]),
                ds.I_g.sel(qubit=qn) * np.sin(node.results["results"][qn]["angle"])
                + ds.Q_g.sel(qubit=qn) * np.cos(node.results["results"][qn]["angle"]),
                ".",
                alpha=0.2,
                label="Ground",
                markersize=1,
            )
            ax.plot(
                ds.I_e.sel(qubit=qn) * np.cos(node.results["results"][qn]["angle"])
                - ds.Q_e.sel(qubit=qn) * np.sin(node.results["results"][qn]["angle"]),
                ds.I_e.sel(qubit=qn) * np.sin(node.results["results"][qn]["angle"])
                + ds.Q_e.sel(qubit=qn) * np.cos(node.results["results"][qn]["angle"]),
                ".",
                alpha=0.2,
                label="Excited",
                markersize=1,
            )
        node.results["figure_IQ_blobs"] = grid.fig

        # %% {Update_state}
        if node.parameters.load_data_id is None:
            with node.record_state_updates():
                for qubit in qubits:
                    qubit.resonator.operations[operation_name].integration_weights_angle -= float(
                        node.results["results"][qubit.name]["angle"]
                    )
                    qubit.resonator.operations[operation_name].threshold = (
                        float(node.results["results"][qubit.name]["threshold"])
                        * qubit.resonator.operations[operation_name].length
                        / 2**12
                    )
                    qubit.resonator.operations[operation_name].rus_exit_threshold = (
                        float(node.results["results"][qubit.name]["rus_threshold"])
                        * qubit.resonator.operations[operation_name].length
                        / 2**12
                    )
                    if operation_name == "readout":
                        qubit.resonator.confusion_matrix = node.results["results"][qubit.name]["confusion_matrix"].tolist()

        # node.machine = machine
        # node.save()
    P1 = [node.results["results"][q.name]["confusion_matrix"][0, 1] for q in qubits]
    Temperature = [node.results["results"][q.name]["temperature"] for q in qubits]
    qubits_name = [q.name for q in qubits]

    return (Temperature, "Temperature"), qubits_name
