# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.lib import find_c_with_q
from quam_libs.macros import qua_declaration, active_reset, active_reset_simple
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
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
import xarray as xr
from sklearn.mixture import GaussianMixture


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = ['q1', 'q2']
    num_runs: int = 2000
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    c_min_v: float = -0.2
    c_max_v: float = 0.2
    v_nums: int = 100
    outliers_threshold: float = 0.98
    plot_raw: bool = True
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    z_rising_time_ns:int = 1000


node = QualibrationNode(name="07c_Readout_Power_Optimization", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
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

qubit_pairs = find_c_with_q(qubit_list=[q.name for q in qubits], coupler_list=machine.active_qubit_pairs)

# %% {QUA_program}
n_runs = node.parameters.num_runs  # Number of runs
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
reset_type = node.parameters.reset_type_thermal_or_active  # "active" or "thermal"
dcs = np.linspace(node.parameters.c_min_v, node.parameters.c_max_v, node.parameters.v_nums)

# Check out the longest pi-pulse duration
xy_dura = 0
for qubit in qubits:
    if qubit.xy.operations["x180"].length > xy_dura:
        xy_dura = qubit.xy.operations["x180"].length

# Check out the longest readout time
ro_dura = 0
for qubit in qubits:
    if qubit.resonator.operations["readout"].length > ro_dura:
        ro_dura = qubit.resonator.operations["readout"].length



with program() as iq_blobs_snr:
    I_g, I_g_st, Q_g, Q_g_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    I_e, I_e_st, Q_e, Q_e_st, _, _ = qua_declaration(num_qubits=num_qubits)
    dc = declare(fixed)

    # --- SNR calculation related ---
    snr_st = [declare_stream() for _ in range(num_qubits)]
    # averaging and square sum accumulator
    sum_Ig = [declare(fixed, value=0.0) for _ in range(num_qubits)]
    sum_Qg = [declare(fixed, value=0.0) for _ in range(num_qubits)]
    sum_Ie = [declare(fixed, value=0.0) for _ in range(num_qubits)]
    sum_Qe = [declare(fixed, value=0.0) for _ in range(num_qubits)]
    sum_sq_g = [declare(fixed, value=0.0) for _ in range(num_qubits)]
    sum_sq_e = [declare(fixed, value=0.0) for _ in range(num_qubits)]
    
    inv_n = 1.0 / n_runs # avoid > 8

    if not node.parameters.simulate:
        machine.apply_all_couplers_to_min()
        for i, qubit in enumerate(qubits):

            # Bring the active qubits to the desired frequency point
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)
            if "c" in qubit.id: qubit.z.set_dc_offset(qubit.z.joint_offset) # for coupler-test case
            qubit.z.settle()
            qubit.align()
    
    align()       
    
    with for_(*from_array(dc, dcs)):
        
        # --- 每個新 dc 開始前，重置累加器 ---
        for i in range(num_qubits):
            assign(sum_Ig[i], 0.0)
            assign(sum_Qg[i], 0.0)
            assign(sum_Ie[i], 0.0)
            assign(sum_Qe[i], 0.0)
            assign(sum_sq_g[i], 0.0)
            assign(sum_sq_e[i], 0.0)
            

        with for_(n, 0, n < n_runs, n + 1):
            # ground iq blobs for all qubits
            save(n, n_st)
        

            for i, qubit in enumerate(qubits):
                if reset_type == "active":
                    active_reset_simple(qubit, "readout")
                elif reset_type == "thermal":
                    if node.parameters.simulate:
                        qubit.wait(16 * u.ns)
                    else:
                        qubit.wait(qubit.thermalization_time * u.ns)
                else:
                    raise ValueError(f"Unrecognized reset type {reset_type}.")

            align()

            for i, qp in enumerate(qubit_pairs):
                qp.coupler.play(
                    "const", 
                    amplitude_scale = dc / qp.coupler.operations["const"].amplitude, 
                    duration = node.parameters.z_rising_time_ns//4 + ro_dura//4
                )

                # flux crosstalk compenstation
                comp_control, comp_target = declare(fixed), declare(fixed)
                if "FCC" in qp.extras:           
                    if 'control' in qp.extras['FCC']:
                        assign(comp_control,  qp.extras["FCC"]["control"] * dc )
                    else:
                        assign(comp_control, 0.0)
                    if 'target' in qp.extras['FCC']:
                        assign(comp_target,  qp.extras["FCC"]["target"] * dc )
                    else:
                        assign(comp_target, 0.0)  
                else:
                    assign(comp_control, 0.0)
                    assign(comp_target, 0.0)

                qp.qubit_control.z.play(
                    "const", 
                    amplitude_scale = comp_control / qp.qubit_control.z.operations["const"].amplitude, 
                    duration = node.parameters.z_rising_time_ns//4 + ro_dura//4
                )
                qp.qubit_target.z.play(
                    "const", 
                    amplitude_scale = comp_target / qp.qubit_control.z.operations["const"].amplitude, 
                    duration = node.parameters.z_rising_time_ns//4 + ro_dura//4
                )

                # end of compensation
            
            for i, qubit in enumerate(qubits):
                qubit.resonator.wait(node.parameters.z_rising_time_ns//4)
                qubit.resonator.measure("readout", qua_vars=(I_g[i], Q_g[i]))

                assign(sum_Ig[i], sum_Ig[i] + I_g[i] * inv_n)
                assign(sum_Qg[i], sum_Qg[i] + Q_g[i] * inv_n)
                assign(sum_sq_g[i], sum_sq_g[i] + (I_g[i] * I_g[i] + Q_g[i] * Q_g[i]) * inv_n)

                save(I_g[i], I_g_st[i])
                save(Q_g[i], Q_g_st[i])
            
            align()

            for i, qubit in enumerate(qubits):
                if reset_type == "active":
                    active_reset_simple(qubit, "readout")
                elif reset_type == "thermal":
                    if node.parameters.simulate:
                        qubit.wait(16 * u.ns)
                    else:
                        qubit.wait(qubit.thermalization_time * u.ns)
                else:
                    raise ValueError(f"Unrecognized reset type {reset_type}.")
            
            align()


            for i, qp in enumerate(qubit_pairs):
                qp.coupler.play(
                    "const", 
                    amplitude_scale = dc / qp.coupler.operations["const"].amplitude, 
                    duration = node.parameters.z_rising_time_ns//4 + xy_dura//4 + ro_dura//4
                )

                # Flux crosstalk compensation
                comp_control, comp_target = declare(fixed), declare(fixed)
                if "FCC" in qp.extras:           
                    if 'control' in qp.extras['FCC']:
                        assign(comp_control,  qp.extras["FCC"]["control"] * dc )
                    else:
                        assign(comp_control, 0.0)
                    if 'target' in qp.extras['FCC']:
                        assign(comp_target,  qp.extras["FCC"]["target"] * dc )
                    else:
                        assign(comp_target, 0.0)  
                else:
                    assign(comp_control, 0.0)
                    assign(comp_target, 0.0)

                qp.qubit_control.z.play(
                    "const", 
                    amplitude_scale = comp_control / qp.qubit_control.z.operations["const"].amplitude, 
                    duration = node.parameters.z_rising_time_ns//4 + ro_dura//4
                )
                qp.qubit_target.z.play(
                    "const", 
                    amplitude_scale = comp_target / qp.qubit_control.z.operations["const"].amplitude, 
                    duration = node.parameters.z_rising_time_ns//4 + ro_dura//4
                )
                # end of compensation
            
            # align()

            for i, qubit in enumerate(qubits):
                qubit.xy.wait(node.parameters.z_rising_time_ns//4)
                qubit.xy.play("x180")
            
            # align()
            
            # for i, qubit in enumerate(qubits):
                qubit.resonator.measure("readout", qua_vars=(I_e[i], Q_e[i]))

                assign(sum_Ie[i], sum_Ie[i] + I_e[i] * inv_n)
                assign(sum_Qe[i], sum_Qe[i] + Q_e[i] * inv_n)
                assign(sum_sq_e[i], sum_sq_e[i] + (I_e[i]**2 + Q_e[i]**2) * inv_n)

                save(I_e[i], I_e_st[i])
                save(Q_e[i], Q_e_st[i])

            align()
        
        # --- Calculate SNR after shotting ---
        for i in range(num_qubits):
            dist_sq = declare(fixed)
            var_g = declare(fixed)
            var_e = declare(fixed)
            var_max = declare(fixed)
            snr_val = declare(fixed)

            # 1. S**2
            assign(dist_sq, (sum_Ie[i] - sum_Ig[i])**2 + (sum_Qe[i] - sum_Qg[i])**2)
            
            # 2. σ**2
            assign(var_g, sum_sq_g[i] - (sum_Ig[i]**2 + sum_Qg[i]**2))
            assign(var_e, sum_sq_e[i] - (sum_Ie[i]**2 + sum_Qe[i]**2))
            
            # 3. Select Max Variance
            with if_(var_e > var_g):
                assign(var_max, var_e)
            with else_():
                assign(var_max, var_g)
            
            # 4. SNR = sqrt(Dist^2 / var_max)
            with if_(var_max > 0):
                assign(snr_val, Math.sqrt(Math.div(dist_sq, var_max)))
            with else_():
                assign(snr_val, 0.0)

            save(snr_val, snr_st[i])

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_g_st[i].buffer(n_runs).buffer(len(dcs)).save(f"I_g{i + 1}")
            Q_g_st[i].buffer(n_runs).buffer(len(dcs)).save(f"Q_g{i + 1}")
            I_e_st[i].buffer(n_runs).buffer(len(dcs)).save(f"I_e{i + 1}")
            Q_e_st[i].buffer(n_runs).buffer(len(dcs)).save(f"Q_e{i + 1}")

            # SNR
            snr_st[i].buffer(len(dcs)).save(f"snr{i + 1}")


if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, iq_blobs_snr, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    samples.con1.plot()
    node.results = {"figure": plt.gcf()}
    wf_report = job.get_simulated_waveform_report()
    wf_report.create_plot(samples, plot=True, save_path=None)
    node.machine = machine
    node.save()

elif node.parameters.load_data_id is None:
    data_list = ["n"]
    for i in range(num_qubits):
        data_list.extend([f"I_g{i+1}", f"Q_g{i+1}", f"I_e{i+1}", f"Q_e{i+1}", f"snr{i+1}"])
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(iq_blobs_snr)
        results = fetching_tool(job, data_list, mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_runs, start_time=results.start_time)


# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        ds = fetch_results_as_xarray(
            job.result_handles, 
            qubits, 
            {"voltage": dcs, "N": np.linspace(1, n_runs, n_runs)}
        )

        ds_rearranged = xr.Dataset()

        # merge I, Q
        ds_rearranged["I"] = xr.concat([ds.I_g, ds.I_e], dim="state").assign_coords(state=[0, 1])
        ds_rearranged["Q"] = xr.concat([ds.Q_g, ds.Q_e], dim="state").assign_coords(state=[0, 1])

    
        ds = ds.assign_coords({"voltage": (["qubit", "voltage"], np.array([dcs * 1 for q in qubits]))})

        for var in ds.coords:
            if var not in ds_rearranged.coords:
                ds_rearranged[var] = ds[var]

        for var in ds.data_vars:
            if var not in ["I_g", "I_e", "Q_g", "Q_e"]:
                ds_rearranged[var] = ds[var]

        ds = ds_rearranged
    else:
        ds, machine, json_data, qubits, node.parameters = load_dataset(node.parameters.load_data_id, parameters = node.parameters)


    node.results = {"ds": ds, "results": {}, "figs": {}}

    #%% Plot
    ## Plot SNR
    for q in ds.qubit.values:
        plt.figure(figsize=(8, 4))
        ds.snr.sel(qubit=q).plot(marker='o')
        plt.title(f"SNR vs Coupler DC - Qubit {q}")
        plt.xlabel("Coupler DC (V)")
        plt.ylabel("SNR")
        plt.grid(True)
        plt.show()
        node.results["figs"][f"Figure_{q}_SNR"] = plt.gcf()

    ## Plot Raw 
    if node.parameters.plot_raw:
        fig, axes = plt.subplots(
            ncols=num_qubits,
            nrows=len(ds.voltage),
            sharex=False,
            sharey=False,
            squeeze=False,
            figsize=(5 * num_qubits, 5 * len(ds.voltage)),
        )
        for voltage, ax1 in zip(ds.voltage, axes):
            for q, ax2 in zip(list(qubits), ax1):
                ds_q = ds.sel(qubit=q.name, voltage=voltage)
                ax2.plot(ds_q.I.sel(state=0), ds_q.Q.sel(state=0), ".", alpha=0.2, label="Ground", markersize=2)
                ax2.plot(ds_q.I.sel(state=1), ds_q.Q.sel(state=1), ".", alpha=0.2, label="Excited", markersize=2)
                ax2.set_xlabel("I")
                ax2.set_ylabel("Q")
                ax2.set_title(f"{q.name}, {float(voltage)}")
                ax2.axis("equal")
        plt.show()
        node.results["figure_raw_data"] = fig

    # Plot Centroid vs coupler bias
    plt.figure(figsize=(8, 6))
    for q in ds.qubit.values:
        # 先對 N 維度取平均，得到每顆球的中心點 (Centroid)
        avg_I = ds.I.sel(qubit=q, state=0).mean(dim="N")
        avg_Q = ds.Q.sel(qubit=q, state=0).mean(dim="N")
        
        plt.plot(avg_I, avg_Q, '.-', label=f"Qubit {q} path")
        # 標註起點 (第一個 DC 點)
        plt.annotate("Start", (avg_I[0], avg_Q[0]))

    plt.title("IQ Centroid Drift with DC Flux")
    plt.xlabel("Average I"); plt.ylabel("Average Q")
    plt.legend()
    plt.axis('equal')
    plt.show()
    node.results["figs"][f"Figure_MovingCentroid"] = plt.gcf()


    # %% {Update_state}
    if node.parameters.load_data_id is None:
        # with node.record_state_updates():
        #     for qubit in qubits:
        #         qubit.resonator.operations["readout"].integration_weights_angle -= float(
        #             node.results["results"][qubit.name]["angle"]
        #         )
        #         qubit.resonator.operations["readout"].threshold = float(node.results["results"][qubit.name]["threshold"])
        #         qubit.resonator.operations["readout"].rus_exit_threshold = float(
        #             node.results["results"][qubit.name]["rus_threshold"]
        #         )
        #         qubit.resonator.operations["readout"].amplitude = float(node.results["results"][qubit.name]["best_amp"])
        #         qubit.resonator.confusion_matrix = node.results["results"][qubit.name]["confusion_matrix"].tolist()
        pass

        # %% {Save_results}
        node.outcomes = {q.name: "successful" for q in qubits}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()

# %%
