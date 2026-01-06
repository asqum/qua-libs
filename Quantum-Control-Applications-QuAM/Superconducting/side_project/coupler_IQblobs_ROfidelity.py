# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.lib import find_c_with_q
from quam_libs.macros import qua_declaration, active_reset, active_reset_simple, readout_state_gef
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
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
import xarray as xr
import matplotlib.animation as animation


# %%
from scipy.interpolate import UnivariateSpline
def auto_optimize_readout_voltage(ds, qubit_name, plot=True):
    """
    自動估算平滑係數並尋找最佳讀取電壓。
    """
    # 1. 取得數據並轉為 dB
    fidelity_data = ds.ROfidelity.sel(qubit=qubit_name).mean(dim="N")
    
    v_data = ds.voltage.values
    snr_values = fidelity_data.values
    m = len(v_data)

    # 2. 自動估算雜訊 (Sigma)
    # 透過計算相鄰點的差值來估算高頻雜訊
    sigma = np.std(np.diff(snr_values)) 
    
    # 3. 根據經驗公式設定自動 s
    # s = m * sigma^2 是統計學上的經典建議值
    auto_s = m * (sigma**2)
    
    # 限制 s 的範圍，避免過度平滑或完全沒平滑
    auto_s = np.clip(auto_s, 0.5, 20) 

    # 4. 進行擬合
    spline = UnivariateSpline(v_data, snr_values, s=auto_s)
    v_fine = np.linspace(v_data.min(), v_data.max(), 1000)
    snr_smooth = spline(v_fine)
    
    best_v = v_fine[np.argmax(snr_smooth)]
    max_snr_db = np.max(snr_smooth)

    # 5. 繪圖
    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(v_data, snr_values, 'b.', label='Raw Data', alpha=0.3)
        plt.plot(v_fine, snr_smooth, 'r-', lw=2, label=f'Auto Spline (s={auto_s:.2f})')
        plt.axvline(best_v, color='green', linestyle='--', 
                   label=f'Best V: {best_v:.3f}V\nSNR: {max_snr_db:.2f} dB')
        plt.title(f"{qubit_name}: Auto-optimized SNR")
        plt.xlabel("Coupler Voltage (V)")
        plt.ylabel("SNR (dB)")
        plt.legend()
        plt.show()

    return float(best_v)


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = ['q1', 'q2']
    num_runs: int = 4096
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    c_min_v: float = -0.2
    c_max_v: float = 0.2
    v_nums: int = 100
    outliers_threshold: float = 0.98
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
    n = declare(int)
    dc = declare(fixed)
    n_st = declare_stream()
    state_pg = [declare(int) for _ in range(num_qubits)]
    state_pe = [declare(int) for _ in range(num_qubits)]
    state_st_pg = [declare_stream() for _ in range(num_qubits)]
    state_st_pe = [declare_stream() for _ in range(num_qubits)]
      

    # --- RO fidelity calculation related ---
    ro_fidelity = [declare(fixed) for _ in range(num_qubits)] 
    ro_fidelity_st = [declare_stream() for _ in range(num_qubits)]
    
    inv_n = 1.0 / (2 * n_runs)

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
        
        # --- reset ---
        for i in range(num_qubits):
            assign(ro_fidelity[i], 0.0)
            

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
                readout_state_gef(qubit, state_pg[i])
                # save(state_pg[i], state_st_pg[i])
                with if_(state_pg[i] == 0):
                    assign(ro_fidelity[i], ro_fidelity[i] + inv_n)
                        

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
            
            for i, qubit in enumerate(qubits):
                readout_state_gef(qubit, state_pe[i])
                # save(state_pe[i], state_st_pe[i])
                with if_(state_pe[i] == 1):
                    assign(ro_fidelity[i], ro_fidelity[i] + inv_n)


            align()
        
        # To match stream_processing buffer
        for i in range(num_qubits):
            with for_(n, 0, n < n_runs, n + 1):
                save(ro_fidelity[i], ro_fidelity_st[i])

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            # retrieve RO fidelity
            ro_fidelity_st[i].buffer(n_runs).buffer(len(dcs)).save(f"ROfidelity{i + 1}")


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
        data_list.extend([f"ROfidelity{i+1}"])
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
            {"N": np.linspace(1, n_runs, n_runs), "voltage": dcs}
        )

        ds_rearranged = xr.Dataset()
        ds = ds.assign_coords({"voltage": (["qubit", "voltage"], np.array([dcs * 1 for q in qubits]))})

        for var in ds.coords:
            if var not in ds_rearranged.coords:
                ds_rearranged[var] = ds[var]

        ds = ds_rearranged
    else:
        ds, machine, json_data, qubits, node.parameters = load_dataset(node.parameters.load_data_id, parameters = node.parameters)


    node.results = {"ds": ds, "results": {}, "figs": {}}

    #%% Plot
    ## Plot SNR
    best_amp = {}
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    ds["RO_fidelity_100"] = 100 * ds.ROfidelity
    for ax, qubit in grid_iter(grid):
        node.results["results"][qubit["qubit"]] = {}
        optimal_voltage = auto_optimize_readout_voltage(ds, qubit["qubit"], False)
        node.results["results"][qubit["qubit"]]["best_coupler_voltage"] = optimal_voltage
        ds.RO_fidelity_100.sel(qubit=qubit["qubit"]).mean(dim="N").plot(ax=ax)
        ax.axvline(optimal_voltage, color="k", linestyle="dashed")
        ax.set_xlabel("Coupler flux pulse amplitude (V)")
        ax.set_ylabel("RO fidelity (%)")
        ax.set_title(f'{qubit["qubit"]}, coupler at {round(optimal_voltage, 3)} is good')
        ax.grid()
    grid.fig.suptitle("Coupler flux pulse vs Readout SNR")
    
    plt.tight_layout()
    plt.show()
    node.results["figs"][f"Figure_RO_fidelity"] = grid.fig

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
