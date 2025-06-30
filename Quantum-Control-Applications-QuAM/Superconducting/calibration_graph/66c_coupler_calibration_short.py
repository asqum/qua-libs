# %%
"""
COUPLER_CALIBRATION_SHORT    
Reference paper: https://arxiv.org/pdf/2410.15041

This protocol corrects flux pulse distortions in tunable couplers using a long-time correction approach:
Long-time correction (32ns-10 μs): Use second-order reversed convolution to deconvolve exponential relaxation (τ ∼25 μs)
from measured step responses.

The method exploits qubit-coupler coupling to map distortions via qubit population shifts, bypassing dedicated coupler readout. 
We inject square flux pulses, measure distorted qubit responses, then generate predistorted waveforms using fitted parameters 
(A, τ) from exponential decay models. 

Prerequisites:
    - 66a calibrate deress pi pulse
    - 66b coupler calibration long
Before proceeding to the next node:
    - 
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, readout_state
from quam_libs.lib.fit_utils import fit_resonator
from quam_libs.lib.qua_datasets import apply_angle, subtract_slope, convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Optional, List
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal, Optional, List
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names


# %% {Node_parameters}
class Parameters(NodeParameters):


    qubit_pairs: Optional[List[str]] = ['coupler_q2_q3','coupler_q1_q2'] # ["coupler_q1_q2"]
    num_averages: int = 2
    flux_point_joint_or_independent_or_pairwise: Literal["joint", "independent", "pairwise"] = "joint"
    reset_type: Literal['active', 'thermal'] = "thermal"
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None
    
    # q1_q2:
    # coupler_flux_min : float = 0.175 #relative to the coupler set point
    # coupler_flux_max : float = 0.240 #relative to the coupler set point
    # q2_q3:
    coupler_flux_min : float = 0.180 #relative to the coupler set point
    coupler_flux_max : float = 0.255 #relative to the coupler set point
    # q3_q4:
    # coupler_flux_min : float = 0.177 #relative to the coupler set point
    # coupler_flux_max : float = 0.230 #relative to the coupler set point

    use_state_discrimination: bool = False
    strong_coupling_amp : float = -0.2
    V_offset_stable_time : int  = 4000//4
    V_offset_max: float = 0.02
    V_offset_min: float = -0.02
    V_offset_step: float = 0.005
    min_wait_time: int = 32//4
    max_wait_time: int = 10000//4
    total_time_step: int = 50

node = QualibrationNode(name="66b_coupler_calibration_short", parameters=Parameters())
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


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

flux_point = node.parameters.flux_point_joint_or_independent_or_pairwise  # 'independent' or 'joint' or 'pairwise'

reset_coupler_bias = False
time_offset = (qubit_pairs[0].qubit_control.resonator.operations['readout'].length+
               qubit_pairs[0].qubit_control.xy.operations['dressed_x180'].length)
coupler_const_amp = qubit_pairs[0].coupler.operations['const'].amplitude
coupler_idle_amp = qubit_pairs[0].coupler.decouple_offset
V_stable_time = node.parameters.V_offset_stable_time+2
strong_coupling_amp = node.parameters.strong_coupling_amp
dcs = np.arange(
    node.parameters.V_offset_min,
    node.parameters.V_offset_max,
    node.parameters.V_offset_step)

ts = np.round(np.geomspace(
    node.parameters.min_wait_time,
    node.parameters.max_wait_time,
    node.parameters.total_time_step)).astype(int)
AT_constnt_list=[]
for qp in qubit_pairs:
    AT_constnt = qp.qubit_control.xy.operations['dressed_x180'].amplitude* qp.qubit_control.xy.operations['dressed_x180'].length
    AT_constnt_list.append(AT_constnt)
with program() as coupler_distortion:
    n = declare(int)
    dc = declare(fixed)
    t = declare(int)
    flux_coupler = declare(float)
    flux_qubit = declare(float)
    comp_flux_qubit = declare(float)
    n_st = declare_stream()
    new_amp=declare(fixed)
    tt = declare(fixed)
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
    
    machine.apply_all_flux_to_min()
    machine.apply_all_couplers_to_min()
    for i, qp in enumerate(qubit_pairs):
        print("qubit control: %s, qubit target: %s" %(qp.qubit_control.name, qp.qubit_target.name))
        # Bring the active qubits to the minimum frequency point
        machine.set_all_fluxes(flux_point, qp)
        if reset_coupler_bias:
            qp.coupler.set_dc_offset(0.0)
        wait(100)
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)  
            with for_each_(t,ts):
                with if_(t > 50):
                    with for_(*from_array(dc,dcs)):
                        qp.coupler.play('const', amplitude_scale=dc/coupler_const_amp,duration=V_stable_time)
                        align(qp.coupler.name,qp.qubit_control.xy.name)
                        qp.coupler.play("const", amplitude_scale = (strong_coupling_amp-coupler_idle_amp)/coupler_const_amp, duration = t + time_offset//4 )
                        wait(t,qp.qubit_control.xy.name)
                        qp.qubit_control.xy.play("dressed_x180")
                        align(qp.qubit_control.xy.name,qp.qubit_control.resonator.name)
                        if node.parameters.use_state_discrimination:
                            readout_state(qp.qubit_control, state_control[i])
                            save(state_control[i], state_st_control[i])
                        else:
                            qp.qubit_control.resonator.measure("readout", qua_vars=(I_control[i], Q_control[i]))
                            save(I_control[i], I_st_control[i])
                            save(Q_control[i], Q_st_control[i])
                with else_():
                    with for_(*from_array(dc,dcs)):
                        align(qp.coupler.name,qp.qubit_control.xy.name)
                        qp.coupler.play('const', amplitude_scale=dc/coupler_const_amp,duration=V_stable_time)
                        qp.coupler.play("const", amplitude_scale = (strong_coupling_amp-coupler_idle_amp)/coupler_const_amp, duration = t + time_offset//4 )
                        wait(V_stable_time-11,qp.qubit_control.xy.name) # 
                        
                        qp.qubit_control.xy.play("dressed_x180",amplitude_scale=AT_constnt_list[i]/Cast.to_fixed(t),duration=t)
                        align(qp.qubit_control.xy.name,qp.qubit_control.resonator.name)
                        if node.parameters.use_state_discrimination:
                            readout_state(qp.qubit_control, state_control[i])
                            save(state_control[i], state_st_control[i])
                        else:
                            qp.qubit_control.resonator.measure("readout", qua_vars=(I_control[i], Q_control[i]))
                            save(I_control[i], I_st_control[i])
                            save(Q_control[i], Q_st_control[i])

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            if node.parameters.use_state_discrimination:
                state_st_control[i].buffer(len(dcs)).buffer(len(ts)).average().save(f"state_control{i + 1}")
            else:
                I_st_control[i].buffer(len(dcs)).buffer(len(ts)).average().save(f"I_control{i + 1}")
                Q_st_control[i].buffer(len(dcs)).buffer(len(ts)).average().save(f"Q_control{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    simulation_config = SimulationConfig(duration=30000)
    job = qmm.simulate(config,coupler_distortion,simulation_config)
    samples = job.get_simulated_samples()
    waveform_report = job.get_simulated_waveform_report()
    waveform_report.create_plot(samples,plot=True)
else:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(coupler_distortion)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

#%% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {  "V_offset": dcs, "t_delay": ts})
    else:
        ds, machine = load_dataset(node.parameters.load_data_id)
        
    node.results = {"ds": ds}
# %%
if not node.parameters.simulate:
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qubit_pair in grid_iter(grid):
        if node.parameters.use_state_discrimination:
            ds.sel(qubit=qubit_pair['qubit']).state_control.plot(ax=ax)
            ax.set_yscale('log')
        else:
            ds.sel(qubit=qubit_pair['qubit']).I_control.plot(ax=ax)
            ax.set_yscale('log')
    plt.tight_layout()
    plt.show()
    node.results["figure_distortion"] = grid.fig
# %% {Save_results}
if not node.parameters.simulate:    
    node.outcomes = {q.name: "successful" for q in qubit_pairs}
    node.results['initial_parameters'] = node.parameters.model_dump()
    node.machine = machine
    node.save()
# %%
