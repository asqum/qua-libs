"""
Doing TWPA calibration by observing IQ-blobs with 2 Gaussian Mixtrue fitting to get the best RO SNR. 
Treat TWPA as a Transmon and use its XY operations to be the pumping source and charging it with 10 us square pulse.

- Prerequisites:
    1. A transmon qubit for TWPA and wired. 
    2. 'TWPA_pump' xy operation in TWPA transmon with square waveform
    3.
"""




# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters

from quam_libs.components import QuAM
from quam_libs.lib.instrument_limits import instrument_limits
from quam_libs.macros import qua_declaration, measure_with_TWPA_beta
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import (
    fetch_results_as_xarray,
    restore_load_data_id,
    resolve_qubits_from_node,
)
from quam_libs.experiments.iq_blobs.fetch_dataset import fetch_dataset
from quam_libs.experiments.simulation import simulate_and_plot
from quam_libs.lib.fit import peaks_dips
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
from quam_libs.lib.qubit_thermometer import repetition_data, StateDiscrimination, prepare_dataset_for_qcat




class Parameters(NodeParameters):

    qubits: Optional[List[str]] = ['q4', 'q5']
    twpa_q:str = 'qTWPA'
    num_runs: int = 1200
    flux_point_joint_or_independent: Literal["joint"] = "joint"
    multiplexed:bool = True
    simulate: bool = False
    simulation_duration_ns: int = 25_000
    timeout: int = 1000
    load_data_id: Optional[int] = None
    use_waveform_report: bool = True
    


node = QualibrationNode(name="beta_TWPA_calibre_verify", parameters=Parameters())

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# node.machine = machine# Open Communication with the QOP

if node.parameters.load_data_id is None:
    qmm = machine.connect(timeout=300)

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits_item = machine.active_qubits
else:
    qubits_item = [machine.qubits[q] for q in node.parameters.qubits]

qubits = machine.get_qubits_used_in_node(node.parameters)
num_qubits = len(qubits)

twpa = machine.qubits[node.parameters.twpa_q]

# Generate the OPX and Octave configurations
config = machine.generate_config()

align_elements = qubits_item
align_elements.append(twpa)

# %% {QUA_program}
n_avg = node.parameters.num_runs 
flux_point = node.parameters.flux_point_joint_or_independent

with program() as qua_separate:
    reset_global_phase()
    I_g, I_g_st, Q_g, Q_g_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    I_e, I_e_st, Q_e, Q_e_st, _, _ = qua_declaration(num_qubits=num_qubits)

    for multiplexed_qubits in qubits.batch():
        if not node.parameters.simulate:
            machine.set_all_fluxes(flux_point=flux_point, target=list(multiplexed_qubits.values())[0])
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)

            # measure |0> IQ blob for all qubits
            if not node.parameters.simulate:
                for i, qubit in multiplexed_qubits.items():
                    qubit.wait(qubit.thermalization_time * u.ns)

            align(*[q.xy.name for q in align_elements] +
                    [q.resonator.name for q in align_elements] +
                    [q.z.name for q in align_elements])

            # pumping ON
            twpa.xy.play('TWPA_pump')
            for i, qubit in multiplexed_qubits.items():
                # measure_with_TWPA_beta(qubit, twpa, (I_g[i], Q_g[i]), i)
                qubit.resonator.wait(twpa.extras['charging_time'] * u.ns)
                qubit.resonator.measure("readout", qua_vars=(I_g[i], Q_g[i]))
                qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                save(I_g[i], I_g_st[i])
                save(Q_g[i], Q_g_st[i])

            align(*[q.xy.name for q in align_elements] +
                    [q.resonator.name for q in align_elements] +
                    [q.z.name for q in align_elements])


            # measure |1> IQ blob for all qubits
            if not node.parameters.simulate:
                for i, qubit in multiplexed_qubits.items():
                    qubit.wait(qubit.thermalization_time * u.ns)

            align(*[q.xy.name for q in align_elements] +
                    [q.resonator.name for q in align_elements] +
                    [q.z.name for q in align_elements])

            for i, qubit in multiplexed_qubits.items():
                qubit.xy.play("x180")

            align(*[q.xy.name for q in align_elements] +
                    [q.resonator.name for q in align_elements] +
                    [q.z.name for q in align_elements])

            twpa.xy.play('TWPA_pump')
            for i, qubit in multiplexed_qubits.items():
                qubit.resonator.wait(twpa.extras['charging_time'] * u.ns)
                qubit.resonator.measure("readout", qua_vars=(I_e[i], Q_e[i]))
                qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)

                # measure_with_TWPA_beta(qubit, twpa, (I_e[i], Q_e[i]), i)
                save(I_e[i], I_e_st[i])
                save(Q_e[i], Q_e_st[i])

            if node.parameters.multiplexed:
                align(*[q.xy.name for q in align_elements] +
                        [q.resonator.name for q in align_elements] +
                        [q.z.name for q in align_elements])
            else:
                align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_g_st[i].save_all(f"I_g{i + 1}")
            Q_g_st[i].save_all(f"Q_g{i + 1}")
            I_e_st[i].save_all(f"I_e{i + 1}")
            Q_e_st[i].save_all(f"Q_e{i + 1}")

with program() as qua_overlap:
    reset_global_phase()
    I_g, I_g_st, Q_g, Q_g_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    I_e, I_e_st, Q_e, Q_e_st, _, _ = qua_declaration(num_qubits=num_qubits)

    for multiplexed_qubits in qubits.batch():
        if not node.parameters.simulate:
            machine.set_all_fluxes(flux_point=flux_point, target=list(multiplexed_qubits.values())[0])
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            

            # measure |0> IQ blob for all qubits
            if not node.parameters.simulate:
                for i, qubit in multiplexed_qubits.items():
                    qubit.wait(qubit.thermalization_time * u.ns)

            align(*[q.xy.name for q in align_elements] +
                    [q.resonator.name for q in align_elements] +
                    [q.z.name for q in align_elements])

            # pumping ON
            twpa.xy.play('TWPA_pump')
            for i, qubit in multiplexed_qubits.items():
                qubit.resonator.wait(twpa.xy.operations['TWPA_pump'].length * u.ns - 2*qubit.resonator.operations.readout.length * u.ns)
                qubit.resonator.measure("readout", qua_vars=(I_g[i], Q_g[i]))
                qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                save(I_g[i], I_g_st[i])
                save(Q_g[i], Q_g_st[i])

            align(*[q.xy.name for q in align_elements] +
                    [q.resonator.name for q in align_elements] +
                    [q.z.name for q in align_elements])

            wait(1 * u.us)

            # measure |1> IQ blob for all qubits
            if not node.parameters.simulate:
                for i, qubit in multiplexed_qubits.items():
                    qubit.wait(qubit.thermalization_time * u.ns)

            align(*[q.xy.name for q in align_elements] +
                    [q.resonator.name for q in align_elements] +
                    [q.z.name for q in align_elements])

            # pumping ON
            twpa.xy.play('TWPA_pump')

            for i, qubit in multiplexed_qubits.items():
                qubit.xy.wait(twpa.xy.operations['TWPA_pump'].length * u.ns - 2*qubit.resonator.operations.readout.length * u.ns - qubit.xy.operations['x180'].length * u.ns)
                qubit.xy.play('x180')
                qubit.resonator.wait(qubit.xy.operations['x180'].length * u.ns)
                qubit.resonator.measure("readout", qua_vars=(I_e[i], Q_e[i]))
                qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                save(I_e[i], I_e_st[i])
                save(Q_e[i], Q_e_st[i])

            if node.parameters.multiplexed:
                align(*[q.xy.name for q in align_elements] +
                        [q.resonator.name for q in align_elements] +
                        [q.z.name for q in align_elements])
            else:
                align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_g_st[i].save_all(f"I_g{i + 1}")
            Q_g_st[i].save_all(f"Q_g{i + 1}")
            I_e_st[i].save_all(f"I_e{i + 1}")
            Q_e_st[i].save_all(f"Q_e{i + 1}")


# %% {Simulate_or_execute}
if twpa.extras['pumping_mode'] == 'overlap':
    qua_prog = qua_overlap
else:
    qua_prog = qua_separate

if node.parameters.simulate:
    samples, fig = simulate_and_plot(qmm, config, qua_prog, node.parameters)
    node.results = {"figure": fig}
    node.machine = machine
    node.save()

else:
    if node.parameters.load_data_id is not  None:
        load_data_id = node.parameters.load_data_id
        node = node.load_from_id(load_data_id)
        ds = node.results["ds"]
        restore_load_data_id(node, load_data_id)
        machine = node.machine
        qubits = resolve_qubits_from_node(machine, node)
    else:
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(qua_prog)
            
            results = fetching_tool(job, ["n"], mode="live")
            while results.is_processing():
                n = results.fetch_all()[0]
                progress_counter(n, n_avg, start_time=results.start_time)

        ds = fetch_dataset(job, qubits, node.parameters)

    node.results = {"ds": ds}

# %% {Data_analysis}


if not node.parameters.simulate:
    models = {q.name:[] for q in qubits}
    trained_params = {}
    blob_sigma = {q.name:[] for q in qubits}
    blob_dist = {q.name:[] for q in qubits}
    snr = {q.name:[] for q in qubits}
    ds_for_qcat = prepare_dataset_for_qcat(ds)

    dss = prepare_dataset_for_qcat(ds)
            
    sep_data = repetition_data(dss, repetition_dim="qubit")


    for sq_data in sep_data:
        qubit_name = sq_data["qubit"].values.item()

        analysis = StateDiscrimination(sq_data)
        analysis._start_analysis()
        models[qubit_name].append(analysis)


# %% {Plot}
## prepare ground 1D histogram
grid = QubitGrid(ds, [q.grid_location for q in qubits])
for ax, qubit in grid_iter(grid):
    Teff = models[qubit['qubit']][0].show_thermal_analysis(qubit['qubit'], machine.qubits[qubit['qubit']].xy.RF_frequency*1e-9, ax)
    
plt.suptitle("Ground State Preparation")
plt.tight_layout()
plt.show()
node.results["Pg_1DHistogran"] = grid.fig

## raw data plot
grid_2 = QubitGrid(ds, [q.grid_location for q in qubits])
for ax, qubit in grid_iter(grid_2): 
    models[qubit['qubit']][0].show_scatter_analysis(qubit['qubit'], ax)
plt.suptitle("IQ Blobs")
plt.tight_layout()
plt.show()
node.results["Colored_Scatter"] = grid_2.fig
## outliers
grid_3 = QubitGrid(ds, [q.grid_location for q in qubits])
for ax, qubit in grid_iter(grid_3): 
    models[qubit['qubit']][0].show_outliers_analysis(qubit['qubit'], ax)
plt.suptitle("Outliers Detection")
plt.tight_layout()
plt.show()
node.results["Outliers_Detection"] = grid_3.fig

## error breakdown
grid_4 = QubitGrid(ds, [q.grid_location for q in qubits])
for ax, qubit in grid_iter(grid_4): 
    models[qubit['qubit']][0].show_error_analysis(qubit['qubit'], ax)

plt.show()
node.results["Error_source"] = grid_4.fig

## error breakdown
grid_5 = QubitGrid(ds, [q.grid_location for q in qubits])
for ax, qubit in grid_iter(grid_5): 
    models[qubit['qubit']][0].show_confusion_matrix(qubit['qubit'], ax)
plt.tight_layout()
plt.show()
node.results["confusion_matrix"] = grid_5.fig

# %%
for ax, qubit in grid_iter(grid_5): 
    print(max(models[qubit['qubit']][0].analysis_result['outlier_probability']))

# %% {Save_results}

node.results["initial_parameters"] = node.parameters.model_dump()
node.save()  
# %%
