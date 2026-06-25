"""
        IQ BLOBS
This sequence involves measuring the state of the resonator 'N' times, first after thermalization (with the qubit
in the |g> state) and then after applying a pi pulse to the qubit (bringing the qubit to the |e> state) successively.
The resulting IQ blobs are displayed, and the data is processed to determine:
    - The rotation angle required for the integration weights, ensuring that the separation between |g> and |e> states
      aligns with the 'I' quadrature.
    - The threshold along the 'I' quadrature for effective qubit state discrimination.
    - The readout fidelity matrix, which is also influenced by the pi pulse fidelity.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit.wait(qubit.thermalization_time * u.ns) spectroscopy, power_rabi and updated the state.
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the rotation angle (rotation_angle) in the state.
    - Update the g -> e thresholds (threshold & rus_threshold) in the state.
    - Update the confusion matrices in the state.
    - Save the current state
"""


# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.animation import FuncAnimation, PillowWriter
from qualibrate import QualibrationNode
from quam_libs.components import QuAM
from quam_libs.experiments.iq_blobs.fetch_dataset import fetch_dataset
from quam_libs.experiments.iq_blobs.parameters import Parameters
from quam_libs.experiments.simulation import simulate_and_plot
from quam_libs.macros import qua_declaration, active_reset, active_reset_simple, active_reset_gef
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from qualang_tools.analysis.discriminator import two_state_discriminator
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *


# %% {Node_parameters}
node = QualibrationNode(
    name="07b_IQ_Blobs",
    parameters=Parameters(
        qubits=None,
        multiplexed=False,
        flux_point_joint_or_independent="independent",
        num_runs=4096*1,
        reset_type_thermal_or_active = 'active',
        load_data_id=None,
        simulate=False,
        simulation_duration_ns=1000,
        use_waveform_report=False
    )
)

# %% {Initialize_QuAM_and_QOP}
u = unit(coerce_to_integer=True)

machine = QuAM.load()
node.machine = machine
# machine.network["port"] = int(access_port)

# print(f"Machine access port :{access_port}")
if node.parameters.load_data_id is None:
    qmm = machine.connect()

qubits = machine.get_qubits_used_in_node(node.parameters)
num_qubits = len(qubits)
print(num_qubits)

config = machine.generate_config()

# %% {QUA_program}
n_runs = node.parameters.num_runs
flux_point = node.parameters.flux_point_joint_or_independent
reset_type = node.parameters.reset_type_thermal_or_active
operation_name = node.parameters.operation_name

with program() as iq_blobs:
    reset_global_phase()
    I_g, I_g_st, Q_g, Q_g_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    I_e, I_e_st, Q_e, Q_e_st, _, _ = qua_declaration(num_qubits=num_qubits)

    for multiplexed_qubits in qubits.batch():
        machine.set_all_fluxes(flux_point=flux_point, target=list(multiplexed_qubits.values())[0])

        with for_(n, 0, n < n_runs, n + 1):
            save(n, n_st)

            if not node.parameters.simulate:
                # measure ground-state IQ blob for all qubits
                for i, qubit in multiplexed_qubits.items():
                    if reset_type == "active":
                        active_reset(qubit)
                        # active_reset_gef(qubit)
                    elif reset_type == "thermal":
                        qubit.wait(2 * qubit.thermalization_time * u.ns)
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

            if not node.parameters.simulate:
                # measure excited-state IQ blob for all qubits
                align(*[q.xy.name for q in multiplexed_qubits.values()] +
                       [q.resonator.name for q in multiplexed_qubits.values()] +
                       [q.z.name for q in multiplexed_qubits.values()])

                for i, qubit in multiplexed_qubits.items():
                        if reset_type == "active":
                            active_reset(qubit)
                        elif reset_type == "thermal":
                            qubit.wait(2*qubit.thermalization_time * u.ns)
                        else:
                            raise ValueError(f"Unrecognized reset type {reset_type}.")

            align(*[q.xy.name for q in multiplexed_qubits.values()] +
                   [q.resonator.name for q in multiplexed_qubits.values()] +
                   [q.z.name for q in multiplexed_qubits.values()])

            for i, qubit in multiplexed_qubits.items():
                qubit.xy.play("x180")
                qubit.resonator.wait(qubit.xy.operations["x180"].length * u.ns) # qubit.align()
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
    node.save()

elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(iq_blobs)
        for i in range(num_qubits):
            results = fetching_tool(job, ["n"], mode="live")
            while results.is_processing():
                n = results.fetch_all()[0]
                progress_counter(n, n_runs, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        # todo: Write docstring
        ds = fetch_dataset(job, qubits, node.parameters)
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    
    # %% {Data_analysis}
    node.results = {"ds": ds, "figs": {}, "results": {}}
    plot_individual = False
    for q in qubits:
        # Perform two state discrimination
        angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(
            ds.I_g.sel(qubit=q.name),
            ds.Q_g.sel(qubit=q.name),
            ds.I_e.sel(qubit=q.name),
            ds.Q_e.sel(qubit=q.name),
            True,
            b_plot=plot_individual,
        )
        # TODO: check the difference between the above and the below
        # Get the rotated 'I' quadrature
        I_rot = ds.I_g.sel(qubit=q.name) * np.cos(angle) - ds.Q_g.sel(qubit=q.name) * np.sin(angle)
        # Get the blobs histogram along the rotated axis
        hist = np.histogram(I_rot, bins=100)
        # Get the discriminating threshold along the rotated axis
        RUS_threshold = hist[1][1:][np.argmax(hist[0])]
        # Save the individual figures if requested
        if plot_individual:
            fig = plt.gcf()
            plt.show()
            node.results["figs"][q.name] = fig
        node.results["results"][q.name] = {}
        node.results["results"][q.name]["angle"] = float(angle)
        node.results["results"][q.name]["threshold"] = float(threshold)
        node.results["results"][q.name]["fidelity"] = float(fidelity)
        node.results["results"][q.name]["confusion_matrix"] = np.array([[gg, ge], [eg, ee]])
        node.results["results"][q.name]["rus_threshold"] = float(RUS_threshold)

    # %% {Plotting}
    
    def update(frame):
        current_idx = min((frame + 1) * step, total_pts)
        changed_artists = []
        
        for i, (scat_g, scat_e) in enumerate(all_scatters):
            (Ig, Qg), (Ie, Qe) = all_data[i]
            
            # 更新位置
            scat_g.set_offsets(np.c_[Ig[:current_idx], Qg[:current_idx]])
            scat_e.set_offsets(np.c_[Ie[:current_idx], Qe[:current_idx]])
            
            changed_artists.extend([scat_g, scat_e])
        
        return changed_artists

    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    fig = grid.fig
    all_scatters = []
    all_data = []
    
    total_pts = node.parameters.num_runs # 假設所有位元數據長度相同
    step = int(total_pts * 0.05)
    for ax, qubit in grid_iter(grid):
    
        q_name = qubit['qubit']
        I_g = ds.I_g.sel(qubit=q_name).values*1000
        Q_g = ds.Q_g.sel(qubit=q_name).values*1000
        I_e = ds.I_e.sel(qubit=q_name).values*1000
        Q_e = ds.Q_e.sel(qubit=q_name).values*1000

        angle = node.results["results"][q_name]["angle"]
        def rotate(i, q, ang):
            return i * np.cos(ang) - q * np.sin(ang), i * np.sin(ang) + q * np.cos(ang)

        Ig_r, Qg_r = rotate(I_g, Q_g, angle)
        Ie_r, Qe_r = rotate(I_e, Q_e, angle)

        scat_g = ax.scatter([], [], color='blue', alpha=0.3, s=1, label='Ground')
        scat_e = ax.scatter([], [], color='red', alpha=0.3, s=1, label='Excited')
        ax.set_xlim(min(Ig_r.min(), Ie_r.min()), max(Ig_r.max(), Ie_r.max()))
        ax.set_ylim(min(Qg_r.min(), Qe_r.min()), max(Qg_r.max(), Qe_r.max()))
        ax.set_xlabel("I [mV]")
        ax.set_ylabel("Q [mV]")
        ax.set_aspect('equal')
        ax.set_title(f"{q_name}")
        ax.legend()
        plt.tight_layout()
        plt.suptitle("IQ Blobs Evolution", fontsize=16)

        all_scatters.append((scat_g, scat_e))
        all_data.append(((Ig_r, Qg_r), (Ie_r, Qe_r)))

        
    total_frames = total_pts // step
    ani = FuncAnimation(fig, update, frames=total_frames, blit=True, interval=50)
    from IPython.display import HTML
    HTML(ani.to_jshtml())

    # %% {Update_state}
    if node.parameters.load_data_id is None:

        node.outcomes = {q.name: "successful" for q in qubits}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.save()          

        from qualibrate_config.resolvers import get_qualibrate_config_path, get_qualibrate_config
        from quam_libs.compat import get_node_dir_path
        import os
        qs = get_qualibrate_config(get_qualibrate_config_path())
        base_path = qs.storage.location

        node_dir = get_node_dir_path(node.snapshot_idx, base_path)
    
        ani.save(os.path.join(node_dir, f"IQ_blobs_live.gif"), writer='pillow', fps=5)
# %%
