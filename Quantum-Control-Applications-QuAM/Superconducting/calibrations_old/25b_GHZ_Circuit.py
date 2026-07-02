from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import *

from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close
from qualang_tools.units import unit
from qualang_tools.analysis import two_state_discriminator
# from oqc import *

import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

from macros import multiplexed_readout, cz_gate, Dynamical_Decoupling

simulate = False
cz_type = "const_wf"
shots = 1024
h_loop = 1
multiplexed = [1,2,3,4,5]
# cz_corr = float(eval(f"cz{4}_{3}_2pi_dev"))

with program() as cz_ops:

    I_g = [declare(fixed) for i in range(len(multiplexed))]
    Q_g = [declare(fixed) for i in range(len(multiplexed))] 
    I_st_g = [declare_stream() for i in range(len(multiplexed))]
    Q_st_g = [declare_stream() for i in range(len(multiplexed))]
    n = declare(int)
    n_st = declare_stream()
    # t = declare(int, value=2)
    a = declare(fixed)
    phi = declare(fixed)
    # global_phase_correction = declare(fixed, value=cz_corr)

    with for_(n, 0, n < shots, n+1):
        save(n, n_st)
        
        if not simulate: wait(thermalization_time * u.ns)

        # align()
        # play("y90", "q1_xy")
        
        # # CX(1,2)
        # play("-y90", "q2_xy")
        # align()
        # cz_gate(2, 1, cz_type)
        # frame_rotation_2pi(eval(f"cz{1}_{2}_2pi_dev"), "q2_xy")
        # frame_rotation_2pi(eval(f"cz{2}_{1}_2pi_dev"), "q1_xy")
        # align()
        # play("y90", "q2_xy")

        # # CX(2,3)
        # play("-y90", "q3_xy")
        # align()
        # cz_gate(3, 2, cz_type)
        # frame_rotation_2pi(eval(f"cz{2}_{3}_2pi_dev"), "q3_xy")
        # frame_rotation_2pi(eval(f"cz{3}_{2}_2pi_dev"), "q2_xy")
        align()
        play("y90", "q3_xy")


        # align()
        # play("y90", "q3_xy")

        # # CX(2,3)
        # play("-y90", "q2_xy")
        # align()
        # cz_gate(3, 2, cz_type)
        # frame_rotation_2pi(eval(f"cz{2}_{3}_2pi_dev"), "q3_xy")
        # frame_rotation_2pi(eval(f"cz{3}_{2}_2pi_dev"), "q2_xy")
        # align()
        # play("y90", "q2_xy")
        
        # # CX(1,2)
        # play("-y90", "q1_xy")
        # align()
        # cz_gate(2, 1, cz_type)
        # frame_rotation_2pi(eval(f"cz{1}_{2}_2pi_dev"), "q2_xy")
        # frame_rotation_2pi(eval(f"cz{2}_{1}_2pi_dev"), "q1_xy")
        # align()
        # play("y90", "q1_xy")


        # CX(3,4)
        play("-y90", "q4_xy")
        align()
        cz_gate(3, 4, cz_type)
        frame_rotation_2pi(eval(f"cz{4}_{3}_2pi_dev"), "q4_xy")
        frame_rotation_2pi(eval(f"cz{3}_{4}_2pi_dev"), "q3_xy")
        align()
        play("y90", "q4_xy")

        # CX(4,5)
        play("-y90", "q5_xy")
        align()
        cz_gate(4, 5, cz_type)
        frame_rotation_2pi(eval(f"cz{5}_{4}_2pi_dev"), "q5_xy")
        frame_rotation_2pi(eval(f"cz{4}_{5}_2pi_dev"), "q4_xy")
        align()
        play("y90", "q5_xy")

        align()
        multiplexed_readout(I_g, I_st_g, Q_g, Q_st_g, resonators=multiplexed, weights="rotated_")
        
    with stream_processing():
        for i in range(len(multiplexed)):
            I_st_g[i].save_all(f"I_g_{i+1}")
            Q_st_g[i].save_all(f"Q_g_{i+1}")
        

# open communication with opx
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

if not simulate:
    qm = qmm.open_qm(config)
    job = qm.execute(cz_ops)
    job.result_handles.wait_for_all_values()
    results = fetching_tool(job, [f"I_g_{x}" for x in multiplexed])
    qm.close()

    q1_states = [str(int(x)) for x in np.array(results.fetch_all()[0])>ge_threshold_q1]
    q2_states = [str(int(x)) for x in np.array(results.fetch_all()[1])>ge_threshold_q2]
    q3_states = [str(int(x)) for x in np.array(results.fetch_all()[2])>ge_threshold_q3]
    q4_states = [str(int(x)) for x in np.array(results.fetch_all()[3])>ge_threshold_q4]
    q5_states = [str(int(x)) for x in np.array(results.fetch_all()[4])>ge_threshold_q5]
    print("q1-states: %s" %Counter(q1_states))
    print("q2-states: %s" %Counter(q2_states))
    print("q3-states: %s" %Counter(q3_states))
    print("q4-states: %s" %Counter(q4_states))
    print("q5-states: %s" %Counter(q5_states))

    # bitstrings = sorted([''.join(x) for x in zip(q5_states,q4_states,q3_states,q2_states,q1_states)])
    # bitstrings = sorted([''.join(x) for x in zip(q5_states, q4_states, q3_states, q2_states)])
    bitstrings = sorted([''.join(x) for x in zip(q5_states, q4_states, q3_states)])
    # bitstrings = sorted([''.join(x) for x in zip(q4_states,q3_states,q2_states)])
    # bitstrings = sorted([''.join(x) for x in zip(q3_states,q2_states,q1_states)])
    print(Counter(bitstrings))

    fig, ax = plt.subplots()
    print(Counter(bitstrings).keys())
    CBits = [x for x in Counter(bitstrings).keys()]
    percentage = [x/shots*100 for x in Counter(bitstrings).values()]
    # bar_colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
    ax.bar(CBits, percentage)#, color=bar_colors)
    ax.set_ylabel('Population (%)')
    # ax.set_title('Quantum Circuit\'s Outcome')
    ax.set_title(f'Bell/GHZ state fidelity: {(percentage[0]+percentage[-1]):.3}%')
    # ax.legend(title='Fruit color')
    plt.show()

else:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=3_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, cz_ops, simulation_config)
    # Plot the simulated samples
    job.get_simulated_samples().con1.plot()
    job.get_simulated_samples().con2.plot()
    plt.show()
