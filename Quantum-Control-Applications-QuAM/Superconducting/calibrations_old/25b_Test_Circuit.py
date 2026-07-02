from pathlib import Path
from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from qualang_tools.units import unit

import matplotlib.pyplot as plt
import numpy as np

from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, multiplexed_readout, node_save
from quam.components import pulses

import matplotlib
import json

matplotlib.use("TKAgg")

from collections import Counter
import pandas as pd

###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handle units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
qubits = machine.active_qubits
num_qubits_full = len(qubits)
q1 = machine.qubits["q5"]
q2 = machine.qubits["q4"]
q1_number = qubits.index(q1) + 1
q2_number = qubits.index(q2) + 1

try: coupler = (q1 @ q2).coupler
except: coupler = (q2 @ q1).coupler 

readout_qubits = [qubit for qubit in machine.qubits.values() if qubit not in [q1, q2]]

####################
# Define variables #
####################

# Qubit to flux-tune to reach some distance of Ec with another qubit, Qubit to meet with:
qubit_to_flux_tune = q1
qubit_to_meet_with = q2
play_cz = True

cz_type = "const_wf"
n_avg = 1024
h_loop = 1
multiplexed = [1,2,3,4,5]
cz_corr = float(eval(f"cz{5}_{4}_2pi_dev"))

with program() as cz_ops:

    I_g = [declare(fixed) for i in range(len(multiplexed))]
    Q_g = [declare(fixed) for i in range(len(multiplexed))] 
    I_st_g = [declare_stream() for i in range(len(multiplexed))]
    Q_st_g = [declare_stream() for i in range(len(multiplexed))]
    n = declare(int)
    n_st = declare_stream()
    t = declare(int)
    a = declare(fixed)
    phi = declare(fixed)
    global_phase_correction = declare(fixed, value=cz_corr)

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()
    machine.apply_all_couplers_to_min()

    with for_(n, 0, n < n_avg, n+1):
        save(n, n_st)
        
        wait(machine.thermalization_time * u.ns)
        align()

        # play("x180", "q3_xy")
        # play("x180", "q1_xy")
        
        # Circuit 1:
        # play("x90", "q1_xy")
        # play("x90", "q2_xy")
        # align()
        # cz_gate(cz_type)
        # align()
        # play("x90", "q2_xy")
        # align()
        # cz_gate(cz_type)
        # align()
        # play("x180", "q2_xy")
        # align()
        # cz_gate(cz_type)

        # Circuit 2: Bell-state
        # play("x180", "q4_xy")
        # align()

        # play("y90", "q3_xy")
        # play("x180", "q3_xy")

        # align()
        # play("y90", "q4_xy")
        # play("x180", "q4_xy")
        # align()

        # play("x180", "q1_xy")
        # play("x180", "q2_xy")
        # play("x180", "q1_xy")
        # play("x180", "q2_xy")
        # align()

        # cz_gate(3, 4, cz_type)
        # frame_rotation_2pi(global_phase_correction, "q4_xy")

        # align()
        # play("y180", "q1_xy")
        # play("y180", "q2_xy")
        # play("y180", "q1_xy")
        # play("y180", "q2_xy")
        # align()

        # play("y90", "q4_xy")
        # play("x180", "q4_xy")
        # align()

        # Circuit 3: Hadamard-test
        # play("y90", "q1_xy")
        # play("x180", "q1_xy")
        # play("y90", "q2_xy")
        # play("x180", "q2_xy")
        # align()
        # play("y90", "q1_xy")
        # play("x180", "q1_xy")
        # play("y90", "q2_xy")
        # play("x180", "q2_xy")

        # align()
        # play("x180", "q4_xy")
        # play("x180", "q5_xy")

        # Circuit 4: CX
        # play("x180", "q4_xy")
        # align()
        # play("x90"*amp(*np.array([0., -1., 1., 0.])), "q5_xy") # y90
        # # play("y90", "q5_xy")
        # play("x180", "q5_xy")
        # align()
        # cz_gate(4, 5, cz_type)
        # frame_rotation_2pi(eval(f"cz{5}_{4}_2pi_dev"), "q5_xy")
        # frame_rotation_2pi(eval(f"cz{4}_{5}_2pi_dev"), "q4_xy")
        # align()
        # play("y90", "q5_xy")
        # play("x180", "q5_xy")
    
        align()
        multiplexed_readout(I_g, I_st_g, Q_g, Q_st_g, resonators=multiplexed, weights="rotated_")
        
    with stream_processing():
        for i in range(len(multiplexed)):
            I_st_g[i].save_all(f"I_g_{i+1}")
            Q_st_g[i].save_all(f"Q_g_{i+1}")
        

# open communication with opx
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

simulate = False
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
    bitstrings = sorted([''.join(x) for x in zip(q5_states,q4_states)])
    print(Counter(bitstrings))

    # n, bins, patches = plt.hist(x=bitstrings, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    # plt.grid(axis='y', alpha=0.75)
    # plt.xlabel('Bitstrings')
    # plt.ylabel('Occurance')
    # plt.title('State Population')
    # plt.text(23, 45, r'$\mu=15, b=3$')
    # maxfreq = n.max()
    # # Set a clean upper y-axis limit.
    # plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    # plt.show()

    fig, ax = plt.subplots()
    print(Counter(bitstrings).keys())
    CBits = [x for x in Counter(bitstrings).keys()]
    percentage = [x/n_avg*100 for x in Counter(bitstrings).values()]
    # bar_colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
    ax.bar(CBits, percentage)#, color=bar_colors)
    ax.set_ylabel('Population (%)')
    ax.set_title('Quantum Circuit\'s Outcome')
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
