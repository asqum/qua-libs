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
q1 = machine.qubits["q1"]
q2 = machine.qubits["q2"]
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

simulate = False
shots = 2048
multiplexed = [1,2,3,4,5]
bitstrings = ['00', '01', '10', '11']

'''NOTE:
1. switch order in turn to compensate both channel consecutively
2. Ascent order: first row of cz*_*_2pi_dev in configuration
3. Descent order: Second row of cz*_*_2pi_dev in configuration
'''
cx_control, cx_target = q1_number, q2_number  
th_control, th_target = q1.resonator.operations["readout"].threshold, q2.resonator.operations["readout"].threshold
phis_corr = np.linspace(-0.9, 0.9, 360)
# phis_corr = np.linspace(0, 2, 360)

check_phase ="01" # 12: to_flux_tune, 01: to_meet_with
if coupler.name=="coupler_q4_q5": 
    phi_to_flux_tune, phi_to_meet_with = -0.278, -0.830
if coupler.name=="coupler_q3_q4": 
    phi_to_flux_tune, phi_to_meet_with = 0.384, 0.449
if coupler.name=="coupler_q2_q3": 
    phi_to_flux_tune, phi_to_meet_with = 0.484, 0.374
if coupler.name=="coupler_q1_q2": 
    phi_to_flux_tune, phi_to_meet_with = 0.263, -0.459

with program() as cz_ops:

    I = [declare(fixed) for i in range(len(multiplexed))]
    Q = [declare(fixed) for i in range(len(multiplexed))] 
    I_st = [declare_stream() for i in range(len(multiplexed))]
    Q_st = [declare_stream() for i in range(len(multiplexed))]
    state = [declare(bool) for _ in range(len(bitstrings))]
    state_st = [declare_stream() for _ in range(len(bitstrings))]
    n = declare(int)
    n_st = declare_stream()
    a = declare(fixed)
    phi_corr = declare(fixed)
    # global_phase_correction = declare(fixed, value=eval(f"cz{qubit_to_flux_tune}_{qubit_to_meet_with}_2pi_dev"))
    phi_to_flux_tune_full = declare(fixed)
    phi_to_meet_with_full = declare(fixed)
    assign(phi_to_flux_tune_full, phi_to_flux_tune)
    assign(phi_to_meet_with_full, phi_to_meet_with)

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()
    machine.apply_all_couplers_to_min()

    with for_(n, 0, n < shots, n+1):
        save(n, n_st)
        
        with for_(*from_array(phi_corr, phis_corr)):
            if check_phase=="12": assign(phi_to_flux_tune_full, phi_to_flux_tune + phi_corr)
            if check_phase=="01": assign(phi_to_meet_with_full, phi_to_meet_with + phi_corr)

            if not simulate: wait(machine.thermalization_time * u.ns)

            # Bell: 
            align()
            if check_phase=="12": 
                q2.xy.play("y90")
                q1.xy.play("-y90")
            if check_phase=="01": 
                q1.xy.play("y90")
                q2.xy.play("-y90")

            # CX: 
            # align()
            # q1.xy.play("x180")
            # align()
            # q2.xy.play("y90")
            # q2.xy.play("x180")
            # q2.xy.play("x90")
            
            align()

            # CZ-gate:  
            q1.z.play("cz%s_%s"%(q1_number,q2_number))
            coupler.play("cz")
            wait(150 * u.ns)
            align()

            # Bell: 
            if check_phase=="12": 
                frame_rotation_2pi(phi_to_flux_tune_full, q1.xy.name)
                q1.xy.play("y90")
            if check_phase=="01": 
                frame_rotation_2pi(phi_to_meet_with_full, q2.xy.name)
                q2.xy.play("y90")

            # CX: 
            # q2.xy.play("x180")
            # q2.xy.play("x90")

            align()
            multiplexed_readout(qubits, I, I_st, Q, Q_st)

            assign(state[0], ((I[qubits.index(q1)]<th_control) & (I[qubits.index(q2)]<th_target)))
            assign(state[1], ((I[qubits.index(q1)]<th_control) & (I[qubits.index(q2)]>th_target)))
            assign(state[2], ((I[qubits.index(q1)]>th_control) & (I[qubits.index(q2)]<th_target)))
            assign(state[3], ((I[qubits.index(q1)]>th_control) & (I[qubits.index(q2)]>th_target)))

            for i in range(len(bitstrings)): save(state[i], state_st[i])
        
        save(n, n_st)
    with stream_processing():
        # for the progress counter
        n_st.save("n")
        for i in range(len(multiplexed)):
            I_st[i].buffer(len(phis_corr)).save_all(f"I_{i+1}")
        for i in range(len(bitstrings)):
            state_st[i].boolean_to_int().buffer(len(phis_corr)).average().save(f"state_{bitstrings[i]}")
        

if not simulate:
    qm = qmm.open_qm(config)
    job = qm.execute(cz_ops)
    # job.result_handles.wait_for_all_values()
    # Get results from QUA program
    results = fetching_tool(job, ["n"] + [f"state_{x}" for x in bitstrings] + [f"I_{x}" for x in multiplexed], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  #  Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        n, state00, state01, state10, state11, I1, I2, I3, I4, I5 = results.fetch_all()
        # Progress bar
        progress_counter(n, shots, start_time=results.start_time)

        Bell_SNR = (state00+state11)/(state01+state10) *(state00*state11)

        plt.suptitle(f"Optimizing Phase compensation for CZ ({n}/{shots})")
        plt.subplot(121)
        plt.cla()
        plt.plot(phis_corr, state00, '.b', phis_corr, state11, '.r', phis_corr, state01, '.g', phis_corr, state10, '.k')
        plt.xlabel("Phase adjustment (2pi)")
        plt.ylabel("I quadrature [V]")
        plt.legend(("00", "11", "01", "10"), loc="upper right")
        plt.subplot(122)
        plt.cla()
        plt.plot(phis_corr, Bell_SNR)
        plt.xlabel("Phase adjustment (2pi)")
        plt.ylabel("Bell SNR")
        plt.title(f"The Best Phi-Adjust: {phis_corr[list(Bell_SNR).index(max(Bell_SNR))]:.3f}")

        plt.tight_layout()
        plt.pause(0.3)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
    plt.show()
    
    # if (qubit_to_flux_tune==4 and qubit_to_meet_with==3) or (qubit_to_flux_tune==5 and qubit_to_meet_with==4):
    #     Phi_config = eval(f"cz{cx_target}_{cx_control}_2pi_dev")  # <---------
    # if (qubit_to_flux_tune==1 and qubit_to_meet_with==2) or (qubit_to_flux_tune==2 and qubit_to_meet_with==3):
    #     Phi_config = eval(f"cz{cx_control}_{cx_target}_2pi_dev")  # <---------
    Phi_config = 0

    collected_shots = len(I1[:,0])
    for ii,j in enumerate([0, list(phis_corr).index(min(phis_corr, key=lambda x:abs(x))), list(Bell_SNR).index(max(Bell_SNR)), -1]):
        q_states = [] #np.zeros((len(multiplexed),collected_shots))
        for i,x in enumerate(multiplexed): 
            q_states += [[str(int(a)) for a in np.array(eval(f"I{x}")[:,j])>eval(f"machine.qubits['q{x}'].resonator.operations['readout'].threshold")]]
            print(f"q{x}-states: %s" %Counter(q_states[i]))
        
        bitstrings = sorted([''.join(x) for x in zip(q_states[multiplexed.index(cx_target)], q_states[multiplexed.index(cx_control)])])
        print(Counter(bitstrings))

        fig, ax = plt.subplots()
        print(Counter(bitstrings).keys())
        CBits = [x for x in Counter(bitstrings).keys()]
        percentage = [x/collected_shots*100 for x in Counter(bitstrings).values()]
        ax.bar(CBits, percentage)#, color=bar_colors)
        ax.set_ylabel('Population (%)')

        note_that = ["First", "Configured", "The BEST", "Last"]
        ax.set_title(f'{note_that[ii]} Bell/GHZ state fidelity: {(percentage[0]+percentage[-1]):.3}% (Phi={Phi_config+phis_corr[j]})')
        plt.show()

    print("=====================================")
    print(f"The Best Phi-Adjust: {phis_corr[list(Bell_SNR).index(max(Bell_SNR))]:.3f}")

    

else:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=3_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, cz_ops, simulation_config)
    # Plot the simulated samples
    job.get_simulated_samples().con1.plot()
    job.get_simulated_samples().con2.plot()
    plt.show()
