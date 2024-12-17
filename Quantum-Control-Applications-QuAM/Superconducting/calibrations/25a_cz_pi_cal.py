# %%
"""
        CZ CHEVRON - 4ns granularity
The goal of this protocol is to find the parameters of the CZ gate between two flux-tunable qubits.
The protocol consists in flux tuning one qubit to bring the |11> state on resonance with |20>.
The two qubits must start in their excited states so that, when |11> and |20> are on resonance, the state |11> will
start acquiring a global phase when varying the flux pulse duration.

By scanning the flux pulse amplitude and duration, the CZ chevron can be obtained and post-processed to extract the
CZ gate parameters corresponding to a single oscillation period such that |11> pick up an overall phase of pi (flux
pulse amplitude and interation time).

This version sweeps the flux pulse duration using real-time QUA, which means that the flux pulse can be arbitrarily long
but the step must be larger than 1 clock cycle (4ns) and the minimum pulse duration is 4 clock cycles (16ns).

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having found the qubits maximum frequency point (qubit_spectroscopy_vs_flux).
    - Having calibrated qubit gates (x180) by running qubit spectroscopy, rabi_chevron, power_rabi, Ramsey and updated the configuration.
    - (Optional) having corrected the flux line distortions by running the Cryoscope protocol and updating the filter taps in the configuration.

Next steps before going to the next node:
    - Update the CZ gate parameters in the configuration.
"""

from pathlib import Path

from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array, get_equivalent_log_array
from qualang_tools.units import unit
from quam_libs.components import QuAM, Transmon
from quam_libs.macros import qua_declaration, multiplexed_readout, node_save

import matplotlib.pyplot as plt
import numpy as np

import matplotlib

matplotlib.use("TKAgg")

# from qm.QuantumMachinesManager import QuantumMachinesManager
# from qm.qua import *
# from qm import SimulationConfig
# from configuration import *
# import matplotlib.pyplot as plt
# from qualang_tools.loops import from_array
# from qualang_tools.results import fetching_tool
# from qualang_tools.plot import interrupt_on_close
# from qualang_tools.results import progress_counter
# import numpy as np
# from macros import qua_declaration, multiplexed_readout
# import warnings
# from scipy.optimize import curve_fit

# warnings.filterwarnings("ignore")

from quam_libs.experiments.cz_pi_calibration.cosine import Cosine

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
num_qubits_full = len(machine.active_qubits)

q1 = machine.qubits["q1"]
q2 = machine.qubits["q2"]

try: coupler = (q1 @ q2).coupler
except: coupler = (q2 @ q1).coupler 

# qubits = [q1, q2]
# num_qubits = len(qubits)

q1_number = machine.active_qubits.index(q1) + 1
q2_number = machine.active_qubits.index(q2) + 1

# readout_qubits = [qubit for qubit in machine.qubits.values() if qubit not in [q1, q2]]

####################
# Define variables #
####################

# Qubit to flux-tune to reach some distance of Ec with another qubit, Qubit to meet with:
qubit_to_flux_tune = q1
qubit_to_meet_with = q2
play_cz = True

# qubit to flux-tune is target
# qubit to meet with is control

qubits = machine.active_qubits

points_per_cycle = 20
cz_corr = 0 # float(eval(f"cz{q2_number}_{q1_number}_2pi_dev"))

simulate = False
with_set_dc = False

n_avg = 10000  # The number of averages
phis = np.arange(0, 3, 1 / points_per_cycle)
amps = np.linspace(0.5, 1.5, 25)
amps = np.linspace(0.7, 1.3, 25)
# amps = np.linspace(0.9, 1.1, 25)
amps = np.linspace(0.95, 1.05, 25)
amps = np.linspace(0.995, 1.005, 25)
# amps = np.linspace(-0.04085/-0.04128, -0.0425/-0.04128, 25)
# amps = np.linspace(-0.040/-0.04128, -0.042/-0.04128, 25) 

# cz_dur = 60
cz_dur = 88

# Ref: 22z_CZ_coupler_flex.py 
if coupler.name=="coupler_q4_q5": 
    cz_point, scale = -0.05884, -0.0039
    # cz_coupler = -0.06815*1.5*1.25  
    cz_coupler = -0.06815*1.4583333 
    phi_to_flux_tune, phi_to_meet_with = 0, 0
if coupler.name=="coupler_q3_q4": 
    cz_point, scale = -0.09082, 0.0338 
    cz_coupler = -0.07457*1.481637 
    # cz_coupler = -0.07457*.958333*.98333
    phi_to_flux_tune, phi_to_meet_with = 0, 0
if coupler.name=="coupler_q2_q3": 
    cz_point, scale = 0.06053, -0.0087
    # cz_coupler = -0.10223*1.0291667*1.0041667
    cz_coupler = -0.10223*.9583333*1.0166667*1.0083333
    phi_to_flux_tune, phi_to_meet_with = 0, 0
if coupler.name=="coupler_q1_q2": 
    cz_point, scale =  0.055533, 0.0828 #0.05594, 0.0397
    # cz_coupler = -0.05457*1.025*1.0175*1.0020833    
    cz_coupler = -0.04912*1.0041667*1.005  
    phi_to_flux_tune, phi_to_meet_with = 0, 0

pulse_dc_factor = 1.0 #(0.00859 - qubit_to_flux_tune.z.min_offset)/(0.00908 - qubit_to_flux_tune.z.min_offset) * 1.08
print("pulse_dc_factor: %s" % pulse_dc_factor)
print("%s's offset: %s" % (qubit_to_flux_tune.name, qubit_to_flux_tune.z.min_offset))

sweep_flux = "qc" # qb or qc
check_phase = "12" # 12: to_flux_tune, 01: to_meet_with
check_cz_pulse = False

print("updated cz_coupler: %s" %cz_coupler)
print("cz_coupler tuning from %s to %s" % (cz_coupler*amps[0], cz_coupler*amps[-1]))
print("machine.thermalization_time: %s" % (machine.thermalization_time*u.ns)) 

###################
# The QUA program #
###################

with program() as cz_pi_cal:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits_full)
    phi = declare(fixed)  # QUA variable angle of the second pi/2 wrt to the first pi/2
    ampp = declare(fixed)  # QUA variable for the flux pulse amplitude pre-factor.
    flag = declare(bool)
    # global_phase_correction = declare(fixed, value=cz_corr)
    phi_to_flux_tune_full = declare(fixed)
    phi_to_meet_with_full = declare(fixed)
    assign(phi_to_flux_tune_full, phi_to_flux_tune)
    assign(phi_to_meet_with_full, phi_to_meet_with)
    

    z_amp = declare(fixed)
    coupler_amp = declare(fixed)

    machine.apply_all_flux_to_min()
    machine.apply_all_couplers_to_min()

    with for_(n, 0, n < n_avg, n + 1):
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

        with for_(*from_array(phi, phis)):
            if check_phase=="12": assign(phi_to_flux_tune_full, phi_to_flux_tune + phi)
            if check_phase=="01": assign(phi_to_meet_with_full, phi_to_meet_with + phi)
            with for_(*from_array(ampp, amps)):
                with for_each_(flag, [True, False]):

                    # control qubit
                    if check_phase=="12": play("x180", qubit_to_meet_with.xy.name, condition=flag)
                    if check_phase=="01": play("x180", qubit_to_flux_tune.xy.name, condition=flag)

                    # ramsey first pi/2
                    align()
                    if check_phase=="12": play("x90", qubit_to_flux_tune.xy.name)
                    if check_phase=="01": play("x90", qubit_to_meet_with.xy.name)

                    align()
                    # Wait some time to ensure that the flux pulse will arrive after the x90 pulse
                    # wait(100 * u.ns)

                    # cz
                    if play_cz:
                        if sweep_flux == "qb":
                            z_pulse_height = pulse_dc_factor*((ampp*cz_point - qubit_to_flux_tune.z.min_offset + scale * cz_coupler))
                            assign(z_amp, Cast.mul_fixed_by_int(z_pulse_height, 5))
                            c_pulse_height = pulse_dc_factor*(cz_coupler - coupler.decouple_offset)
                            assign(coupler_amp, Cast.mul_fixed_by_int(c_pulse_height, 5))
                        else:
                            z_pulse_height = pulse_dc_factor*((cz_point - qubit_to_flux_tune.z.min_offset + scale * ampp*cz_coupler))
                            assign(z_amp, Cast.mul_fixed_by_int(z_pulse_height, 5))
                            c_pulse_height = pulse_dc_factor*(ampp*cz_coupler - coupler.decouple_offset)
                            assign(coupler_amp, Cast.mul_fixed_by_int(c_pulse_height, 5))
                        ########### Pulsed Version
                        # wait(24 * u.ns)  
                        # wait( (24) * u.ns, qubit_to_flux_tune.z.name, coupler.name) # another bug
                        if check_cz_pulse:
                            # qubit_to_flux_tune.z.play("flux_pulse", duration=cz_dur//4, amplitude_scale=-0.06828336307877787*5)
                            # coupler.play("flux_pulse", duration=cz_dur//4, amplitude_scale=-0.059020520846625*5)
                            # from state.json: 
                            qubit_to_flux_tune.z.play(("cz%s_%s"%(qubit_to_flux_tune.name,qubit_to_meet_with.name)).replace("q",""))
                            coupler.play("cz")
                        else:
                            qubit_to_flux_tune.z.play("flux_pulse", duration=cz_dur//4, amplitude_scale=z_amp)
                            coupler.play("flux_pulse", duration=cz_dur//4, amplitude_scale=coupler_amp)
                        #############################

                        # Wait some time to ensure that the flux pulse will end before the readout pulse
                        wait(150 * u.ns)

                    # ramsey second pi/2
                    align()
                    if check_phase=="12": 
                        frame_rotation_2pi(phi_to_flux_tune_full, qubit_to_flux_tune.xy.name)
                        play("x90", qubit_to_flux_tune.xy.name)
                    if check_phase=="01": 
                        frame_rotation_2pi(phi_to_meet_with_full, qubit_to_meet_with.xy.name)
                        play("x90", qubit_to_meet_with.xy.name)
                    align()

                    wait(30 * u.ns) # to prevent the readout being overlapped by the flux tail  
                    # Play the readout on the other resonators to measure in the same condition as when optimizing readout
                    # for other_qubit in readout_qubits:
                    #     other_qubit.resonator.play("readout")
                    # Measure the state of the resonators
                    multiplexed_readout(qubits, I, I_st, Q, Q_st)

                    # Wait for the qubit to decay to the ground state
                    if not simulate: wait(1 * machine.thermalization_time * u.ns)

    with stream_processing():
        # for the progress counter
        n_st.save("n")

        # Target:
        I_st[qubits.index(qubit_to_flux_tune)].buffer(len(phis), len(amps), 2).average().save("I1")
        Q_st[qubits.index(qubit_to_flux_tune)].buffer(len(phis), len(amps), 2).average().save("Q1")

        # Control:
        I_st[qubits.index(qubit_to_meet_with)].buffer(len(phis), len(amps), 2).average().save("I2")
        I_st[qubits.index(qubit_to_meet_with)].buffer(len(phis), len(amps), 2).average().save("Q2")

###########################
# Run or Simulate Program #
###########################
# simulate = True

if simulate:
#     from qm import generate_qua_script

#     sourceFile = open('debug.py', 'w')
#     print(generate_qua_script(cz_pi_cal, config), file=sourceFile) 
#     sourceFile.close()

    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=500)  # In clock cycles = 4ns
    job = qmm.simulate(config, cz_pi_cal, simulation_config)
    job.get_simulated_samples().con1.plot()
    # job.get_simulated_samples()
    # con3.plot()
    plt.show()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(cz_pi_cal)

    # import time
    # time.sleep(300)
    # I1 =job.result_handles.I1.fetch_all()
    # print(f"len of amps {len(amps)}")
    # print(f"len of phis {len(phis)}")

    # fig = plt.figure()
    fig, ax = plt.subplots(len(amps) // 5, 5)

    # fig2, ax2 = plt.subplots(len(amps)//5, 5)
    # CZ_sign = np.zeros([len(amps),len(phis)])

    interrupt_on_close(fig, job)
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    # Live plotting
    while results.is_processing():
        # Fetch results
        n, I1, Q1, I2, Q2 = results.fetch_all()
        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)

        plt.suptitle(f"q{q1_number}->q{q2_number}: amp_scale, pha_diff_deg ({n}/{n_avg})")
        for i in range(len(amps)):
            ax[int(i // 5), int(i % 5)].cla()

            # Fitting for phase
            if check_phase=="12": 
                I_control_g = I1[:, i, 1]
                I_control_e = I1[:, i, 0]
                # I_control_g = Q1[:, i, 1]
                # I_control_e = Q1[:, i, 0]
            if check_phase=="01": 
                I_control_g = I2[:, i, 1]
                I_control_e = I2[:, i, 0]
            try:
                fit = Cosine(phis, I_control_g, plot=False)
                phase_g = fit.out.get('phase')[0]
                ax[int(i // 5), int(i % 5)].plot(fit.x_data, fit.fit_type(fit.x, fit.popt) * fit.y_normal, '-b',
                                                 alpha=0.5)
                fit = Cosine(phis, I_control_e, plot=False)
                phase_e = fit.out.get('phase')[0]
                ax[int(i // 5), int(i % 5)].plot(fit.x_data, fit.fit_type(fit.x, fit.popt) * fit.y_normal, '-r',
                                                 alpha=0.5)
                dphase = (phase_g - phase_e) / np.pi * 180
            except Exception as e:
                print(e)
            ax[int(i // 5), int(i % 5)].plot(phis, I_control_e, '.r', phis, I_control_g, '.b')
            ax[int(i // 5), int(i % 5)].set_title("%.7f, %.1f" % (amps[i], dphase))

            # I10 = I1[:,i,0]
            # I10 /= np.max(I10)
            # I11 = I1[:,i,1]
            # I11 /= np.max(I11)
            # CZ_sign[i,:] = I10 - I11
            # ax2[int(i//5), int(i%5)].cla()
            # ax2[int(i//5), int(i%5)].plot(I11, I10, '.')
            # ax2[int(i//5), int(i%5)].set_aspect('equal')
            # ax2[int(i//5), int(i%5)].set_title(f"amp scale: {amps[i]}")

        plt.tight_layout()
        plt.pause(3)

    plt.show()

    # plt.plot(amps, [np.max(CZ_sign[x,:]) for x in range(len(amps))] )
    # plt.show()

    # def cosine_function(t, A, f, phi, C):
    #     return A * np.cos(2 * np.pi * f * (t - phi)) + C

    # for i in range(len(amps)):
    #     # Initial guess for parameters
    #     initial_guess = [np.abs(np.max(I1[:,i,0])-np.min(I1[:,i,0]))/2, 1/15, 0.0, np.mean(I1[:,i,0])]
    #     # initial_guess = [1,1,1,1]
    #     params, covariance = curve_fit(cosine_function, phis, I1[:,i,0], p0=initial_guess)
    #     print(params)
    #     plt.plot(cosine_function(phis, params[0], params[1], params[2], params[3]))
    # plt.show()

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
    # plt.show()

    print("cz_dur: %s" %cz_dur)
    z_pulse_height = pulse_dc_factor*((cz_point - qubit_to_flux_tune.z.min_offset + scale * cz_coupler))
    print("z_pulse_height: %s" %z_pulse_height)
    c_pulse_height = pulse_dc_factor*(cz_coupler - coupler.decouple_offset)
    print("c_pulse_height: %s" %c_pulse_height)
    if int(input("Update QUAM STATES for cz-pulse: (1/0) ")):
        qubit_to_flux_tune.z.operations["cz%s_%s"%(q1.name.replace("q",""),q2.name.replace("q",""))].length = cz_dur
        coupler.operations["cz"].length = cz_dur
        qubit_to_flux_tune.z.operations["cz%s_%s"%(q1.name.replace("q",""),q2.name.replace("q",""))].amplitude = z_pulse_height
        coupler.operations["cz"].amplitude = c_pulse_height

    save = True
    if save:
        filename = f"CZ_Pi_Cal_c{q2_number}_t{q1_number}"

        data = {}
        data["I1"] = I1
        data["figure"] = fig
        # np.savez(save_dir / filename, I1=I1)
        # print("Data saved as %s.npz" % filename)

        # np.savez(save_dir/'cz', I1=I1, Q1=Q1, I2=I2, Q2=Q2, ts=ts, amps=amps)

        node_save(machine, filename, data, additional_files=True)

# %%
