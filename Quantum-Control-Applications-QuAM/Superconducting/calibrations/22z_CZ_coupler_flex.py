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
    - Having calibrated qubit gates (x180) by running qubit spectroscopy, rabi_chevron, power_rabi, Ramsey and updated the state.
    - (Optional) having corrected the flux line distortions by running the Cryoscope protocol and updating the filter taps in the state.

Next steps before going to the next node:
    - Update the CZ gate parameters in the state.
    - Save the current state by calling machine.save("quam")
"""

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

###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Define a path relative to this script, i.e., ../configuration/quam_state
config_path = Path(__file__).parent.parent / "configuration" / "quam_state"
# Instantiate the QuAM class from the state file
machine = QuAM.load(config_path)
# Generate the OPX and Octave configurations
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
q1 = machine.qubits["q1"]
q2 = machine.qubits["q2"]

readout_qubits = [qubit for qubit in machine.qubits.values() if qubit not in [q1, q2]]
try: coupler = (q1 @ q2).coupler
except: coupler = (q2 @ q1).coupler 
qb = q1  # The qubit whose flux will be swept

mode = "pulse" # dc or pulse
sweep_flux = "qc" # qb or qc or others
coupler_point = 0 # coupler.decouple_offset # -0.020 
# NOTE: always start from 0, turn ~20-40mV left to the FAST LANE. 

config = machine.generate_config()

# debugging qua: 
# with open("qua_config.json", "w+") as f: json.dump(config, f, indent=4)
# raise Exception

###################
# The QUA program #
###################

simulate = False
n_avg = 1373 #137000
# The flux pulse durations in clock cycles (4ns) - Must be larger than 4 clock cycles.
ts = np.arange(4, 30, 1)
# ts = np.arange(4, 600, 4)

# The flux bias sweep in V
if sweep_flux == "qb": 
    if coupler.name=="coupler_q4_q5": dcs = np.linspace(-0.070, -0.044, 301) # (q5<q4, Top-Left) 
    if coupler.name=="coupler_q3_q4": dcs = np.linspace(-0.100, -0.072, 301) # (q3<q4, Top-Left) 
    if coupler.name=="coupler_q2_q3": dcs = np.linspace(0.050, 0.072, 301) # (q3>q2, Top-Left)
    if coupler.name=="coupler_q1_q2": dcs = np.linspace(0.0519, 0.0591, 301) # (q1>q2, Top-Left) 
    # dcs = np.linspace(-0.3, 0.3, 501) # default wide-sweep 
elif sweep_flux == "qc": 
    if coupler.name=="coupler_q4_q5": dcs = np.linspace(-0.084, -0.051, 301) 
    if coupler.name=="coupler_q3_q4": dcs = np.linspace(-0.082, -0.048, 301) 
    if coupler.name=="coupler_q2_q3": dcs = np.linspace(-0.108, -0.060, 301) 
    if coupler.name=="coupler_q1_q2": dcs = np.linspace(-0.062, -0.015, 301) 
    # dcs = np.linspace(-0.4, 0.4, 501) # default wide-sweep 
    # dcs = np.linspace(-0.15, 0.4, 501) # Catching Sweet-Spot 
else: 
    ts = [30]
    dcs = [-0.045]

if coupler.name=="coupler_q4_q5": cz_point, scale = -0.05884, -0.0039
if coupler.name=="coupler_q3_q4": cz_point, scale = -0.09082, 0.0338 
if coupler.name=="coupler_q2_q3": cz_point, scale = 0.06053, -0.0087
if coupler.name=="coupler_q1_q2": cz_point, scale = 0.055533, 0.0828 

print("%s: %s" % (q1.name, q1.xy.RF_frequency))
print("%s: %s" % (q2.name, q2.xy.RF_frequency))
print("sweeping %s's offset: %s" % (qb.name,qb.z.min_offset))
print("%s's decouple-offset: %s" %(coupler.name,coupler.decouple_offset))

with program() as cz:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=2)
    t = declare(int)  # QUA variable for the flux pulse duration
    dc = declare(fixed)  # QUA variable for the flux pulse amplitude

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()
    machine.apply_all_couplers_to_min()

    # turn off neighboring coupler(s)
    # coupler_n1.set_dc_offset(-0.02257)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(t, ts)):
            with for_(*from_array(dc, dcs)):
                # Put the two qubits in their excited states
                q1.xy.play("x180")
                q2.xy.play("x180")
                align() # this makes the following flux pulse to have an extra delay of q1.xy length.. weird.. 

                z_amp = declare(fixed)
                coupler_amp = declare(fixed)
                qb_dc_point = declare(fixed)
                coupler_dc_point = declare(fixed)

                if sweep_flux == "qb":
                    # Pulse: 
                    assign(z_amp, Cast.mul_fixed_by_int((dc - qb.z.min_offset + scale * coupler_point), 5))
                    assign(coupler_amp, Cast.mul_fixed_by_int(coupler_point-coupler.decouple_offset, 5))
                    # dc: 
                    assign(qb_dc_point, dc + scale * coupler_point)
                    assign(coupler_dc_point, coupler_point)
                else:
                    # Pulse: 
                    assign(z_amp, Cast.mul_fixed_by_int((cz_point - qb.z.min_offset + scale * dc), 5))
                    assign(coupler_amp, Cast.mul_fixed_by_int(dc-coupler.decouple_offset, 5))
                    # dc: 
                    assign(qb_dc_point, cz_point + scale * dc)
                    assign(coupler_dc_point, dc)

                if mode == "pulse":
                    ########### Pulsed Version
                    wait( (24) * u.ns, qb.z.name, coupler.name) # another bug 
                    qb.z.play("flux_pulse", duration=t, amplitude_scale=z_amp)
                    coupler.play("flux_pulse", duration=t, amplitude_scale=coupler_amp)
                    # q1.z.play("flux_pulse", duration=t, amplitude_scale=0)
                    # wait(64 * u.ns, qb.z.name)
                    #############################

                if mode == "dc":
                    ########## Set DC Offset Version
                    wait(24 * u.ns)
                    qb.z.set_dc_offset(qb_dc_point) # 0.0175
                    coupler.set_dc_offset(coupler_dc_point)
                    wait(t)
                    # coupler.set_dc_offset(0)
                    coupler.to_decouple_idle()
                    q1.z.to_min()
                    q2.z.to_min()

                    # machine.apply_all_flux_to_min()
                    # machine.apply_all_couplers_to_min()

                    # wait(t - 36 * u.ns)
                    #############################

                # Wait some time to ensure that the flux pulse will end before the readout pulse
                # wait(16 * u.ns)
                # Align the elements to measure after having waited a time "tau" after the qubit pulses.
                align()
                # Play the readout on the other resonators to measure in the same condition as when optimizing readout
                for other_qubit in readout_qubits:
                    other_qubit.resonator.play("readout")
                # Measure the state of the resonators
                multiplexed_readout([q1, q2], I, I_st, Q, Q_st)
                # Wait for the qubits to decay to the ground state
                if not simulate: wait(machine.thermalization_time * u.ns)

    with stream_processing():
        # for the progress counter
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(len(dcs)).buffer(len(ts)).average().save("I1")
        Q_st[0].buffer(len(dcs)).buffer(len(ts)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(len(dcs)).buffer(len(ts)).average().save("I2")
        Q_st[1].buffer(len(dcs)).buffer(len(ts)).average().save("Q2")


###########################
# Run or Simulate Program #
###########################

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=750)  # In clock cycles = 4ns
    job = qmm.simulate(config, cz, simulation_config)
    job.get_simulated_samples().con1.plot()
    plt.show()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Calibrate the active qubits
    # machine.calibrate_octave_ports(qm)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(cz)
    # Prepare the figure for live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)
    # Tool to easily fetch results from the OPX (results_handle used in it)
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    # Live plotting
    while results.is_processing():
        # Fetch results
        n, I1, Q1, I2, Q2 = results.fetch_all()
        # Convert the results into Volts
        I1 = u.demod2volts(I1, q1.resonator.operations["readout"].length)
        Q1 = u.demod2volts(Q1, q1.resonator.operations["readout"].length)
        I2 = u.demod2volts(I2, q2.resonator.operations["readout"].length)
        Q2 = u.demod2volts(Q2, q2.resonator.operations["readout"].length)
        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)

        if sweep_flux == "qc":
            cz_dur = 88 # coupler.operations["cz"].length
        if sweep_flux == "qb": 
            cz_dur = 240 

        # Plot
        plt.suptitle("CZ chevron (compensation: %s, cz_dur: %sns, %s/%s)" % (scale, cz_dur, n, n_avg ))

        plt.subplot(321)
        plt.cla()
        plt.plot(dcs, I1[:][list(ts).index(cz_dur//4)])
        
        plt.subplot(322)
        plt.cla()
        plt.plot(dcs, I2[:][list(ts).index(cz_dur//4)])

        plt.subplot(323)
        plt.cla()
        plt.pcolor(dcs, 4 * ts, I1)
        # plt.plot(cz_point, wait_time, color="r", marker="*")
        # plt.title(f"{q1.name} - I, f_01={int(q1.f_01 / u.MHz)} MHz")
        plt.ylabel("Interaction time [ns]")

        # feedback from CZ-Pi: 
        if sweep_flux=="qc":
            if coupler.decouple_offset>dcs[0] and coupler.decouple_offset<dcs[-1]:
                # plt.axvline( coupler.operations["cz"].amplitude, color="r", linestyle="--", linewidth=1.5)
                plt.axvline(coupler.decouple_offset, color="b", linestyle="--", linewidth=1.0)
        else:
            if cz_point>dcs[0] and cz_point<dcs[-1]:
                # plt.axvline( q1.z.operations["cz"].amplitude + q1.z.min_offset - scale*coupler.operations["cz"].amplitude, color="r", linestyle="--", linewidth=1.5)
                plt.axvline(cz_point, color="b", linestyle="--", linewidth=1.0)
        # plt.axhline( q1.z.operations["cz"].length, color="r", linestyle="--", linewidth=1.5)
        plt.axhline( cz_dur, color="k", linestyle="--", linewidth=0.57)
        # plt.axhline( 40, color="y", linestyle="--", linewidth=0.57)
        # plt.axhline( 48, color="r", linestyle="--", linewidth=0.57)
        
        plt.subplot(325)
        plt.cla()
        plt.pcolor(dcs, 4 * ts, Q1)
        # plt.plot(cz_point, wait_time, color="r", marker="*")
        plt.title(f"{q1.name} - Q")
        plt.xlabel("Flux amplitude [V]")
        plt.ylabel("Interaction time [ns]")

        # feedback from CZ-Pi: 
        if sweep_flux=="qc":
            if coupler.decouple_offset>dcs[0] and coupler.decouple_offset<dcs[-1]:
                # plt.axvline( coupler.operations["cz"].amplitude, color="r", linestyle="--", linewidth=1.5)
                plt.axvline(coupler.decouple_offset, color="b", linestyle="--", linewidth=1.0)
        else:
            if cz_point>dcs[0] and cz_point<dcs[-1]:
                # plt.axvline( q1.z.operations["cz"].amplitude + q1.z.min_offset - scale*coupler.operations["cz"].amplitude, color="r", linestyle="--", linewidth=1.5)
                plt.axvline(cz_point, color="b", linestyle="--", linewidth=1.0)
        # plt.axhline( q1.z.operations["cz"].length, color="r", linestyle="--", linewidth=1.5)
        plt.axhline( cz_dur, color="k", linestyle="--", linewidth=0.57)
        # plt.axhline( 40, color="y", linestyle="--", linewidth=0.57)
        # plt.axhline( 48, color="r", linestyle="--", linewidth=0.57)

        plt.subplot(324)
        plt.cla()
        plt.pcolor(dcs, 4 * ts, I2)
        # plt.plot(cz_point, wait_time, color="r", marker="*")
        # plt.title(f"{q2.name} - I, f_01={int(q2.f_01 / u.MHz)} MHz")

        # feedback from CZ-Pi: 
        if sweep_flux=="qc":
            if coupler.decouple_offset>dcs[0] and coupler.decouple_offset<dcs[-1]:
                # plt.axvline( coupler.operations["cz"].amplitude, color="r", linestyle="--", linewidth=1.5)
                plt.axvline(coupler.decouple_offset, color="b", linestyle="--", linewidth=1.0)
        else:
            if cz_point>dcs[0] and cz_point<dcs[-1]:
                # plt.axvline( q1.z.operations["cz"].amplitude + q1.z.min_offset - scale*coupler.operations["cz"].amplitude, color="r", linestyle="--", linewidth=1.5)
                plt.axvline(cz_point, color="b", linestyle="--", linewidth=1.0)
        # plt.axhline( q1.z.operations["cz"].length, color="r", linestyle="--", linewidth=1.5)
        plt.axhline( cz_dur, color="k", linestyle="--", linewidth=0.57)
        # plt.axhline( 40, color="y", linestyle="--", linewidth=0.57)
        # plt.axhline( 48, color="r", linestyle="--", linewidth=0.57)

        plt.subplot(326)
        plt.cla()
        plt.pcolor(dcs, 4 * ts, Q2)
        # plt.plot(cz_point, wait_time, color="r", marker="*")
        plt.title(f"{q2.name} - Q")
        plt.xlabel("Flux amplitude [V]")

        plt.tight_layout()
        plt.pause(0.3)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # plt.show()

    # Save data from the node
    data = {
        f"{q1.name}_flux_pulse_amplitude": dcs,
        f"{q1.name}_flux_pulse_duration": 4 * ts,
        f"{q1.name}_I": I1.T,
        f"{q1.name}_Q": Q1.T,
        f"{q2.name}_flux_pulse_amplitude": dcs,
        f"{q2.name}_flux_pulse_duration": 4 * ts,
        f"{q2.name}_I": I2.T,
        f"{q2.name}_Q": Q2.T,
        f"qubit_flux": qb.name,
        "figure": fig,
    }
    node_save(machine, "CZ_chevron_coupler_working", data)

# %%
