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
q1 = machine.qubits["q4"]
q2 = machine.qubits["q5"]
qn1 = machine.qubits["q3"]

try: 
    coupler = (q1 @ q2).coupler
except:
    coupler = (q2 @ q1).coupler 

qb = q1  # The qubit whose flux will be swept

print("q1: %s" %q1.xy.RF_frequency)
print("q2: %s" %q2.xy.RF_frequency)

# neighbour coupling off:
coupler_n1 = (qn1 @ q1).coupler

# compensations = {
#     q1: coupler.opx_output.crosstalk[q1.z.opx_output.port_id],
#     q2: coupler.opx_output.crosstalk[q2.z.opx_output.port_id]
# }

import numpy as np
compensation_arr = np.array([[1, 0.177], [0.408, 1]])
inv_arr = np.linalg.inv(compensation_arr)

# Add qubit pulses
q1.z.operations["flux_pulse"] = pulses.SquarePulse(length=100, amplitude=0.1)
q2.z.operations["flux_pulse"] = pulses.SquarePulse(length=100, amplitude=0.1)
coupler.operations["flux_pulse"] = pulses.SquarePulse(length=100, amplitude=0.1)

config = machine.generate_config()

# debugging qua: 
# with open("qua_config.json", "w+") as f:
#     json.dump(config, f, indent=4)
# raise Exception

###################
# The QUA program #
###################

n_avg = 137000
# The flux pulse durations in clock cycles (4ns) - Must be larger than 4 clock cycles.
# ts = np.arange(4, 40, 1)
ts = np.arange(4, 50, 1)
# ts = [30]
# The flux bias sweep in V
dcs = np.linspace(-0.0425, -0.03, 501) # for qc (4_5)
# dcs = np.linspace(-0.0445, -0.025, 501) # for qc (4_5, n1)
# dcs = np.linspace(-0.028, -0.01, 501) # for qc (3_4)
# dcs = np.linspace(-0.025, 0.003, 501) # for qc (3_2) 
# dcs = np.linspace(0.02, 0.025, 401) # for q4_5
# dcs = np.linspace(0.018, 0.024, 401) # for q4_5_cn1
# dcs = np.linspace(-0.12, -0.09, 401) # for q3_4
# dcs = np.linspace(-0.10, -0.08, 401) # for q3_2
# dcs = [-0.045]
cz_point = 0.02275 #q3_2:-0.09529 #q3_4:-0.105 #q4_5:0.02275, 0.02075
coupler_point = -0.04128*1.0113857 #*1.0178961 #q3_4_off: -0.02257, q4_5_off: -0.03479, -0.03663 
scale = 0.0448 # q4_5: 0.0448, q3_4: 0.051, q3_2: -0.117 

simulate = True
mode = "pulse" # dc or pulse
sweep_flux = "qc" # q4 or qc
pulse_dc_factor = 1.0 #(0.00859 - q1.z.min_offset)/(0.00908 - q1.z.min_offset) * 1.08
print("pulse_dc_factor: %s" % pulse_dc_factor)
print("q4's offset: %s" % q1.z.min_offset)


with program() as cz:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=2)
    t = declare(int)  # QUA variable for the flux pulse duration
    dc = declare(fixed)  # QUA variable for the flux pulse amplitude

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()

    # turn off neighboring coupler(s)
    # coupler_n1.set_dc_offset(-0.02257)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(t, ts)):
            # assign(dc, -0.035)
            # scale = declare(fixed)
            # scales = np.linspace(0.02, 0.12, 301)
            # dcs = scales
            # with for_(*from_array(scale, scales)):
            with for_(*from_array(dc, dcs)):
                # assign(v1, Cast.mul_fixed_by_int(-0.15, dc))
                # Put the two qubits in their excited states
                # wait(300 * u.ns, q1.xy.name, q2.xy.name)
                q1.xy.play("x180")
                q2.xy.play("x180")
                align() # this makes the following flux pulse to have an extra delay of q1.xy length.. weird.. 

                # q1.z.set_dc_offset(cz_point)
                # wait(20 * u.ns)

                z_amp = declare(fixed)
                coupler_amp = declare(fixed)
                q1_dc_point = declare(fixed)
                coupler_dc_point = declare(fixed)

                if sweep_flux == "q4":
                    assign(z_amp, Cast.mul_fixed_by_int(pulse_dc_factor*((dc - qb.z.min_offset + scale * coupler_point)), 10))
                    assign(coupler_amp, Cast.mul_fixed_by_int(pulse_dc_factor*(coupler_point), 10))
                    assign(q1_dc_point, dc + scale * coupler_point)
                    assign(coupler_dc_point, coupler_point)
                else:
                    assign(z_amp, Cast.mul_fixed_by_int(pulse_dc_factor*((cz_point - qb.z.min_offset + scale * dc)), 10))
                    assign(coupler_amp, Cast.mul_fixed_by_int(pulse_dc_factor*(dc), 10))
                    assign(q1_dc_point, cz_point + scale * dc)
                    assign(coupler_dc_point, dc)

                if mode == "pulse":
                    ########### Pulsed Version
                    wait( (24) * u.ns, qb.z.name, coupler.name)
                    qb.z.play("flux_pulse", duration=t, amplitude_scale=z_amp)
                    coupler.play("flux_pulse", duration=t, amplitude_scale=coupler_amp)
                    # wait(64 * u.ns, qb.z.name)
                    #############################

                if mode == "dc":
                    ########## Set DC Offset Version
                    wait(64 * u.ns)
                    qb.z.set_dc_offset(q1_dc_point) # 0.0175
                    coupler.set_dc_offset(coupler_dc_point)
                    wait(t)
                    coupler.set_dc_offset(0)
                    qb.z.to_min()
                    q2.z.to_min()
                    # wait(t - 36 * u.ns)
                    #############################

                # Wait some time to ensure that the flux pulse will end before the readout pulse
                wait(600 * u.ns)
                # Align the elements to measure after having waited a time "tau" after the qubit pulses.
                align()
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
        # Plot
        plt.suptitle("CZ chevron (compensation: %s)" %scale)
        plt.subplot(221)
        plt.cla()
        plt.pcolor(dcs, 4 * ts, I1)
        # plt.plot(cz_point, wait_time, color="r", marker="*")
        # plt.title(f"{q1.name} - I, f_01={int(q1.f_01 / u.MHz)} MHz")
        plt.ylabel("Interaction time [ns]")
        plt.subplot(223)
        plt.cla()
        plt.pcolor(dcs, 4 * ts, Q1)
        # plt.plot(cz_point, wait_time, color="r", marker="*")
        plt.title(f"{q1.name} - Q")
        plt.xlabel("Flux amplitude [V]")
        plt.ylabel("Interaction time [ns]")

        # feedback from CZ-Pi: 
        if sweep_flux=="qc":
            plt.axvline( coupler_point, color="r", linestyle="--", linewidth=0.5)
        else:
            plt.axvline( cz_point, color="r", linestyle="--", linewidth=0.5)
        plt.axhline( 60, color="r", linestyle="--", linewidth=0.5)
        plt.axhline( 40, color="g", linestyle="--", linewidth=0.5)

        plt.subplot(222)
        plt.cla()
        plt.pcolor(dcs, 4 * ts, I2)
        # plt.plot(cz_point, wait_time, color="r", marker="*")
        # plt.title(f"{q2.name} - I, f_01={int(q2.f_01 / u.MHz)} MHz")
        plt.subplot(224)
        plt.cla()
        plt.pcolor(dcs, 4 * ts, Q2)
        # plt.plot(cz_point, wait_time, color="r", marker="*")
        plt.title(f"{q2.name} - Q")
        plt.xlabel("Flux amplitude [V]")

        # feedback from CZ-Pi: 
        if sweep_flux=="qc":
            plt.axvline( coupler_point, color="r", linestyle="--", linewidth=0.5)
        else:
            plt.axvline( cz_point, color="r", linestyle="--", linewidth=0.5)
        plt.axhline( 60, color="r", linestyle="--", linewidth=0.5)
        plt.axhline( 40, color="g", linestyle="--", linewidth=0.5)

        plt.tight_layout()
        plt.pause(0.3)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # plt.show()

    # q1.z.cz.length =
    # q1.z.cz.level =

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
