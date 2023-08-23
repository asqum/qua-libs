"""
        RESONATOR SPECTROSCOPY VERSUS FLUX
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to
extract the 'I' and 'Q' quadratures.
This is done across various readout intermediate frequencies and flux biases.
Based on the results, one can determine the resonator frequency as a function of flux bias.

This information can then be used to adjust the readout frequency for the maximum frequency point.

Prerequisites:

    - Calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibration of the IQ mixer connected to the readout line (be it an external mixer or an Octave port).
    - Identification of the resonator's resonance frequency (referred to as "resonator_spectroscopy").
    - Configuration of the readout pulse amplitude and duration.
    - Specification of the expected resonator depletion time in the configuration.
Before proceeding to the next node:
    - Update the readout frequency, labeled as "resonator_IF", in the configuration.
    - Adjust the flux bias to the maximum frequency point, labeled as "max_frequency_point", in the configuration.
    - Update the resonator frequency versus flux fit parameters (amplitude_fit, frequency_fit, phase_fit, offset_fit) in the configuration
"""

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from qualang_tools.loops import from_array
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings("ignore")

##############################
# Program-specific variables #
##############################

n_avg = 6000  # Number of averaging loops

# Frequency sweep in Hz
f_min = 55 * u.MHz
f_max = 65 * u.MHz
df = 500 * u.kHz
frequencies = np.arange(f_min, f_max + df / 2, df)  # +df/2 to add f_max to the scan
# Flux amplitude sweep (as a pre-factor of the flux amplitude)
dc_min = -0.49
dc_max = 0.49
step = 0.01
flux = np.arange(dc_min, dc_max + step / 2, step)  # +da/2 to add a_max to the scan

###################
# The QUA program #
###################

with program() as resonator_spec_2D:
    n = declare(int)  # QUA variable for the averaging loop
    f = declare(int)  # QUA variable for the readout frequency
    dc = declare(fixed)  # QUA variable for the flux bias pre-factor
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(f, frequencies)):
            # Update the frequency of the digital oscillator linked to the resonator element
            update_frequency("resonator", f)
            with for_(*from_array(dc, flux)):
                # Flux sweeping by tuning the OPX dc offset
                set_dc_offset("flux_line", "single", dc)
                wait(flux_settle_time * u.ns, "resonator", "qubit")
                # Measure the resonator (send a readout pulse whose amplitude is rescaled by the pre-factor 'a' [-2, 2)
                # and demodulate the signals to get the 'I' & 'Q' quadratures)
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                # Wait for the resonator to cooldown
                wait(depletion_time * u.ns, "resonator")
                # Save the 'I' & 'Q' quadratures to their respective streams
                save(I, I_st)
                save(Q, Q_st)
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        # Cast the data into a 2D matrix, average the 2D matrices together and store the results on the OPX processor
        I_st.buffer(len(flux)).buffer(len(frequencies)).average().save("I")
        Q_st.buffer(len(flux)).buffer(len(frequencies)).average().save("Q")
        n_st.save("iteration")


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip, cluster_name=cluster_name, octave=octave_config)

#######################
# Simulate or execute #
#######################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, resonator_spec_2D, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(resonator_spec_2D)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  #  Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        I, Q, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Convert results into Volts and normalize
        S = u.demod2volts(I + 1j * Q, readout_len)
        R = np.abs(S)  # Amplitude
        phase = np.angle(S)  # Phase
        # 2D spectroscopy plot
        plt.subplot(211)
        plt.cla()
        plt.title(r"resonator spectroscopy $R=\sqrt{I^2 + Q^2}$")
        plt.pcolor(flux, frequencies / u.MHz, R)
        plt.xlabel("flux level [V]")
        plt.ylabel("frequency [MHz]")
        plt.subplot(212)
        plt.cla()
        plt.title("Resonator spectroscopy phase")
        plt.pcolor(flux, frequencies / u.MHz, signal.detrend(np.unwrap(phase)))
        plt.xlabel("Flux level [V]")
        plt.ylabel("Readout frequency [MHz]")
        plt.pause(0.1)
        plt.tight_layout()
    plt.show()

    # Fitting to cosine resonator frequency response
    def cosine_func(x, amplitude, frequency, phase, offset):
        return amplitude * np.cos(2 * np.pi * frequency * x + phase) + offset

    Z = I + 1j * Q
    mag = np.abs(Z)
    minima = np.zeros(len(flux))
    for i in range(len(flux)):
        minima[i] = frequencies[np.argmin(mag.T[i])] / u.MHz

    initial_guess = [1, 0.5, 0, 0]  # Initial guess for the parameters
    fit_params, _ = curve_fit(cosine_func, flux, minima, p0=initial_guess)

    # Get the fitted values
    amplitude_fit, frequency_fit, phase_fit, offset_fit = fit_params
    print("fitting parameters", fit_params)

    # Generate the fitted curve using the fitted parameters
    fitted_curve = cosine_func(flux, amplitude_fit, frequency_fit, phase_fit, offset_fit)

    plt.figure()
    plt.pcolor(flux, frequencies / u.MHz, np.abs(Z))
    plt.plot(flux, minima, "x-", color="red", label="Flux minima")
    plt.plot(flux, fitted_curve, label="Fitted Cosine", color="orange")
    plt.xlabel("Flux level [V]")
    plt.ylabel("Readout frequency [MHz]")
    plt.legend()

    print("DC flux value corresponding to the maximum frequency point", flux[np.argmax(fitted_curve)])
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
