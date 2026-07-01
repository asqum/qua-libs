# %%
from pathlib import Path
from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array, get_equivalent_log_array
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, multiplexed_readout, node_save
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("TKAgg")

u = unit(coerce_to_integer=True)
machine = QuAM.load()
config = machine.generate_config()
qmm = machine.connect()
qubits = machine.active_qubits
num_qubits = len(qubits)

n_avg = 200
t_delay = np.arange(4, 20000, 200)

with program() as T1:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    t = declare(int)  # QUA variable for the wait time
    machine.apply_all_flux_to_min()

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(t, t_delay)):
            for qubit in qubits:
                qubit.xy.play("x180")
                qubit.xy.wait(t)
            align()
            multiplexed_readout(qubits, I, I_st, Q, Q_st)
            wait(machine.thermalization_time * u.ns)

    with stream_processing():
        if np.isclose(np.std(t_delay[1:] / t_delay[:-1]), 0, atol=1e-3):
            t_delay = get_equivalent_log_array(t_delay)
        for i in range(len(machine.active_qubits)):
            I_st[i].buffer(len(t_delay)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(t_delay)).average().save(f"Q{i + 1}")
        n_st.save("n")

qm = qmm.open_qm(config)
plt.figure(figsize=(15, 9))
plt.suptitle("T1")

def get_T1():
    job = qm.execute(T1)
    data_list = sum([[f"I{i + 1}", f"Q{i + 1}"] for i in range(num_qubits)], ["n"])
    results = fetching_tool(job, data_list, mode="live")
    while results.is_processing():
        # Fetch results
        fetched_data = results.fetch_all()
        n = fetched_data[0]
        I_data = fetched_data[1::2]
        I_volts = [
            u.demod2volts(I, qubit.resonator.operations["readout"].length)
            for I, qubit in zip(I_data, qubits)
        ]
        progress_counter(n, n_avg, start_time=results.start_time)
        
    data = {}

    plt.clf()
    for i, qubit in enumerate(qubits):
        try:
            from qualang_tools.plot.fitting import Fit
            fit = Fit()

            plt.subplot(1, num_qubits, i + 1)

            fit_res = fit.T1(4 * t_delay, I_volts[i], plot=True)
            qubit.T1 = int(np.round(np.abs(fit_res["T1"][0]) / 4) * 4)
            print("%s's T1: %s" %(qubit.name, qubit.T1))

            plt.xlabel("Wait time [ns]")
            plt.ylabel("I quadrature [V]")
            plt.title(f"{qubit.name}")
            plt.legend((f"T1 = {np.round(np.abs(fit_res['T1'][0]) / 4) * 4:.0f} ns",))
            plt.pause(1)
        except (Exception,):
            print("error fitting / plotting for %s" %qubit.name)
            pass
    return [q.T1 for q in qubits]

import time
import csv
from datetime import datetime

# Open CSV file in append mode to continuously save data
file_name="Quantum-Control-Applications-QuAM/Superconducting/data/T1-stat/QPU_T1_01.csv"
with open(file_name, mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write the header only once if the file is new
    file.seek(0, 2)  # Move to end of the file to check if it is empty
    if file.tell() == 0:
        print(["Timestamp"] + [f"q{i + 1}-T1" for i in range(num_qubits)])
        writer.writerow(["Timestamp"] + [f"q{i + 1}-T1" for i in range(num_qubits)])  # Write header if file is empty
    try:
        while True:
            # Get current timestamp and temperature
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            qubit_lifetime = get_T1()

            # Only Write the legit data to the CSV file: 
            if 0 in qubit_lifetime or sum(qubit_lifetime) > 100000000:
                print("!!!Anomaly!!! detected in qubit_lifetime: %s" %qubit_lifetime)
            else:
                writer.writerow([current_time] + qubit_lifetime)

            # Force the data to be written to the file immediately
            file.flush()
            # Print the data for user feedback
            print(f"Recorded qubit_lifetime at {current_time}: {qubit_lifetime} ns")
    except KeyboardInterrupt:
        print("T1 logging stopped.")

qm.close()
# %%
