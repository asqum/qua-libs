from pathlib import Path
# from qm.qua import *
from qualang_tools.units import unit
from quam_libs.components import QuAM
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use("TKAgg")
# Set global font size
plt.rcParams.update({'font.size': 15.7})

###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handle units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
# config = machine.generate_config()
# Open Communication with the QOP
# qmm = machine.connect()

# Get the relevant QuAM components
qubits = machine.active_qubits
num_qubits = len(qubits)

for q in qubits: 
    print("\n%s: " %(q.name))
    print("qb.f01: %s" %(q.xy.RF_frequency))
    print("ro.length: %s" %(q.resonator.operations["readout"].length))

x_data = [q.name for q in qubits]
y_data = [q.xy.RF_frequency for q in qubits]

QPU_Map = plt.figure()
plt.suptitle("qubit frequencies")
plt.xlabel("qubit.name")
plt.ylabel("qubit.xy.frequency (GHz)")
plt.plot(x_data, y_data, marker='o', color='red')

for i, (x, y) in enumerate(zip(x_data, y_data)):
    plt.annotate(f"{y*1e-9:.4f}GHz", (x, y), textcoords="offset points", xytext=(21, 12), ha="center") 

plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Read data from the CSV file
# Replace 'data_log.csv' with the path to your actual CSV file
file_name="Quantum-Control-Applications-QuAM/Superconducting/data/T1-stat/QPU_T1_00.csv"
df = pd.read_csv(file_name)

# Display the first few rows to verify
print(df.head())

# Convert 'Timestamp' column to datetime type for better plotting
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Step 2: Plot Time Series for Temperature and Pressure
plt.figure(figsize=(15, 9))

# Plot q1-T1 over time
plt.subplot(5, 1, 1)
plt.plot(df['Timestamp'], df['q1-T1'], label='q1-T1', color='blue')
plt.title('Time Series of T1 for AS-DR2-5q4c')
plt.ylabel('q1-T1 (ns)')
plt.grid(True)

# Plot q2-T1 over time
plt.subplot(5, 1, 2)
plt.plot(df['Timestamp'], df['q2-T1'], label='q2-T1', color='blue')
plt.ylabel('q2-T1 (ns)')
plt.grid(True)

# Plot q3-T1 over time
plt.subplot(5, 1, 3)
plt.plot(df['Timestamp'], df['q3-T1'], label='q3-T1', color='blue')
plt.ylabel('q3-T1 (ns)')
plt.grid(True)

# Plot q4-T1 over time
plt.subplot(5, 1, 4)
plt.plot(df['Timestamp'], df['q4-T1'], label='q4-T1', color='blue')
plt.ylabel('q4-T1 (ns)')
plt.grid(True)

# Plot q5-T1 over time
plt.subplot(5, 1, 5)
plt.plot(df['Timestamp'], df['q5-T1'], label='q5-T1', color='blue')
plt.xlabel('Time')
plt.ylabel('q5-T1 (ns)')
plt.grid(True)
plt.xticks(rotation=45)

# Adjust layout for better readability
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 9))
plt.plot(df['Timestamp'], (df['q1-T1']+df['q2-T1']+df['q3-T1']+df['q4-T1']+df['q5-T1'])/5, color='blue')
plt.title('Time Series of Average-T1 for AS-DR2-5q4c')
plt.xlabel('Time')
plt.ylabel('q1-T1 (ns)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot Distribution of Temperature and Pressure
plt.figure(figsize=(15, 9))

plt.subplot(1, 5, 1)
plt.title('q1:')
sns.histplot(df['q1-T1']/1000, kde=True, color='black')
plt.xlabel('T1 (us)')

plt.subplot(1, 5, 2)
plt.title('q2:')
sns.histplot(df['q2-T1']/1000, kde=True, color='black')
plt.xlabel('T1 (us)')

plt.subplot(1, 5, 3)
plt.title('q3:')
sns.histplot(df['q3-T1']/1000, kde=True, color='black')
plt.xlabel('T1 (us)')

plt.subplot(1, 5, 4)
plt.title('q4:')
sns.histplot(df['q4-T1']/1000, kde=True, color='black')
plt.xlabel('T1 (us)')

plt.subplot(1, 5, 5)
plt.title('q5:')
sns.histplot(df['q5-T1']/1000, kde=True, color='black')
plt.xlabel('T1 (us)')

# Adjust layout
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 9))

plt.subplot(1, 2, 1)
plt.title('All:')
sns.histplot(df['q1-T1']/1000, kde=True, legend="q1")
sns.histplot(df['q2-T1']/1000, kde=True, legend='q2')
sns.histplot(df['q3-T1']/1000, kde=True, legend='q3')
sns.histplot(df['q4-T1']/1000, kde=True, legend='q4')
sns.histplot(df['q5-T1']/1000, kde=True, legend='q5')
plt.legend(["q1","q2","q3","q4","q5"])
plt.xlabel('T1 (us)')

plt.subplot(1, 2, 2)
plt.title('Averaged T1 across all 5q:')
sns.histplot((df['q1-T1']+df['q2-T1']+df['q3-T1']+df['q4-T1']+df['q5-T1'])/5000, kde=True, color='black')
plt.xlabel('T1 (us)')

plt.tight_layout()
plt.show()
