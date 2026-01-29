# Winter School — QUAM Setup

## Background
This guide prepares the software environment, configures Qualibrate paths, and creates a QUAM state for the OPX1000 with MW-FEM and LF-FEM.

### Create a new Conda environment
Replace `<env_name>` and `<python_version>` with your preferred values (e.g., `AS_winter_school` and `3.11`).

````bash
conda create -n <env_name> python=<python_version>
conda activate <env_name>
````

## 1) Install environment
Go to `Quantum-Control-Applications-QuAM/Superconducting` and install the package:

````bash
cd /Users/jackchao/Desktop/Project/QM/AS/qua-libs/Quantum-Control-Applications-QuAM/Superconducting
pip install -e .
````
It will install all packages listed in pyproject.toml.

## 2) Configure Qualibrate
Run the following in your terminal and set the paths:

````bash
setup-qualibrate-config
````

When prompted, enter:
- `data` path: store your data
- `calibration_node` path: all the calibration files will store here
- `quam_state` path: your QUAM states will store here

You can check the config at `~/.qualibrate/config.json` on macOS/Linux or `C:\Users\<username>\.qualibrate\config.json` on Windows.

## 3) Build QUAM state
From the `configuration` directory, run:

````bash
cd /Users/jackchao/Desktop/Project/QM/AS/qua-libs/Quantum-Control-Applications-QuAM/Superconducting/configuration

python make_wiring_lffem_mwfem_INIT.py
python make_quam.py
python modify_quam.py
````
This helps you create your initial QUAM state for your QPU.
### Notes
- `make_wiring_lffem_mwfem_INIT.py`: creates the base QUAM wiring/state for OPX1000 (MW-FEM + LF-FEM). Before running, update these fields to match your setup:
	- `host_ip`: your QOP/OPX1000 IP address.
	- `cluster_name`: cluster name used by your QOP.
	- `port`: leave as `None` unless your QOP uses a custom port.
	- `path`: output folder for `wiring.json` and `state.json` (the folder must not contain other JSON files).
	- `qubits` and `qubit_pairs`: how many qubits and couplers you want to include.
	- `instruments`: define which FEMs are installed (e.g., MW-FEM slots and LF-FEM slots).
	- Also verify the connectivity section: `add_resonator_line`, `add_qubit_drive_lines`, and `add_qubit_flux_lines` should match your QPU wiring.
- `make_quam.py`: builds and saves the initial QUAM structure from the wiring/state. Update as needed:
	- `path`: must point to the same QUAM state folder created by the wiring step.
	- `octave_settings`: set your Octave IP/port if you use an Octave; otherwise leave `{}`.
- `modify_quam.py`: sets expected values/initial guesses for calibration. Update as needed:
	- Load the correct `path` and ensure the machine is loaded properly.
	- Set initial guesses for resonator/XY frequencies, LO/IF, power levels, pulse lengths, and flux channel modes.
	- Use these as starting points for calibration; refine after measurements.

## 4) Calibration graph nodes (hello_qua)
These demo nodes show basic QUA timing behaviors:
- `00_hello_qua_1.py`: plays `xy` and `z` pulses at the same time (parallel channels).
- `00_hello_qua_2.py`: plays `xy` then `z` sequentially using alignment.
- `00_hello_qua_3.py`: shows how to insert a `wait` between pulses.

## 5) Time of flight (readout delay)
- `01b_time_of_flight_mw_fem.py`: measures raw ADC traces from the readout pulse to estimate the time-of-flight (delay) you should add so the acquisition window aligns with the resonator response. Use the extracted delay to update the readout timing in the state before further calibration.