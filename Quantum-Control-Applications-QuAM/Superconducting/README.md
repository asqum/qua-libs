# Academia Sinica - QPU Calibration Library

---

## Installation
This folder contains an installable module called `quam_libs`, which provides a collection of tailored components for controlling and calibrating Acadmia Sinica's QPU architecture. These components extend the functionality of [QuAM](https://qua-platform.github.io/quam/), making it easier to design and execute calibration nodes.

### Prerequisites
 - **Python**: Version 3.9 to 3.12 is supported.
 - **Python Virtual Environment**: Strongly recommended to avoid dependency conflicts. You can create one using **[conda](https://www.anaconda.com/download)**: Anaconda's virtual environment manager as follows:
     ```bash
     conda create -n qualibrate_env python=3.10
     conda activate qualibrate_env
     ```
   
 - [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git): (Optional but Recommended) For version control, easier updates (pulling changes), and collaboration (forking and contributing). Install Git.
 - Access to Quantum Orchestration Platform (QOP) hardware
   - Required for running experiments on hardware.
   - We recommend upgrading to [QOP3.4.1](https://docs.quantum-machines.co/latest/docs/Releases/qop3_releases/) or later for the OPX1000, and [QOP2.4.4](https://docs.quantum-machines.co/latest/docs/Releases/qop2_releases/) or later of the OPX+
      - The minimum corresponding QUA SDK (`qm-qua`) version will be shown on the pages linked above.
    
### Getting Started
First, activate your conda environment using
```bash
conda activate qualibrate_env
```

Then, install quam-libs

```sh
# Install `quam_libs` (locally, from this directory)
pip install -e path/to/asqum/Quantum-Control-Applications-QuAM/Superconducting
```
> **_NOTE:_**  The `-e` flag means you *don't* have to reinstall if you make a local change to `quam_libs`!


## Initial Setup (QUAlibrate Configuration)

The QUAlibrate framework needs some initial configuration to know where to find calibration scripts, store data, and manage the system state (QUAM).

1.  **Run the Configuration Script:** Execute the provided script from within the `Superconducting` directory:

    ```bash
    setup-qualibrate-config
    ```

    If this command does not work, you may need to first restart your terminal or IDE.

2.  **Follow Prompts:** The script will interactively ask for the following details:

   - `project name`: A unique name for your project or QPU chip (e.g., `MyQPU_Chip1`).  
     Default: `QPU_project`.
   - `storage location`: The root directory where measurement data will be saved.  
     Default: `data/{project_name}` relative to the current directory.
   - `calibration library folder`: The path to the directory containing calibration nodes/graphs.  
     Default: `./calibration_graph` relative to the current directory.
   - `QUAM state path`: The location where the QUAM state file (containing system parameters, connectivity, etc.) is stored.  
     Default: `./configuration/quam_state` relative to the current directory.

         You can press `Enter` or type `y` to accept the defaults, or `n` to provide custom paths.

3.  **Confirm Full Config:** The script will show the complete QUAlibrate configuration for final confirmation.
    For detailed explanations of all settings, refer to the [QUAlibrate Configuration File Documentation](https://qua-platform.github.io/qualibrate/configuration/).

### Verify Setup

To ensure QUAlibrate is installed and configured correctly:

1.  **Launch the Web Interface:** Run the following command in your terminal:

    ```bash
    qualibrate start
    ```

2.  **Open in Browser:** Navigate to [http://127.0.0.1:8001](http://127.0.0.1:8001).

You should see the QUAlibrate web UI, listing the calibration nodes found in your configured `calibrations` directory.


## Creating the QUAM State

QUAM (Quantum Abstract Machine) provides an abstraction layer over the low-level QUA configuration. It allows you to define your quantum system (hardware, connectivity, qubit parameters, pulses, etc.) in a structured, physicist-friendly way. The QUAM state is stored in the `./quam_state/` directory, separated into a static part `./quam_state/wiring.json` for the wiring and network, and the main contents in `./quam_state/state.json`. The QUAM state serves as a persistent digital model of your entire setup, one that is continuously updated with calibrations.

**Interaction with Calibration Nodes:**

- **Loading:** Calibration nodes (scripts in `calibration_graph/`) typically load the latest QUAM state at the beginning of their execution. This provides them with all the necessary parameters (e.g., frequencies, amplitudes, timings) required to run the specific calibration experiment.
- **Updating:** After a calibration node runs and analyzes the results, it often calculates updated parameters (e.g., a newly calibrated qubit frequency or an optimized pulse amplitude). The node then modifies the corresponding values within the loaded QUAM object.
- **Saving:** QUAlibrate nodes save the modified QUAM state, often alongside the experiment results. This ensures that subsequent nodes in a calibration graph or future runs use the most up-to-date, calibrated parameters. This also updates the latest QUAM state in the `./quam_state/` directory.

**How to Create the State:**

The process of creating the initial QUAM state file involves defining your specific hardware components (OPXs, Octaves, mixers, LOs), as well as the QPU layout that the hardware is attached to. Detailed instructions are found in **[configuration/README.md](configuration/README.md)**

This directory contains scripts (`make_quam.py`, `modify_quam.py`, `make_wiring_...`, etc.) that demonstrate how to build the QUAM object programmatically.

## Calibration Nodes and Graphs

The scripts within the `calibrations` directory are the building blocks for automated calibration routines.
Each script typically performs a specific measurement (e.g., Resonator Spectroscopy, Rabi Oscillations, T1 measurement).
They are designed to be run via the QUAlibrate framework, either individually or as part of a larger calibration sequence (graph), but can also be executed as a standalone script from your favorite Python IDE (e.g. PyCharm, VScode...).

Refer to the [calibrations/README.md](calibrations/README.md) for detailed information on the structure and conventions used for these nodes.

## Project Structure

The library is organized into the following main directories:

```
Superconducting/
├── calibration_graph/      # Individual calibration scripts (nodes) runnable by QUAlibrate.
│   ├── 00_close_other_qms.py
│   ├── 01a_mixer_calibration.py
│   └── ... (many calibration routines)
│
├── calibrations/           # Non-standard or experimental calibration scripts.
│   ├── 18_allxy.py
│   ├── 25b_two_qubit_rb.py
│   └── ... (many calibration routines)
│
├── data/                   # Default location for storing experiment results.
│   └── {project_name}/     # Data organized by project name.
│       └── YYYY-MM-DD/     # Data organized by date.
│           └── #idx_{node_name}_HHMMSS/ # Data for a specific run.
│               └── quam_state/
│                   ├── state.json      # Contains the QUAM state except the wiring and network.
│                   └── wiring.json     # Contains the static part of the QUAM state (wiring and network).
│               ├── data.json       # Structure containing the data outpoutted by the node (fit results, figures,...).
│               ├── ds_raw.h5       # HDF5 dataset containing the raw data.
│               ├── ds_fit.h5       # HDF5 dataset containg the post-processed data.
│               ├── figures.png     # Generated figures.
│               └── node.json       # Metadat about the node used by QUAlibrate.
│
├── configuration           # Scripts and configurations for generating/managing QUAM state files.
│   ├── wiring_examples/    # Example configurations for different hardware setups.
│   ├── generate_quam.py        # Script to generate a QUAM file.
│   ├── populate_quam_xx.py     # Script to populate the newly generated QUAM file with initial values.
│   └── quam_state/         # Default location for the main QUAM state file.
│       ├── state.json      # Contains the QUAM state except the wiring and network
│       └── wiring.json     # Contains the static part of the QUAM state (wiring and network)
│
├── quam_libs               # QUAM utilties library for superconducting qubits
│   ├───components          # QUAM component definitions
│   ├───lib                 # Helper functions for running expeirments
│   └───quam_builder        # Tool for building the QUAM state.
│       ├───transmons
│       └───wiring
│
├── README.md               # This file.
└── pyproject.toml          # Installation configuration for the package.
```

**calibration_graph**  
The `calibration_graph/` folder contains individual Python scripts, each representing a calibration "node".
These scripts typically import functionality from **quam_libs**, define parameters, run a QUA program, analyze results, and update the QUAM state. See the README.md within this folder for more details on node structure.

**data**  
The `data/` folder is the default output directory where QUAlibrate saves results (plots, raw data, QUAM state snapshots) from calibration runs, organized by project, date, and run index/name.

**configuration**  
The configuration folder contains the python scripts used to build the QUAM before starting the experiments.
It contains three files whose working principles are explained in more details below:
* __make_wiring__: create the port mapping between the control hardware (OPX+, Octave, OPX1000 LF fem, MW fem) and the quantum elements (qubits, resonators, flux lines...).
    * [make_wiring_lffem_mwfem.py](./configuration/make_wiring_lffem_mwfem.py) for a cluster made of LF and MW FEMs (OPX1000).
    * [make_wiring_lffem_octave.py](./configuration/make_wiring_lffem_octave.py) for a cluster made of LF-FEMs and Octaves (OPX1000).
    * [make_wiring_opxp_octave.py](./configuration/make_wiring_opxp_octave.py) for a cluster made of OPX+ and Octaves.
* [make_quam.py](./configuration/make_quam.py): create the state of the system based on the generated wiring and QUAM components and containing all the information necessary to calibrate the chip and run experiments. This state is used to generate the OPX configuration.
* [modify_quam.py](./configuration/modify_quam.py): update the parameters of the state programmatically based on defaults values (previous calibration, chip manufacturer specification...).

**configuration/quam_state**  
The `configuration/quam_state/` directory is where the main QUAM state files are stored. These files are crucial for maintaining the current state of the quantum system, excluding the wiring and network configurations. The `state.json` file contains dynamic aspects of the QUAM state, while the `wiring.json` file holds static information about the system's wiring and network setup.

**quam_libs**  
The `quam_libs` folder contains all the utility functions necessary to create the wiring or build the QUAM, as well as QUA macros and data processing tools:
* [components](./quam_libs/components): this is where the QUAM root and custom QUAM components are defined. A set of basic QUAM components are already present, but advanced user can easily modify them or create new ones.
* [lib](./quam_libs/lib): contains several utility functions for saving, fitting and post-processing data.
* [quam_builder](./quam_libs/quam_builder): contains the main functions called in [machine.py](./quam_libs/quam_builder/machine.py) and used to generate the wiring and build the QUAM structure from it and the QUAM components declared in the [components](./quam_libs/components) folder. It also contains the [pulses.py](./quam_libs/quam_builder/pulses.py) file where the default qubits pulses are defined.
