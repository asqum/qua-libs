import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

from quam.components import Octave

from qualang_tools.wirer import Connectivity
from quam_libs.quam_builder.pulses import add_default_transmon_pulses, add_default_transmon_pair_pulses
from quam_libs.quam_builder.transmons.add_transmon_drive_component import add_transmon_drive_component
from quam_libs.quam_builder.transmons.add_transmon_flux_component import add_transmon_flux_component
from quam_libs.quam_builder.transmons.add_transmon_pair_component import add_transmon_pair_component
from quam_libs.quam_builder.transmons.add_transmon_resonator_component import add_transmon_resonator_component
from qualang_tools.wirer.connectivity.wiring_spec import WiringLineType
from quam_libs.components import OPXPlusQuAM, FEMQuAM, QuAM, Transmon
from quam_libs.quam_builder.wiring.create_wiring import create_wiring


def build_quam(machine: QuAM, quam_state_path: Union[Path, str], octave_names: Sequence[str] = ()) -> QuAM:
    add_octaves(machine, octave_names)
    add_ports(machine)
    add_transmons(machine)
    add_pulses(machine)

    save_machine(machine, quam_state_path)

    return machine


def build_quam_wiring(
    connectivity: Connectivity,
    host_ip: str,
    cluster_name: str,
    quam_state_path: Union[Path, str],
    port: int = None,
    octaves: Optional[Dict[str, Tuple[str, int]]] = None,
) -> QuAM:
    if os.path.exists(quam_state_path) and "state.json" in os.listdir(quam_state_path):
        # if there is a non-empty QuAM state already
        machine = QuAM.load(quam_state_path)
    else:
        machine = create_base_machine(connectivity)

    add_name_and_ip(machine, host_ip, cluster_name, port, octaves)
    machine.wiring = create_wiring(connectivity)
    save_machine(machine, quam_state_path)

    return machine


def create_base_machine(connectivity: Connectivity):
    """
    Detects whether the `connectivity` is using OPX+ or OPX1000 and returns
    the corresponding base object. Otherwise, raises a TypeError.
    """
    for element in connectivity.elements.values():
        for channels in element.channels.values():
            for channel in channels:
                if channel.instrument_id in ["lf-fem", "mw-fem"]:
                    return FEMQuAM()
                elif channel.instrument_id in ["opx+"]:
                    return OPXPlusQuAM()

    raise TypeError(
        "Couldn't identify connectivity as OPX+ or LF-FEM. " "Are channels were allocated for the connectivity?"
    )


def add_name_and_ip(
    machine: QuAM,
    host_ip: str,
    cluster_name: str,
    port: Union[int, None],
    octaves: Optional[Dict[str, Tuple[str, int]]] = None,
):
    """Stores the minimal information to connect to a QuantumMachinesManager."""
    machine.network = {"host": host_ip, "port": port, "cluster_name": cluster_name}
    if octaves:
        machine.network["octaves"] = {name: list(address) for name, address in octaves.items()}


def add_ports(machine: QuAM):
    """
    Creates and stores all input/output ports according to what has been
    allocated to each element in the machine's wiring.
    """
    for wiring_by_element in machine.wiring.values():
        for wiring_by_line_type in wiring_by_element.values():
            for ports in wiring_by_line_type.values():
                for port in ports:
                    if "ports" in ports.get_unreferenced_value(port):
                        machine.ports.reference_to_port(ports.get_unreferenced_value(port), create=True)


def add_transmons(machine: QuAM):
    for element_type, wiring_by_element in machine.wiring.items():
        if element_type == "qubits":
            for qubit_id, wiring_by_line_type in wiring_by_element.items():
                transmon = Transmon(id=qubit_id)
                machine.qubits[qubit_id] = transmon
                for line_type, ports in wiring_by_line_type.items():
                    wiring_path = f"#/wiring/{element_type}/{qubit_id}/{line_type}"
                    if line_type == WiringLineType.RESONATOR.value:
                        add_transmon_resonator_component(transmon, wiring_path, ports, machine)
                    elif line_type == WiringLineType.DRIVE.value:
                        add_transmon_drive_component(transmon, wiring_path, ports)
                    elif line_type == WiringLineType.FLUX.value:
                        add_transmon_flux_component(transmon, wiring_path, ports)
                    else:
                        raise ValueError(f"Unknown line type: {line_type}")
                machine.active_qubit_names.append(transmon.name)

        elif element_type == "qubit_pairs":
            for qubit_pair_id, wiring_by_line_type in wiring_by_element.items():
                for line_type, ports in wiring_by_line_type.items():
                    wiring_path = f"#/wiring/{element_type}/{qubit_pair_id}/{line_type}"
                    if line_type == WiringLineType.COUPLER.value:
                        transmon_pair = add_transmon_pair_component(machine, wiring_path, ports)
                    else:
                        raise ValueError(f"Unknown line type: {line_type}")
                    machine.qubit_pairs[transmon_pair.name] = transmon_pair
                    machine.active_qubit_pair_names.append(transmon_pair.name)


def add_pulses(machine: QuAM):
    if hasattr(machine, "qubits"):
        for transmon in machine.qubits.values():
            add_default_transmon_pulses(transmon)

    if hasattr(machine, "qubit_pairs"):
        for qubit_pair in machine.qubit_pairs.values():
            add_default_transmon_pair_pulses(qubit_pair)


def add_octaves(machine: QuAM, octave_names: Sequence[str]):
    for name in octave_names:
        octave = Octave(name=name)
        machine.octaves[name] = octave
        octave.initialize_frequency_converters()

    return machine


def save_machine(machine: QuAM, quam_state_path: Union[Path, str]):
    machine.save(
        path=quam_state_path,
        content_mapping={
            "wiring.json": ["wiring", "network"],
        },
    )
