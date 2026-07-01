from typing import Literal
from more_itertools import flatten
import numpy as np
from quam_libs.components import Transmon, TransmonPair
from qm.qua import * 
from qm.qua._expressions import QuaVariable, QuaArrayVariable
from quam_libs.lib.data_utils import split_list_by_integer_count
from quam_libs.macros import active_reset, active_reset_gef, readout_state, qua_declaration, align, assign, reset_frame, readout_state_gef
from qualibrate import NodeParameters, QualibrationNode
from quam_libs.components import QuAM
from qualang_tools.units import unit

def reset_qubits(node, control: Transmon, target: Transmon, thermalization_time: float | None = None):
    reset_mode = getattr(node.parameters, "reset_type_thermal_or_active", "active")
    if reset_mode == "active":
        active_reset(control, "readout")
        active_reset(target, "readout")
    elif reset_mode == "active_gef":
        active_reset_gef(control, "readout")
        active_reset_gef(target, "readout")
    else:
        control.resonator.wait(thermalization_time // 4)


def measure_two_qubit_state(node, qubit_pair: TransmonPair, state: QuaVariable, state_control: QuaVariable, state_target: QuaVariable):
    readout_mode = getattr(node.parameters, "readout_mode", "ge")

    if readout_mode == "gef":
        readout_state_gef(qubit_pair.qubit_control, state_control)
        readout_state_gef(qubit_pair.qubit_target, state_target)
        with if_((state_control < 2) & (state_target < 2)):
            assign(state, state_control * 2 + state_target)
        with else_():
            assign(state, 4)
    else:
        readout_state(qubit_pair.qubit_control, state_control)
        readout_state(qubit_pair.qubit_target, state_target)
        assign(state, state_control * 2 + state_target)


def play_gate(gate: QuaVariable, qubit_pair: TransmonPair, state: QuaVariable, state_control: QuaVariable, state_target: QuaVariable, state_st: "_ResultSource", reset_type: Literal["thermal", "active", "active_gef"], readout_mode: Literal["ge", "gef"], node: QualibrationNode):
    with switch_(gate, unsafe=True):
                               
        with case_(0):
            qubit_pair.qubit_control.xy.play("x90")
            qubit_pair.qubit_target.xy.play("x90")
        with case_(1):
            qubit_pair.qubit_control.xy.play("x90")
            qubit_pair.qubit_target.xy.play("x180")
        with case_(2):
            qubit_pair.qubit_control.xy.play("x90")
            qubit_pair.qubit_target.xy.play("y90")
        with case_(3):
            qubit_pair.qubit_control.xy.play("x90")
            qubit_pair.qubit_target.xy.play("y180")
        with case_(4):
            qubit_pair.qubit_control.xy.play("x90")
            qubit_pair.qubit_target.xy.frame_rotation(np.pi/2)
        with case_(5):
            qubit_pair.qubit_control.xy.play("x90")
            qubit_pair.qubit_target.xy.frame_rotation(np.pi)
        with case_(6):
            qubit_pair.qubit_control.xy.play("x90")
            qubit_pair.qubit_target.xy.frame_rotation(3*np.pi/2)
        with case_(7):
            qubit_pair.qubit_control.xy.play("x90")
        with case_(8):
            qubit_pair.qubit_control.xy.play("x180")
            qubit_pair.qubit_target.xy.play("x90")
        with case_(9):
            qubit_pair.qubit_control.xy.play("x180")
            qubit_pair.qubit_target.xy.play("x180")
        with case_(10):
            qubit_pair.qubit_control.xy.play("x180")
            qubit_pair.qubit_target.xy.play("y90")
        with case_(11):
            qubit_pair.qubit_control.xy.play("x180")
            qubit_pair.qubit_target.xy.play("y180")
        with case_(12):
            qubit_pair.qubit_control.xy.play("x180")
            qubit_pair.qubit_target.xy.frame_rotation(np.pi/2)
        with case_(13):
            qubit_pair.qubit_control.xy.play("x180")
            qubit_pair.qubit_target.xy.frame_rotation(np.pi)
        with case_(14):
            qubit_pair.qubit_control.xy.play("x180")
            qubit_pair.qubit_target.xy.frame_rotation(3*np.pi/2)
        with case_(15):
            qubit_pair.qubit_control.xy.play("x180")
        with case_(16):
            qubit_pair.qubit_control.xy.play("y90")
            qubit_pair.qubit_target.xy.play("x90")
        with case_(17):
            qubit_pair.qubit_control.xy.play("y90")
            qubit_pair.qubit_target.xy.play("x180")
        with case_(18):
            qubit_pair.qubit_control.xy.play("y90")
            qubit_pair.qubit_target.xy.play("y90")
        with case_(19):
            qubit_pair.qubit_control.xy.play("y90")
            qubit_pair.qubit_target.xy.play("y180")
        with case_(20):
            qubit_pair.qubit_control.xy.play("y90")
            qubit_pair.qubit_target.xy.frame_rotation(np.pi/2)
        with case_(21):
            qubit_pair.qubit_control.xy.play("y90")
            qubit_pair.qubit_target.xy.frame_rotation(np.pi)
        with case_(22):
            qubit_pair.qubit_control.xy.play("y90")
            qubit_pair.qubit_target.xy.frame_rotation(3*np.pi/2)
        with case_(23):
            qubit_pair.qubit_control.xy.play("y90")
        with case_(24):
            qubit_pair.qubit_control.xy.play("y180")
            qubit_pair.qubit_target.xy.play("x90")
        with case_(25):
            qubit_pair.qubit_control.xy.play("y180")
            qubit_pair.qubit_target.xy.play("x180")
        with case_(26):
            qubit_pair.qubit_control.xy.play("y180")
            qubit_pair.qubit_target.xy.play("y90")
        with case_(27):
            qubit_pair.qubit_control.xy.play("y180")
            qubit_pair.qubit_target.xy.play("y180")
        with case_(28):
            qubit_pair.qubit_control.xy.play("y180")
            qubit_pair.qubit_target.xy.frame_rotation(np.pi/2)
        with case_(29):
            qubit_pair.qubit_control.xy.play("y180")
            qubit_pair.qubit_target.xy.frame_rotation(np.pi)
        with case_(30):
            qubit_pair.qubit_control.xy.play("y180")
            qubit_pair.qubit_target.xy.frame_rotation(3*np.pi/2)
        with case_(31):
            qubit_pair.qubit_control.xy.play("y180")
        with case_(32):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi/2)
            qubit_pair.qubit_target.xy.play("x90")
        with case_(33):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi/2)
            qubit_pair.qubit_target.xy.play("x180")
        with case_(34):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi/2)
            qubit_pair.qubit_target.xy.play("y90")
        with case_(35):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi/2)
            qubit_pair.qubit_target.xy.play("y180")
        with case_(36):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi/2)
            qubit_pair.qubit_target.xy.frame_rotation(np.pi/2)
        with case_(37):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi/2)
            qubit_pair.qubit_target.xy.frame_rotation(np.pi)
        with case_(38):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi/2)
            qubit_pair.qubit_target.xy.frame_rotation(3*np.pi/2)
        with case_(39):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi/2)
        with case_(40):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi)
            qubit_pair.qubit_target.xy.play("x90")
        with case_(41):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi)
            qubit_pair.qubit_target.xy.play("x180")
        with case_(42):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi)
            qubit_pair.qubit_target.xy.play("y90")
        with case_(43):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi)
            qubit_pair.qubit_target.xy.play("y180")
        with case_(44):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi)
            qubit_pair.qubit_target.xy.frame_rotation(np.pi/2)
        with case_(45):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi)
            qubit_pair.qubit_target.xy.frame_rotation(np.pi)
        with case_(46):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi)
            qubit_pair.qubit_target.xy.frame_rotation(3*np.pi/2)
        with case_(47):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi)
        with case_(48):
            qubit_pair.qubit_control.xy.frame_rotation(3*np.pi/2)
            qubit_pair.qubit_target.xy.play("x90")
        with case_(49):
            qubit_pair.qubit_control.xy.frame_rotation(3*np.pi/2)
            qubit_pair.qubit_target.xy.play("x180")
        with case_(50):
            qubit_pair.qubit_control.xy.frame_rotation(3*np.pi/2)
            qubit_pair.qubit_target.xy.play("y90")
        with case_(51):
            qubit_pair.qubit_control.xy.frame_rotation(3*np.pi/2)
            qubit_pair.qubit_target.xy.play("y180")
        with case_(52):
            qubit_pair.qubit_control.xy.frame_rotation(3*np.pi/2)
            qubit_pair.qubit_target.xy.frame_rotation(np.pi/2)
        with case_(53):
            qubit_pair.qubit_control.xy.frame_rotation(3*np.pi/2)
            qubit_pair.qubit_target.xy.frame_rotation(np.pi)
        with case_(54):
            qubit_pair.qubit_control.xy.frame_rotation(3*np.pi/2)
            qubit_pair.qubit_target.xy.frame_rotation(3*np.pi/2)
        with case_(55):
            qubit_pair.qubit_control.xy.frame_rotation(3*np.pi/2)
        with case_(56):
            qubit_pair.qubit_target.xy.play("x90")
        with case_(57):
            qubit_pair.qubit_target.xy.play("x180")
        with case_(58):
            qubit_pair.qubit_target.xy.play("y90")
        with case_(59):
            qubit_pair.qubit_target.xy.play("y180")
        with case_(60):
            qubit_pair.qubit_target.xy.frame_rotation(np.pi/2)
        with case_(61):
            qubit_pair.qubit_target.xy.frame_rotation(np.pi)
        with case_(62):
            qubit_pair.qubit_target.xy.frame_rotation(3*np.pi/2)
        with case_(63):
            qubit_pair.qubit_control.wait(20)
            qubit_pair.qubit_target.wait(20)
        with case_(64): #CZ
            # qubit_pair.macros['cz'].apply()
            qubit_pair.gates['Cz'].execute()
        with case_(65): # idle_2q
            # # wait CZ duratoin
            # qubit_pair.qubit_control.wait(qubit_pair.gates['Cz'].flux_pulse_control.length // 4)
            # qubit_pair.qubit_target.wait(qubit_pair.gates['Cz'].flux_pulse_control.length // 4)
            # # original ver 
            # qubit_pair.qubit_control.wait(4)
            # qubit_pair.qubit_target.wait(4)
            # # wait sq gate duraition
            qubit_pair.qubit_control.wait(qubit_pair.qubit_control.xy.operations['x180'].length//4)
            qubit_pair.qubit_target.wait(qubit_pair.qubit_target.xy.operations['x180'].length//4)
        
        with case_(66):
            
            align()
            wait(4)
            
            measure_two_qubit_state(node, qubit_pair, state, state_control, state_target)
            save(state, state_st)
            
            # Initialize the qubits
            if reset_type == "active":
                active_reset(qubit_pair.qubit_control, "readout")
                active_reset(qubit_pair.qubit_target, "readout")
            elif reset_type == "active_gef":
                active_reset_gef(qubit_pair.qubit_control, "readout")
                active_reset_gef(qubit_pair.qubit_target, "readout")
            else:
                qubit_pair.qubit_control.resonator.wait(qubit_pair.qubit_control.thermalization_time // 4)
                qubit_pair.qubit_target.resonator.wait(qubit_pair.qubit_target.thermalization_time // 4)
            # Reset the frame of the qubits in order not to accumulate rotations
            reset_frame(qubit_pair.qubit_control.xy.name, qubit_pair.qubit_target.xy.name)
            
            align()
            
def play_sequence(sequence: QuaArrayVariable, depth: int, qubit_pair: TransmonPair, state: list[QuaVariable], state_control: QuaVariable, state_target: QuaVariable, state_st, reset_type: Literal["thermal", "active", "active_gef"], readout_mode: Literal["ge", "gef"], node: QualibrationNode): 
    
    i = declare(int)
    with for_(i, 0, i < depth, i + 1):
        play_gate(sequence[i], qubit_pair, state, state_control, state_target, state_st, reset_type, readout_mode, node)    

class QuaProgramHandler:
    
    def __init__(self, node: QualibrationNode, num_pairs: int, circuits_as_ints: list[int], machine: QuAM, qubit_pairs: list[TransmonPair], max_sequence_length: int = 6000):
        
        self.u = unit(coerce_to_integer=True)
        self.node = node
        self.num_pairs = num_pairs
        self.circuits_as_ints = circuits_as_ints
        self.machine = machine
        self.qubit_pairs = qubit_pairs
        self.max_sequence_length = max_sequence_length
        self.readout_mode = getattr(self.node.parameters, "readout_mode", "ge")
        self.reset_mode = getattr(self.node.parameters, "reset_type_thermal_or_active", "active")
         
    def _get_qua_program_with_input_stream(self):
        
        with program() as rb:
    
            n = declare(int)
            n_st = declare_stream()
            
            sequence = declare_input_stream(int, name=f"sequence", size=self.max_current_sequence_length)
            
            # The relevant streams
            state_control = declare(int)
            state_target = declare(int)
            state = declare(int)
            state_st = [declare_stream() for _ in range(self.num_pairs)]

            for i, qubit_pair in enumerate(self.qubit_pairs):

                # Bring the active qubits to the desired frequency point
                self.machine.set_all_fluxes(flux_point=self.node.parameters.flux_point_joint_or_independent, target=qubit_pair.qubit_control)

                # Initialize the qubits
                if self.reset_mode == "active":
                    active_reset(qubit_pair.qubit_control, "readout")
                    active_reset(qubit_pair.qubit_target, "readout")
                elif self.reset_mode == "active_gef":
                    active_reset_gef(qubit_pair.qubit_control, "readout")
                    active_reset_gef(qubit_pair.qubit_target, "readout")
                else:
                    # qubit_pair.qubit_control.resonator.wait(4)
                    qubit_pair.qubit_control.resonator.wait(qubit_pair.qubit_control.thermalization_time * self.u.ns)
                    qubit_pair.qubit_target.resonator.wait(qubit_pair.qubit_target.thermalization_time * self.u.ns)
                
                # Align the two elements to play the sequence after qubit initialization
                align()
                
                for l in self.sequence_lengths:
                    advance_input_stream(sequence)

                    with for_(n, 0, n < self.node.parameters.num_averages, n + 1):
                        
                        play_sequence(sequence, l, qubit_pair, state, state_control, state_target, state_st[i], self.reset_mode, self.readout_mode, self.node)
                                    
                        save(n, n_st)

            with stream_processing():
                n_st.save("iteration")
                for i in range(len(self.qubit_pairs)):
                    state_st[i].buffer(self.node.parameters.num_circuits_per_length).buffer(len(self.node.parameters.circuit_lengths)).buffer(self.node.parameters.num_averages).save(
                        f"state{i + 1}"
                    )
        return rb
    
    def _get_qua_program_without_input_stream(self):
        
        job_sequence = list(flatten(self.circuits_as_ints))
        sequence_length = len(job_sequence)
        
        
        with program() as rb:
    
            n_st = declare_stream()
            job_sequence_qua = declare(int, value=job_sequence)
            # The relevant streams
            state_st = [declare_stream() for _ in range(self.num_pairs)]

            for i, qubit_pair in enumerate(self.qubit_pairs):
                
                n = declare(int)
                state_control = declare(int)
                state_target = declare(int)
                state = declare(int)
                # Bring the active qubits to the desired frequency point
                self.machine.set_all_fluxes(flux_point=self.node.parameters.flux_point_joint_or_independent, target=qubit_pair.qubit_control)

                # Initialize the qubits
                if self.reset_mode == "active":
                    active_reset(qubit_pair.qubit_control, "readout")
                    active_reset(qubit_pair.qubit_target, "readout")
                elif self.reset_mode == "active_gef":
                    active_reset_gef(qubit_pair.qubit_control, "readout")
                    active_reset_gef(qubit_pair.qubit_target, "readout")
                else:
                    # qubit_pair.qubit_control.resonator.wait(4)
                    qubit_pair.qubit_control.resonator.wait(qubit_pair.qubit_control.thermalization_time * self.u.ns)
                    qubit_pair.qubit_target.resonator.wait(qubit_pair.qubit_target.thermalization_time * self.u.ns)
                
                # Align the two elements to play the sequence after qubit initialization
                align()
                
                with for_(n, 0, n < self.node.parameters.num_averages, n + 1):
                    
                    play_sequence(job_sequence_qua, sequence_length, qubit_pair, state, state_control, state_target, state_st[i], self.reset_mode, self.readout_mode, self.node)
                                
                    save(n, n_st)
                    
                align() # align between pairs. No multiplexing support for the moment.

            with stream_processing():
                n_st.save("iteration")
                for i in range(len(self.qubit_pairs)):
                    state_st[i].buffer(self.node.parameters.num_circuits_per_length).buffer(len(self.node.parameters.circuit_lengths)).buffer(self.node.parameters.num_averages).save(
                        f"state{i + 1}"
                    )
        return rb
    
    
    def get_qua_program(self):
        return self._get_qua_program_without_input_stream()
        

circ1 = [0]
circ2 = [1]

qua_circ = [0,66,1]
