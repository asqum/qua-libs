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

# Padding value used to fill input-stream chunks up to the declared array size.
# case_(63) is an idle wait(20ns) on both qubits in play_gate (see below) -- a safe
# no-op, so an off-by-one under switch_case(..., unsafe=True) lands on a harmless
# idle instead of undefined behavior.
INPUT_STREAM_PAD_VALUE = 63

# Rough OPX QUA variable budget; leave headroom for streams, counters, sub-chunk
# length arrays, etc. when sizing input-stream chunks / the non-input-stream array.
OPX_QUA_VARIABLE_BUDGET = 16000


def build_single_depth_chunks(
    circuits_as_ints: list[list[int]],
    circuit_lengths: list[int],
    num_circuits_per_length: int,
    max_chunk_ints: int,
) -> tuple[list[list[list[int]]], int]:
    """
    Pack RB circuits into per-depth sub-chunks for the input-stream QUA path.

    Circuits are grouped strictly by circuit depth: a sub-chunk never mixes circuits
    coming from two different depths. For each depth, its ``num_circuits_per_length``
    circuits are greedily packed into sub-chunks so that each sub-chunk's total int
    count stays <= ``max_chunk_ints``. This keeps the on-OPX save order aligned with
    depth boundaries, which is required for the ``.buffer(...)`` reshape done in
    ``_get_qua_program_with_input_stream``.

    Args:
        circuits_as_ints: Flat list of circuits (each a list of ints terminated by
            the readout opcode), ordered depth-major then circuit-index (as built
            in the calibration node scripts).
        circuit_lengths: Ordered circuit depths (Cliffords) being benchmarked.
        num_circuits_per_length: Number of random circuits sampled per depth.
        max_chunk_ints: Maximum number of ints allowed per sub-chunk (should stay
            comfortably below the OPX QUA variable budget).

    Returns:
        chunks_per_depth: One entry per depth; each entry is a list of sub-chunks
            (each sub-chunk a flat ``list[int]`` of one or more concatenated circuits).
        declared_size: The max sub-chunk length across all depths. This sizes the
            ``declare_input_stream`` array; shorter sub-chunks are padded to it before
            being pushed.
    """
    expected = len(circuit_lengths) * num_circuits_per_length
    if len(circuits_as_ints) != expected:
        raise ValueError(
            f"circuits_as_ints length ({len(circuits_as_ints)}) does not match "
            f"circuit_lengths x num_circuits_per_length ({len(circuit_lengths)} x "
            f"{num_circuits_per_length} = {expected})."
        )

    chunks_per_depth: list[list[list[int]]] = []
    declared_size = 0

    for depth_idx, depth in enumerate(circuit_lengths):
        start = depth_idx * num_circuits_per_length
        end = start + num_circuits_per_length
        circuits_for_depth = circuits_as_ints[start:end]

        sub_chunks: list[list[int]] = []
        current_chunk: list[int] = []

        for circ_idx, circuit in enumerate(circuits_for_depth):
            if len(circuit) > max_chunk_ints:
                raise ValueError(
                    f"Single circuit too large for an input-stream chunk: depth={depth} "
                    f"Cliffords, circuit_index={circ_idx}, circuit_ints={len(circuit)}, "
                    f"max_chunk_ints={max_chunk_ints}. Reduce the depth, raise max_chunk_ints "
                    "(must stay below the OPX QUA variable budget), or reduce num_circuits_per_length."
                )
            if current_chunk and len(current_chunk) + len(circuit) > max_chunk_ints:
                sub_chunks.append(current_chunk)
                current_chunk = []
            current_chunk.extend(circuit)

        if current_chunk:
            sub_chunks.append(current_chunk)

        for sub_chunk in sub_chunks:
            declared_size = max(declared_size, len(sub_chunk))

        chunks_per_depth.append(sub_chunks)

    return chunks_per_depth, declared_size


def validate_without_input_stream_budget(circuits_as_ints: list[list[int]], max_chunk_ints: int) -> None:
    """Fail fast (before QUA compilation) if the flattened sequence would exceed the OPX budget."""
    total_ints = sum(len(circuit) for circuit in circuits_as_ints)
    if total_ints <= max_chunk_ints:
        return
    raise ValueError(
        f"Flattened RB sequence has {total_ints} ints, which exceeds the budget for the "
        f"non-input-stream path (max_chunk_ints={max_chunk_ints}, OPX QUA variable budget "
        f"~{OPX_QUA_VARIABLE_BUDGET}). Set use_input_stream=True in the node parameters, "
        "reduce circuit_lengths, or reduce num_circuits_per_length."
    )


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
        self.use_input_stream = getattr(self.node.parameters, "use_input_stream", False)
        self.max_chunk_ints = getattr(self.node.parameters, "max_chunk_ints", 15000)

        # Total number of distinct RB circuits (across all depths) played per qubit pair.
        self.total_circuits = len(self.circuits_as_ints)

        self.chunks_per_depth = None
        self.declared_size = None
        # Number of sub-chunks pushed via advance_input_stream, per qubit pair. Only
        # meaningful when use_input_stream=True; used to size the progress bar (which
        # tracks host pushes, i.e. depth/sub-chunk boundaries) instead of averages.
        self.n_sub_chunks = None

        if self.use_input_stream:
            self.chunks_per_depth, self.declared_size = build_single_depth_chunks(
                circuits_as_ints=self.circuits_as_ints,
                circuit_lengths=list(self.node.parameters.circuit_lengths),
                num_circuits_per_length=self.node.parameters.num_circuits_per_length,
                max_chunk_ints=self.max_chunk_ints,
            )
            self.n_sub_chunks = sum(len(depth_chunks) for depth_chunks in self.chunks_per_depth)
        else:
            validate_without_input_stream_budget(self.circuits_as_ints, self.max_chunk_ints)

    @property
    def total_pushes(self) -> int:
        """Total number of advance_input_stream calls across all qubit pairs (progress bar total)."""
        if self.n_sub_chunks is None:
            raise RuntimeError("total_pushes is only defined when use_input_stream is True.")
        return self.n_sub_chunks * self.num_pairs
         
    def _get_qua_program_with_input_stream(self):

        # Flatten chunks_per_depth into a single ordered list of sub-chunks: depth-major,
        # then sub-chunk-index within depth -- the same order the QUA program consumes
        # them (via advance_input_stream) and the order push_all_chunks() pushes them in.
        flat_sub_chunks = [sub_chunk for depth_chunks in self.chunks_per_depth for sub_chunk in depth_chunks]
        sub_chunk_lengths = [len(sub_chunk) for sub_chunk in flat_sub_chunks]
        n_sub_chunks = len(flat_sub_chunks)

        with program() as rb:

            n_st = declare_stream()

            # sub_lens lives in a tiny QUA array (one int per sub-chunk) so the play_gate
            # switch_case body is compiled once per qubit pair, not once per sub-chunk.
            sub_lens = declare(int, value=sub_chunk_lengths)

            sequence = declare_input_stream(int, name="sequence", size=self.declared_size)

            state_st = [declare_stream() for _ in range(self.num_pairs)]

            # Counts host pushes (advance_input_stream calls) across all qubit pairs, so
            # "iteration" reports progress in terms of pushed sub-chunks/depths instead of
            # raw shots -- avoids the progress bar being scaled by num_averages.
            push_counter = declare(int, value=0)

            for i, qubit_pair in enumerate(self.qubit_pairs):

                n = declare(int)
                j = declare(int)
                k = declare(int)
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
                    qubit_pair.qubit_control.resonator.wait(qubit_pair.qubit_control.thermalization_time * self.u.ns)
                    qubit_pair.qubit_target.resonator.wait(qubit_pair.qubit_target.thermalization_time * self.u.ns)

                # Align the two elements to play the sequence after qubit initialization
                align()

                # Chunk outer, shot inner: one advance (one host push) per sub-chunk;
                # all shots replay the same sub-chunk on the OPX without extra host pushes.
                with for_(j, 0, j < n_sub_chunks, j + 1):
                    advance_input_stream(sequence)

                    # One push happened; report it immediately so the progress bar advances
                    # as soon as each sub-chunk lands on the OPX, before its averages run.
                    assign(push_counter, push_counter + 1)
                    save(push_counter, n_st)

                    with for_(n, 0, n < self.node.parameters.num_averages, n + 1):

                        with for_(k, 0, k < sub_lens[j], k + 1):
                            play_gate(sequence[k], qubit_pair, state, state_control, state_target, state_st[i], self.reset_mode, self.readout_mode, self.node)

                align()  # align between pairs. No multiplexing support for the moment.

            with stream_processing():
                n_st.save("iteration")
                for i in range(len(self.qubit_pairs)):
                    # Input-stream save order is depth-major (outer) -> shot (middle) ->
                    # circuit-within-depth (inner), unlike the without-input-stream path
                    # which is shot-major. See the Data_fetching section of the calling
                    # node script, which builds the sweep axes accordingly.
                    state_st[i].buffer(self.node.parameters.num_circuits_per_length).buffer(self.node.parameters.num_averages).buffer(len(self.node.parameters.circuit_lengths)).save(
                        f"state{i + 1}"
                    )
        return rb

    def _padded_chunks(self) -> list[list[int]]:
        """Flat depth-major list of sub-chunks, each padded with INPUT_STREAM_PAD_VALUE up to declared_size."""
        declared_size = self.declared_size
        return [
            sub_chunk + [INPUT_STREAM_PAD_VALUE] * (declared_size - len(sub_chunk))
            for depth_chunks in self.chunks_per_depth
            for sub_chunk in depth_chunks
        ]

    def get_all_padded_chunks_for_all_pairs(self) -> list[list[int]]:
        """
        Padded sub-chunks repeated once per qubit pair, in the exact order the QUA
        program consumes/pushes them (qubit pairs run sequentially, not multiplexed).
        Used both by push_all_chunks() and to generate the cloud sync_hook script.
        """
        if not self.use_input_stream:
            raise RuntimeError(
                "get_all_padded_chunks_for_all_pairs called but use_input_stream is False; "
                "the QUA program does not declare an input stream."
            )
        return self._padded_chunks() * len(self.qubit_pairs)

    def push_all_chunks(self, job) -> None:
        """Push input-stream chunks to a running job, in the order the QUA program consumes them."""
        for chunk in self.get_all_padded_chunks_for_all_pairs():
            job.push_to_input_stream("sequence", chunk)
    
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
        if self.use_input_stream:
            return self._get_qua_program_with_input_stream()
        return self._get_qua_program_without_input_stream()
        

circ1 = [0]
circ2 = [1]

qua_circ = [0,66,1]
