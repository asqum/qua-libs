import inspect
from pathlib import Path
from typing import Optional, Union, Literal
import warnings

from qm.qua import *
from quam_libs.components import QuAM
from quam_libs.components import Transmon, TransmonPair
from quam.utils.qua_types import QuaVariable, ScalarFloat

__all__ = [
    "qua_declaration",
    "multiplexed_readout",
    "node_save",
    "active_reset",
    "readout_state",
]


def qua_declaration(num_qubits):
    """
    Macro to declare the necessary QUA variables

    :param num_qubits: Number of qubits used in this experiment
    :return:
    """
    n = declare(int)
    n_st = declare_stream()
    I = [declare(fixed) for _ in range(num_qubits)]
    Q = [declare(fixed) for _ in range(num_qubits)]
    I_st = [declare_stream() for _ in range(num_qubits)]
    Q_st = [declare_stream() for _ in range(num_qubits)]
    # Workaround to manually assign the results variables to the readout elements
    # for i in range(num_qubits):
    #     assign_variables_to_element(f"rr{i}", I[i], Q[i])
    return I, I_st, Q, Q_st, n, n_st


def multiplexed_readout(qubits, I, I_st, Q, Q_st, sequential=False, amplitude=1.0, weights=""):
    """Perform multiplexed readout on two resonators"""

    for ind, q in enumerate(qubits):
        q.resonator.measure("readout", qua_vars=(I[ind], Q[ind]), amplitude_scale=amplitude)

        if I_st is not None:
            save(I[ind], I_st[ind])
        if Q_st is not None:
            save(Q[ind], Q_st[ind])

        if sequential and ind < len(qubits) - 1:
            align(q.resonator.name, qubits[ind + 1].resonator.name)


def node_save(
    quam: QuAM,
    name: str,
    data: dict,
    additional_files: Optional[Union[dict, bool]] = None,
):
    # Save results
    if isinstance(additional_files, dict):
        quam.data_handler.additional_files = additional_files
    elif additional_files is True:
        files = ["../calibration_db.json", "optimal_weights.npz"]

        try:
            files.append(inspect.currentframe().f_back.f_locals["__file__"])
        except Exception:
            warnings.warn("Could not find the script file path to save it in the data folder")

        additional_files = {}
        for file in files:
            filepath = Path(file)
            if not filepath.exists():
                warnings.warn(f"File {file} does not exist, unable to save file")
                continue
            additional_files[str(filepath)] = filepath.name
    else:
        additional_files = {}
    quam.data_handler.additional_files = additional_files
    quam.data_handler.save_data(data=data, name=name)

    # Save QuAM to the data folder
    quam.save(
        path=quam.data_handler.path / "state.json",
    )
    quam.save(
        path=quam.data_handler.path / "quam_state",
        content_mapping={"wiring.json": {"wiring", "network"}},
    )

    # Save QuAM to configuration directory / `state.json`
    quam.save(content_mapping={"wiring.json": {"wiring", "network"}})


def readout_state(qubit, state, pulse_name: str = "readout", threshold: float = None, save_qua_var: StreamType = None):
    I = declare(fixed)
    Q = declare(fixed)
    if threshold is None:
        threshold = qubit.resonator.operations[pulse_name].threshold
    qubit.resonator.measure(pulse_name, qua_vars=(I, Q))
    assign(state, Cast.to_int(I > threshold))
    wait(qubit.resonator.depletion_time // 4, qubit.resonator.name)


def readout_state_gef(
    qubit: Transmon, state: QuaVariable, pulse_name: str = "readout", save_qua_var: StreamType = None
):
    I = declare(fixed)
    Q = declare(fixed)
    diff = declare(fixed, size=3)

    qubit.resonator.update_frequency(qubit.resonator.intermediate_frequency + qubit.resonator.GEF_frequency_shift, keep_phase=True)
    wait(4)
    qubit.resonator.measure(pulse_name, qua_vars=(I, Q))
    wait(4)
    qubit.resonator.update_frequency(qubit.resonator.intermediate_frequency, keep_phase=True)
    wait(4)

    gef_centers = [qubit.resonator.gef_centers[0], qubit.resonator.gef_centers[1], qubit.resonator.gef_centers[2]]
    for p in range(3):
        assign(
            diff[p],
            Math.abs(I - gef_centers[p][0]) + Math.abs(Q - gef_centers[p][1]),
        )
    wait(4)
    assign(state, Math.argmin(diff))
    qubit.wait(qubit.resonator.depletion_time // 4)


def active_reset_gef(
    qubit: Transmon,
    readout_pulse_name: str = "readout",
    pi_01_pulse_name: str = "x180",
    pi_12_pulse_name: str = "EF_x180",
    max_attempts: int = 10,
):
    res_ar = declare(int)
    success = declare(int)
    assign(success, 0)
    attempts = declare(int)
    assign(attempts, 0)
    qubit.align()
    qubit.resonator.wait(4*(qubit.resonator.depletion_time//4))       
    wait(4)
    with while_((success < 2) & (attempts < max_attempts)):
        readout_state_gef(qubit, res_ar, readout_pulse_name)
        qubit.align()
        with if_(res_ar == 0):
            assign(success, success + 1)  # we need to measure 'g' two times in a row to increase our confidence
        with if_(res_ar == 1):
            update_frequency(qubit.xy.name, int(qubit.xy.intermediate_frequency))
            qubit.xy.play(pi_01_pulse_name)
            assign(success, 0)
        with if_(res_ar == 2):
            update_frequency(
                qubit.xy.name,
                int(qubit.xy.intermediate_frequency - qubit.anharmonicity),
            )
            qubit.xy.play(pi_12_pulse_name)
            update_frequency(qubit.xy.name, int(qubit.xy.intermediate_frequency))
            qubit.xy.play(pi_01_pulse_name)
            assign(success, 0)
        qubit.align()
        assign(attempts, attempts + 1)

def active_reset_simple(
        qubit: Transmon,
        save_qua_var: Optional[StreamType] = None,
        pi_pulse_name: str = "x180",
        readout_pulse_name: str = "readout"):
    """
    Simple active reset for a qubit
    """
    pulse = qubit.resonator.operations[readout_pulse_name]

    I = declare(fixed)
    Q = declare(fixed)
    state = declare(bool)
    qubit.align()
    qubit.resonator.measure("readout", qua_vars=(I, Q))
    assign(state, I > pulse.threshold)
    wait(qubit.resonator.depletion_time // 4, qubit.resonator.name)
    qubit.align()
    qubit.xy.play(pi_pulse_name, condition=state)
    qubit.align()


def active_reset(
        qubit: Transmon,
        save_qua_var: Optional[StreamType] = None,
        pi_pulse_name: str = "x180",
        readout_pulse_name: str = "readout",
        max_attempts: int = 15):
    pulse = qubit.resonator.operations[readout_pulse_name]

    I = declare(fixed, value=pulse.rus_exit_threshold * 1.1)
    Q = declare(fixed)
    state = declare(bool)
    attempts = declare(int, value=1)
    assign(attempts, 1)
    qubit.align()
    qubit.resonator.measure("readout", qua_vars=(I, Q))
    assign(state, I > pulse.threshold)
    wait(qubit.resonator.depletion_time // 4, qubit.resonator.name)
    qubit.align()
    qubit.xy.play(pi_pulse_name, condition=state)
    qubit.align()
    with while_(broadcast.and_(I > pulse.rus_exit_threshold, attempts < max_attempts)):
        qubit.align()
        wait(4)
        reset_if_phase(qubit.resonator.name)
        wait(4)
        qubit.resonator.measure("readout", qua_vars=(I, Q))
        assign(state, I > pulse.threshold)
        wait(qubit.resonator.depletion_time, qubit.resonator.name)
        qubit.align()
        qubit.xy.play(pi_pulse_name, condition=state)
        qubit.align()
        assign(attempts, attempts + 1)
    wait(500, qubit.xy.name)
    qubit.align()
    if save_qua_var is not None:
        save(attempts, save_qua_var)

def active_reset_coupler(
        drive_qubit: Transmon,
        read_qubit: Transmon,
        pi_pulse_name: str,
        flux_applied_target:Transmon|TransmonPair|None = None,
        save_qua_var: Optional[StreamType] = None,
        max_attempts: int = 15,
        method:Literal['standard', 'aswap'] = 'aswap'):
    if method == 'standard':
        state = declare(int)
        true_int = declare(int, value=1)
        cond = declare(bool)
        attempts = declare(int, value=1)
        align()
        readout_state_coupler(read_qubit, state, method='aswap', flux_applied_target=flux_applied_target)
        assign(cond, state==true_int)
        align()
        drive_qubit.xy.play(pi_pulse_name, condition=cond)
        align()
        with while_(broadcast.and_(cond, attempts < max_attempts)):
            align()
            wait(4)
            reset_if_phase(read_qubit.resonator.name)
            wait(4)
            readout_state_coupler(read_qubit, state, method='aswap', flux_applied_target=flux_applied_target)
            assign(cond, state==true_int)
            align()
            drive_qubit.xy.play(pi_pulse_name, condition=cond)
            align()
            assign(attempts, attempts + 1)
        wait(250)
        if save_qua_var is not None:
            save(attempts, save_qua_var)
    else:

        align()
        # # active_reset(drive_qubit)
        
        readout_state_coupler(read_qubit, state=None, method='aswap', flux_applied_target=flux_applied_target, active_reset_readout_q=True)
        align()
        # active_reset(drive_qubit)
        active_reset(read_qubit, max_attempts=1)
        align()
        

def split_bipolar_macro(
        qbORqp:Transmon|TransmonPair,
        amplitude_scale:float|ScalarFloat = 1.0,
        neg_pole_amp_ratio:float|ScalarFloat = 1.0,
        debug:bool=False):

    ### ========== Composing ==========
    if isinstance(qbORqp, Transmon):
        channel = qbORqp.z
    elif isinstance(qbORqp, TransmonPair):
        channel = qbORqp.coupler
    else:
        raise TypeError(f"Target assigned for split_bipolar macro must be Transmon or TransmonPair ! Go checking it plz.")
    
    
    if not debug:
        ## Half Cosine Raise
        channel.play('flattopV2', amplitude_scale=amplitude_scale)
        ## Half Cosine Fall
        channel.play('flattopV2', amplitude_scale=-1*neg_pole_amp_ratio*amplitude_scale)
    else:
        ## Half Cosine Raise
        channel.play('Cz_flattop', amplitude_scale=amplitude_scale)
        ## Half Cosine Fall
        channel.play('Cz_flattop', amplitude_scale=-1*neg_pole_amp_ratio*amplitude_scale)
        
    # qbORqp.align()

def readout_state_coupler(
        qb2read:Transmon,
        state: QuaVariable|None,
        method:Literal["aswap", "3tone", "zz-pi"],
        flux_applied_target:Transmon|TransmonPair|None = None,
        readout_gef:bool = False,
        buffer_b4_readout:bool = True,
        zz_pi_pulse_duration_scale:int = 100,
        assign_aswap_duration:int|None = None,
        active_reset_readout_q:bool = False
    ):
    '''
    A readout macro for reading a coupler, can be achieved by either performing an aswap and reading the qubit, or by performing a 3-tone spectroscopy.
    * Note: For the aSWAP method, the pulse for performing the aSWAP should be pre-configured in the state.json with the name "aSWAP". The pulse should be designed to bring the qubit and coupler into resonance for the swap. The flux_applied_target parameter should be set accordingly based on whether the flux pulse is applied to the coupler or the qubit. From Li-Chieh's experiences, the amplitude and duration can be set to a half flux period and 400ns.
    :param qb2read: readout qubit.
    ### Warning: Here is a global align() in the beginning.\n
    :param state: the readout result will be assigned to this variable.
    :param method: supports both methods of reading the coupler: "aswap", "zz-pi" and "3tone". The 'zz-pi' method uses a 10 times long pi-pulse to drive readout qubit, and the '3tone' method uses a saturation pulse to drive the readout qubit.
    :param flux_applied_target: If method is 'aswap', we should apply a flux pulse on either the coupler (TransmonPair) or the qubit (Transmon) to bring them into resonance for the swap. This parameter specifies which one to apply the flux pulse on, and it's determined by the frequency position. The None (default) was given, the flux pulse will be applied on the readout qubit itself.
    :param readout_gef: whether to use GEF-readout or GE-readout.
    :param buffer_b4_readout: Add some buffer time before the readout. The buffer time is fixed at this moment.
    :param zz_pi_pulse_duration_scale: Only for 'zz-pi' method. The duration of the long pi pulse is determined by multiplying the original pi pulse duration with this scale factor. The amplitude is set to be the inverse of this scale factor to make sure the total pulse area is the same as the original pi pulse.
    :param assign_aswap_duration: The pulse duration for the aSWAP pulse, unit in QUA cycle (4ns).

    '''
    align() # to make sure the timing is correct when performing the aSWAP and the readout
    if active_reset_readout_q:
        active_reset(qb2read, max_attempts=2)
        align()
    match method:
        case "aswap":
            try:
                
                align()
                if flux_applied_target is not None:
                    if isinstance(flux_applied_target, TransmonPair):
                        if assign_aswap_duration is None:
                            flux_applied_target.coupler.play('aSWAP', amplitude_scale= 1.0)
                        else:
                            flux_applied_target.coupler.play('aSWAP', amplitude_scale= 1.0, duration=assign_aswap_duration)
                    elif isinstance(flux_applied_target, Transmon):
                        if assign_aswap_duration is None:
                            flux_applied_target.z.play('aSWAP', amplitude_scale= 1.0)
                        else:
                            flux_applied_target.z.play('aSWAP', amplitude_scale= 1.0, duration=assign_aswap_duration)
                else:
                    if assign_aswap_duration is None:
                        qb2read.z.play('aSWAP', amplitude_scale= 1.0)
                    else:
                        qb2read.z.play('aSWAP', amplitude_scale= 1.0, duration=assign_aswap_duration)
                    
                align()
                if buffer_b4_readout:
                    wait(25) # optional wait time, 40ns is recommended by Li-Chieh  
                if state is not None:
                    if not readout_gef:
                        readout_state(qb2read, state=state, pulse_name='readout')
                    else:
                        readout_state_gef(qb2read, state=state, pulse_name='readout')
            except:
                print("Got an issue when performing the aSWAP readout. Please check if the flux pulse is properly configured with the name 'aSWAP' in the state.json")
        case "zz-pi":
            long_pi_dura = (qb2read.xy.operations['x180'].length * zz_pi_pulse_duration_scale)//4 # to QUA clicks
            qb2read.xy.play('x180', amplitude_scale=1/zz_pi_pulse_duration_scale, duration=long_pi_dura) # long pi-pulse with 10*pi duration and 0.1*amplitude
            qb2read.align()
            if buffer_b4_readout:
                wait(4) # buffer
            if not readout_gef:
                readout_state(qb2read, state=state, pulse_name='readout')
                assign(state, 1-state) # flip the state since the qubit is in excited state after the long pi pulse
            else:
                readout_state_gef(qb2read, state=state, pulse_name='readout')
            
    
        case _:
            pass # (extend it in the near future for 3-tone spectroscopy (saturation driving) method)
        