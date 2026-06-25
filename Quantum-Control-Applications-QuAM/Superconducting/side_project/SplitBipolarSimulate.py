"""
        IQ BLOBS
This sequence involves measuring the state of the resonator 'N' times, first after thermalization (with the qubit
in the |g> state) and then after applying a pi pulse to the qubit (bringing the qubit to the |e> state) successively.
The resulting IQ blobs are displayed, and the data is processed to determine:
    - The rotation angle required for the integration weights, ensuring that the separation between |g> and |e> states
      aligns with the 'I' quadrature.
    - The threshold along the 'I' quadrature for effective qubit state discrimination.
    - The readout fidelity matrix, which is also influenced by the pi pulse fidelity.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit.wait(qubit.thermalization_time * u.ns) spectroscopy, power_rabi and updated the state.
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the rotation angle (rotation_angle) in the state.
    - Update the g -> e thresholds (threshold & rus_threshold) in the state.
    - Update the confusion matrices in the state.
    - Save the current state
"""


# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from typing import Literal, Optional, List
from qualibrate import QualibrationNode
from quam_libs.components import QuAM
from quam_libs.components import Transmon, TransmonPair
from quam.utils.qua_types import QuaVariable, ScalarInt, ScalarFloat
from qualibrate import NodeParameters, QualibrationNode
from quam_libs.macros import qua_declaration, active_reset
from qm import SimulationConfig
from qm.qua import *
from qualang_tools.units import unit

# %% {Node_parameters}


class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ['q3']
    num_runs: int = 500
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    reset_type_thermal_or_active: Literal["active", "thermal"] = "thermal"
    use_state_discrimination: bool = False
    simulation_duration_ns: int = 51_000
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = True
    pulse_tot_len_ns:int = 32_766
    falt_len_ratio:float = 0.8
    switch_pt_ratio:float = 0.75
    pulse_align_debug:bool = True

node = QualibrationNode(name="BipolarOpimize", parameters=Parameters())

def int_fixed_multiply(a_int:ScalarInt, a_fixed:ScalarFloat):
    """
    Safely multiply a QUA-int and a QUA-fixed (By Gemini), BUT int must < 32767
    """
    
    ratio_raw = declare(int)
    result = declare(int)

    
    assign(ratio_raw, Cast.unsafe_cast_int(a_fixed))
    
    
    assign(result, (a_int * (ratio_raw >> 12)) >> 16)
    
    return result

def split_bipolar_macro(
        qbORqp:Transmon|TransmonPair,
        total_len_ns:ScalarInt,
        switch_point_ratio:ScalarFloat,
        flat_ratio:ScalarFloat,
        amplitude_scale:ScalarFloat = 1.0,
        debug:bool=False):
        
    
    neg = declare(fixed, value=-1.0)
    total_len = declare(int)
    assign(total_len, total_len_ns)

    ###  ========== Length definition ==========

    ##  positive total length
    pos_len, pos_flat_len = declare(int), declare(int)
    assign(pos_len, int_fixed_multiply(total_len, switch_point_ratio))
    assign(pos_flat_len, int_fixed_multiply(pos_len, flat_ratio))
    ##  negative total length 
    neg_len, neg_flat_len = declare(int), declare(int)
    assign(neg_len, total_len-pos_len)
    assign(neg_flat_len, int_fixed_multiply(neg_len, flat_ratio))

    ## length for Rise and Fall region (positive part)
    rise_len, fall_to_zero_len = declare(int), declare(int)
    assign(rise_len, Cast.to_int((pos_len-pos_flat_len)>> 1))
    assign(fall_to_zero_len, pos_len-pos_flat_len-rise_len)

    ## length for Rise and Fall region (negative part)
    return_zero_len, fall_from_zero_len = declare(int), declare(int)
    assign(fall_from_zero_len, Cast.to_int((neg_len-neg_flat_len)>> 1))
    assign(return_zero_len, neg_len-neg_flat_len-fall_from_zero_len)

    ### check total length conserve
    # composed_total_len, padding_length = declare(int), declare(int)
    # assign(composed_total_len, rise_len+pos_flat_len+fall_to_zero_len+fall_from_zero_len+neg_flat_len+return_zero_len)
    # assign(padding_length, total_len-composed_total_len) # must >= 0
    # with if_(padding_length==0):
    #     pass
    # with else_():
    #     # odd padding
    #     with if_((padding_length & 1) == 1):
    #         assign(pos_flat_len, pos_flat_len + (padding_length >> 1) + 1)
    #         assign(neg_flat_len, neg_flat_len + (padding_length >> 1))
            
    #     # even padding
    #     with else_():
    #         assign(pos_flat_len, pos_flat_len+Cast.to_int(padding_length>> 1))
    #         assign(neg_flat_len, neg_flat_len+Cast.to_int(padding_length>> 1))




    ### ========== Composing ==========
    if isinstance(qbORqp, Transmon):
        channel = qbORqp.z
    elif isinstance(qbORqp, TransmonPair):
        channel = qbORqp.coupler
    else:
        raise TypeError(f"Target assigned for split_bipolar macro must be Transmon or TransmonPair ! Go checking it plz.")
    
    if not debug:
        ## Half Cosine Raise
        channel.play('hcrp', amplitude_scale=amplitude_scale, duration=rise_len>> 2)
        ## Flat
        channel.play('fp', amplitude_scale=amplitude_scale, duration=pos_flat_len>> 2)
        ## Half Cosine Fall
        channel.play('hcfp', amplitude_scale=amplitude_scale, duration=fall_to_zero_len>> 2)
        ### Negative part
        ## Half Cosine Raise (-1.0*amplitude)
        channel.play('hcrp', amplitude_scale=amplitude_scale*neg, duration=fall_from_zero_len>> 2)
        ## Flat (-1.0*amplitude)
        channel.play('fp', amplitude_scale=amplitude_scale*neg, duration=neg_flat_len>> 2)
        ## Half Cosine Fall (-1.0*amplitude)
        channel.play('hcfp', amplitude_scale=amplitude_scale*neg, duration=return_zero_len>> 2)
    else:
        # ## Half Cosine Raise
        # channel.play('fp', amplitude_scale=amplitude_scale, duration=rise_len>> 2)
        # ## Flat
        # channel.play('fp', amplitude_scale=amplitude_scale, duration=pos_flat_len>> 2)
        # ## Half Cosine Fall
        # channel.play('fp', amplitude_scale=amplitude_scale, duration=fall_to_zero_len>> 2)
        # ### Negative part
        # ## Half Cosine Raise (-1.0*amplitude)
        # channel.play('fp', amplitude_scale=amplitude_scale*neg, duration=fall_from_zero_len>> 2)
        # ## Flat (-1.0*amplitude)
        # channel.play('fp', amplitude_scale=amplitude_scale*neg, duration=neg_flat_len>> 2)
        # ## Half Cosine Fall (-1.0*amplitude)
        # channel.play('fp', amplitude_scale=amplitude_scale*neg, duration=return_zero_len>> 2)
        """ All Const """
        ## Half Cosine Raise
        channel.play('const', amplitude_scale=amplitude_scale, duration=rise_len>> 2)
        ## Flat
        channel.play('const', amplitude_scale=amplitude_scale, duration=pos_flat_len>> 2)
        ## Half Cosine Fall
        channel.play('const', amplitude_scale=amplitude_scale, duration=fall_to_zero_len>> 2)
        ### Negative part
        ## Half Cosine Raise (-1.0*amplitude)
        channel.play('const', amplitude_scale=amplitude_scale*neg, duration=fall_from_zero_len>> 2)
        ## Flat (-1.0*amplitude)
        channel.play('const', amplitude_scale=amplitude_scale*neg, duration=neg_flat_len>> 2)
        ## Half Cosine Fall (-1.0*amplitude)
        channel.play('const', amplitude_scale=amplitude_scale*neg, duration=return_zero_len>> 2)
    
    
    # qbORqp.align()



# %% {Initialize_QuAM_and_QOP}
u = unit(coerce_to_integer=True)
if node.parameters.pulse_tot_len_ns >= 32767:
    raise ValueError("Maximum int breaked ! The maximum int must lower than 32767 !")
machine = QuAM.load()
node.machine = machine
# machine.network["port"] = int(access_port)

# print(f"Machine access port :{access_port}")
if node.parameters.load_data_id is None:
    qmm = machine.connect()

qubits = machine.get_qubits_used_in_node(node.parameters)
num_qubits = len(qubits)


config = machine.generate_config()

# %% {QUA_program}
n_runs = node.parameters.num_runs
flux_point = node.parameters.flux_point_joint_or_independent
reset_type = node.parameters.reset_type_thermal_or_active
operation_name = 'readout'

with program() as iq_blobs:
    reset_global_phase()
    sw_ratio, flat_ratio = declare(fixed, value=node.parameters.switch_pt_ratio), declare(fixed, value=node.parameters.falt_len_ratio)
    tot_len = declare(int, value=node.parameters.pulse_tot_len_ns)
    amp_sca = declare(fixed, value=0.1)
    I_g, I_g_st, Q_g, Q_g_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    

    for multiplexed_qubits in qubits.batch():

        with for_(n, 0, n < n_runs, n + 1):
            save(n, n_st)

            for i, qubit in multiplexed_qubits.items():
                if reset_type == "active":
                    active_reset(qubit)
                elif reset_type == "thermal":
                    qubit.wait(16 * u.ns)
                else:
                    raise ValueError(f"Unrecognized reset type {reset_type}.")

            align(*[q.xy.name for q in multiplexed_qubits.values()] +
                   [q.resonator.name for q in multiplexed_qubits.values()] +
                   [q.z.name for q in multiplexed_qubits.values()])

            for i, qubit in multiplexed_qubits.items():
                split_bipolar_macro(qubit, total_len_ns=tot_len, switch_point_ratio=sw_ratio, flat_ratio=flat_ratio, amplitude_scale=amp_sca, debug=node.parameters.pulse_align_debug)
                qubit.align()
                qubit.xy.play('x180')
                qubit.align()
                qubit.resonator.measure(operation_name, qua_vars=(I_g[i], Q_g[i]))
                qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                save(I_g[i], I_g_st[i])
                save(Q_g[i], Q_g_st[i])


    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_g_st[i].save_all(f"I_g{i + 1}")
            Q_g_st[i].save_all(f"Q_g{i + 1}")
            


# %% {Simulate_or_execute}

# Simulates the QUA program for the specified duration
simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns//4)  # In clock cycles = 4ns
job = qmm.simulate(config, iq_blobs, simulation_config)
# Get the simulated samples and plot them for all controllers

samples = job.get_simulated_samples()
samples.con1.plot()
node.results = {"figure": plt.gcf()}
wf_report = job.get_simulated_waveform_report()
wf_report.create_plot(samples, plot=True, save_path=None)
node.save()




# %%
