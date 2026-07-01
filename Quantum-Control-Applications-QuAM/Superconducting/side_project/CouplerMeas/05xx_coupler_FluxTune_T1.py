"""
COUPLER T1 MEASUREMENT

Prerequisites:
    - Coupler's π pulse calibrated.
    - 'bias_to_sweet' and 'neighboring_qubit_detune_flux_amp' for the coupler are both known and also recorded in coupler.extras['Fx']['bias_to_sweet']. This action requires you run the node '03x_coupler_FluxSpectroscopy' first and update the state.
Status:
    - Tracking some strange results.
    - load_data is working.    
"""

# %% {Imports}
from quam_libs.experiments.coupler_RD_related.T1_Biased import CP_zT1_EXP, node

# %% {Node_parameters}
node.parameters.coupler = ['coupler_q6_q7']
node.parameters.num_averages = 300
node.parameters.coupler_flux_amp = None
''' If coupler_flux_amp is None, use the value saved in extras["Fx"] '''
node.parameters.flux_point_joint_or_independent_or_arbitrary = 'independent'
node.parameters.min_wait_time_in_ns = 16
node.parameters.max_wait_time_in_ns = 180016
node.parameters.wait_time_step_in_ns = 1800
node.parameters.reset_type = 'active'
node.parameters.load_data_id = None
node.parameters.simulate = False
node.parameters.histo_num = 1



# %% {Initialize}
EXP = CP_zT1_EXP(node)
if node.parameters.load_data_id is None:
    EXP.easy_preparation()


# %% {Run and Fetch}
if node.parameters.load_data_id is None:
    EXP.iteratively_run(node.parameters.histo_num)
    EXP.dataset_post_proccess()


# %% {Data analyze}
EXP.analyze()


# %% {Visualization}
EXP.visualize()


# %% {Update state}
EXP.state_management(node.parameters.UPDATE_STATE)


# %% {Run all above}