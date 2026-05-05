# %%
"""
The Power Rabi for the target coupler.

Prerequisites:
    - the driving frequency for the target coupler.

Updates:
    - the pi pulse amplitude for this coupler.

Next step:
    - You may check the population again for the case in aSWAP's slope direction to +1.

Status:
    - All good
    - Load_data is working

"""


# %% {Imports}
from quam_libs.experiments.coupler_RD_related.power_Rabi import CP_PR_EXP, node

# %% {Node_parameters}
node.parameters.coupler = ['coupler_q6_q7']
node.parameters.num_averages = 500
node.parameters.operation_x180_or_any_90 = 'x180'
node.parameters.min_amp_factor = 0.0
node.parameters.max_amp_factor = 1.99
node.parameters.amp_factor_step = 0.02
node.parameters.update_x90 = True
node.parameters.flux_point_joint_or_independent = 'independent'
node.parameters.max_number_rabi_pulses_per_sweep = 1

node.parameters.reset_type = 'active'
node.parameters.load_data_id = None
node.parameters.simulate = False



# %% {Initialize}
EXP = CP_PR_EXP(node)
if node.parameters.load_data_id is None:
    EXP.easy_preparation()


# %% {Run and Fetch}
if node.parameters.load_data_id is None:
    EXP.iteratively_run(1)
    EXP.dataset_post_proccess()


# %% {Data analyze}
EXP.analyze()


# %% {Visualization}
EXP.visualize()


# %% {Update state}
node.parameters.UPDATE_STATE = False
EXP.state_management(node.parameters.UPDATE_STATE)


# %% {Run all above}
