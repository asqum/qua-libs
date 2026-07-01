# %%
"""
RABI CHEVRON flux version
Use the same pi-pulse condition with the flux varied for the coupler, optimize the flux offset this current driving frequency. Noticed, the loger x180_cp_duration is, the sharper a signal will be shown.

Prerequisites:
    - x180_cp calibrated
    - node 67bxx JAZZ statistics checked.

Recommended node parameters
    - x180cp_dura_scaling = [3, 9] # use may need to change it and see.
    - flux_span_V: float = 0.025

Next steps before going to the next node:
    - Update coupler's decouple offset in the state.
    - Save the current state by calling machine.save("quam")

Status:
    - All good
    - Load_data is working

"""

# %% {Imports}
from quam_libs.experiments.coupler_RD_related.FluxRabiChevron import CP_FRC_EXP, node

# %% {Node_parameters}
node.parameters.coupler = ['coupler_q4_q5']
node.parameters.num_averages = 200
node.parameters.x180cp_dura_scaling = [15,9]
node.parameters.flux_span_V = 0.04 # 0.05
node.parameters.flux_pts = 100

node.parameters.flux_point_joint_or_independent = 'independent'
node.parameters.reset_type = 'active'
node.parameters.load_data_id = None
node.parameters.simulate = False



# %% {Initialize}
EXP = CP_FRC_EXP(node)
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

