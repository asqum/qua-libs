# %%
"""
CONFUSION MATRIX

Prerequisites:
    - the π-pulse calibrated for the target coupler.

Updates:
    - Figures only.

Status:
    - All good
    - Load_data is working
"""


# %% {Imports}
from quam_libs.experiments.coupler_RD_related.ConfusionMatrix import CP_CM_EXP, node

# %% {Node_parameters}
node.parameters.coupler = ['coupler_q6_q7']
node.parameters.shots = 2048*2
node.parameters.reset_type = 'active'
node.parameters.load_data_id = None
node.parameters.simulate = False




# %% {Initialize}
EXP = CP_CM_EXP(node)
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
EXP.state_management()


# %% {Run all above}