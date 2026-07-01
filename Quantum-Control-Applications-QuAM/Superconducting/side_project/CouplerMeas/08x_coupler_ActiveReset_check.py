# %%
"""
RESET METHOD VALIDATION.
Prepares the coupler at excited then apply different reset methods. Finally readout it's ground state populations.

Prerequisites:
    - the π-pulse calibrated for the target coupler.

Updates:
    - Figures only.

Next step:
    - You may check the population again for the case in aSWAP's slope direction to +1.
Status:
    - All good
    - Load_data is working
"""


# %% {Imports}
from quam_libs.experiments.coupler_RD_related.ActiveResetCheck import CP_ARC_EXP, node

# %% {Node_parameters}
node.parameters.coupler = ['coupler_q6_q7']
node.parameters.shots = 2048*2
node.parameters.prepared_state = 1
node.parameters.load_data_id = None
node.parameters.simulate = False




# %% {Initialize}
EXP = CP_ARC_EXP(node)
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