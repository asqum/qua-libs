from typing import Literal
from quam_libs.experiments.batched_SI2QRB import run_batched_rb


# # 20 random circuits takes 10 mins
target_operation:Literal['cz', 'idle_2q'] = 'cz'
random_gates_per_depth:int = 750
depth_rank:Literal["short", "standard", "long"] = "short"
job_id:int|None = None #1271
remove_zero_to_analyze:bool = False


fidelities, names = run_batched_rb(depth_rank, target_operation, random_gates_per_depth, job_id, time_mark=True, zero_removal_plot=remove_zero_to_analyze)