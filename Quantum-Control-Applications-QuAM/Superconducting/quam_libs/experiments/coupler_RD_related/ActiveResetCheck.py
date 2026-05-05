from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset_coupler, readout_state_coupler, active_reset
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.fit import fit_decay_exp, decay_exp
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import norm
from quam_libs.components.macros.coupler_RD_decoding import CouplerReadoutDecoder
from quam_libs.experiments.coupler_RD_related import ExpTemplate
from qm.jobs.running_qm_job import RunningQmJob
from typing import Literal, Optional

u = unit(coerce_to_integer=True)

class Parameters(NodeParameters):
    coupler: list[str] = ['']
    shots: int = 2048
    prepared_state: Literal[0, 1] = 0
    flux_point_joint_or_independent_or_arbitrary: Literal["joint", "independent"] = "independent"
    simulate: bool = False
    simulation_duration_ns: int = 10000
    timeout: int = 100
    load_data_id: Optional[int] = None
    UPDATE_STATE:bool = True


node = QualibrationNode(name="08x_coupler_ActiveReset_check", parameters=Parameters())



class CP_ARC_EXP(ExpTemplate):

    def __init__(self, node:QualibrationNode):
        super().__init__()
        self.node = node
        self._check_coupler_num_()
        self.machine = QuAM.load()


    # Checking all the required participants
    def participants_stand_by(self):
        self.CRD = CouplerReadoutDecoder(self.machine, self.node.parameters.coupler, coupler_arbi_LO_manual=None, simulate=self.node.parameters.simulate, load_data_id=self.node.parameters.load_data_id)
        self.config = self.machine.generate_config()
        self.participant_num = len(list(self.CRD.paired_elements.keys()))
        if self.node.parameters.load_data_id is None:
            self.qmm = self.machine.connect()

    # Counting variables 
    def exp_variable_arangement(self):
        self.variables["N"] = np.linspace(1, self.node.parameters.shots, self.node.parameters.shots)
        self.variables["instructions"] = ["10*T1\nThermalize", "Active", "aSWAP", "Skip"]

    # Composing QUA program
    def qua_composer(self):
        with program() as self.qua_prog:
            _, _, _, _, n, n_st = qua_declaration(num_qubits=self.participant_num)
            instr_idx = declare(int)  # QUA variable for the idle time
            
            state = [declare(int) for _ in range(self.participant_num)]
            state_st = [declare_stream() for _ in range(self.participant_num)]
            
            if not self.node.parameters.simulate:
                self.machine.apply_all_couplers_to_min()
            
            for i, c_name in enumerate(self.CRD.paired_elements):

                # Bring the active qubits to the desired frequency point
                if not self.node.parameters.simulate:
                    for q_type in ["drive_q", "readout_q"]:
                        self.machine.set_all_fluxes(flux_point=self.node.parameters.flux_point_joint_or_independent_or_arbitrary, target=self.CRD.paired_elements[c_name][q_type])
                    
                    wait(400)
                
                t1_ns = int(self.CRD.get_obj_with_type(c_name,'coupler').extras['T1']*1e9)


                with for_(n, 0, n < self.node.parameters.shots, n + 1):
                    save(n, n_st)
                    with for_(*from_array(instr_idx, np.arange(len(self.variables["instructions"])))):
                        self.CRD.get_obj_with_type(c_name,"drive_q").xy.update_frequency(self.CRD.get_obj_with_type(c_name,"drive_q").xy.intermediate_frequency)
                        wait(4)
                        align()


                        if not self.node.parameters.simulate:
                            wait(10*t1_ns//4)
                            active_reset(self.CRD.get_obj_with_type(c_name,'readout_q'))
                            active_reset(self.CRD.get_obj_with_type(c_name,'drive_q')) 

                        align()
                        self.CRD.get_obj_with_type(c_name,"drive_q").xy.update_frequency(self.CRD.get_obj_with_type(c_name,"coupler").extras["RD"]["IF"])
                        wait(4)
                        align()
                        # prepare state
                        self.CRD.get_obj_with_type(c_name,"drive_q").xy.play(f"x180_{c_name}", amplitude_scale=self.node.parameters.prepared_state)
                        align()
                        
                        with if_(instr_idx<1):
                            wait(10*t1_ns//4)
                        with elif_(instr_idx==1):
                            active_reset_coupler(self.CRD.get_obj_with_type(c_name,"drive_q"), self.CRD.get_obj_with_type(c_name,'readout_q'), f"x180_{c_name}", method='standard')
                        with elif_(instr_idx==2):
                            readout_state_coupler(self.CRD.get_obj_with_type(c_name,'readout_q'), None, flux_applied_target=self.CRD.paired_elements[c_name]["aswap_supplier"], method=self.CRD.paired_elements[c_name]["readout_method"])
                        with else_():
                            pass

                        # Measure the state of the resonators
                        readout_state_coupler(self.CRD.paired_elements[c_name]["readout_q"], state[i], flux_applied_target=self.CRD.paired_elements[c_name]["aswap_supplier"], method=self.CRD.paired_elements[c_name]["readout_method"], active_reset_readout_q=True)
                        save(state[i], state_st[i])


            with stream_processing():
                n_st.save("n")
                for i, c_name in enumerate(self.CRD.paired_elements):
                    state_st[i].buffer(len(self.variables["instructions"])).buffer(len(self.variables['N'])).save(f"state{i + 1}")

    # execute or simulate
    def qua_executor(self):

        if self.node.parameters.simulate:
            # Simulates the QUA program for the specified duration
            simulation_config = SimulationConfig(duration=self.node.parameters.simulation_duration_ns // 4)  # In clock cycles = 4ns
            job = self.qmm.simulate(self.config, self.qua_prog, simulation_config)
            # Get the simulated samples and plot them for all controllers
            samples = job.get_simulated_samples()
            samples.con1.plot()
            self.node.results = {"figure": plt.gcf()}
            wf_report = job.get_simulated_waveform_report()
            wf_report.create_plot(samples, plot=True, save_path=None)
            self.node.save()

            return None

        else:
            with qm_session(self.qmm, self.config, timeout=self.node.parameters.timeout) as self.qm:
                job = self.qm.execute(self.qua_prog)
                results = fetching_tool(job, ["n"], mode="live")
                while results.is_processing():
                    # Fetch results
                    n = results.fetch_all()[0]
                    # Progress bar
                    if self.progress_bar_display:
                        progress_counter(n, self.node.parameters.shots, start_time=results.start_time)
                    
            ds = self.data_catcher(job)
            return ds


    # fetch data
    def data_catcher(self, job:RunningQmJob) -> xr.Dataset:
        
        ds = fetch_results_as_xarray(job.result_handles, self.CRD.get_all_TransmonPairs(), self._sort_variables_(["instructions", "N"]))
        
        return ds
    

    # assign new coordinates
    def dataset_post_proccess(self):
        ''' Once the dataset had been saved into node.results '''
        pass


    # analyze data
    def analyze(self):
        if self.node.parameters.load_data_id is not None:
            self.load_data = True
            from quam_libs.experiments.coupler_RD_related import load_data_only
            self.node, self.machine, _, self.CRD = load_data_only(self.node)


    
    # plot
    def visualize(self):
        if not self.node.parameters.simulate:
            ds:xr.Dataset = self.node.results['ds']
            grid_names, qubit_pair_names = grid_pair_names(self.CRD.get_all_TransmonPairs())
            grid = QubitPairGrid(grid_names, qubit_pair_names)
            for ax, qubit in grid_iter(grid):
                current_state = ds.state.sel(qubit=qubit['qubit'])
                prob_zero = (current_state == 0).astype(float).mean(dim='N')

                bars = ax.bar(
                    prob_zero.instructions.values, 
                    prob_zero.values,
                    width=0.6, 
                    color='steelblue', 
                    edgecolor='black',
                    alpha=0.8
                )

                ax.bar_label(bars, padding=0, fmt='%.3f', fontsize=9, fontweight='bold')
                ax.set_ylim(0,1.2)
                ax.set_xlabel("Method", fontweight='bold')
                ax.set_ylabel(r"$|0\rangle$ population")
                ax.set_title(f"{qubit['qubit']}")
                ax.grid(alpha=0.3)


            plt.suptitle(f"Reset Method comparison\n prepared {'Excite' if self.node.parameters.prepared_state else 'Ground'}")
            plt.tight_layout()
            plt.show()

            
            self.node.results["figure"] = grid.fig
            
    
    # Save state and results
    def state_management(self, update_state_also:bool=False):
        if not self.node.parameters.simulate:
            if not self.load_data:
                for q in self.CRD.get_all_Transmons('drive_q'):
                    q.xy.opx_output.upconverter_frequency = self.CRD.original_lo[q.name] # revert the driving LO
                for c_name in self.CRD.paired_elements:
                    if self.CRD.paired_elements[c_name]["aswap_supplier"] is None:
                        self.CRD.get_obj_with_type(c_name, 'readout_q').z.operations['aSWAP'].slope_direction = -1 
                    else:
                        self.CRD.get_obj_with_type(c_name, 'coupler').coupler.operations['aSWAP'].slope_direction = -1
            self.node.results["initial_parameters"] = self.node.parameters.model_dump()
            self.node.machine = self.machine
            self.node.save()

