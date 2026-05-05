from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset_coupler, readout_state_coupler
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.fit import peaks_dips
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from quam_libs.components.macros.coupler_RD_decoding import CouplerReadoutDecoder
from quam_libs.experiments.coupler_RD_related import ExpTemplate
from qm.jobs.running_qm_job import RunningQmJob
from typing import Literal, Optional, List


u = unit(coerce_to_integer=True)

class Parameters(NodeParameters):

    coupler: list[str] = ['coupler_q6_q7']
    num_averages: int = 500
    flux_span_V: float = 0.01 # 0.05
    flux_pts: int = 100
    x180cp_dura_scaling:List[int] = [9, 15] # [3, 9]
    reset_type: Literal['active', 'thermal'] = 'active'
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    simulate: bool = False
    timeout: int = 100
    load_data_id:Optional[int] = None
    UPDATE_STATE:bool = True


node = QualibrationNode(name="03xx_coupler_FluxRabiChevron", parameters=Parameters())




class CP_FRC_EXP(ExpTemplate):

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
        self.z_rising_buffer_time_ns = 200
        dcs = np.linspace(-self.node.parameters.flux_span_V / 2, +self.node.parameters.flux_span_V / 2, self.node.parameters.flux_pts, dtype=float)
        self.current_x180cp_length_qua = {}
        pi_dura_scal = self.node.parameters.x180cp_dura_scaling
        
        for c_name in self.CRD.paired_elements:
            # Check if the qubit has the required operations
            if hasattr(self.CRD.get_obj_with_type(c_name,"drive_q").xy.operations, f"x180_{c_name}"):
                self.current_x180cp_length_qua[c_name] = int(self.CRD.get_obj_with_type(c_name,"drive_q").xy.operations[f"x180_{c_name}"].length)//4
            else:
                raise ValueError(f"x180_{c_name} hadn't been calibrated for {c_name}! ")

        self.variables['flux'] = dcs
        self.variables['dura_scales'] = pi_dura_scal
        

    # Composing QUA program
    def qua_composer(self):
        with program() as self.qua_prog:
            _, _, _, _, n, n_st = qua_declaration(num_qubits=self.participant_num)
            dc = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
            state = [declare(int) for _ in range(self.participant_num)]
            state_st = [declare_stream() for _ in range(self.participant_num)]
            dura_scal = declare(int)
            
            if not self.node.parameters.simulate:
                self.machine.apply_all_couplers_to_min()
            
            for i, c_name in enumerate(self.CRD.paired_elements):

                # Bring the active qubits to the desired frequency point
                if not self.node.parameters.simulate:
                    for q_type in ["drive_q", "readout_q"]:
                        self.machine.set_all_fluxes(flux_point=self.node.parameters.flux_point_joint_or_independent, target=self.CRD.paired_elements[c_name][q_type])
                    
                    wait(400)


                with for_(n, 0, n < self.node.parameters.num_averages, n + 1):
                    save(n, n_st)
                    with for_(*from_array(dc, self.variables["flux"])):
                        with for_each_(dura_scal, self.variables['dura_scales']):
                            self.CRD.get_obj_with_type(c_name,"drive_q").xy.update_frequency(self.CRD.get_obj_with_type(c_name,"drive_q").xy.intermediate_frequency)
                            wait(4)
                            align()

                            if self.node.parameters.reset_type == "active":
                                active_reset_coupler(self.CRD.paired_elements[c_name]["drive_q"], self.CRD.paired_elements[c_name]["readout_q"], f"x180_{c_name}", flux_applied_target=self.CRD.paired_elements[c_name]["aswap_supplier"], method='aswap')
                                
                            else:
                                
                                if not self.node.parameters.simulate:
                                    if max(self.CRD.get_obj_with_type(c_name,"drive_q").thermalization_time//5, self.CRD.get_obj_with_type(c_name,"readout_q").thermalization_time//5) < self.CRD.get_obj_with_type(c_name,"coupler").extras['T1']*1e9:
                                        wait(10*int(self.CRD.get_obj_with_type(c_name,"coupler").extras['T1']*1e9)//4)
                                    else:
                                        wait(max(self.CRD.get_obj_with_type(c_name,"drive_q").thermalization_time, self.CRD.get_obj_with_type(c_name,"readout_q").thermalization_time)//4)

                            align()
                            self.CRD.get_obj_with_type(c_name,"drive_q").xy.update_frequency(self.CRD.get_obj_with_type(c_name,"coupler").extras["RD"]["IF"])
                            wait(4)
                            align()

                            self.CRD.get_obj_with_type(c_name,'coupler').coupler.play("const", amplitude_scale=dc/ self.CRD.get_obj_with_type(c_name,'coupler').coupler.operations['const'].amplitude, duration=self.current_x180cp_length_qua[c_name]*dura_scal+self.z_rising_buffer_time_ns//4)
                            self.CRD.get_obj_with_type(c_name,'drive_q').xy.wait(self.z_rising_buffer_time_ns//4)
                            self.CRD.get_obj_with_type(c_name,'drive_q').xy.play(f"x180_{c_name}", amplitude_scale=1/dura_scal, duration=self.current_x180cp_length_qua[c_name]*dura_scal)
                            align()

                            # Measure the state of the resonators
                            readout_state_coupler(self.CRD.paired_elements[c_name]["readout_q"], state[i], flux_applied_target=self.CRD.paired_elements[c_name]["aswap_supplier"], method=self.CRD.paired_elements[c_name]["readout_method"], active_reset_readout_q=False)
                            save(state[i], state_st[i])


            with stream_processing():
                n_st.save("n")
                for i, c_name in enumerate(self.CRD.paired_elements):
                    state_st[i].buffer(len(self.variables['dura_scales'])).buffer(len(self.variables["flux"])).average().save(f"state{i + 1}")

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
                        progress_counter(n, self.node.parameters.num_averages, start_time=results.start_time)
                    
            ds = self.data_catcher(job)
            return ds


    # fetch data
    def data_catcher(self, job:RunningQmJob) -> xr.Dataset:
        
        ds = fetch_results_as_xarray(job.result_handles, self.CRD.get_all_TransmonPairs(), self._sort_variables_(["dura_scales", "flux"]))
        
        return ds
    

    # assign new coordinates
    def dataset_post_proccess(self):
        ''' Once the dataset had been saved into node.results '''
        pass


    # analyze data
    def analyze(self, peak_prominence_factor:float=5):
        """
        - peak_prominence_factor: the arg used in `quam_libs.lib.fit.peaks_dips()`
        """
        if not self.node.parameters.simulate:
            if self.node.parameters.load_data_id is not None:
                self.load_data = True
                from quam_libs.experiments.coupler_RD_related import load_data_only
                self.node, self.machine, _, self.CRD = load_data_only(self.node)
                
            ds:xr.Dataset = self.node.results['ds']

            result = peaks_dips(
                ds.state, dim="flux", prominence_factor=peak_prominence_factor, remove_baseline=False
            )
            
            
            fit_results = {}
            for c_name in self.CRD.paired_elements:
                fit_results[c_name] = {}
            
                if not np.isnan(result.sel(qubit=c_name).position.values.all()):
                    fit_results[c_name]["fit_successful"] = True
                    fit_results[c_name]["correct_additional_flux"] = np.average(result.sel(qubit=c_name).position.values)
                else:
                    fit_results[c_name]["fit_successful"] = False
                    print(f"Failed to find a peak for {c_name}\n")

            self.node.results["fit_results"] = fit_results
            self.analyzed_items["fits"] = result

    
    # plot
    def visualize(self):
        
        if not self.node.parameters.simulate:
            ds:xr.Dataset = self.node.results['ds']
            grid_names, qubit_pair_names = grid_pair_names(self.CRD.get_all_TransmonPairs())
            grid = QubitPairGrid(grid_names, qubit_pair_names)
            for ax, qubit in grid_iter(grid):
                ds.sel(qubit=qubit['qubit']).state.plot(
                x="flux", 
                hue="dura_scales",
                ax=ax
                )
                ax.axvline(
                    np.average(self.analyzed_items["fits"].sel(qubit=qubit["qubit"]).position.values),
                    color="r",
                    linestyle="--",
                )
                ax.grid()
                ax.set_xlabel("Additional Flux (V)")
                
            plt.suptitle("Flux Rabi Chevron")
            plt.tight_layout()
            plt.show()
            self.node.results["figure"] = grid.fig


    
    # Save state and results
    def state_management(self, update_state_also:bool):
        if not self.node.parameters.simulate:
            if update_state_also:
                with self.node.record_state_updates():
                    for c in self.CRD.get_all_TransmonPairs():
                        if self.node.results["fit_results"][c.name]["fit_successful"]:
                            c.coupler.decouple_offset += self.node.results["fit_results"][c.name]["correct_additional_flux"]
            
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

