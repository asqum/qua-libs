from qualibrate import QualibrationNode, NodeParameters
from typing import Optional, Literal, List
from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, readout_state_coupler, active_reset_coupler
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.experiments.ramsey.plotting import add_fit_text
from quam_libs.experiments.ramsey.analysis.fitting import fit_ramsey_oscillations_with_exponential_decay, extract_relevant_fit_parameters, calculate_fit_results, RamseyFit
from quam_libs.components.macros.coupler_RD_decoding import CouplerReadoutDecoder
from quam_libs.experiments.coupler_RD_related import ExpTemplate
from qm.jobs.running_qm_job import RunningQmJob
from typing import Literal, Optional
u = unit(coerce_to_integer=True)

class Parameters(NodeParameters):
    coupler: list[str] = ['coupler_q7_q8']
    num_averages: int = 500
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 6008
    wait_time_step_in_ns: int = 120
    frequency_detuning_in_mhz: float = 6.0
    reset_type: Literal['active', 'thermal'] = 'active'
    flux_point_joint_or_independent_or_arbitrary: Literal['joint', 'independent'] = 'independent'   
    load_data_id: Optional[int] = None
    simulate: bool = False
    timeout: int = 100
    histo_num:int = 1
    UPDATE_STATE: bool = True

node = QualibrationNode( name="06xa_coupler_Ramsey", parameters=Parameters())

class CP_Ramsey_EXP(ExpTemplate):

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
        # Dephasing time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
        idle_times = np.arange(
            self.node.parameters.min_wait_time_in_ns // 4,
            self.node.parameters.max_wait_time_in_ns // 4,
            self.node.parameters.wait_time_step_in_ns // 4,
        )

        self.variables['time'] = idle_times

        self.detuning = node.parameters.frequency_detuning_in_mhz * 1e6
        self.variables['sign'] = [-1, 1]


    # Composing QUA program
    def qua_composer(self):
        with program() as self.qua_prog:
            _, _, _, _, n, n_st = qua_declaration(num_qubits=self.participant_num)
            t = declare(int)  # QUA variable for the idle time
            
            state = [declare(int) for _ in range(self.participant_num)]
            state_st = [declare_stream() for _ in range(self.participant_num)]
            detuning_sign = declare(int)
            virtual_detuning_phases = [declare(fixed) for _ in range(self.participant_num)]
            
            if not self.node.parameters.simulate:
                self.machine.apply_all_couplers_to_min()
            
            for i, c_name in enumerate(self.CRD.paired_elements):

                # Bring the active qubits to the desired frequency point
                if not self.node.parameters.simulate:
                    for q_type in ["drive_q", "readout_q"]:
                        self.machine.set_all_fluxes(flux_point=self.node.parameters.flux_point_joint_or_independent_or_arbitrary, target=self.CRD.paired_elements[c_name][q_type])
                    
                    wait(400)


                with for_(n, 0, n < self.node.parameters.num_averages, n + 1):
                    save(n, n_st)
                    with for_(*from_array(detuning_sign, self.variables['sign'])):
                        with for_(*from_array(t, self.variables['time'])):
                            self.CRD.get_obj_with_type(c_name,"drive_q").xy.update_frequency(self.CRD.get_obj_with_type(c_name,"drive_q").xy.intermediate_frequency)
                            wait(4)

                            with if_(detuning_sign == 1):
                                assign(virtual_detuning_phases[i], Cast.mul_fixed_by_int(self.detuning * 1e-9, 4 * t))
                            with else_():
                                assign(virtual_detuning_phases[i], Cast.mul_fixed_by_int(-self.detuning * 1e-9, 4 * t))
                            
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
                            
                            
                            self.CRD.get_obj_with_type(c_name,"drive_q").xy.play(f"x90_{c_name}")
                    
                            self.CRD.get_obj_with_type(c_name,"drive_q").wait(t)
                            self.CRD.get_obj_with_type(c_name,"drive_q").xy.frame_rotation_2pi(virtual_detuning_phases[i])

                            self.CRD.get_obj_with_type(c_name,"drive_q").xy.play(f"x90_{c_name}")
                            align()

                            # Measure the state of the resonators
                            readout_state_coupler(self.CRD.paired_elements[c_name]["readout_q"], state[i], flux_applied_target=self.CRD.paired_elements[c_name]["aswap_supplier"], method=self.CRD.paired_elements[c_name]["readout_method"], active_reset_readout_q=False)
                            save(state[i], state_st[i])
                            reset_frame(self.CRD.get_obj_with_type(c_name,"drive_q").xy.name)


            with stream_processing():
                n_st.save("n")
                for i, c_name in enumerate(self.CRD.paired_elements):
                    state_st[i].buffer(len(self.variables['time'])).buffer(len(self.variables['sign'])).average().save(f"state{i + 1}")

    # execute or simulate
    def qua_executor(self):

        if self.node.parameters.simulate:
            # Simulates the QUA program for the specified duration
            simulation_config = SimulationConfig(duration=self.node.parameters.simulation_duration_ns//4)  # In clock cycles = 4ns
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
        
        ds = fetch_results_as_xarray(job.result_handles, self.CRD.get_all_TransmonPairs(), self._sort_variables_(['time','sign']))
        
        return ds
    

    # assign new coordinates
    def dataset_post_proccess(self):
        ''' Once the dataset had been saved into node.results '''

        ds:xr.Dataset = self.node.results["ds"]
        ds = ds.assign_coords(time= 4 * ds.time)
        ds.time.attrs = {"long_name": "idle time", "units": "ns"}
        self.node.results["ds"] = ds


    # analyze data
    def analyze(self):
        
        if not self.node.parameters.simulate:
            if self.node.parameters.load_data_id is not None:
                self.load_data = True
                from quam_libs.experiments.coupler_RD_related import load_data_only
                self.node, self.machine, _, self.CRD = load_data_only(self.node)

            ds:xr.Dataset = self.node.results['ds']


            fit = fit_ramsey_oscillations_with_exponential_decay(ds, True)

            frequency, decay, tau, tau_error = extract_relevant_fit_parameters(fit)

            detuning = int(self.node.parameters.frequency_detuning_in_mhz * 1e6)

            freq_offset, decay, decay_error = calculate_fit_results(
                frequency, tau, tau_error, fit, detuning
            )
            fits = {
                    q.name: RamseyFit(
                        qubit_name = q.name,
                        freq_offset=1e9 * freq_offset.loc[q.name].values,
                        decay=decay.loc[q.name].values,
                        decay_error=decay_error.loc[q.name].values,
                        raw_fit_results=fit.to_dataset(name="fit")
                    )

                    for q in self.CRD.get_all_TransmonPairs()
                }

            self.analyzed_items = {"fits":fits}

    
    # plot
    def visualize(self):
        
        def plot_ramsey_data_with_fit(ax, ds, qubit, fit, c_name):
            """
            Plot individual qubit data on a given axis.

            """
            def oscillation_decay_exp(t, a, f, phi, offset, decay):
                return a * np.exp(-t * decay) * np.cos(2 * np.pi * f * t + phi) + offset

            fitted_ramsey_data = oscillation_decay_exp(
                ds.time,
                fit.raw_fit_results.sel(fit_vals="a"),
                fit.raw_fit_results.sel(fit_vals="f"),
                fit.raw_fit_results.sel(fit_vals="phi"),
                fit.raw_fit_results.sel(fit_vals="offset"),
                fit.raw_fit_results.sel(fit_vals="decay"),
            )

            
            plot_state(ax, ds, qubit, fitted_ramsey_data)
            ax.set_ylabel("State")
            

            ax.set_xlabel("Idle time [ns]")
            ax.set_title(c_name)
            add_fit_text(ax, fit)
            ax.legend()

        def plot_state(ax, ds, qubit, fitted):
            """Plot state data for a qubit."""
            ds.sel(sign=1).loc[qubit].state.plot(
                ax=ax, x="time", c="C0", marker=".", ms=5.0, ls="", label="$\Delta$ = +"
            )
            ds.sel(sign=-1).loc[qubit].state.plot(
                ax=ax, x="time", c="C1", marker=".", ms=5.0, ls="", label="$\Delta$ = -"
            )
            ax.plot(ds.time, fitted.fit.loc[qubit].sel(sign=1), c="C0", ls="-", lw=1)
            ax.plot(ds.time, fitted.fit.loc[qubit].sel(sign=-1), c="C1", ls="-", lw=1)
    
        if not self.node.parameters.simulate:
            ds:xr.Dataset = self.node.results['ds']
            grid_names, qubit_pair_names = grid_pair_names(self.CRD.get_all_TransmonPairs())
            grid = QubitPairGrid(grid_names, qubit_pair_names)
            self.node.results["freq_shift"] = {}
            
            for ax, qubit in grid_iter(grid):
                plot_ramsey_data_with_fit(ax, ds, qubit, self.analyzed_items['fits'][qubit['qubit']], qubit['qubit'])
                self.node.results["freq_shift"][qubit['qubit']] = -1*float(self.analyzed_items['fits'][qubit['qubit']].freq_offset)

            self.node.results["Ramsey"] = grid.fig

            

    
    # Save state and results
    def state_management(self, update_state_also:bool):
        if not self.node.parameters.simulate:
            if update_state_also:
                with self.node.record_state_updates():
                    for c in self.CRD.get_all_TransmonPairs():
                        c.extras["RD"]["IF"] += self.node.results["freq_shift"][c.name]
            
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


