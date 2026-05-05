from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset_coupler, readout_state_coupler
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
from time import time
import xarray as xr
from scipy.stats import norm
from quam_libs.components.macros.coupler_RD_decoding import CouplerReadoutDecoder
from quam_libs.experiments.coupler_RD_related import ExpTemplate
from qm.jobs.running_qm_job import RunningQmJob
from typing import Literal, Optional



u = unit(coerce_to_integer=True)

class Parameters(NodeParameters):
    coupler: list[str] = ['']
    coupler_flux_amp:float = None
    num_averages: int = 250
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 200016
    wait_time_step_in_ns: int = 2000
    reset_type: Literal["active", "thermal"] = "active"
    flux_point_joint_or_independent_or_arbitrary: Literal["joint", "independent"] = "independent"
    simulate: bool = False
    simulation_duration_ns: int = 10000
    timeout: int = 100
    load_data_id: Optional[int] = None
    histo_num:int = 1
    UPDATE_STATE:bool = True


node = QualibrationNode(name="05xx_biased_coupler_T1", parameters=Parameters())



class CP_zT1_EXP(ExpTemplate):

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
        self.detuned_qs = [self.machine.qubits[q] for q in node.parameters.coupler.split("_")[1:]]

    # Counting variables 
    def exp_variable_arangement(self):
        # Dephasing time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
        idle_times = np.arange(
            self.node.parameters.min_wait_time_in_ns // 4,
            self.node.parameters.max_wait_time_in_ns // 4,
            self.node.parameters.wait_time_step_in_ns // 4,
        )

        self.variables['idle_time'] = idle_times


        if self.node.parameters.coupler_flux_amp is None:
            if 'neighboring_qubit_detune_flux_amp' in self.CRD.get_all_TransmonPairs()[0].extras['Fx'] and 'bias_to_sweet' in self.CRD.get_all_TransmonPairs()[0].extras['Fx']:
                self.node.parameters.coupler_flux_amp = self.CRD.get_all_TransmonPairs()[0].extras["Fx"]['bias_to_sweet']
                self.neighboring_qubits_detune_flux_amp = self.CRD.get_all_TransmonPairs()[0].extras["Fx"]['neighboring_qubit_detune_flux_amp']
            else:
                print("Required info in coupler.extras['Fx'] not found, using 0 for both coupler tuning flux and neighboring qubits detuning flux.")
                self.node.parameters.coupler_flux_amp = 0.0
                self.neighboring_qubits_detune_flux_amp = 0.0
        # 
        else:
            self.neighboring_qubits_detune_flux_amp = 0.0
        

        

    # Composing QUA program
    def qua_composer(self):
        with program() as self.qua_prog:
            _, _, _, _, n, n_st = qua_declaration(num_qubits=self.participant_num)
            t = declare(int)  # QUA variable for the idle time
            
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


                with for_(n, 0, n < self.node.parameters.num_averages, n + 1):
                    save(n, n_st)
                    with for_(*from_array(t, self.variables['idle_time'])):
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
                        
                        self.CRD.get_obj_with_type(c_name,"drive_q").xy.play(f"x180_{c_name}")
                        align()
                        self.CRD.get_obj_with_type(c_name, 'coupler').coupler.play("const", amplitude_scale=self.node.parameters.coupler_flux_amp/self.CRD.get_obj_with_type(c_name, 'coupler').coupler.operations["const"].amplitude, duration=t)
                        for q in self.detuned_qs:
                            q.z.play("const", amplitude_scale=self.neighboring_qubits_detune_flux_amp / q.z.operations["const"].amplitude, duration=t)
                        
                        align()
                        wait(100)

                        # Measure the state of the resonators
                        readout_state_coupler(self.CRD.paired_elements[c_name]["readout_q"], state[i], flux_applied_target=self.CRD.paired_elements[c_name]["aswap_supplier"], method=self.CRD.paired_elements[c_name]["readout_method"], active_reset_readout_q=False)
                        save(state[i], state_st[i])


            with stream_processing():
                n_st.save("n")
                for i, c_name in enumerate(self.CRD.paired_elements):
                    state_st[i].buffer(len(self.variables['idle_time'])).average().save(f"state{i + 1}")

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
        
        ds = fetch_results_as_xarray(job.result_handles, self.CRD.get_all_TransmonPairs(), self.variables)
        
        return ds
    

    # assign new coordinates
    def dataset_post_proccess(self):
        ''' Once the dataset had been saved into node.results '''

        ds:xr.Dataset = self.node.results["ds"]
        ds = ds.assign_coords(idle_time=4 * ds.idle_time / u.us)
        ds.idle_time.attrs = {"long_name": "idle time", "units": "µs"}
        self.node.results["ds"] = ds


    # analyze data
    def analyze(self):
        if not self.node.parameters.simulate:
            if self.node.parameters.load_data_id is not None:
                self.load_data = True
                from quam_libs.experiments.coupler_RD_related import load_data_only
                self.node, self.machine, _, self.CRD = load_data_only(self.node)

            ds:xr.Dataset = self.node.results['ds']


            qbs = ds.qubit.values
            

            
            t1_collection = {q: [] for q in qbs}
            fit_collection = {q: [] for q in qbs}

            for q_name in qbs:
                try:
                    iterations = ds.iteration.values
                    for iter_val in iterations:
                        
                        ds_sub = ds.sel(qubit=q_name, iteration=iter_val)
                        
                        fit_data = fit_decay_exp(ds_sub.state, "idle_time")

                        fit_collection[q_name] = fit_data
                        
                        
                        decay_val = fit_data.sel(fit_vals="decay").values
                        tau_val = -1 / decay_val
                        t1_collection[q_name].append(tau_val)
                except:
                    print("No iteration found !")
                    ds_sub = ds.sel(qubit=q_name)
                        
                    fit_data = fit_decay_exp(ds_sub.state, "idle_time")

                    fit_collection[q_name] = fit_data
                    
                    
                    decay_val = fit_data.sel(fit_vals="decay").values
                    tau_val = -1 / decay_val
                    t1_collection[q_name].append(tau_val)

            self.analyzed_items = {"t1s":t1_collection, "fits":fit_collection}

    
    # plot
    def visualize(self, histogram_outlier_percentage:float=1):
        '''
        - histogram_outlier_percentage: A float in the range [0, 100).
        '''
        if not self.node.parameters.simulate:
            ds:xr.Dataset = self.node.results['ds']
            grid_names, qubit_pair_names = grid_pair_names(self.CRD.get_all_TransmonPairs())
            grid = QubitPairGrid(grid_names, qubit_pair_names)
            mu_collection, sig_collection = {}, {}
            
            coupler = self.CRD.get_all_TransmonPairs()
            for ax, qubit in grid_iter(grid):
                
                if self.node.parameters.histo_num > 1:
                    data = np.array(self.analyzed_items['t1s'][qubit['qubit']])
                    if histogram_outlier_percentage > 0 and histogram_outlier_percentage < 100:
                        lower_bound = np.percentile(data, histogram_outlier_percentage)   # 下界
                        upper_bound = np.percentile(data, 100-histogram_outlier_percentage)  # 上界
                        data = data[(data >= lower_bound) & (data <= upper_bound)]
                    tot_c = len(data)
                    counts, bins, _ = ax.hist(data, bins=15, alpha=0.7, color='skyblue', edgecolor='white', label='Counts')
                    ### Normal distribution
                    mu, sigma = norm.fit(data)
                    bin_width = bins[1] - bins[0]
                    scaling_factor = len(data) * bin_width
                    x = np.linspace(min(data), max(data), 100)
                    p = norm.pdf(x, mu, sigma) * scaling_factor  
                    mu_collection[qubit['qubit']], sig_collection[qubit['qubit']] = mu, sigma
                
                    ### Plot
                    ax.plot(x, p, 'r-', lw=2, label='Normal Fit')
                    ax.set_title(f"{qubit['qubit']}\n Z-pulse amp = {round(self.node.parameters.coupler_flux_amp,3)}V\n#={tot_c}")
                    ax.set_xlabel("T1 (µs)")
                    ax.set_ylabel("Counts")
                    ax.grid(axis='y', alpha=0.3)
                    
                    stats_text = (
                        f"$T_{1} = {mu:.1f} \pm {sigma:.2f}$ µs"
                    )
                    ax.text(
                        0.05, 0.95, stats_text,
                        transform=ax.transAxes,
                        fontsize=11,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
                    )
                    
                else:
                    fitted = decay_exp(
                        ds.idle_time,
                        self.analyzed_items['fits'][qubit['qubit']].sel(fit_vals="a"),
                        self.analyzed_items['fits'][qubit['qubit']].sel(fit_vals="offset"),
                        self.analyzed_items['fits'][qubit['qubit']].sel(fit_vals="decay"),
                    )
                    decay = self.analyzed_items['fits'][qubit['qubit']].sel(fit_vals="decay")
                    decay_res = self.analyzed_items['fits'][qubit['qubit']].sel(fit_vals="decay_decay")
                    tau = -1 / self.analyzed_items['fits'][qubit['qubit']].sel(fit_vals="decay")
                    tau_error = -tau * (np.sqrt(decay_res) / decay)
                    
                    ds.sel(qubit=qubit["qubit"]).state.plot(ax=ax, marker='o', linestyle='', ms=3, alpha=0.65)
                    ax.set_ylabel("State")
                    
                    ax.plot(ds.idle_time, fitted, "r--")
                    ax.set_title(f'{qubit["qubit"]}\n Z-pulse amp = {round(self.node.parameters.coupler_flux_amp,3)}V')
                    ax.set_xlabel("Idle_time (uS)")
                    ax.text(
                        0.1,
                        0.9,
                        f'T1 = {tau.values:.1f} ± {tau_error.values:.1f} µs',
                        transform=ax.transAxes,
                        fontsize=10,
                        verticalalignment="top",
                        bbox=dict(facecolor="white", alpha=0.5),
                    )
                    mu_collection[qubit['qubit']], sig_collection[qubit['qubit']] = float(tau.values), float(tau_error.values)

            plt.suptitle(f"Coupler T1 with {self.node.parameters.reset_type} reset")
            plt.tight_layout()
            plt.show()

            self.node.results["t1_stats"] = {q: {"mu_us": mu_collection[q], "sigma_us": sig_collection[q]} for q in list(self.analyzed_items['t1s'].keys())}
            self.node.results["figure_histogram"] = grid.fig
            self.analyzed_items['mu_collection'] = mu_collection
            self.analyzed_items['sig_collection'] = sig_collection

    
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


