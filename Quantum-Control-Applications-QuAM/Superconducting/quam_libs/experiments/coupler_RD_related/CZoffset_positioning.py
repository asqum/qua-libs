from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, readout_state, active_reset
from quam_libs.lib.fit import extract_dominant_frequencies
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
    num_averages: int = 250
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 416
    wait_time_step_in_ns: int = 4
    reset_type: Literal["active", "thermal"] = "active"
    coupler_flux_min : float = 0
    coupler_flux_max : float = 0.3
    coupler_flux_step : float = 0.003
    
    simulate: bool = False
    simulation_duration_ns: int = 10000
    timeout: int = 100
    load_data_id: Optional[int] = None
    debug: bool = False
    histo_num:int = 1
    UPDATE_STATE:bool = True
    operation:Literal['Cz'] = 'Cz'


node = QualibrationNode(name="12x_CZ_coupler_offset_positioning", parameters=Parameters())



class CP_CZposition_EXP(ExpTemplate):

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

        self.variables['idle_time'] = idle_times

        fluxes_coupler = np.arange(node.parameters.coupler_flux_min, node.parameters.coupler_flux_max, node.parameters.coupler_flux_step)
        self.variables['flux_coupler'] = fluxes_coupler

    # Composing QUA program
    def qua_composer(self):
        with program() as self.qua_prog:
            _, _, _, _, n, n_st = qua_declaration(num_qubits=self.participant_num)
            t = declare(int)  # QUA variable for the idle time
            flux_coupler = declare(float)
            comp_flux_qubit = declare(float)
            state_control = [declare(int) for _ in range(self.participant_num)]
            state_target = [declare(int) for _ in range(self.participant_num)]
            state = [declare(int) for _ in range(self.participant_num)]
            state_st_control = [declare_stream() for _ in range(self.participant_num)]
            state_st_target = [declare_stream() for _ in range(self.participant_num)]
            state_st = [declare_stream() for _ in range(self.participant_num)]
            
                
            for i, qp in enumerate(self.CRD.get_all_TransmonPairs()):

                # Bring the active qubits to the desired frequency point
                if not self.node.parameters.simulate:
                    self.machine.set_all_fluxes("joint", qp)
                    wait(400)


                with for_(n, 0, n < self.node.parameters.num_averages, n + 1):
                    save(n, n_st)
                    with for_(*from_array(flux_coupler, self.variables['flux_coupler'])):
                        with for_(*from_array(t, self.variables['idle_time'])):
                            

                            if self.node.parameters.reset_type == "active":
                                active_reset(qp.qubit_control)
                                active_reset(qp.qubit_target)
                                
                            else:
                                wait(qp.qubit_control.thermalization_time * u.ns)
                                wait(qp.qubit_target.thermalization_time * u.ns)
                            qp.align()

                            if "coupler_qubit_crosstalk" in qp.extras:
                                assign(comp_flux_qubit, qp.detuning  +  qp.extras["coupler_qubit_crosstalk"] * flux_coupler )
                            else:
                                assign(comp_flux_qubit, qp.detuning)
                            qp.align()
                            
                            # setting both qubits ot the initial state
                            qp.qubit_control.xy.play("x180")
                            qp.qubit_target.xy.play("x180")   

                            qp.align()         
                            
                            # Play the flux pulse on the qubit control and coupler
                            qp.qubit_control.z.play("const", amplitude_scale = comp_flux_qubit / qp.qubit_control.z.operations["const"].amplitude, duration = t)
                            qp.coupler.play("const", amplitude_scale = flux_coupler / qp.coupler.operations["const"].amplitude, duration = t)
                            
                            qp.align()
                            # readout
                            
                            readout_state(qp.qubit_control, state_control[i])
                            readout_state(qp.qubit_target, state_target[i])
                            assign(state[i], state_control[i]*2 + state_target[i])
                            save(state_control[i], state_st_control[i])
                            save(state_target[i], state_st_target[i])
                            save(state[i], state_st[i])

                        
            with stream_processing():
                n_st.save("n")
                for i, c_name in enumerate(self.CRD.get_all_TransmonPairs()):
                    state_st_control[i].buffer(len(self.variables['idle_time'])).buffer(len(self.variables['flux_coupler'])).average().save(f"state_control{i + 1}")
                    state_st_target[i].buffer(len(self.variables['idle_time'])).buffer(len(self.variables['flux_coupler'])).average().save(f"state_target{i + 1}")
                    state_st[i].buffer(len(self.variables['idle_time'])).buffer(len(self.variables['flux_coupler'])).average().save(f"state{i + 1}")
                

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
        
        ds = fetch_results_as_xarray(job.result_handles, self.CRD.get_all_TransmonPairs(), self._sort_variables_(['idle_time', 'flux_coupler']))
        
        return ds
    

    # assign new coordinates
    def dataset_post_proccess(self):
        ''' Once the dataset had been saved into node.results '''

        ds:xr.Dataset = self.node.results["ds"]
        ds = ds.assign_coords(idle_time=4 * ds.idle_time / u.us)
        ds = ds.assign({"res_sum" : ds.state_control - ds.state_target})
        flux_coupler_full = np.array([self.variables['flux_coupler'] + qp.coupler.decouple_offset for qp in self.CRD.get_all_TransmonPairs()])
        ds = ds.assign_coords({"flux_coupler_full": (["qubit", "flux_coupler"], flux_coupler_full)})    
        ds['dominant_frequency'] = extract_dominant_frequencies(ds.res_sum)
        ds.dominant_frequency.attrs['units'] = 'GHz'
        
        self.node.results["ds"] = ds


    # analyze data
    def analyze(self):
        if not self.node.parameters.simulate:
            if self.node.parameters.load_data_id is not None:
                self.load_data = True
                from quam_libs.experiments.coupler_RD_related import load_data_only
                self.node, self.machine, _, self.CRD = load_data_only(self.node)

            ds:xr.Dataset = self.node.results['ds']


            
            # Plot the dominant frequencies
            # Find the values of flux_coupler_full for which the dominant frequencies are max and min
            for qp in self.CRD.get_all_TransmonPairs():

                target_coupling_strength_MHz = round(1e3/(qp.gates[self.node.parameters.operation].coupler_flux_pulse.length*u.ns),1)
                # interaction_max = (ds.dominant_frequency * (ds.dominant_frequency<target_coupling_strength_MHz*1e-3)).max(dim='flux_coupler')
                coupler_flux_pulse = ds.flux_coupler.isel(flux_coupler=(ds.dominant_frequency * (ds.dominant_frequency<=target_coupling_strength_MHz*1e-3)).argmax(dim='flux_coupler'))
                # coupler_flux_min = ds.flux_coupler_full.isel(flux_coupler=ds.dominant_frequency.argmin(dim='flux_coupler'))


                self.analyzed_items[qp.name] = {"target_coupling_MHz":target_coupling_strength_MHz, "coupler_flux_pulse":coupler_flux_pulse}

    
    # plot
    def visualize(self):
        ds:xr.Dataset = self.node.results['ds']
        grid_names, qubit_pair_names = grid_pair_names(self.CRD.get_all_TransmonPairs())
        
        grid = QubitPairGrid(grid_names, qubit_pair_names)    
        for ax, qp in grid_iter(grid):     
            
            values_to_plot = ds.state_control.sel(qubit=qp['qubit'])
            
            values_to_plot.plot(ax = ax, cmap = 'viridis', y = 'idle_time', x = 'flux_coupler')
            
            ax.set_title(f"{qp['qubit']}, coupler set point: {self.CRD.get_obj_with_type(qp['qubit'], 'coupler').coupler.decouple_offset}", fontsize = 10)
        
        grid.fig.suptitle('Control State')
        plt.tight_layout()
        plt.show()
        self.node.results['figure_Control_state'] = grid.fig
        
        grid = QubitPairGrid(grid_names, qubit_pair_names)    
        for ax, qp in grid_iter(grid):
            
            values_to_plot = ds.state_target.sel(qubit=qp['qubit'])
           
            values_to_plot.plot(ax = ax, cmap = 'viridis', y = 'idle_time', x = 'flux_coupler')
            
            ax.set_title(f"{qp['qubit']}, coupler set point: {self.CRD.get_obj_with_type(qp['qubit'], 'coupler').coupler.decouple_offset}", fontsize = 10)
        grid.fig.suptitle('Target State')
        plt.tight_layout()
        plt.show()
        self.node.results['figure_Target_target'] = grid.fig
        
        grid = QubitPairGrid(grid_names, qubit_pair_names)    
        for ax, qp in grid_iter(grid):
            (1e3*ds.dominant_frequency.sel(qubit=qp['qubit'])).plot(ax = ax, marker = '.', ls = 'None', x = 'flux_coupler')
            
            # ax.axvline(x = qubit_pair.coupler.decouple_offset, color = 'black')
            # ax.axvline(x = coupler_flux_pulse.sel(qubit=qp['qubit']), color = 'red', lw = 0.5, ls = '--', )
            ax.scatter(float(self.analyzed_items[qp['qubit']]["coupler_flux_pulse"].sel(qubit=qp["qubit"]).values), self.analyzed_items[qp['qubit']]["target_coupling_MHz"], marker="X", s=100, color='red')
            ax.set_title(f"{qp['qubit']}, coupler set point: {self.CRD.get_obj_with_type(qp['qubit'], 'coupler').coupler.decouple_offset}", fontsize = 10)
            ax.set_xlabel('Flux Coupler')
            ax.set_ylabel('Frequency (MHz)')
            # ax.set_ylim(0, 20)
            # ax.set_xlim(0, 0.35)
        grid.fig.suptitle(f'g={self.analyzed_items[qp['qubit']]["target_coupling_MHz"]} MHz at flux pulse amplitude = {round(float(self.analyzed_items[qp['qubit']]["coupler_flux_pulse"].sel(qubit=qp["qubit"]).values),3)} V')
        plt.tight_layout()
        plt.show()
        self.node.results['figure_dominant_frequency'] = grid.fig


    
    # Save state and results
    def state_management(self, update_state_also:bool):
        if not self.node.parameters.simulate:
            if update_state_also:
                with self.node.record_state_updates():
                    for qp in self.CRD.get_all_TransmonPairs():
                        if "CZg_bias_conversion" not in qp.extras:
                            qp.extras["CZgMHz_bias_conversion"] = {}
                        qp.extras["CZgMHz_bias_conversion"][str(self.analyzed_items[qp.name]["target_coupling_MHz"])] = float(self.analyzed_items[qp.name]["coupler_flux_pulse"].sel(qubit=qp.name).values)
            
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

