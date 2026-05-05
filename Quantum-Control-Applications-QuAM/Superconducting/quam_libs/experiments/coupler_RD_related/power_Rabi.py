from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset_coupler, readout_state_coupler
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
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
from quam_libs.lib.fit import fit_oscillation, oscillation

u = unit(coerce_to_integer=True)

class Parameters(NodeParameters):
    coupler: list[str] = ['coupler_q6_q7']
    num_averages: int = 200 #10
    operation_x180_or_any_90: Literal["x180", "x90"] = "x180"
    update_x90:bool = True
    min_amp_factor: float = 0.8
    max_amp_factor: float = 1.2
    amp_factor_step: float = 0.004
    max_number_rabi_pulses_per_sweep: int = 44
    reset_type: Literal['active', 'thermal'] = 'active'
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    simulate: bool = False
    timeout: int = 100
    load_data_id:int|None = None
    UPDATE_STATE: bool = True


node = QualibrationNode(name="04x_coupler_PowerRabi", parameters=Parameters())



class CP_PR_EXP(ExpTemplate):

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
        
        self.operation = self.node.parameters.operation_x180_or_any_90
        
        amps = np.arange(
            self.node.parameters.min_amp_factor,
            self.node.parameters.max_amp_factor,
            self.node.parameters.amp_factor_step,
        )
        self.variables['amp'] = amps
        
        if self.node.parameters.max_number_rabi_pulses_per_sweep > 1:
            self.N_pi = (self.node.parameters.max_number_rabi_pulses_per_sweep)
            if self.operation == "x180":
                N_pi_vec = np.arange(1, self.N_pi, 2).astype("int")
            elif self.operation in ["x90"]:
                N_pi_vec = np.arange(2, self.N_pi, 4).astype("int")
            else:
                raise ValueError(f"Unrecognized operation {self.operation}.")
        else:
            self.N_pi = self.node.parameters.max_number_rabi_pulses_per_sweep
            N_pi_vec = np.linspace(1, self.N_pi, self.N_pi).astype("int")[::2]
        self.variables['N'] = N_pi_vec
        print(self.variables['N'])

    # Composing QUA program
    def qua_composer(self):
        with program() as self.qua_prog:
            _, _, _, _, n, n_st = qua_declaration(num_qubits=self.participant_num)
            a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
            npi = declare(int)
            state = [declare(int) for _ in range(self.participant_num)]
            state_st = [declare_stream() for _ in range(self.participant_num)]
            count = declare(int)
            
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
                    with for_(*from_array(npi, self.variables["N"])):
                        with for_(*from_array(a, self.variables["amp"])):
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

                            with for_(count, 0, count < npi, count + 1):
                                self.CRD.get_obj_with_type(c_name,"drive_q").xy.play(f"{self.operation}_{c_name}", amplitude_scale=a)
                            align() 

                            # Measure the state of the resonators
                            readout_state_coupler(self.CRD.paired_elements[c_name]["readout_q"], state[i], flux_applied_target=self.CRD.paired_elements[c_name]["aswap_supplier"], method=self.CRD.paired_elements[c_name]["readout_method"], active_reset_readout_q=False)
                            save(state[i], state_st[i])


            with stream_processing():
                n_st.save("n")
                for i, c_name in enumerate(self.CRD.paired_elements):
                    if self.operation == "x180":
                        state_st[i].buffer(len(self.variables["amp"])).buffer(
                            len(self.variables["N"])
                        ).average().save(f"state{i + 1}")
                    elif self.operation in ["x90"]:
                        state_st[i].buffer(len(self.variables["amp"])).buffer(
                            len(self.variables["N"])
                        ).average().save(f"state{i + 1}")
                    else:
                        raise ValueError(f"Unrecognized operation {self.operation}.")
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
        
        ds = fetch_results_as_xarray(job.result_handles, self.CRD.get_all_TransmonPairs(), self._sort_variables_(["amp", "N"]))
        
        return ds
    

    # assign new coordinates
    def dataset_post_proccess(self):
        ''' Once the dataset had been saved into node.results '''

        ds:xr.Dataset = self.node.results["ds"]
        ds = ds.assign_coords(
            {
                "abs_amp": (
                    ["qubit", "amp"],
                    np.array([self.CRD.get_obj_with_type(c_name,"drive_q").xy.operations[f"x180_{c_name}"].amplitude * self.variables["amp"] for c_name in self.CRD.paired_elements]),
                )
            }
        )
        self.node.results["ds"] = ds


    # analyze data
    def analyze(self):
        if not self.node.parameters.simulate:
            if self.node.parameters.load_data_id is not None:
                self.load_data = True
                from quam_libs.experiments.coupler_RD_related import load_data_only
                self.node, self.machine, _, self.CRD = load_data_only(self.node)
                self.N_pi = self.node.parameters.max_number_rabi_pulses_per_sweep
                self.operation =  self.node.parameters.operation_x180_or_any_90

            ds:xr.Dataset = self.node.results['ds']

            fit_results = {}
            if self.N_pi == 1:
                # Fit the power Rabi oscillations
                fit = fit_oscillation(ds.state, "amp")
                fit_evals = oscillation(
                    ds.amp,
                    fit.sel(fit_vals="a"),
                    fit.sel(fit_vals="f"),
                    fit.sel(fit_vals="phi"),
                    fit.sel(fit_vals="offset"),
                )
                
            # Save fitting results
                for c_name in self.CRD.paired_elements:
                    fit_results[c_name] = {}
                    f_fit = fit.loc[c_name].sel(fit_vals="f")
                    phi_fit = fit.loc[c_name].sel(fit_vals="phi")
                    phi_fit = phi_fit - np.pi * (phi_fit > np.pi / 2)
                    factor = float(1.0 * (np.pi - phi_fit) / (2 * np.pi * f_fit))
                    new_pi_amp = self.CRD.get_obj_with_type(c_name,"drive_q").xy.operations[f"x180_{c_name}"].amplitude * factor
                    if new_pi_amp < 0.3:  # TODO: 1 for OPX1000 MW
                        print(f"amplitude for Pi pulse is modified by a factor of {factor:.2f}")
                        print(
                            f"new amplitude is {1e3 * new_pi_amp:.2f} mV \n"
                        )  # TODO: 1 for OPX1000 MW
                        fit_results[c_name]["Pi_amplitude"] = float(new_pi_amp)
                    else:
                        print(f"Fitted amplitude too high, new amplitude is 300 mV \n")
                        fit_results[c_name]["Pi_amplitude"] = 0.3  # TODO: 1 for OPX1000 MW
                self.node.results["fit_results"] = fit_results
                self.analyzed_items = {"data_max_idx":None, "fits":fit_evals}

            elif self.N_pi > 1:
                # Get the average along the number of pulses axis to identify the best pulse amplitude
                I_n = ds.state.mean(dim="N")
                data_max_idx = I_n.argmax(dim="amp")
                
                
            # Save fitting results
                for c_name in self.CRD.paired_elements:
                    new_pi_amp = ds.abs_amp.sel(qubit=c_name)[data_max_idx.sel(qubit=c_name)]
                    fit_results[c_name] = {}
                    if new_pi_amp < 1:  # TODO: 1 for OPX1000 MW
                        fit_results[c_name]["Pi_amplitude"] = float(new_pi_amp)
                        print(
                            f"amplitude for Pi pulse is modified by a factor of {I_n.idxmax(dim='amp').sel(qubit = c_name):.2f}"
                        )
                        print(
                            f"new amplitude is {1e3 * new_pi_amp:.2f} mV \n"
                        )  # TODO: 1 for OPX1000 MW
                    else:
                        print(f"Fitted amplitude too high, new amplitude is 1000 mV \n")
                        fit_results[c_name]["Pi_amplitude"] = 1  # TODO: 1 for OPX1000 MW
                self.node.results["fit_results"] = fit_results
                self.analyzed_items = {"data_max_idx":data_max_idx, "fits":None}

    
    # plot
    def visualize(self):
        
        if not self.node.parameters.simulate:
            ds:xr.Dataset = self.node.results['ds']
            grid = QubitGrid(ds, [q.grid_location for q in self.CRD.get_all_Transmons()])
            
            coupler = self.CRD.get_all_TransmonPairs()
            for ax, qubit in grid_iter(grid):
                ## HARDcoded:
                qubit['qubit'] = coupler[0].name
                if self.N_pi == 1:
                    ds.assign_coords(amp_mV=ds.abs_amp * 1e3).loc[qubit].state.plot(
                        ax=ax, x="amp_mV"
                    )
                    ax.plot(ds.abs_amp.loc[qubit] * 1e3, self.analyzed_items["fits"].loc[qubit][0])
                    ax.set_ylabel("State")
                elif self.N_pi > 1:
                    ds.assign_coords(amp_mV=ds.abs_amp * 1e3).loc[qubit].state.plot(
                        ax=ax, x="amp_mV", y="N"
                    )
                    ax.axvline(1e3 * ds.abs_amp.loc[qubit][self.analyzed_items["data_max_idx"].loc[qubit]], color="r")
                    ax.set_ylabel("num. of pulses")
                ax.set_xlabel("Amplitude [mV]")
                ax.set_title(qubit["qubit"])
            if self.N_pi > 1:
                grid.fig.suptitle(f"Coupler {self.operation} Power Rabi State")
            else:
                grid.fig.suptitle(f"Coupler {self.operation} Power Rabi")
            plt.tight_layout()
            plt.show()
            self.node.results["figure"] = grid.fig


    
    # Save state and results
    def state_management(self, update_state_also:bool):
        if not self.node.parameters.simulate:
            if update_state_also:
                with self.node.record_state_updates():
                    for c in self.CRD.get_all_TransmonPairs():
                        self.CRD.get_obj_with_type(c.name,"drive_q").xy.operations[f"x180_{c.name}"].amplitude = self.node.results["fit_results"][c.name]["Pi_amplitude"]
                        if self.operation == 'x180' and self.node.parameters.update_x90:
                            self.CRD.get_obj_with_type(c.name,"drive_q").xy.operations[f'x90_{c.name}'].amplitude = self.node.results["fit_results"][c.name]["Pi_amplitude"]/2
            
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

