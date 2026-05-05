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
from scipy.stats import norm
from quam_libs.components.macros.coupler_RD_decoding import CouplerReadoutDecoder
from quam_libs.experiments.coupler_RD_related import ExpTemplate
from qm.jobs.running_qm_job import RunningQmJob
from typing import Literal, Optional, List
from quam.components import pulses

u = unit(coerce_to_integer=True)

class Parameters(NodeParameters):
    coupler: list[str] = ['coupler_q7_q8']
    num_averages: int = 500
    operation: str = "saturation"
    operation_amplitude_factor: Optional[float] = 0.05    # 0.05 , 0.1 good
    operation_len_in_ns: Optional[int] = None
    Driving_LO_GHz:float|None = 3.0      # None use state recorded. Otherwise, use this value as the new LO (and will be updated into state)
    frequency_span_in_mhz: float = 300 #200, 4, 800
    frequency_step_in_mhz: float = 1 #0.25, 0.01
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    target_peak_width: Optional[float] = 1e6 #1e6
    arbitrary_flux_bias: Optional[float] = None
    arbitrary_qubit_frequency_in_ghz: Optional[float] = None
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False
    readme_password: str|None = None
    UPDATE_STATE:bool=True


node = QualibrationNode(name="03x_Coupler_Spectroscopy", parameters=Parameters())

def encode():
    MASK = 0xFFFF
    cal = (13 - 37) & MASK
    return hex(cal)


class CP_Spectrum_EXP(ExpTemplate):

    def __init__(self, node:QualibrationNode):
        super().__init__()
        self.node = node
        self._check_coupler_num_()
        self.machine = QuAM.load()
        if self.node.parameters.arbitrary_flux_bias is not None:
            if self.node.parameters.arbitrary_flux_bias != 0:
                assert self.node.parameters.readme_password == encode(), PermissionError("WARNING: Please go Checking README.md in CouplerMeas directory and use the correct readme_password !!!!")



    # Checking all the required participants
    def participants_stand_by(self):
        self.CRD = CouplerReadoutDecoder(self.machine, self.node.parameters.coupler, coupler_arbi_LO_manual=self.node.parameters.Driving_LO_GHz, simulate=self.node.parameters.simulate, load_data_id=self.node.parameters.load_data_id)
        self.config = self.machine.generate_config()
        self.participant_num = len(list(self.CRD.paired_elements.keys()))
        if self.node.parameters.load_data_id is None:
            self.qmm = self.machine.connect()

    # Counting variables 
    def exp_variable_arangement(self):
        self.operation_len = self.node.parameters.operation_len_in_ns

        if self.node.parameters.operation_amplitude_factor:
            self.operation_amp = self.node.parameters.operation_amplitude_factor
        else:
            self.operation_amp = 1.0

        dfs = np.arange(-self.node.parameters.frequency_span_in_mhz * 1e6 // 2, +self.node.parameters.frequency_span_in_mhz * 1e6 // 2, self.node.parameters.frequency_step_in_mhz*1e6, dtype=np.int32)

        self.variables['freq'] = dfs


        if self.node.parameters.arbitrary_flux_bias is None:
            self.flux_offset_pulse ={c_name: 0 for c_name in self.CRD.paired_elements}
        else:
            self.flux_offset_pulse = {c_name: self.node.parameters.arbitrary_flux_bias for c_name in self.CRD.paired_elements}
        
        

    # Composing QUA program
    def qua_composer(self):
        with program() as self.qua_prog:
            _, _, _, _, n, n_st = qua_declaration(num_qubits=self.participant_num)
            
            state = [declare(int) for _ in range(self.participant_num)]
            state_st = [declare_stream() for _ in range(self.participant_num)]
            df = declare(int)
            
            if not self.node.parameters.simulate:
                self.machine.apply_all_couplers_to_min()
            
            for i, c_name in enumerate(self.CRD.paired_elements):

                max_freq = self.variables['freq'][-1] + self.CRD.get_obj_with_type(c_name,'coupler').extras["RD"]["IF"]
                min_freq = self.variables['freq'][0] + self.CRD.get_obj_with_type(c_name,'coupler').extras["RD"]["IF"]

                assert max_freq <= 400e6 and min_freq >= -400e6, (
                    f"{c_name} IF span out of range: min={min_freq/1e6:.2f} MHz, "
                    f"max={max_freq/1e6:.2f} MHz (limit ±400 MHz), please adjust the frequency span.")

                # Bring the active qubits to the desired frequency point
                if not self.node.parameters.simulate:
                    for q_type in ["drive_q", "readout_q"]:
                        self.machine.set_all_fluxes(flux_point=self.node.parameters.flux_point_joint_or_independent, target=self.CRD.paired_elements[c_name][q_type])
                    
                    wait(400)


                with for_(n, 0, n < self.node.parameters.num_averages, n + 1):
                    save(n, n_st)
                    with for_(*from_array(df, self.variables["freq"])):
                        
                        self.CRD.get_obj_with_type(c_name,"drive_q").xy.update_frequency(df + self.CRD.get_obj_with_type(c_name,"coupler").extras["RD"]["IF"])
                        wait(4)
                        align()
                        if not node.parameters.simulate:
                            duration = self.operation_len //4 if self.operation_len is not None else self.CRD.get_obj_with_type(c_name,'drive_q').xy.operations[self.node.parameters.operation].length//4
                        else:
                            duration = 100 //4


                        align()
                        self.CRD.get_obj_with_type(c_name,'coupler').coupler.play("const", amplitude_scale=self.flux_offset_pulse[c_name] / self.CRD.get_obj_with_type(c_name,'coupler').coupler.operations["const"].amplitude, duration=duration)
                        self.CRD.get_obj_with_type(c_name,'drive_q').xy.play(
                            self.node.parameters.operation,
                            amplitude_scale=self.operation_amp,
                            duration=duration,
                        )
                        align()

                        # Measure the state of the resonators
                        readout_state_coupler(self.CRD.paired_elements[c_name]["readout_q"], state[i], flux_applied_target=self.CRD.paired_elements[c_name]["aswap_supplier"], method=self.CRD.paired_elements[c_name]["readout_method"], active_reset_readout_q=True)
                        save(state[i], state_st[i])


            with stream_processing():
                n_st.save("n")
                for i, c_name in enumerate(self.CRD.paired_elements):
                    state_st[i].buffer(len(self.variables["freq"])).average().save(f"state{i + 1}")

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
        ds:xr.Dataset = self.node.results['ds']
        
        ds = ds.assign_coords(
            {
                "freq_full": (
                    ["qubit", "freq"],
                    np.array([self.variables['freq'] + c.extras["RD"]["IF"] + (self.node.parameters.Driving_LO_GHz*1e9 if self.node.parameters.Driving_LO_GHz is not None else c.extras["RD"]["LO"]) for c in self.CRD.get_all_TransmonPairs()]),
                )
            }
        )
        ds.freq_full.attrs["long_name"] = "Frequency"
        ds.freq_full.attrs["units"] = "GHz"
        self.node.results["ds"] = ds



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

            result = peaks_dips(ds.state, dim="freq", prominence_factor=peak_prominence_factor)
            abs_freqs = dict(
                [
                    (
                        q.name,
                        ds.freq_full.sel(freq = result.position.sel(qubit=q.name).values).sel(qubit=q.name).values,
                    )
                    for q in self.CRD.get_all_TransmonPairs() if not np.isnan(result.sel(qubit=q.name).position.values)
                ]
            )
            fit_results = {}
            for c in self.CRD.get_all_TransmonPairs():
                fit_results[c.name] = {}
                LO_to_plot = self.node.parameters.Driving_LO_GHz*1e9 if self.node.parameters.Driving_LO_GHz is not None else c.extras["RD"]["LO"]
                if not np.isnan(result.sel(qubit=c.name).position.values):
                    fit_results[c.name]["fit_successful"] = True
                    print(
                        f"Drive frequency for {c.name} is "
                        f"{(result.sel(qubit = c.name).position.values +  c.extras['RD']['IF'] + LO_to_plot) / 1e9:.6f} GHz"
                    )
                    fit_results[c.name]["drive_freq"] = result.sel(qubit=c.name).position.values +  c.extras["RD"]["IF"] + LO_to_plot
                    print(f"(shift of {result.sel(qubit = c.name).position.values/1e6:.3f} MHz)\n")
    
                else:
                    fit_results[c.name]["fit_successful"] = False
                    print(f"Failed to find a peak for {c.name}\n")
                    
            self.node.results["fit_results"] = fit_results
            self.analyzed_items["fits"] = result
            self.analyzed_items["abs_freqs"] = abs_freqs

    
    # plot
    def visualize(self):
        
        if not self.node.parameters.simulate:
            ds:xr.Dataset = self.node.results['ds']
            grid_names, qubit_pair_names = grid_pair_names(self.CRD.get_all_TransmonPairs())
            grid = QubitPairGrid(grid_names, qubit_pair_names)
            approx_peak = self.analyzed_items["fits"].base_line + self.analyzed_items["fits"].amplitude * (1 / (1 + ((ds.freq - self.analyzed_items["fits"].position) / self.analyzed_items["fits"].width) ** 2))
            for ax, qubit in grid_iter(grid):
                # Plot the line
                (ds.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit].state).plot(ax=ax, x="freq_GHz")
                # Identify the resonance peak
                if not np.isnan(self.analyzed_items["fits"].sel(qubit=qubit["qubit"]).position.values):
                    ax.plot(
                        self.analyzed_items["abs_freqs"][qubit["qubit"]] / 1e9,
                        ds.loc[qubit].sel(freq=self.analyzed_items["fits"].loc[qubit].position.values, method="nearest").state,
                        ".r",
                    )
                    # # Identify the width
                    if np.average(approx_peak.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit].values) > 0:
                        (approx_peak.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit]).plot(
                            ax=ax, x="freq_GHz", linewidth=0.5, linestyle="--"
                        )
                ax.set_xlabel("Driving freq [GHz]")
                ax.set_ylabel("State Porbability")
                if self.node.parameters.arbitrary_flux_bias is None:
                    ax.set_title(qubit["qubit"])
                else:
                    ax.set_title(f'{qubit["qubit"]}\nadditional Flux={self.node.parameters.arbitrary_flux_bias}V')
            
            grid.fig.suptitle("Coupler spectroscopy")
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
                            drive_q = self.CRD.get_obj_with_type(c.name, 'drive_q')
                            
                            c.extras["RD"]["IF"] += float(self.analyzed_items["fits"].sel(qubit=c.name).position.values)
                            if self.node.parameters.Driving_LO_GHz is not None:
                                c.extras["RD"]["LO"] = self.node.parameters.Driving_LO_GHz * 1e9
                            if "T1" not in c.extras:
                                c.extras["T1"] = 30e-6 
                            drive_q.xy.operations[f"x180_{c.name}"] = pulses.DragCosinePulse(
                                amplitude=0.3,
                                alpha=0.0,
                                anharmonicity=f"#/qubits/{drive_q.name}/anharmonicity",
                                length=32,
                                axis_angle=0,
                                detuning=0,
                                digital_marker="ON",
                            )
                            drive_q.xy.operations[f"x90_{c.name}"] = pulses.DragCosinePulse(
                                amplitude=0.15,
                                alpha=0.0,
                                anharmonicity=f"#/qubits/{drive_q.name}/anharmonicity",
                                length=32,
                                axis_angle=0,
                                detuning=0,
                                digital_marker="ON",
                            )
                            

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

