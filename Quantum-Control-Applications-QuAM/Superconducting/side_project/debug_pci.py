
# Single QUA script generated at 2026-01-15 13:35:48.123400
# QUA library version: 1.2.4a1


from qm import CompilerOptionArguments
from qm.qua import *

with program() as prog:
    v1 = declare(int, )
    v2 = declare(fixed, )
    v3 = declare(fixed, )
    v4 = declare(fixed, )
    v5 = declare(fixed, )
    v6 = declare(int, )
    v7 = declare(int, )
    v8 = declare(int, )
    v9 = declare(int, )
    v10 = declare(fixed, )
    v11 = declare(fixed, )
    v12 = declare(fixed, )
    v13 = declare(fixed, )
    v14 = declare(fixed, )
    v15 = declare(fixed, )
    v16 = declare(fixed, )
    v17 = declare(fixed, )
    v18 = declare(fixed, )
    v19 = declare(fixed, )
    reset_global_phase()
    set_dc_offset("coupler_q1_q2", "single", 0.275)
    set_dc_offset("q1.z", "single", -0.2385333232548495)
    set_dc_offset("q2.z", "single", -0.023621635438666397)
    set_dc_offset("c1.z", "single", 0.275)
    set_dc_offset("coupler_q1_q2", "single", 0.275)
    set_dc_offset("q1.z", "single", -0.2385333232548495)
    wait(24, "q1.z")
    align("q1.xy", "q1.resonator", "q1.z")
    wait(24, "q1.z")
    align("q1.xy", "q1.resonator", "q1.z")
    with for_(v1,0,(v1<150),(v1+1)):
        r0 = declare_stream()
        save(v1, r0)
        with for_(v11,-0.05,(v11<0.050500000000000086),(v11+0.0010000000000000009)):
            with for_(v9,4,(v9<=124),(v9+2)):
                measure("readout", "q1.resonator", dual_demod.full("iw1", "iw2", v12), dual_demod.full("iw3", "iw1", v13))
                assign(v6, Cast.to_int((v12>-0.0018752122661306952)))
                wait(250, "q1.resonator")
                align("q1.xy", "q1.resonator", "q1.z")
                assign(v10, Cast.mul_fixed_by_int(0.004,(4*v9)))
                play("x180"*amp(0.5), "q1.xy")
                align("q1.xy", "q1.resonator", "q1.z")
                wait(20, "q1.z")
                play("const"*amp((v11/2.5)), "q1.z", duration=v9)
                wait(20, "q1.z")
                frame_rotation_2pi(v10, "q1.xy")
                align("q1.xy", "q1.resonator", "q1.z")
                play("x180"*amp(0.5), "q1.xy")
                align()
                measure("readout", "q1.resonator", dual_demod.full("iw1", "iw2", v14), dual_demod.full("iw3", "iw1", v15))
                assign(v7, Cast.to_int((v14>-0.0018752122661306952)))
                wait(250, "q1.resonator")
                assign(v7, (v6^v7))
                r1 = declare_stream()
                save(v7, r1)
                reset_frame("q1.xy")
                align("q1.xy", "q1.resonator", "q1.z")
    align()
    set_dc_offset("q1.z", "single", -0.2385333232548495)
    set_dc_offset("q2.z", "single", -0.023621635438666397)
    set_dc_offset("c1.z", "single", 0.275)
    set_dc_offset("coupler_q1_q2", "single", 0.275)
    set_dc_offset("q2.z", "single", -0.023621635438666397)
    wait(24, "q2.z")
    align("q2.xy", "q2.resonator", "q2.z")
    wait(24, "q2.z")
    align("q2.xy", "q2.resonator", "q2.z")
    with for_(v1,0,(v1<150),(v1+1)):
        save(v1, r0)
        with for_(v11,-0.05,(v11<0.050500000000000086),(v11+0.0010000000000000009)):
            with for_(v9,4,(v9<=124),(v9+2)):
                measure("readout", "q2.resonator", dual_demod.full("iw1", "iw2", v16), dual_demod.full("iw3", "iw1", v17))
                assign(v6, Cast.to_int((v16>0.0003978685289192598)))
                wait(250, "q2.resonator")
                align("q2.xy", "q2.resonator", "q2.z")
                assign(v10, Cast.mul_fixed_by_int(0.004,(4*v9)))
                play("x180"*amp(0.5), "q2.xy")
                align("q2.xy", "q2.resonator", "q2.z")
                wait(20, "q2.z")
                play("const"*amp((v11/2.5)), "q2.z", duration=v9)
                wait(20, "q2.z")
                frame_rotation_2pi(v10, "q2.xy")
                align("q2.xy", "q2.resonator", "q2.z")
                play("x180"*amp(0.5), "q2.xy")
                align()
                measure("readout", "q2.resonator", dual_demod.full("iw1", "iw2", v18), dual_demod.full("iw3", "iw1", v19))
                assign(v8, Cast.to_int((v18>0.0003978685289192598)))
                wait(250, "q2.resonator")
                assign(v8, (v6^v8))
                r2 = declare_stream()
                save(v8, r2)
                reset_frame("q2.xy")
                align("q2.xy", "q2.resonator", "q2.z")
    align()
    with stream_processing():
        r0.save("n")
        r1.buffer(61).buffer(101).average().save("state1")
        r2.buffer(61).buffer(101).average().save("state2")

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "fems": {
                "1": {
                    "type": "LF",
                    "analog_outputs": {
                        "1": {
                            "delay": 130,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                        "2": {
                            "delay": 130,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                        "3": {
                            "delay": 130,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                    },
                },
                "6": {
                    "type": "MW",
                    "analog_outputs": {
                        "1": {
                            "band": 2,
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": -2,
                            "upconverter_frequency": 6060000000,
                        },
                        "2": {
                            "band": 1,
                            "delay": 15,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 10,
                            "upconverter_frequency": 4600000000.0,
                        },
                        "3": {
                            "band": 1,
                            "delay": 19,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 13,
                            "upconverter_frequency": 4600000000.0,
                        },
                        "4": {
                            "band": 1,
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 10,
                            "upconverter_frequency": 3500000000.0,
                        },
                    },
                    "analog_inputs": {
                        "1": {
                            "band": 2,
                            "downconverter_frequency": 6060000000,
                            "sampling_rate": 1000000000.0,
                            "shareable": False,
                            "gain_db": 0,
                        },
                    },
                },
            },
        },
    },
    "elements": {
        "q1.xy": {
            "operations": {
                "x180_DragCosine": "q1.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "q1.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "q1.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "q1.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "q1.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "q1.xy.-y90_DragCosine.pulse",
                "x180_Square": "q1.xy.x180_Square.pulse",
                "x90_Square": "q1.xy.x90_Square.pulse",
                "-x90_Square": "q1.xy.-x90_Square.pulse",
                "y180_Square": "q1.xy.y180_Square.pulse",
                "y90_Square": "q1.xy.y90_Square.pulse",
                "-y90_Square": "q1.xy.-y90_Square.pulse",
                "x180": "q1.xy.x180_DragCosine.pulse",
                "x90": "q1.xy.x90_DragCosine.pulse",
                "-x90": "q1.xy.-x90_DragCosine.pulse",
                "y180": "q1.xy.y180_DragCosine.pulse",
                "y90": "q1.xy.y90_DragCosine.pulse",
                "-y90": "q1.xy.-y90_DragCosine.pulse",
                "saturation": "q1.xy.saturation.pulse",
                "EF_x180": "q1.xy.EF_x180.pulse",
            },
            "intermediate_frequency": -22030337.61833399,
            "MWInput": {
                "port": ('con1', 6, 3),
                "upconverter": 1,
            },
        },
        "q1.z": {
            "operations": {
                "const": "q1.z.const.pulse",
            },
            "singleInput": {
                "port": ('con1', 1, 2),
            },
        },
        "q1.resonator": {
            "operations": {
                "readout": "q1.resonator.readout.pulse",
                "const": "q1.resonator.const.pulse",
            },
            "intermediate_frequency": -232093075.0,
            "MWOutput": {
                "port": ('con1', 6, 1),
            },
            "smearing": 0,
            "time_of_flight": 392,
            "MWInput": {
                "port": ('con1', 6, 1),
                "upconverter": 1,
            },
        },
        "q2.xy": {
            "operations": {
                "x180_DragCosine": "q2.xy.x180_DragCosine.pulse",
                "x180_Long": "q2.xy.x180_Long.pulse",
                "x90_DragCosine": "q2.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "q2.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "q2.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "q2.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "q2.xy.-y90_DragCosine.pulse",
                "x180_Square": "q2.xy.x180_Square.pulse",
                "x90_Square": "q2.xy.x90_Square.pulse",
                "-x90_Square": "q2.xy.-x90_Square.pulse",
                "y180_Square": "q2.xy.y180_Square.pulse",
                "y90_Square": "q2.xy.y90_Square.pulse",
                "-y90_Square": "q2.xy.-y90_Square.pulse",
                "x180": "q2.xy.x180_DragCosine.pulse",
                "x90": "q2.xy.x90_DragCosine.pulse",
                "-x90": "q2.xy.-x90_DragCosine.pulse",
                "y180": "q2.xy.y180_DragCosine.pulse",
                "y90": "q2.xy.y90_DragCosine.pulse",
                "-y90": "q2.xy.-y90_DragCosine.pulse",
                "saturation": "q2.xy.saturation.pulse",
                "EF_x180": "q2.xy.EF_x180.pulse",
            },
            "intermediate_frequency": 199122277.7482808,
            "MWInput": {
                "port": ('con1', 6, 2),
                "upconverter": 1,
            },
        },
        "q2.z": {
            "operations": {
                "const": "q2.z.const.pulse",
                "Cz_unipolar": "q2.z.Cz_unipolar.pulse",
                "Cz_flattop": "q2.z.Cz_flattop.pulse",
                "Cz_unipolar.flux_pulse_control_q1_q2": "q2.z.Cz_unipolar.flux_pulse_control_q1_q2.pulse",
                "Cz_flattop.flux_pulse_control_q1_q2": "q2.z.Cz_flattop.flux_pulse_control_q1_q2.pulse",
            },
            "singleInput": {
                "port": ('con1', 1, 1),
            },
        },
        "q2.resonator": {
            "operations": {
                "readout": "q2.resonator.readout.pulse",
                "const": "q2.resonator.const.pulse",
            },
            "intermediate_frequency": -150129053.0,
            "MWOutput": {
                "port": ('con1', 6, 1),
            },
            "smearing": 0,
            "time_of_flight": 392,
            "MWInput": {
                "port": ('con1', 6, 1),
                "upconverter": 1,
            },
        },
        "c1.xy": {
            "operations": {
                "x180_DragCosine": "c1.xy.x180_DragCosine.pulse",
                "x180_Long": "c1.xy.x180_Long.pulse",
                "x90_DragCosine": "c1.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "c1.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "c1.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "c1.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "c1.xy.-y90_DragCosine.pulse",
                "x180_Square": "c1.xy.x180_Square.pulse",
                "x90_Square": "c1.xy.x90_Square.pulse",
                "-x90_Square": "c1.xy.-x90_Square.pulse",
                "y180_Square": "c1.xy.y180_Square.pulse",
                "y90_Square": "c1.xy.y90_Square.pulse",
                "-y90_Square": "c1.xy.-y90_Square.pulse",
                "x180": "c1.xy.x180_Long.pulse",
                "x90": "c1.xy.x90_DragCosine.pulse",
                "-x90": "c1.xy.-x90_DragCosine.pulse",
                "y180": "c1.xy.y180_DragCosine.pulse",
                "y90": "c1.xy.y90_DragCosine.pulse",
                "-y90": "c1.xy.-y90_DragCosine.pulse",
                "saturation": "c1.xy.saturation.pulse",
            },
            "intermediate_frequency": 47200000.0,
            "MWInput": {
                "port": ('con1', 6, 4),
                "upconverter": 1,
            },
        },
        "c1.z": {
            "operations": {
                "const": "c1.z.const.pulse",
            },
            "singleInput": {
                "port": ('con1', 1, 3),
            },
        },
        "c1.resonator": {
            "operations": {
                "readout": "c1.resonator.readout.pulse",
                "const": "c1.resonator.const.pulse",
            },
            "intermediate_frequency": -67369238,
            "MWOutput": {
                "port": ('con1', 6, 1),
            },
            "smearing": 0,
            "time_of_flight": 392,
            "MWInput": {
                "port": ('con1', 6, 1),
                "upconverter": 1,
            },
        },
        "coupler_q1_q2": {
            "operations": {
                "const": "coupler_q1_q2.const.pulse",
                "Cz_unipolar.coupler_flux_pulse_q1_q2": "coupler_q1_q2.Cz_unipolar.coupler_flux_pulse_q1_q2.pulse",
                "Cz_flattop.coupler_flux_pulse_q1_q2": "coupler_q1_q2.Cz_flattop.coupler_flux_pulse_q1_q2.pulse",
            },
            "singleInput": {
                "port": ('con1', 1, 3),
            },
        },
    },
    "pulses": {
        "const_pulse": {
            "operation": "control",
            "length": 1000,
            "waveforms": {
                "I": "const_wf",
                "Q": "zero_wf",
            },
        },
        "q1.xy.x180_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.x180_DragCosine.wf.I",
                "Q": "q1.xy.x180_DragCosine.wf.Q",
            },
        },
        "q1.xy.x90_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.x90_DragCosine.wf.I",
                "Q": "q1.xy.x90_DragCosine.wf.Q",
            },
        },
        "q1.xy.-x90_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.-x90_DragCosine.wf.I",
                "Q": "q1.xy.-x90_DragCosine.wf.Q",
            },
        },
        "q1.xy.y180_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.y180_DragCosine.wf.I",
                "Q": "q1.xy.y180_DragCosine.wf.Q",
            },
        },
        "q1.xy.y90_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.y90_DragCosine.wf.I",
                "Q": "q1.xy.y90_DragCosine.wf.Q",
            },
        },
        "q1.xy.-y90_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.-y90_DragCosine.wf.I",
                "Q": "q1.xy.-y90_DragCosine.wf.Q",
            },
        },
        "q1.xy.x180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.x180_Square.wf.I",
                "Q": "q1.xy.x180_Square.wf.Q",
            },
        },
        "q1.xy.x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.x90_Square.wf.I",
                "Q": "q1.xy.x90_Square.wf.Q",
            },
        },
        "q1.xy.-x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.-x90_Square.wf.I",
                "Q": "q1.xy.-x90_Square.wf.Q",
            },
        },
        "q1.xy.y180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.y180_Square.wf.I",
                "Q": "q1.xy.y180_Square.wf.Q",
            },
        },
        "q1.xy.y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.y90_Square.wf.I",
                "Q": "q1.xy.y90_Square.wf.Q",
            },
        },
        "q1.xy.-y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.-y90_Square.wf.I",
                "Q": "q1.xy.-y90_Square.wf.Q",
            },
        },
        "q1.xy.saturation.pulse": {
            "operation": "control",
            "length": 20000,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.saturation.wf.I",
                "Q": "q1.xy.saturation.wf.Q",
            },
        },
        "q1.xy.EF_x180.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.EF_x180.wf.I",
                "Q": "q1.xy.EF_x180.wf.Q",
            },
        },
        "q1.z.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "q1.z.const.wf",
            },
        },
        "q1.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 1200,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.resonator.readout.wf.I",
                "Q": "q1.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "q1.resonator.readout.iw1",
                "iw2": "q1.resonator.readout.iw2",
                "iw3": "q1.resonator.readout.iw3",
            },
        },
        "q1.resonator.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "I": "q1.resonator.const.wf.I",
                "Q": "q1.resonator.const.wf.Q",
            },
        },
        "q2.xy.x180_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.x180_DragCosine.wf.I",
                "Q": "q2.xy.x180_DragCosine.wf.Q",
            },
        },
        "q2.xy.x180_Long.pulse": {
            "operation": "control",
            "length": 128,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.x180_Long.wf.I",
                "Q": "q2.xy.x180_Long.wf.Q",
            },
        },
        "q2.xy.x90_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.x90_DragCosine.wf.I",
                "Q": "q2.xy.x90_DragCosine.wf.Q",
            },
        },
        "q2.xy.-x90_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.-x90_DragCosine.wf.I",
                "Q": "q2.xy.-x90_DragCosine.wf.Q",
            },
        },
        "q2.xy.y180_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.y180_DragCosine.wf.I",
                "Q": "q2.xy.y180_DragCosine.wf.Q",
            },
        },
        "q2.xy.y90_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.y90_DragCosine.wf.I",
                "Q": "q2.xy.y90_DragCosine.wf.Q",
            },
        },
        "q2.xy.-y90_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.-y90_DragCosine.wf.I",
                "Q": "q2.xy.-y90_DragCosine.wf.Q",
            },
        },
        "q2.xy.x180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.x180_Square.wf.I",
                "Q": "q2.xy.x180_Square.wf.Q",
            },
        },
        "q2.xy.x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.x90_Square.wf.I",
                "Q": "q2.xy.x90_Square.wf.Q",
            },
        },
        "q2.xy.-x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.-x90_Square.wf.I",
                "Q": "q2.xy.-x90_Square.wf.Q",
            },
        },
        "q2.xy.y180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.y180_Square.wf.I",
                "Q": "q2.xy.y180_Square.wf.Q",
            },
        },
        "q2.xy.y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.y90_Square.wf.I",
                "Q": "q2.xy.y90_Square.wf.Q",
            },
        },
        "q2.xy.-y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.-y90_Square.wf.I",
                "Q": "q2.xy.-y90_Square.wf.Q",
            },
        },
        "q2.xy.saturation.pulse": {
            "operation": "control",
            "length": 20000,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.saturation.wf.I",
                "Q": "q2.xy.saturation.wf.Q",
            },
        },
        "q2.xy.EF_x180.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.EF_x180.wf.I",
                "Q": "q2.xy.EF_x180.wf.Q",
            },
        },
        "q2.z.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "q2.z.const.wf",
            },
        },
        "q2.z.Cz_unipolar.pulse": {
            "operation": "control",
            "length": 48,
            "waveforms": {
                "single": "q2.z.Cz_unipolar.wf",
            },
        },
        "q2.z.Cz_flattop.pulse": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "single": "q2.z.Cz_flattop.wf",
            },
        },
        "q2.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 2000,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.resonator.readout.wf.I",
                "Q": "q2.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "q2.resonator.readout.iw1",
                "iw2": "q2.resonator.readout.iw2",
                "iw3": "q2.resonator.readout.iw3",
            },
        },
        "q2.resonator.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "I": "q2.resonator.const.wf.I",
                "Q": "q2.resonator.const.wf.Q",
            },
        },
        "c1.xy.x180_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "c1.xy.x180_DragCosine.wf.I",
                "Q": "c1.xy.x180_DragCosine.wf.Q",
            },
        },
        "c1.xy.x180_Long.pulse": {
            "operation": "control",
            "length": 128,
            "digital_marker": "ON",
            "waveforms": {
                "I": "c1.xy.x180_Long.wf.I",
                "Q": "c1.xy.x180_Long.wf.Q",
            },
        },
        "c1.xy.x90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "c1.xy.x90_DragCosine.wf.I",
                "Q": "c1.xy.x90_DragCosine.wf.Q",
            },
        },
        "c1.xy.-x90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "c1.xy.-x90_DragCosine.wf.I",
                "Q": "c1.xy.-x90_DragCosine.wf.Q",
            },
        },
        "c1.xy.y180_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "c1.xy.y180_DragCosine.wf.I",
                "Q": "c1.xy.y180_DragCosine.wf.Q",
            },
        },
        "c1.xy.y90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "c1.xy.y90_DragCosine.wf.I",
                "Q": "c1.xy.y90_DragCosine.wf.Q",
            },
        },
        "c1.xy.-y90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "c1.xy.-y90_DragCosine.wf.I",
                "Q": "c1.xy.-y90_DragCosine.wf.Q",
            },
        },
        "c1.xy.x180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "c1.xy.x180_Square.wf.I",
                "Q": "c1.xy.x180_Square.wf.Q",
            },
        },
        "c1.xy.x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "c1.xy.x90_Square.wf.I",
                "Q": "c1.xy.x90_Square.wf.Q",
            },
        },
        "c1.xy.-x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "c1.xy.-x90_Square.wf.I",
                "Q": "c1.xy.-x90_Square.wf.Q",
            },
        },
        "c1.xy.y180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "c1.xy.y180_Square.wf.I",
                "Q": "c1.xy.y180_Square.wf.Q",
            },
        },
        "c1.xy.y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "c1.xy.y90_Square.wf.I",
                "Q": "c1.xy.y90_Square.wf.Q",
            },
        },
        "c1.xy.-y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "c1.xy.-y90_Square.wf.I",
                "Q": "c1.xy.-y90_Square.wf.Q",
            },
        },
        "c1.xy.saturation.pulse": {
            "operation": "control",
            "length": 20000,
            "digital_marker": "ON",
            "waveforms": {
                "I": "c1.xy.saturation.wf.I",
                "Q": "c1.xy.saturation.wf.Q",
            },
        },
        "c1.z.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "c1.z.const.wf",
            },
        },
        "c1.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 2000,
            "digital_marker": "ON",
            "waveforms": {
                "I": "c1.resonator.readout.wf.I",
                "Q": "c1.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "c1.resonator.readout.iw1",
                "iw2": "c1.resonator.readout.iw2",
                "iw3": "c1.resonator.readout.iw3",
            },
        },
        "c1.resonator.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "I": "c1.resonator.const.wf.I",
                "Q": "c1.resonator.const.wf.Q",
            },
        },
        "coupler_q1_q2.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "coupler_q1_q2.const.wf",
            },
        },
        "q2.z.Cz_unipolar.flux_pulse_control_q1_q2.pulse": {
            "operation": "control",
            "length": 48,
            "waveforms": {
                "single": "q2.z.Cz_unipolar.flux_pulse_control_q1_q2.wf",
            },
        },
        "coupler_q1_q2.Cz_unipolar.coupler_flux_pulse_q1_q2.pulse": {
            "operation": "control",
            "length": 48,
            "waveforms": {
                "single": "coupler_q1_q2.Cz_unipolar.coupler_flux_pulse_q1_q2.wf",
            },
        },
        "q2.z.Cz_flattop.flux_pulse_control_q1_q2.pulse": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "single": "q2.z.Cz_flattop.flux_pulse_control_q1_q2.wf",
            },
        },
        "coupler_q1_q2.Cz_flattop.coupler_flux_pulse_q1_q2.pulse": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "single": "coupler_q1_q2.Cz_flattop.coupler_flux_pulse_q1_q2.wf",
            },
        },
    },
    "waveforms": {
        "zero_wf": {
            "type": "constant",
            "sample": 0.0,
        },
        "const_wf": {
            "type": "constant",
            "sample": 0.1,
        },
        "q1.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0006173280808725863, 0.0024440388393264516, 0.0054053465222810935, 0.009380014845257467, 0.01420532042010923, 0.01968371466965261, 0.025590911489627312, 0.03168506954815742, 0.03771669329829353, 0.0434388473550194, 0.04861726605892807, 0.053039944339843215, 0.056525817229301493, 0.05893217268163582] + [0.060160494221892874] * 2 + [0.05893217268163582, 0.056525817229301493, 0.05303994433984322, 0.04861726605892808, 0.04343884735501939, 0.03771669329829356, 0.031685069548157446, 0.025590911489627312, 0.01968371466965261, 0.01420532042010923, 0.00938001484525747, 0.0054053465222810935, 0.0024440388393264516, 0.000617328080872593, 0.0],
        },
        "q1.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0008857106768594052, -0.0017351602545416006, -0.0025135721675301897, -0.003189078140647903, -0.0037340228799867207, -0.004126096283889822, -0.004349246821134746, -0.00439433868250761, -0.004259525801905741, -0.003950327434500819, -0.0034794021977836704, -0.0028660298262768564, -0.0021353218569380625, -0.0013171935598865596, -0.0004451392037296311, 0.00044513920372962805, 0.0013171935598865604, 0.0021353218569380617, 0.0028660298262768555, 0.0034794021977836696, 0.00395032743450082, 0.00425952580190574, 0.00439433868250761, 0.004349246821134746, 0.004126096283889822, 0.0037340228799867207, 0.0031890781406479036, 0.0025135721675301906, 0.0017351602545416013, 0.0008857106768594098, 1.0776857623282754e-18],
        },
        "q1.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0003064061033179365, 0.0012131159036326198, 0.002683114638042165, 0.004656392725021253, 0.0070523961299658695, 0.00977331557227668, 0.012708080222675404, 0.015736892767100233, 0.01873612060626968, 0.021583343420916796, 0.024162350955529363, 0.026367886918357612, 0.0281099452784387, 0.029317443536663084, 0.029941123009334353, 0.029955557748355236, 0.02936019014101973, 0.028179350995915564, 0.026461263404612387, 0.024276071176488577, 0.021712972480277166, 0.018876575863234487, 0.015882627564812044, 0.012853284699894512, 0.009912127409564313, 0.007179113709210926, 0.004765683071446945, 0.002770208671783026, 0.001273983949392676, 0.0003379072782645619, 3.8924160035476876e-20],
        },
        "q1.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0005483549694920474, -0.001077707822663543, -0.0015697409725369505, -0.0020074816802543825, -0.002375914788121007, -0.0026625261347110415, -0.002857757315564628, -0.0029553555214895007, -0.00295260582501773, -0.0028504373720345637, -0.0026533993323578833, -0.0023695070213745274, -0.002009963169723411, -0.001588763733216842, -0.00112220174838982, -0.0006282864069314518, -0.0001260976152874759, 0.000364901287275493, 0.0008255683468733768, 0.001237780250766896, 0.0015850970026626266, 0.0018533699604141884, 0.0020312709959735076, 0.0021107231566632797, 0.0020872165459982985, 0.001959997093005342, 0.0017321203101146963, 0.0014103669013825848, 0.0010050220113840408, 0.000529524828514157, 6.653677400775176e-19],
        },
        "q1.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.00030743918313297445, -0.0012171701362518636, -0.002691948367247132, -0.004671396282062746, -0.007074475050490907, -0.009802802345402392, -0.01274468012677868, -0.01577966757259427, -0.01878351194022158, -0.02163323548831323, -0.024212170199982036, -0.02641473418512033, -0.02815075421538626, -0.029349157426767046] + [-0.029960881050982366] * 2 + [-0.029349157426767046, -0.02815075421538626, -0.026414734185120334, -0.02421217019998204, -0.021633235488313227, -0.018783511940221594, -0.015779667572594285, -0.01274468012677868, -0.009802802345402392, -0.007074475050490907, -0.004671396282062748, -0.002691948367247132, -0.001217170136251864, -0.0003074391831329779, -8.175842214353895e-35],
        },
        "q1.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0005486828481647913, 0.001074902556058219, 0.0015571156270109356, 0.0019755802012405062, 0.002313164289910474, 0.002556047160766243, 0.002694285160537453, 0.002722218809271194, 0.0026387045001062446, 0.002447161318672331, 0.002155431065322727, 0.0017754572109064126, 0.0013227959296682255, 0.0008159792276474994, 0.0002757562401733974, -0.00027575624017338817, -0.0008159792276474926, -0.0013227959296682181, -0.0017754572109064057, -0.0021554310653227205, -0.002447161318672326, -0.00263870450010624, -0.0027222188092711906, -0.0026942851605374495, -0.0025560471607662403, -0.0023131642899104727, -0.001975580201240506, -0.0015571156270109351, -0.001074902556058219, -0.0005486828481647942, -6.676081805828616e-19],
        },
        "q1.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0008857106768594052, 0.0017351602545416008, 0.00251357216753019, 0.0031890781406479036, 0.0037340228799867215, 0.004126096283889823, 0.004349246821134748, 0.004394338682507611, 0.004259525801905743, 0.003950327434500822, 0.0034794021977836735, 0.0028660298262768594, 0.002135321856938066, 0.0013171935598865632, 0.0004451392037296348, -0.00044513920372962437, -0.0013171935598865567, -0.002135321856938058, -0.0028660298262768525, -0.0034794021977836665, -0.003950327434500817, -0.004259525801905737, -0.004394338682507608, -0.004349246821134745, -0.0041260962838898215, -0.00373402287998672, -0.003189078140647903, -0.00251357216753019, -0.001735160254541601, -0.0008857106768594098, -1.0776857623282754e-18],
        },
        "q1.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0006173280808725862, 0.0024440388393264516, 0.0054053465222810935, 0.009380014845257467, 0.01420532042010923, 0.01968371466965261, 0.025590911489627312, 0.03168506954815742, 0.03771669329829353, 0.0434388473550194, 0.04861726605892807, 0.053039944339843215, 0.056525817229301493, 0.05893217268163582] + [0.060160494221892874] * 2 + [0.05893217268163582, 0.056525817229301493, 0.05303994433984322, 0.04861726605892808, 0.04343884735501939, 0.03771669329829356, 0.031685069548157446, 0.025590911489627312, 0.01968371466965261, 0.01420532042010923, 0.00938001484525747, 0.0054053465222810935, 0.0024440388393264516, 0.0006173280808725931, 6.598922096609988e-35],
        },
        "q1.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0005483549694920474, 0.001077707822663543, 0.0015697409725369507, 0.002007481680254383, 0.0023759147881210075, 0.002662526134711042, 0.002857757315564629, 0.0029553555214895015, 0.0029526058250177313, 0.002850437372034565, 0.0026533993323578846, 0.002369507021374529, 0.002009963169723413, 0.0015887637332168438, 0.0011222017483898217, 0.0006282864069314536, 0.0001260976152874777, -0.00036490128727549126, -0.0008255683468733751, -0.0012377802507668944, -0.0015850970026626253, -0.0018533699604141873, -0.0020312709959735067, -0.002110723156663279, -0.002087216545998298, -0.0019599970930053415, -0.001732120310114696, -0.0014103669013825846, -0.0010050220113840408, -0.000529524828514157, -6.653677400775176e-19],
        },
        "q1.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.00030640610331793646, 0.0012131159036326198, 0.002683114638042165, 0.004656392725021253, 0.0070523961299658695, 0.00977331557227668, 0.012708080222675404, 0.015736892767100233, 0.01873612060626968, 0.021583343420916796, 0.024162350955529363, 0.026367886918357612, 0.0281099452784387, 0.029317443536663084, 0.029941123009334353, 0.029955557748355236, 0.02936019014101973, 0.028179350995915564, 0.026461263404612387, 0.024276071176488577, 0.021712972480277166, 0.018876575863234487, 0.015882627564812044, 0.012853284699894512, 0.009912127409564313, 0.007179113709210926, 0.004765683071446945, 0.002770208671783026, 0.001273983949392676, 0.000337907278264562, 3.892416003547692e-20],
        },
        "q1.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0005483549694920474, -0.001077707822663543, -0.0015697409725369503, -0.002007481680254382, -0.0023759147881210066, -0.002662526134711041, -0.002857757315564627, -0.0029553555214895, -0.0029526058250177287, -0.0028504373720345624, -0.002653399332357882, -0.0023695070213745257, -0.0020099631697234094, -0.0015887637332168403, -0.0011222017483898182, -0.0006282864069314499, -0.0001260976152874741, 0.00036490128727549473, 0.0008255683468733784, 0.0012377802507668974, 0.0015850970026626279, 0.0018533699604141895, 0.0020312709959735085, 0.0021107231566632806, 0.002087216545998299, 0.0019599970930053424, 0.0017321203101146965, 0.001410366901382585, 0.0010050220113840408, 0.000529524828514157, 6.653677400775176e-19],
        },
        "q1.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.00030640610331793657, -0.0012131159036326198, -0.002683114638042165, -0.004656392725021253, -0.0070523961299658695, -0.00977331557227668, -0.012708080222675404, -0.015736892767100233, -0.01873612060626968, -0.021583343420916796, -0.024162350955529363, -0.026367886918357612, -0.0281099452784387, -0.029317443536663084, -0.029941123009334353, -0.029955557748355236, -0.02936019014101973, -0.028179350995915564, -0.026461263404612387, -0.024276071176488577, -0.021712972480277166, -0.018876575863234487, -0.015882627564812044, -0.012853284699894512, -0.009912127409564313, -0.007179113709210926, -0.004765683071446945, -0.002770208671783026, -0.001273983949392676, -0.00033790727826456187, -3.8924160035476834e-20],
        },
        "q1.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q1.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q1.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05,
        },
        "q1.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q1.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "q1.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q1.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "q1.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q1.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "q1.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q1.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "q1.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q1.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.05,
        },
        "q1.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q1.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.007854789785471543, 0.030060994629453126, 0.06277895997738259, 0.10035145238721895, 0.1362818465859237, 0.16435745099915694] + [0.17972373856501173] * 2 + [0.16435745099915694, 0.13628184658592377, 0.10035145238721903, 0.06277895997738261, 0.030060994629453105, 0.007854789785471532, 0.0],
        },
        "q1.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
        },
        "q1.z.const.wf": {
            "type": "constant",
            "sample": 2.5,
        },
        "q1.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.007349054643675365,
        },
        "q1.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q1.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "q1.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.003056738220897418, 0.012101809661674639, 0.02676490803479547, 0.04644572437002343, 0.07033852373419659, 0.09746513211406094, 0.1267149829753037, 0.15689058398695405, 0.18675654249811238, 0.2150901426582309, 0.24073140354769748, 0.26263056893297737, 0.27989108441015437, 0.2918063024475684] + [0.2978884126187891] * 2 + [0.2918063024475684, 0.27989108441015437, 0.2626305689329774, 0.2407314035476975, 0.21509014265823087, 0.18675654249811252, 0.15689058398695419, 0.1267149829753037, 0.09746513211406094, 0.07033852373419659, 0.04644572437002344, 0.02676490803479547, 0.012101809661674639, 0.0030567382208974508, 0.0],
        },
        "q2.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0014143661568459074, -0.002770827997049576, -0.004013851813495569, -0.00509254806508949, -0.005962754800549657, -0.00658884565388138, -0.006945188391986707, -0.00701719430129911, -0.0068019154514310076, -0.006308165383790187, -0.005556158284159039, -0.004576681411553536, -0.003409834664221801, -0.0021033888850982637, -0.0007108301178811064, 0.0007108301178811018, 0.002103388885098265, 0.003409834664221799, 0.004576681411553534, 0.005556158284159038, 0.006308165383790188, 0.006801915451431006, 0.00701719430129911, 0.006945188391986707, 0.00658884565388138, 0.005962754800549658, 0.005092548065089491, 0.00401385181349557, 0.002770827997049577, 0.0014143661568459147, 1.7209257038161997e-18],
        },
        "q2.xy.x180_Long.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 5.283897927734285e-05, 0.00021122661127277061, 0.00047477529490954723, 0.0008428400823955668, 0.0013145202575186943, 0.0018886615398502511, 0.002563858909462676, 0.003338460045248709, 0.004210569368428022, 0.005178052681346141, 0.006238542390213756, 0.007389443299005441, 0.008627938960339213, 0.009950998567795235, 0.011355384372807014, 0.012837659607974696, 0.01439419689741075, 0.01602118713353669, 0.017714648798607548, 0.019470437708153033, 0.02128425715249134, 0.02315166841149776, 0.02506810161689665, 0.027028866935494864, 0.029029166045989444, 0.031064103881263926, 0.03312870060743784, 0.03521790381035475, 0.03732660085968623, 0.03944963142039508, 0.0415818000809399, 0.043717889067317675, 0.045852671011831145, 0.047980921745333506, 0.05009743308164579, 0.05219702556286109, 0.05427456113434606, 0.05632495571842158, 0.05834319165595277, 0.060324329985401694, 0.06226352252929393, 0.0641560237585212, 0.06599720240544636, 0.06778255279739138, 0.06950770588277334, 0.07116843992290586, 0.07276069082330114, 0.07428056207918998, 0.07572433431092221, 0.07708847436591172, 0.07836964396485298, 0.07956470787104991, 0.08067074156286543, 0.08168503839051593, 0.08260511619969697, 0.08342872340583064, 0.08415384450407053, 0.08477870500157958, 0.08530177576001158, 0.08572177673756892, 0.08603768012147936, 0.08624871284322605] + [0.0863543584703758] * 2 + [0.08624871284322605, 0.08603768012147936, 0.08572177673756894, 0.08530177576001158, 0.08477870500157957, 0.08415384450407053, 0.08342872340583066, 0.08260511619969696, 0.08168503839051594, 0.08067074156286543, 0.07956470787104991, 0.07836964396485299, 0.07708847436591171, 0.07572433431092222, 0.07428056207918998, 0.07276069082330112, 0.07116843992290589, 0.06950770588277336, 0.06778255279739138, 0.06599720240544638, 0.06415602375852118, 0.062263522529293944, 0.060324329985401715, 0.05834319165595279, 0.0563249557184216, 0.05427456113434608, 0.05219702556286109, 0.05009743308164583, 0.047980921745333534, 0.04585267101183113, 0.043717889067317696, 0.04158180008093992, 0.0394496314203951, 0.03732660085968624, 0.03521790381035475, 0.03312870060743784, 0.031064103881263905, 0.02902916604598946, 0.02702886693549488, 0.02506810161689663, 0.02315166841149777, 0.02128425715249134, 0.01947043770815303, 0.017714648798607575, 0.01602118713353671, 0.014394196897410741, 0.012837659607974707, 0.01135538437280703, 0.00995099856779524, 0.008627938960339232, 0.007389443299005437, 0.006238542390213752, 0.005178052681346155, 0.004210569368428028, 0.0033384600452487138, 0.0025638589094626953, 0.0018886615398502511, 0.0013145202575186895, 0.0008428400823955717, 0.00047477529490955205, 0.0002112266112727754, 5.283897927734285e-05, 0.0],
        },
        "q2.xy.x180_Long.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -2.8310817294644954e-05, -5.655235327528443e-05, -8.46554961708979e-05, -0.00011255147288153462, -0.00014017201727761362, -0.00016744953725858226, -0.0001943172801621128, -0.00022070949611905293, -0.00024656159895437336, -0.00027181032424036106, -0.0002963938841152749, -0.0003202521184885977, -0.0003433266422628583, -0.0003655609882117489, -0.0003869007451648904, -0.00040729369116108407, -0.00042668992124420094, -0.0004450419695889727, -0.0004623049256578199, -0.00047843654410446303, -0.000493397348155364, -0.0005071507262160061, -0.0005196630214656012, -0.0005309036142209698, -0.0005408449968680365, -0.0005494628411775721, -0.0005567360578404459, -0.0005626468480767005, -0.0005671807471921478, -0.0005703266599759002, -0.0005720768878522105, -0.000572427147720177, -0.0005713765824352095, -0.000568927762906605, -0.0005650866818061022, -0.0005598627389028097, -0.0005532687180603956, -0.0005453207559528313, -0.0005360383025752457, -0.0005254440736465281, -0.0005135639950201578, -0.0005004271392392957, -0.00048606565439140005, -0.00047051468543646865, -0.0004538122882014333, -0.00043599933625117356, -0.0004171194208640539, -0.0003972187443567627, -0.00037634600701949843, -0.00035455228793820106, -0.0003318909199954683, -0.0003084173593560545, -0.0002841890497563442, -0.00025926528192990125, -0.00023370704851310897, -0.0002075768947859702, -0.00018093876561332326, -0.00015385784896104372, -0.00012640041637016996, -9.863366077933385e-05, -7.062553209238195e-05, -4.2444570893575756e-05, -1.4159740717293008e-05, 1.415974071729312e-05, 4.2444570893575356e-05, 7.062553209238209e-05, 9.863366077933346e-05, 0.0001264004163701698, 0.00015385784896104385, 0.00018093876561332286, 0.0002075768947859701, 0.00023370704851310907, 0.00025926528192990087, 0.0002841890497563441, 0.00030841735935605453, 0.00033189091999546804, 0.0003545522879382011, 0.0003763460070194983, 0.00039721874435676266, 0.000417119420864054, 0.0004359993362511733, 0.00045381228820143326, 0.0004705146854364686, 0.0004860656543914, 0.0005004271392392957, 0.0005135639950201576, 0.000525444073646528, 0.0005360383025752456, 0.0005453207559528312, 0.0005532687180603956, 0.0005598627389028097, 0.000565086681806102, 0.000568927762906605, 0.0005713765824352095, 0.000572427147720177, 0.0005720768878522105, 0.0005703266599759002, 0.0005671807471921478, 0.0005626468480767005, 0.0005567360578404459, 0.0005494628411775721, 0.0005408449968680366, 0.0005309036142209699, 0.0005196630214656011, 0.0005071507262160062, 0.000493397348155364, 0.00047843654410446303, 0.00046230492565782017, 0.0004450419695889729, 0.0004266899212442007, 0.00040729369116108424, 0.0003869007451648906, 0.00036556098821174894, 0.00034332664226285865, 0.00032025211848859766, 0.00029639388411527485, 0.00027181032424036133, 0.0002465615989543736, 0.00022070949611905307, 0.0001943172801621134, 0.00016744953725858228, 0.00014017201727761357, 0.00011255147288153501, 8.465549617089824e-05, 5.65523532752847e-05, 2.8310817294644653e-05, 1.4021493968622868e-19],
        },
        "q2.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0015292892188787611, 0.006054547595210294, 0.0133905105193723, 0.023236842807310535, 0.035190434458261914, 0.04876190402554907, 0.06339563395050733, 0.07849251761061862, 0.09343448681502947, 0.10760981558890452, 0.12043816430671082, 0.1313943388646745, 0.14002979218909664, 0.14599098780625266] + [0.14903387366716553] * 2 + [0.14599098780625266, 0.14002979218909664, 0.13139433886467453, 0.12043816430671084, 0.1076098155889045, 0.09343448681502954, 0.07849251761061869, 0.06339563395050733, 0.04876190402554907, 0.035190434458261914, 0.023236842807310545, 0.0133905105193723, 0.006054547595210294, 0.0015292892188787776, 0.0],
        },
        "q2.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0003265886845946167, -0.0006398067900693889, -0.0009268311303846143, -0.0011759108953238598, -0.0013768487300448828, -0.0015214182159850114, -0.001603700661403664, -0.0016203274133176236, -0.0015706177705499724, -0.001456606851716253, -0.0012829622772292327, -0.0010567930763707798, -0.0007873586427977934, -0.0004856896538779531, -0.00016413647346223882, 0.0001641364734622377, 0.00048568965387795333, 0.000787358642797793, 0.0010567930763707794, 0.0012829622772292323, 0.0014566068517162531, 0.001570617770549972, 0.0016203274133176236, 0.001603700661403664, 0.0015214182159850114, 0.001376848730044883, 0.0011759108953238598, 0.0009268311303846145, 0.0006398067900693891, 0.00032658868459461846, 3.9737578502850895e-19],
        },
        "q2.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0015292892188787611, -0.006054547595210294, -0.0133905105193723, -0.023236842807310535, -0.035190434458261914, -0.04876190402554907, -0.06339563395050733, -0.07849251761061862, -0.09343448681502947, -0.10760981558890452, -0.12043816430671082, -0.1313943388646745, -0.14002979218909664, -0.14599098780625266] + [-0.14903387366716553] * 2 + [-0.14599098780625266, -0.14002979218909664, -0.13139433886467453, -0.12043816430671084, -0.1076098155889045, -0.09343448681502954, -0.07849251761061869, -0.06339563395050733, -0.04876190402554907, -0.035190434458261914, -0.023236842807310545, -0.0133905105193723, -0.006054547595210294, -0.0015292892188787776, -4.866449831938302e-35],
        },
        "q2.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0003265886845946169, 0.0006398067900693896, 0.0009268311303846159, 0.0011759108953238626, 0.0013768487300448871, 0.0015214182159850175, 0.0016037006614036719, 0.0016203274133176331, 0.001570617770549984, 0.0014566068517162661, 0.0012829622772292475, 0.0010567930763707959, 0.0007873586427978106, 0.000485689653877971, 0.00016413647346225706, -0.00016413647346221946, -0.00048568965387793544, -0.0007873586427977759, -0.0010567930763707633, -0.0012829622772292176, -0.00145660685171624, -0.0015706177705499605, -0.001620327413317614, -0.0016037006614036562, -0.0015214182159850053, -0.0013768487300448787, -0.001175910895323857, -0.0009268311303846129, -0.0006398067900693883, -0.0003265886845946183, -3.9737578502850895e-19],
        },
        "q2.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0014143661568459076, 0.002770827997049577, 0.004013851813495571, 0.0050925480650894925, 0.005962754800549661, 0.006588845653881386, 0.0069451883919867145, 0.007017194301299119, 0.006801915451431019, 0.0063081653837902, 0.005556158284159053, 0.0045766814115535525, 0.003409834664221818, 0.0021033888850982814, 0.0007108301178811246, -0.0007108301178810836, -0.002103388885098247, -0.003409834664221782, -0.004576681411553518, -0.005556158284159023, -0.006308165383790175, -0.0068019154514309945, -0.0070171943012991, -0.006945188391986699, -0.006588845653881374, -0.005962754800549653, -0.005092548065089488, -0.004013851813495568, -0.002770827997049576, -0.0014143661568459145, -1.7209257038161997e-18],
        },
        "q2.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.003056738220897418, 0.012101809661674639, 0.02676490803479547, 0.04644572437002343, 0.07033852373419659, 0.09746513211406094, 0.1267149829753037, 0.15689058398695405, 0.18675654249811238, 0.2150901426582309, 0.24073140354769748, 0.26263056893297737, 0.27989108441015437, 0.2918063024475684] + [0.2978884126187891] * 2 + [0.2918063024475684, 0.27989108441015437, 0.2626305689329774, 0.2407314035476975, 0.21509014265823087, 0.18675654249811252, 0.15689058398695419, 0.1267149829753037, 0.09746513211406094, 0.07033852373419659, 0.04644572437002344, 0.02676490803479547, 0.012101809661674639, 0.0030567382208974508, 1.0537630773744575e-34],
        },
        "q2.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.00032658868459461683, 0.0006398067900693892, 0.0009268311303846151, 0.0011759108953238613, 0.001376848730044885, 0.0015214182159850145, 0.001603700661403668, 0.0016203274133176284, 0.001570617770549978, 0.0014566068517162594, 0.0012829622772292401, 0.0010567930763707878, 0.000787358642797802, 0.00048568965387796206, 0.00016413647346224795, -0.00016413647346222857, -0.0004856896538779444, -0.0007873586427977844, -0.0010567930763707714, -0.001282962277229225, -0.0014566068517162466, -0.0015706177705499663, -0.0016203274133176188, -0.0016037006614036601, -0.0015214182159850084, -0.0013768487300448808, -0.0011759108953238583, -0.0009268311303846136, -0.0006398067900693887, -0.00032658868459461835, -3.9737578502850895e-19],
        },
        "q2.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0015292892188787611, 0.006054547595210294, 0.0133905105193723, 0.023236842807310535, 0.035190434458261914, 0.04876190402554907, 0.06339563395050733, 0.07849251761061862, 0.09343448681502947, 0.10760981558890452, 0.12043816430671082, 0.1313943388646745, 0.14002979218909664, 0.14599098780625266] + [0.14903387366716553] * 2 + [0.14599098780625266, 0.14002979218909664, 0.13139433886467453, 0.12043816430671084, 0.1076098155889045, 0.09343448681502954, 0.07849251761061869, 0.06339563395050733, 0.04876190402554907, 0.035190434458261914, 0.023236842807310545, 0.0133905105193723, 0.006054547595210294, 0.0015292892188787776, 2.433224915969151e-35],
        },
        "q2.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0003265886845946166, -0.0006398067900693885, -0.0009268311303846134, -0.0011759108953238583, -0.0013768487300448806, -0.0015214182159850084, -0.0016037006614036601, -0.0016203274133176188, -0.0015706177705499668, -0.0014566068517162464, -0.0012829622772292254, -0.0010567930763707718, -0.0007873586427977849, -0.00048568965387794417, -0.00016413647346222968, 0.00016413647346224684, 0.0004856896538779623, 0.0007873586427978016, 0.0010567930763707874, 0.0012829622772292397, 0.0014566068517162596, 0.0015706177705499776, 0.0016203274133176284, 0.001603700661403668, 0.0015214182159850145, 0.0013768487300448852, 0.0011759108953238613, 0.0009268311303846154, 0.0006398067900693894, 0.00032658868459461857, 3.9737578502850895e-19],
        },
        "q2.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0015292892188787611, -0.006054547595210294, -0.0133905105193723, -0.023236842807310535, -0.035190434458261914, -0.04876190402554907, -0.06339563395050733, -0.07849251761061862, -0.09343448681502947, -0.10760981558890452, -0.12043816430671082, -0.1313943388646745, -0.14002979218909664, -0.14599098780625266] + [-0.14903387366716553] * 2 + [-0.14599098780625266, -0.14002979218909664, -0.13139433886467453, -0.12043816430671084, -0.1076098155889045, -0.09343448681502954, -0.07849251761061869, -0.06339563395050733, -0.04876190402554907, -0.035190434458261914, -0.023236842807310545, -0.0133905105193723, -0.006054547595210294, -0.0015292892188787776, 2.433224915969151e-35],
        },
        "q2.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q2.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05,
        },
        "q2.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "q2.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "q2.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q2.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "q2.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q2.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "q2.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q2.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.5,
        },
        "q2.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.002445685357886733, 0.009682614785639218, 0.021414507542182167, 0.03716105855962382, 0.056277602189168996, 0.07798150488852554, 0.10138420633108153, 0.12552759717007808, 0.14942324414783864, 0.17209285666947452, 0.19260833813478337, 0.21012978232210897, 0.22393985924773974, 0.23347318273972528] + [0.23833945741418575] * 2 + [0.23347318273972528, 0.22393985924773974, 0.210129782322109, 0.1926083381347834, 0.1720928566694745, 0.14942324414783875, 0.1255275971700782, 0.10138420633108153, 0.07798150488852554, 0.056277602189168996, 0.03716105855962384, 0.021414507542182167, 0.009682614785639218, 0.0024456853578867593, 0.0],
        },
        "q2.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.004368349936092402, -0.008557859112541859, -0.012397008531418102, -0.01572862296443143, -0.01841630572524621, -0.020350022765845365, -0.021450607483384662, -0.02167300181021483, -0.021008100896465216, -0.01948312586366727, -0.017160509368840702, -0.01413533960417955, -0.010531467375291407, -0.006496435634663962, -0.002195439055854606, 0.0021954390558545918, 0.0064964356346639655, 0.010531467375291403, 0.014135339604179545, 0.0171605093688407, 0.01948312586366727, 0.021008100896465213, 0.02167300181021483, 0.021450607483384662, 0.020350022765845365, 0.018416305725246215, 0.01572862296443143, 0.012397008531418103, 0.008557859112541862, 0.0043683499360924255, 5.315176449816806e-18],
        },
        "q2.z.const.wf": {
            "type": "constant",
            "sample": 2.5,
        },
        "q2.z.Cz_unipolar.wf": {
            "type": "constant",
            "sample": 0.059,
        },
        "q2.z.Cz_flattop.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.005917563276925436, 0.021409945066494435, 0.040559582091781565, 0.056051963881350565] + [0.061969527158276] * 71 + [0.056051963881350565, 0.040559582091781565, 0.021409945066494438, 0.005917563276925439],
        },
        "q2.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.0036215299041330576,
        },
        "q2.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "q2.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "c1.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.017087752868164427, 0.0653963837643907, 0.13657289153658733, 0.21831021137266807, 0.2964752181127355, 0.3575524719846385] + [0.39098116092449303] * 2 + [0.3575524719846385, 0.29647521811273564, 0.21831021137266823, 0.1365728915365874, 0.06539638376439066, 0.017087752868164406, 0.0],
        },
        "c1.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
        },
        "c1.xy.x180_Long.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 2.4922517611573513e-05, 9.962908086939387e-05, 0.00022393687029448296, 0.0003975416838956756, 0.0006200186816028761, 0.000890823424921892, 0.0012092932092666738, 0.0015746486857084786, 0.0019859957681732953, 0.0024423278214202905, 0.0029425281244469374, 0.0034853726032924256, 0.004069532826551766, 0.004693579256270052, 0.00535598474626138, 0.006055128279291481, 0.0067892989339785065, 0.007556700071704382, 0.008355453733290547, 0.009183605234678772, 0.010039127950370588, 0.010919928272919488, 0.011823850736339184, 0.012748683290890074, 0.01369216271633561, 0.014651980160421482, 0.015625786789023977, 0.01661119953414071, 0.017605806925657295, 0.018607174992618863, 0.019612853219564838, 0.0206203805433509, 0.021627291375782894, 0.022631121637324315, 0.02362941478711187, 0.02461972783452256, 0.025599637317581147, 0.02656674523357772, 0.027518684907382126, 0.028453126783094626, 0.029367784124859547, 0.03026041861289111, 0.031128845821017014, 0.03197094056233536, 0.03278464208990307, 0.033567959139729014, 0.0343189748037307, 0.03503585122072947, 0.03571683407400492, 0.036360256884401775, 0.03696454508848367, 0.03752821989175377, 0.03804990188751264, 0.03852831443249762, 0.03896228677104288, 0.03935075690011456, 0.03969277416821035, 0.03998750160176287, 0.040234217953354294, 0.040432319466729656, 0.04058132135428957, 0.04068085898344671] + [0.04073068876894286] * 2 + [0.04068085898344671, 0.04058132135428957, 0.040432319466729656, 0.040234217953354294, 0.03998750160176286, 0.039692774168210355, 0.03935075690011457, 0.038962286771042874, 0.038528314432497636, 0.03804990188751264, 0.03752821989175377, 0.03696454508848368, 0.03636025688440177, 0.03571683407400493, 0.035035851220729476, 0.034318974803730695, 0.03356795913972903, 0.032784642089903074, 0.03197094056233536, 0.03112884582101702, 0.030260418612891103, 0.029367784124859558, 0.028453126783094636, 0.027518684907382136, 0.02656674523357773, 0.025599637317581157, 0.02461972783452256, 0.023629414787111887, 0.022631121637324326, 0.021627291375782884, 0.020620380543350907, 0.01961285321956485, 0.01860717499261887, 0.0176058069256573, 0.01661119953414071, 0.015625786789023977, 0.014651980160421471, 0.013692162716335618, 0.012748683290890083, 0.011823850736339176, 0.010919928272919491, 0.010039127950370588, 0.00918360523467877, 0.008355453733290559, 0.007556700071704392, 0.006789298933978502, 0.0060551282792914865, 0.005355984746261386, 0.0046935792562700545, 0.004069532826551776, 0.0034853726032924234, 0.002942528124446935, 0.002442327821420297, 0.0019859957681732975, 0.0015746486857084808, 0.0012092932092666827, 0.000890823424921892, 0.0006200186816028739, 0.00039754168389567784, 0.00022393687029448523, 9.962908086939613e-05, 2.4922517611573513e-05, 0.0],
        },
        "c1.xy.x180_Long.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 128,
        },
        "c1.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.008543876434082213, 0.03269819188219535, 0.06828644576829367, 0.10915510568633403, 0.14823760905636774, 0.17877623599231926] + [0.19549058046224652] * 2 + [0.17877623599231926, 0.14823760905636782, 0.10915510568633412, 0.0682864457682937, 0.03269819188219533, 0.008543876434082203, 0.0],
        },
        "c1.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.030066968357884298, -0.05493508473685802, -0.07030442589525175, -0.07351749312069815, -0.06401871790011814, -0.04345052476281385, -0.015369341158393722, 0.015369341158393704, 0.04345052476281383, 0.06401871790011811, 0.07351749312069814, 0.07030442589525175, 0.054935084736858014, 0.030066968357884298, 8.376210093969753e-17],
        },
        "c1.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.00854387643408221, -0.032698191882195346, -0.06828644576829365, -0.10915510568633402, -0.14823760905636774, -0.17877623599231926] + [-0.19549058046224652] * 2 + [-0.17877623599231926, -0.14823760905636782, -0.10915510568633413, -0.06828644576829371, -0.03269819188219534, -0.008543876434082207, -1.0257898880565809e-32],
        },
        "c1.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.030066968357884298, 0.05493508473685803, 0.07030442589525177, 0.07351749312069816, 0.06401871790011815, 0.04345052476281387, 0.015369341158393746, -0.01536934115839368, -0.04345052476281381, -0.0640187179001181, -0.07351749312069812, -0.07030442589525174, -0.05493508473685801, -0.030066968357884298, -8.376210093969753e-17],
        },
        "c1.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 1.0463230927309285e-18, 4.00437360264365e-18, 8.362677723529016e-18, 1.3367645078936002e-17, 1.8153871344413744e-17, 2.1893774517160563e-17] + [2.394069136265483e-17] * 2 + [2.1893774517160563e-17, 1.8153871344413753e-17, 1.3367645078936013e-17, 8.36267772352902e-18, 4.004373602643648e-18, 1.0463230927309271e-18, 0.0],
        },
        "c1.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.017087752868164427, 0.0653963837643907, 0.13657289153658733, 0.21831021137266807, 0.2964752181127355, 0.3575524719846385] + [0.39098116092449303] * 2 + [0.3575524719846385, 0.29647521811273564, 0.21831021137266823, 0.1365728915365874, 0.06539638376439066, 0.017087752868164406, 0.0],
        },
        "c1.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.030066968357884298, 0.05493508473685802, 0.07030442589525175, 0.07351749312069815, 0.06401871790011815, 0.043450524762813865, 0.015369341158393734, -0.015369341158393692, -0.043450524762813816, -0.0640187179001181, -0.07351749312069814, -0.07030442589525175, -0.054935084736858014, -0.030066968357884298, -8.376210093969753e-17],
        },
        "c1.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.008543876434082212, 0.03269819188219535, 0.06828644576829367, 0.10915510568633403, 0.14823760905636774, 0.17877623599231926] + [0.19549058046224652] * 2 + [0.17877623599231926, 0.14823760905636782, 0.10915510568633412, 0.0682864457682937, 0.03269819188219533, 0.008543876434082205, 5.1289494402829045e-33],
        },
        "c1.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.030066968357884298, -0.05493508473685802, -0.07030442589525175, -0.07351749312069815, -0.06401871790011812, -0.04345052476281384, -0.01536934115839371, 0.015369341158393716, 0.043450524762813844, 0.06401871790011812, 0.07351749312069814, 0.07030442589525175, 0.054935084736858014, 0.030066968357884298, 8.376210093969753e-17],
        },
        "c1.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.008543876434082215, -0.03269819188219535, -0.06828644576829367, -0.10915510568633403, -0.14823760905636774, -0.17877623599231926] + [-0.19549058046224652] * 2 + [-0.17877623599231926, -0.14823760905636782, -0.10915510568633412, -0.0682864457682937, -0.03269819188219533, -0.008543876434082201, 5.1289494402829045e-33],
        },
        "c1.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "c1.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "c1.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05,
        },
        "c1.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "c1.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "c1.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "c1.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "c1.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "c1.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "c1.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "c1.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "c1.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "c1.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.15,
        },
        "c1.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "c1.z.const.wf": {
            "type": "constant",
            "sample": 2.5,
        },
        "c1.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.011674147072951396,
        },
        "c1.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "c1.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "c1.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "coupler_q1_q2.const.wf": {
            "type": "constant",
            "sample": 2.5,
        },
        "q2.z.Cz_unipolar.flux_pulse_control_q1_q2.wf": {
            "type": "constant",
            "sample": 0.059,
        },
        "coupler_q1_q2.Cz_unipolar.coupler_flux_pulse_q1_q2.wf": {
            "type": "constant",
            "sample": -0.225,
        },
        "q2.z.Cz_flattop.flux_pulse_control_q1_q2.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.009075227138867558, 0.030984763579137996, 0.05289430001940844] + [0.061969527158276] * 73 + [0.05289430001940844, 0.030984763579138, 0.009075227138867561],
        },
        "coupler_q1_q2.Cz_flattop.coupler_flux_pulse_q1_q2.wf": {
            "type": "arbitrary",
            "samples": [-0.0, -0.0025363429867735834, -0.010047901708510151, -0.02224601117606403, -0.038661904883375724, -0.0586647292414125, -0.08148578692780814, -0.10624807749387107, -0.13199999999999998, -0.15775192250612893, -0.18251421307219184, -0.20533527075858746, -0.22533809511662428, -0.241753988823936, -0.25395209829148985, -0.26146365701322644] + [-0.264] * 49 + [-0.26146365701322644, -0.25395209829148985, -0.241753988823936, -0.22533809511662428, -0.20533527075858748, -0.18251421307219187, -0.15775192250612896, -0.132, -0.10624807749387108, -0.08148578692780815, -0.05866472924141254, -0.03866190488337574, -0.022246011176064014, -0.010047901708510151, -0.0025363429867735834],
        },
    },
    "digital_waveforms": {
        "ON": {
            "samples": [[1, 0]],
        },
    },
    "integration_weights": {
        "q1.resonator.readout.iw1": {
            "cosine": [(0.3829858928526527, 1200)],
            "sine": [(0.9237541912629443, 1200)],
        },
        "q1.resonator.readout.iw2": {
            "cosine": [(-0.9237541912629443, 1200)],
            "sine": [(0.3829858928526527, 1200)],
        },
        "q1.resonator.readout.iw3": {
            "cosine": [(0.9237541912629443, 1200)],
            "sine": [(-0.3829858928526527, 1200)],
        },
        "q2.resonator.readout.iw1": {
            "cosine": [(-0.9709198669465744, 2000)],
            "sine": [(0.23940470331312666, 2000)],
        },
        "q2.resonator.readout.iw2": {
            "cosine": [(-0.23940470331312666, 2000)],
            "sine": [(-0.9709198669465744, 2000)],
        },
        "q2.resonator.readout.iw3": {
            "cosine": [(0.23940470331312666, 2000)],
            "sine": [(0.9709198669465744, 2000)],
        },
        "c1.resonator.readout.iw1": {
            "cosine": [(0.016549154040551798, 2000)],
            "sine": [(0.9998630533730817, 2000)],
        },
        "c1.resonator.readout.iw2": {
            "cosine": [(-0.9998630533730817, 2000)],
            "sine": [(0.016549154040551798, 2000)],
        },
        "c1.resonator.readout.iw3": {
            "cosine": [(0.9998630533730817, 2000)],
            "sine": [(-0.016549154040551798, 2000)],
        },
    },
    "mixers": {},
    "oscillators": {},
}

loaded_config = {
    "controllers": {
        "con1": {
            "type": "opx1000",
            "fems": {
                "1": {
                    "type": "LF",
                    "analog_outputs": {
                        "1": {
                            "offset": 0.0,
                            "delay": 130,
                            "shareable": False,
                            "filter": {
                                "feedforward": [],
                                "exponential": [],
                                "high_pass": None,
                                "exponential_dc_gain": None,
                            },
                            "crosstalk": {},
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                        },
                        "2": {
                            "offset": 0.0,
                            "delay": 130,
                            "shareable": False,
                            "filter": {
                                "feedforward": [],
                                "exponential": [],
                                "high_pass": None,
                                "exponential_dc_gain": None,
                            },
                            "crosstalk": {},
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                        },
                        "3": {
                            "offset": 0.0,
                            "delay": 130,
                            "shareable": False,
                            "filter": {
                                "feedforward": [],
                                "exponential": [],
                                "high_pass": None,
                                "exponential_dc_gain": None,
                            },
                            "crosstalk": {},
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                        },
                    },
                },
                "6": {
                    "type": "MW",
                    "analog_outputs": {
                        "1": {
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": -2,
                            "band": 2,
                            "delay": 0,
                            "shareable": False,
                            "upconverters": {
                                "1": {
                                    "frequency": 6060000000.0,
                                },
                            },
                        },
                        "2": {
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 10,
                            "band": 1,
                            "delay": 15,
                            "shareable": False,
                            "upconverters": {
                                "1": {
                                    "frequency": 4600000000.0,
                                },
                            },
                        },
                        "3": {
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 13,
                            "band": 1,
                            "delay": 19,
                            "shareable": False,
                            "upconverters": {
                                "1": {
                                    "frequency": 4600000000.0,
                                },
                            },
                        },
                        "4": {
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 10,
                            "band": 1,
                            "delay": 0,
                            "shareable": False,
                            "upconverters": {
                                "1": {
                                    "frequency": 3500000000.0,
                                },
                            },
                        },
                    },
                    "analog_inputs": {
                        "1": {
                            "band": 2,
                            "shareable": False,
                            "gain_db": 0,
                            "sampling_rate": 1000000000.0,
                            "downconverter_frequency": 6060000000.0,
                        },
                    },
                },
            },
        },
    },
    "oscillators": {},
    "elements": {
        "q1.xy": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "x180_DragCosine": "q1.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "q1.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "q1.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "q1.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "q1.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "q1.xy.-y90_DragCosine.pulse",
                "x180_Square": "q1.xy.x180_Square.pulse",
                "x90_Square": "q1.xy.x90_Square.pulse",
                "-x90_Square": "q1.xy.-x90_Square.pulse",
                "y180_Square": "q1.xy.y180_Square.pulse",
                "y90_Square": "q1.xy.y90_Square.pulse",
                "-y90_Square": "q1.xy.-y90_Square.pulse",
                "x180": "q1.xy.x180_DragCosine.pulse",
                "x90": "q1.xy.x90_DragCosine.pulse",
                "-x90": "q1.xy.-x90_DragCosine.pulse",
                "y180": "q1.xy.y180_DragCosine.pulse",
                "y90": "q1.xy.y90_DragCosine.pulse",
                "-y90": "q1.xy.-y90_DragCosine.pulse",
                "saturation": "q1.xy.saturation.pulse",
                "EF_x180": "q1.xy.EF_x180.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "MWInput": {
                "port": ('con1', 6, 3),
                "upconverter": 1,
            },
            "intermediate_frequency": -22030337.61833399,
        },
        "q1.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "q1.z.const.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "singleInput": {
                "port": ('con1', 1, 2),
            },
        },
        "q1.resonator": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "readout": "q1.resonator.readout.pulse",
                "const": "q1.resonator.const.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "MWInput": {
                "port": ('con1', 6, 1),
                "upconverter": 1,
            },
            "MWOutput": {
                "port": ('con1', 6, 1),
            },
            "smearing": 0,
            "time_of_flight": 392,
            "intermediate_frequency": -232093075.0,
        },
        "q2.xy": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "x180_DragCosine": "q2.xy.x180_DragCosine.pulse",
                "x180_Long": "q2.xy.x180_Long.pulse",
                "x90_DragCosine": "q2.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "q2.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "q2.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "q2.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "q2.xy.-y90_DragCosine.pulse",
                "x180_Square": "q2.xy.x180_Square.pulse",
                "x90_Square": "q2.xy.x90_Square.pulse",
                "-x90_Square": "q2.xy.-x90_Square.pulse",
                "y180_Square": "q2.xy.y180_Square.pulse",
                "y90_Square": "q2.xy.y90_Square.pulse",
                "-y90_Square": "q2.xy.-y90_Square.pulse",
                "x180": "q2.xy.x180_DragCosine.pulse",
                "x90": "q2.xy.x90_DragCosine.pulse",
                "-x90": "q2.xy.-x90_DragCosine.pulse",
                "y180": "q2.xy.y180_DragCosine.pulse",
                "y90": "q2.xy.y90_DragCosine.pulse",
                "-y90": "q2.xy.-y90_DragCosine.pulse",
                "saturation": "q2.xy.saturation.pulse",
                "EF_x180": "q2.xy.EF_x180.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "MWInput": {
                "port": ('con1', 6, 2),
                "upconverter": 1,
            },
            "intermediate_frequency": 199122277.7482808,
        },
        "q2.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "q2.z.const.pulse",
                "Cz_unipolar": "q2.z.Cz_unipolar.pulse",
                "Cz_flattop": "q2.z.Cz_flattop.pulse",
                "Cz_unipolar.flux_pulse_control_q1_q2": "q2.z.Cz_unipolar.flux_pulse_control_q1_q2.pulse",
                "Cz_flattop.flux_pulse_control_q1_q2": "q2.z.Cz_flattop.flux_pulse_control_q1_q2.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "singleInput": {
                "port": ('con1', 1, 1),
            },
        },
        "q2.resonator": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "readout": "q2.resonator.readout.pulse",
                "const": "q2.resonator.const.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "MWInput": {
                "port": ('con1', 6, 1),
                "upconverter": 1,
            },
            "MWOutput": {
                "port": ('con1', 6, 1),
            },
            "smearing": 0,
            "time_of_flight": 392,
            "intermediate_frequency": -150129053.0,
        },
        "c1.xy": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "x180_DragCosine": "c1.xy.x180_DragCosine.pulse",
                "x180_Long": "c1.xy.x180_Long.pulse",
                "x90_DragCosine": "c1.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "c1.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "c1.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "c1.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "c1.xy.-y90_DragCosine.pulse",
                "x180_Square": "c1.xy.x180_Square.pulse",
                "x90_Square": "c1.xy.x90_Square.pulse",
                "-x90_Square": "c1.xy.-x90_Square.pulse",
                "y180_Square": "c1.xy.y180_Square.pulse",
                "y90_Square": "c1.xy.y90_Square.pulse",
                "-y90_Square": "c1.xy.-y90_Square.pulse",
                "x180": "c1.xy.x180_Long.pulse",
                "x90": "c1.xy.x90_DragCosine.pulse",
                "-x90": "c1.xy.-x90_DragCosine.pulse",
                "y180": "c1.xy.y180_DragCosine.pulse",
                "y90": "c1.xy.y90_DragCosine.pulse",
                "-y90": "c1.xy.-y90_DragCosine.pulse",
                "saturation": "c1.xy.saturation.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "MWInput": {
                "port": ('con1', 6, 4),
                "upconverter": 1,
            },
            "intermediate_frequency": 47200000.0,
        },
        "c1.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "c1.z.const.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "singleInput": {
                "port": ('con1', 1, 3),
            },
        },
        "c1.resonator": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "readout": "c1.resonator.readout.pulse",
                "const": "c1.resonator.const.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "MWInput": {
                "port": ('con1', 6, 1),
                "upconverter": 1,
            },
            "MWOutput": {
                "port": ('con1', 6, 1),
            },
            "smearing": 0,
            "time_of_flight": 392,
            "intermediate_frequency": -67369238.0,
        },
        "coupler_q1_q2": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "coupler_q1_q2.const.pulse",
                "Cz_unipolar.coupler_flux_pulse_q1_q2": "coupler_q1_q2.Cz_unipolar.coupler_flux_pulse_q1_q2.pulse",
                "Cz_flattop.coupler_flux_pulse_q1_q2": "coupler_q1_q2.Cz_flattop.coupler_flux_pulse_q1_q2.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "singleInput": {
                "port": ('con1', 1, 3),
            },
        },
    },
    "pulses": {
        "const_pulse": {
            "length": 1000,
            "waveforms": {
                "I": "const_wf",
                "Q": "zero_wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q1.xy.x180_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "q1.xy.x180_DragCosine.wf.I",
                "Q": "q1.xy.x180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.x90_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "q1.xy.x90_DragCosine.wf.I",
                "Q": "q1.xy.x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.-x90_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "q1.xy.-x90_DragCosine.wf.I",
                "Q": "q1.xy.-x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.y180_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "q1.xy.y180_DragCosine.wf.I",
                "Q": "q1.xy.y180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.y90_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "q1.xy.y90_DragCosine.wf.I",
                "Q": "q1.xy.y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.-y90_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "q1.xy.-y90_DragCosine.wf.I",
                "Q": "q1.xy.-y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.x180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q1.xy.x180_Square.wf.I",
                "Q": "q1.xy.x180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q1.xy.x90_Square.wf.I",
                "Q": "q1.xy.x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.-x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q1.xy.-x90_Square.wf.I",
                "Q": "q1.xy.-x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.y180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q1.xy.y180_Square.wf.I",
                "Q": "q1.xy.y180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q1.xy.y90_Square.wf.I",
                "Q": "q1.xy.y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.-y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q1.xy.-y90_Square.wf.I",
                "Q": "q1.xy.-y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.saturation.pulse": {
            "length": 20000,
            "waveforms": {
                "I": "q1.xy.saturation.wf.I",
                "Q": "q1.xy.saturation.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.EF_x180.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q1.xy.EF_x180.wf.I",
                "Q": "q1.xy.EF_x180.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.z.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "q1.z.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q1.resonator.readout.pulse": {
            "length": 1200,
            "waveforms": {
                "I": "q1.resonator.readout.wf.I",
                "Q": "q1.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "q1.resonator.readout.iw1",
                "iw2": "q1.resonator.readout.iw2",
                "iw3": "q1.resonator.readout.iw3",
            },
            "operation": "measurement",
            "digital_marker": "ON",
        },
        "q1.resonator.const.pulse": {
            "length": 100,
            "waveforms": {
                "I": "q1.resonator.const.wf.I",
                "Q": "q1.resonator.const.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q2.xy.x180_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "q2.xy.x180_DragCosine.wf.I",
                "Q": "q2.xy.x180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.x180_Long.pulse": {
            "length": 128,
            "waveforms": {
                "I": "q2.xy.x180_Long.wf.I",
                "Q": "q2.xy.x180_Long.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.x90_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "q2.xy.x90_DragCosine.wf.I",
                "Q": "q2.xy.x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.-x90_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "q2.xy.-x90_DragCosine.wf.I",
                "Q": "q2.xy.-x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.y180_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "q2.xy.y180_DragCosine.wf.I",
                "Q": "q2.xy.y180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.y90_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "q2.xy.y90_DragCosine.wf.I",
                "Q": "q2.xy.y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.-y90_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "q2.xy.-y90_DragCosine.wf.I",
                "Q": "q2.xy.-y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.x180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q2.xy.x180_Square.wf.I",
                "Q": "q2.xy.x180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q2.xy.x90_Square.wf.I",
                "Q": "q2.xy.x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.-x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q2.xy.-x90_Square.wf.I",
                "Q": "q2.xy.-x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.y180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q2.xy.y180_Square.wf.I",
                "Q": "q2.xy.y180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q2.xy.y90_Square.wf.I",
                "Q": "q2.xy.y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.-y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q2.xy.-y90_Square.wf.I",
                "Q": "q2.xy.-y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.saturation.pulse": {
            "length": 20000,
            "waveforms": {
                "I": "q2.xy.saturation.wf.I",
                "Q": "q2.xy.saturation.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.EF_x180.pulse": {
            "length": 32,
            "waveforms": {
                "I": "q2.xy.EF_x180.wf.I",
                "Q": "q2.xy.EF_x180.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.z.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "q2.z.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q2.z.Cz_unipolar.pulse": {
            "length": 48,
            "waveforms": {
                "single": "q2.z.Cz_unipolar.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q2.z.Cz_flattop.pulse": {
            "length": 80,
            "waveforms": {
                "single": "q2.z.Cz_flattop.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q2.resonator.readout.pulse": {
            "length": 2000,
            "waveforms": {
                "I": "q2.resonator.readout.wf.I",
                "Q": "q2.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "q2.resonator.readout.iw1",
                "iw2": "q2.resonator.readout.iw2",
                "iw3": "q2.resonator.readout.iw3",
            },
            "operation": "measurement",
            "digital_marker": "ON",
        },
        "q2.resonator.const.pulse": {
            "length": 100,
            "waveforms": {
                "I": "q2.resonator.const.wf.I",
                "Q": "q2.resonator.const.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "c1.xy.x180_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "c1.xy.x180_DragCosine.wf.I",
                "Q": "c1.xy.x180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "c1.xy.x180_Long.pulse": {
            "length": 128,
            "waveforms": {
                "I": "c1.xy.x180_Long.wf.I",
                "Q": "c1.xy.x180_Long.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "c1.xy.x90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "c1.xy.x90_DragCosine.wf.I",
                "Q": "c1.xy.x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "c1.xy.-x90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "c1.xy.-x90_DragCosine.wf.I",
                "Q": "c1.xy.-x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "c1.xy.y180_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "c1.xy.y180_DragCosine.wf.I",
                "Q": "c1.xy.y180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "c1.xy.y90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "c1.xy.y90_DragCosine.wf.I",
                "Q": "c1.xy.y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "c1.xy.-y90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "c1.xy.-y90_DragCosine.wf.I",
                "Q": "c1.xy.-y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "c1.xy.x180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "c1.xy.x180_Square.wf.I",
                "Q": "c1.xy.x180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "c1.xy.x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "c1.xy.x90_Square.wf.I",
                "Q": "c1.xy.x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "c1.xy.-x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "c1.xy.-x90_Square.wf.I",
                "Q": "c1.xy.-x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "c1.xy.y180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "c1.xy.y180_Square.wf.I",
                "Q": "c1.xy.y180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "c1.xy.y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "c1.xy.y90_Square.wf.I",
                "Q": "c1.xy.y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "c1.xy.-y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "c1.xy.-y90_Square.wf.I",
                "Q": "c1.xy.-y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "c1.xy.saturation.pulse": {
            "length": 20000,
            "waveforms": {
                "I": "c1.xy.saturation.wf.I",
                "Q": "c1.xy.saturation.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "c1.z.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "c1.z.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "c1.resonator.readout.pulse": {
            "length": 2000,
            "waveforms": {
                "I": "c1.resonator.readout.wf.I",
                "Q": "c1.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "c1.resonator.readout.iw1",
                "iw2": "c1.resonator.readout.iw2",
                "iw3": "c1.resonator.readout.iw3",
            },
            "operation": "measurement",
            "digital_marker": "ON",
        },
        "c1.resonator.const.pulse": {
            "length": 100,
            "waveforms": {
                "I": "c1.resonator.const.wf.I",
                "Q": "c1.resonator.const.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q1_q2.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "coupler_q1_q2.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q2.z.Cz_unipolar.flux_pulse_control_q1_q2.pulse": {
            "length": 48,
            "waveforms": {
                "single": "q2.z.Cz_unipolar.flux_pulse_control_q1_q2.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q1_q2.Cz_unipolar.coupler_flux_pulse_q1_q2.pulse": {
            "length": 48,
            "waveforms": {
                "single": "coupler_q1_q2.Cz_unipolar.coupler_flux_pulse_q1_q2.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q2.z.Cz_flattop.flux_pulse_control_q1_q2.pulse": {
            "length": 80,
            "waveforms": {
                "single": "q2.z.Cz_flattop.flux_pulse_control_q1_q2.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q1_q2.Cz_flattop.coupler_flux_pulse_q1_q2.pulse": {
            "length": 80,
            "waveforms": {
                "single": "coupler_q1_q2.Cz_flattop.coupler_flux_pulse_q1_q2.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
    },
    "waveforms": {
        "zero_wf": {
            "type": "constant",
            "sample": 0.0,
        },
        "const_wf": {
            "type": "constant",
            "sample": 0.1,
        },
        "q1.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0006173280808725863, 0.0024440388393264516, 0.0054053465222810935, 0.009380014845257467, 0.01420532042010923, 0.01968371466965261, 0.025590911489627312, 0.03168506954815742, 0.03771669329829353, 0.0434388473550194, 0.04861726605892807, 0.053039944339843215, 0.056525817229301493, 0.05893217268163582] + [0.060160494221892874] * 2 + [0.05893217268163582, 0.056525817229301493, 0.05303994433984322, 0.04861726605892808, 0.04343884735501939, 0.03771669329829356, 0.031685069548157446, 0.025590911489627312, 0.01968371466965261, 0.01420532042010923, 0.00938001484525747, 0.0054053465222810935, 0.0024440388393264516, 0.000617328080872593, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0008857106768594052, -0.0017351602545416006, -0.0025135721675301897, -0.003189078140647903, -0.0037340228799867207, -0.004126096283889822, -0.004349246821134746, -0.00439433868250761, -0.004259525801905741, -0.003950327434500819, -0.0034794021977836704, -0.0028660298262768564, -0.0021353218569380625, -0.0013171935598865596, -0.0004451392037296311, 0.00044513920372962805, 0.0013171935598865604, 0.0021353218569380617, 0.0028660298262768555, 0.0034794021977836696, 0.00395032743450082, 0.00425952580190574, 0.00439433868250761, 0.004349246821134746, 0.004126096283889822, 0.0037340228799867207, 0.0031890781406479036, 0.0025135721675301906, 0.0017351602545416013, 0.0008857106768594098, 1.0776857623282754e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0003064061033179365, 0.0012131159036326198, 0.002683114638042165, 0.004656392725021253, 0.0070523961299658695, 0.00977331557227668, 0.012708080222675404, 0.015736892767100233, 0.01873612060626968, 0.021583343420916796, 0.024162350955529363, 0.026367886918357612, 0.0281099452784387, 0.029317443536663084, 0.029941123009334353, 0.029955557748355236, 0.02936019014101973, 0.028179350995915564, 0.026461263404612387, 0.024276071176488577, 0.021712972480277166, 0.018876575863234487, 0.015882627564812044, 0.012853284699894512, 0.009912127409564313, 0.007179113709210926, 0.004765683071446945, 0.002770208671783026, 0.001273983949392676, 0.0003379072782645619, 3.8924160035476876e-20],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0005483549694920474, -0.001077707822663543, -0.0015697409725369505, -0.0020074816802543825, -0.002375914788121007, -0.0026625261347110415, -0.002857757315564628, -0.0029553555214895007, -0.00295260582501773, -0.0028504373720345637, -0.0026533993323578833, -0.0023695070213745274, -0.002009963169723411, -0.001588763733216842, -0.00112220174838982, -0.0006282864069314518, -0.0001260976152874759, 0.000364901287275493, 0.0008255683468733768, 0.001237780250766896, 0.0015850970026626266, 0.0018533699604141884, 0.0020312709959735076, 0.0021107231566632797, 0.0020872165459982985, 0.001959997093005342, 0.0017321203101146963, 0.0014103669013825848, 0.0010050220113840408, 0.000529524828514157, 6.653677400775176e-19],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.00030743918313297445, -0.0012171701362518636, -0.002691948367247132, -0.004671396282062746, -0.007074475050490907, -0.009802802345402392, -0.01274468012677868, -0.01577966757259427, -0.01878351194022158, -0.02163323548831323, -0.024212170199982036, -0.02641473418512033, -0.02815075421538626, -0.029349157426767046] + [-0.029960881050982366] * 2 + [-0.029349157426767046, -0.02815075421538626, -0.026414734185120334, -0.02421217019998204, -0.021633235488313227, -0.018783511940221594, -0.015779667572594285, -0.01274468012677868, -0.009802802345402392, -0.007074475050490907, -0.004671396282062748, -0.002691948367247132, -0.001217170136251864, -0.0003074391831329779, -8.175842214353895e-35],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0005486828481647913, 0.001074902556058219, 0.0015571156270109356, 0.0019755802012405062, 0.002313164289910474, 0.002556047160766243, 0.002694285160537453, 0.002722218809271194, 0.0026387045001062446, 0.002447161318672331, 0.002155431065322727, 0.0017754572109064126, 0.0013227959296682255, 0.0008159792276474994, 0.0002757562401733974, -0.00027575624017338817, -0.0008159792276474926, -0.0013227959296682181, -0.0017754572109064057, -0.0021554310653227205, -0.002447161318672326, -0.00263870450010624, -0.0027222188092711906, -0.0026942851605374495, -0.0025560471607662403, -0.0023131642899104727, -0.001975580201240506, -0.0015571156270109351, -0.001074902556058219, -0.0005486828481647942, -6.676081805828616e-19],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0008857106768594052, 0.0017351602545416008, 0.00251357216753019, 0.0031890781406479036, 0.0037340228799867215, 0.004126096283889823, 0.004349246821134748, 0.004394338682507611, 0.004259525801905743, 0.003950327434500822, 0.0034794021977836735, 0.0028660298262768594, 0.002135321856938066, 0.0013171935598865632, 0.0004451392037296348, -0.00044513920372962437, -0.0013171935598865567, -0.002135321856938058, -0.0028660298262768525, -0.0034794021977836665, -0.003950327434500817, -0.004259525801905737, -0.004394338682507608, -0.004349246821134745, -0.0041260962838898215, -0.00373402287998672, -0.003189078140647903, -0.00251357216753019, -0.001735160254541601, -0.0008857106768594098, -1.0776857623282754e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0006173280808725862, 0.0024440388393264516, 0.0054053465222810935, 0.009380014845257467, 0.01420532042010923, 0.01968371466965261, 0.025590911489627312, 0.03168506954815742, 0.03771669329829353, 0.0434388473550194, 0.04861726605892807, 0.053039944339843215, 0.056525817229301493, 0.05893217268163582] + [0.060160494221892874] * 2 + [0.05893217268163582, 0.056525817229301493, 0.05303994433984322, 0.04861726605892808, 0.04343884735501939, 0.03771669329829356, 0.031685069548157446, 0.025590911489627312, 0.01968371466965261, 0.01420532042010923, 0.00938001484525747, 0.0054053465222810935, 0.0024440388393264516, 0.0006173280808725931, 6.598922096609988e-35],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0005483549694920474, 0.001077707822663543, 0.0015697409725369507, 0.002007481680254383, 0.0023759147881210075, 0.002662526134711042, 0.002857757315564629, 0.0029553555214895015, 0.0029526058250177313, 0.002850437372034565, 0.0026533993323578846, 0.002369507021374529, 0.002009963169723413, 0.0015887637332168438, 0.0011222017483898217, 0.0006282864069314536, 0.0001260976152874777, -0.00036490128727549126, -0.0008255683468733751, -0.0012377802507668944, -0.0015850970026626253, -0.0018533699604141873, -0.0020312709959735067, -0.002110723156663279, -0.002087216545998298, -0.0019599970930053415, -0.001732120310114696, -0.0014103669013825846, -0.0010050220113840408, -0.000529524828514157, -6.653677400775176e-19],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.00030640610331793646, 0.0012131159036326198, 0.002683114638042165, 0.004656392725021253, 0.0070523961299658695, 0.00977331557227668, 0.012708080222675404, 0.015736892767100233, 0.01873612060626968, 0.021583343420916796, 0.024162350955529363, 0.026367886918357612, 0.0281099452784387, 0.029317443536663084, 0.029941123009334353, 0.029955557748355236, 0.02936019014101973, 0.028179350995915564, 0.026461263404612387, 0.024276071176488577, 0.021712972480277166, 0.018876575863234487, 0.015882627564812044, 0.012853284699894512, 0.009912127409564313, 0.007179113709210926, 0.004765683071446945, 0.002770208671783026, 0.001273983949392676, 0.000337907278264562, 3.892416003547692e-20],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0005483549694920474, -0.001077707822663543, -0.0015697409725369503, -0.002007481680254382, -0.0023759147881210066, -0.002662526134711041, -0.002857757315564627, -0.0029553555214895, -0.0029526058250177287, -0.0028504373720345624, -0.002653399332357882, -0.0023695070213745257, -0.0020099631697234094, -0.0015887637332168403, -0.0011222017483898182, -0.0006282864069314499, -0.0001260976152874741, 0.00036490128727549473, 0.0008255683468733784, 0.0012377802507668974, 0.0015850970026626279, 0.0018533699604141895, 0.0020312709959735085, 0.0021107231566632806, 0.002087216545998299, 0.0019599970930053424, 0.0017321203101146965, 0.001410366901382585, 0.0010050220113840408, 0.000529524828514157, 6.653677400775176e-19],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.00030640610331793657, -0.0012131159036326198, -0.002683114638042165, -0.004656392725021253, -0.0070523961299658695, -0.00977331557227668, -0.012708080222675404, -0.015736892767100233, -0.01873612060626968, -0.021583343420916796, -0.024162350955529363, -0.026367886918357612, -0.0281099452784387, -0.029317443536663084, -0.029941123009334353, -0.029955557748355236, -0.02936019014101973, -0.028179350995915564, -0.026461263404612387, -0.024276071176488577, -0.021712972480277166, -0.018876575863234487, -0.015882627564812044, -0.012853284699894512, -0.009912127409564313, -0.007179113709210926, -0.004765683071446945, -0.002770208671783026, -0.001273983949392676, -0.00033790727826456187, -3.8924160035476834e-20],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q1.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q1.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05,
        },
        "q1.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q1.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "q1.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q1.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "q1.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q1.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "q1.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q1.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "q1.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q1.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.05,
        },
        "q1.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q1.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.007854789785471543, 0.030060994629453126, 0.06277895997738259, 0.10035145238721895, 0.1362818465859237, 0.16435745099915694] + [0.17972373856501173] * 2 + [0.16435745099915694, 0.13628184658592377, 0.10035145238721903, 0.06277895997738261, 0.030060994629453105, 0.007854789785471532, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.z.const.wf": {
            "type": "constant",
            "sample": 2.5,
        },
        "q1.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.007349054643675365,
        },
        "q1.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q1.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "q1.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.003056738220897418, 0.012101809661674639, 0.02676490803479547, 0.04644572437002343, 0.07033852373419659, 0.09746513211406094, 0.1267149829753037, 0.15689058398695405, 0.18675654249811238, 0.2150901426582309, 0.24073140354769748, 0.26263056893297737, 0.27989108441015437, 0.2918063024475684] + [0.2978884126187891] * 2 + [0.2918063024475684, 0.27989108441015437, 0.2626305689329774, 0.2407314035476975, 0.21509014265823087, 0.18675654249811252, 0.15689058398695419, 0.1267149829753037, 0.09746513211406094, 0.07033852373419659, 0.04644572437002344, 0.02676490803479547, 0.012101809661674639, 0.0030567382208974508, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0014143661568459074, -0.002770827997049576, -0.004013851813495569, -0.00509254806508949, -0.005962754800549657, -0.00658884565388138, -0.006945188391986707, -0.00701719430129911, -0.0068019154514310076, -0.006308165383790187, -0.005556158284159039, -0.004576681411553536, -0.003409834664221801, -0.0021033888850982637, -0.0007108301178811064, 0.0007108301178811018, 0.002103388885098265, 0.003409834664221799, 0.004576681411553534, 0.005556158284159038, 0.006308165383790188, 0.006801915451431006, 0.00701719430129911, 0.006945188391986707, 0.00658884565388138, 0.005962754800549658, 0.005092548065089491, 0.00401385181349557, 0.002770827997049577, 0.0014143661568459147, 1.7209257038161997e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.x180_Long.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 5.283897927734285e-05, 0.00021122661127277061, 0.00047477529490954723, 0.0008428400823955668, 0.0013145202575186943, 0.0018886615398502511, 0.002563858909462676, 0.003338460045248709, 0.004210569368428022, 0.005178052681346141, 0.006238542390213756, 0.007389443299005441, 0.008627938960339213, 0.009950998567795235, 0.011355384372807014, 0.012837659607974696, 0.01439419689741075, 0.01602118713353669, 0.017714648798607548, 0.019470437708153033, 0.02128425715249134, 0.02315166841149776, 0.02506810161689665, 0.027028866935494864, 0.029029166045989444, 0.031064103881263926, 0.03312870060743784, 0.03521790381035475, 0.03732660085968623, 0.03944963142039508, 0.0415818000809399, 0.043717889067317675, 0.045852671011831145, 0.047980921745333506, 0.05009743308164579, 0.05219702556286109, 0.05427456113434606, 0.05632495571842158, 0.05834319165595277, 0.060324329985401694, 0.06226352252929393, 0.0641560237585212, 0.06599720240544636, 0.06778255279739138, 0.06950770588277334, 0.07116843992290586, 0.07276069082330114, 0.07428056207918998, 0.07572433431092221, 0.07708847436591172, 0.07836964396485298, 0.07956470787104991, 0.08067074156286543, 0.08168503839051593, 0.08260511619969697, 0.08342872340583064, 0.08415384450407053, 0.08477870500157958, 0.08530177576001158, 0.08572177673756892, 0.08603768012147936, 0.08624871284322605] + [0.0863543584703758] * 2 + [0.08624871284322605, 0.08603768012147936, 0.08572177673756894, 0.08530177576001158, 0.08477870500157957, 0.08415384450407053, 0.08342872340583066, 0.08260511619969696, 0.08168503839051594, 0.08067074156286543, 0.07956470787104991, 0.07836964396485299, 0.07708847436591171, 0.07572433431092222, 0.07428056207918998, 0.07276069082330112, 0.07116843992290589, 0.06950770588277336, 0.06778255279739138, 0.06599720240544638, 0.06415602375852118, 0.062263522529293944, 0.060324329985401715, 0.05834319165595279, 0.0563249557184216, 0.05427456113434608, 0.05219702556286109, 0.05009743308164583, 0.047980921745333534, 0.04585267101183113, 0.043717889067317696, 0.04158180008093992, 0.0394496314203951, 0.03732660085968624, 0.03521790381035475, 0.03312870060743784, 0.031064103881263905, 0.02902916604598946, 0.02702886693549488, 0.02506810161689663, 0.02315166841149777, 0.02128425715249134, 0.01947043770815303, 0.017714648798607575, 0.01602118713353671, 0.014394196897410741, 0.012837659607974707, 0.01135538437280703, 0.00995099856779524, 0.008627938960339232, 0.007389443299005437, 0.006238542390213752, 0.005178052681346155, 0.004210569368428028, 0.0033384600452487138, 0.0025638589094626953, 0.0018886615398502511, 0.0013145202575186895, 0.0008428400823955717, 0.00047477529490955205, 0.0002112266112727754, 5.283897927734285e-05, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.x180_Long.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -2.8310817294644954e-05, -5.655235327528443e-05, -8.46554961708979e-05, -0.00011255147288153462, -0.00014017201727761362, -0.00016744953725858226, -0.0001943172801621128, -0.00022070949611905293, -0.00024656159895437336, -0.00027181032424036106, -0.0002963938841152749, -0.0003202521184885977, -0.0003433266422628583, -0.0003655609882117489, -0.0003869007451648904, -0.00040729369116108407, -0.00042668992124420094, -0.0004450419695889727, -0.0004623049256578199, -0.00047843654410446303, -0.000493397348155364, -0.0005071507262160061, -0.0005196630214656012, -0.0005309036142209698, -0.0005408449968680365, -0.0005494628411775721, -0.0005567360578404459, -0.0005626468480767005, -0.0005671807471921478, -0.0005703266599759002, -0.0005720768878522105, -0.000572427147720177, -0.0005713765824352095, -0.000568927762906605, -0.0005650866818061022, -0.0005598627389028097, -0.0005532687180603956, -0.0005453207559528313, -0.0005360383025752457, -0.0005254440736465281, -0.0005135639950201578, -0.0005004271392392957, -0.00048606565439140005, -0.00047051468543646865, -0.0004538122882014333, -0.00043599933625117356, -0.0004171194208640539, -0.0003972187443567627, -0.00037634600701949843, -0.00035455228793820106, -0.0003318909199954683, -0.0003084173593560545, -0.0002841890497563442, -0.00025926528192990125, -0.00023370704851310897, -0.0002075768947859702, -0.00018093876561332326, -0.00015385784896104372, -0.00012640041637016996, -9.863366077933385e-05, -7.062553209238195e-05, -4.2444570893575756e-05, -1.4159740717293008e-05, 1.415974071729312e-05, 4.2444570893575356e-05, 7.062553209238209e-05, 9.863366077933346e-05, 0.0001264004163701698, 0.00015385784896104385, 0.00018093876561332286, 0.0002075768947859701, 0.00023370704851310907, 0.00025926528192990087, 0.0002841890497563441, 0.00030841735935605453, 0.00033189091999546804, 0.0003545522879382011, 0.0003763460070194983, 0.00039721874435676266, 0.000417119420864054, 0.0004359993362511733, 0.00045381228820143326, 0.0004705146854364686, 0.0004860656543914, 0.0005004271392392957, 0.0005135639950201576, 0.000525444073646528, 0.0005360383025752456, 0.0005453207559528312, 0.0005532687180603956, 0.0005598627389028097, 0.000565086681806102, 0.000568927762906605, 0.0005713765824352095, 0.000572427147720177, 0.0005720768878522105, 0.0005703266599759002, 0.0005671807471921478, 0.0005626468480767005, 0.0005567360578404459, 0.0005494628411775721, 0.0005408449968680366, 0.0005309036142209699, 0.0005196630214656011, 0.0005071507262160062, 0.000493397348155364, 0.00047843654410446303, 0.00046230492565782017, 0.0004450419695889729, 0.0004266899212442007, 0.00040729369116108424, 0.0003869007451648906, 0.00036556098821174894, 0.00034332664226285865, 0.00032025211848859766, 0.00029639388411527485, 0.00027181032424036133, 0.0002465615989543736, 0.00022070949611905307, 0.0001943172801621134, 0.00016744953725858228, 0.00014017201727761357, 0.00011255147288153501, 8.465549617089824e-05, 5.65523532752847e-05, 2.8310817294644653e-05, 1.4021493968622868e-19],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0015292892188787611, 0.006054547595210294, 0.0133905105193723, 0.023236842807310535, 0.035190434458261914, 0.04876190402554907, 0.06339563395050733, 0.07849251761061862, 0.09343448681502947, 0.10760981558890452, 0.12043816430671082, 0.1313943388646745, 0.14002979218909664, 0.14599098780625266] + [0.14903387366716553] * 2 + [0.14599098780625266, 0.14002979218909664, 0.13139433886467453, 0.12043816430671084, 0.1076098155889045, 0.09343448681502954, 0.07849251761061869, 0.06339563395050733, 0.04876190402554907, 0.035190434458261914, 0.023236842807310545, 0.0133905105193723, 0.006054547595210294, 0.0015292892188787776, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0003265886845946167, -0.0006398067900693889, -0.0009268311303846143, -0.0011759108953238598, -0.0013768487300448828, -0.0015214182159850114, -0.001603700661403664, -0.0016203274133176236, -0.0015706177705499724, -0.001456606851716253, -0.0012829622772292327, -0.0010567930763707798, -0.0007873586427977934, -0.0004856896538779531, -0.00016413647346223882, 0.0001641364734622377, 0.00048568965387795333, 0.000787358642797793, 0.0010567930763707794, 0.0012829622772292323, 0.0014566068517162531, 0.001570617770549972, 0.0016203274133176236, 0.001603700661403664, 0.0015214182159850114, 0.001376848730044883, 0.0011759108953238598, 0.0009268311303846145, 0.0006398067900693891, 0.00032658868459461846, 3.9737578502850895e-19],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0015292892188787611, -0.006054547595210294, -0.0133905105193723, -0.023236842807310535, -0.035190434458261914, -0.04876190402554907, -0.06339563395050733, -0.07849251761061862, -0.09343448681502947, -0.10760981558890452, -0.12043816430671082, -0.1313943388646745, -0.14002979218909664, -0.14599098780625266] + [-0.14903387366716553] * 2 + [-0.14599098780625266, -0.14002979218909664, -0.13139433886467453, -0.12043816430671084, -0.1076098155889045, -0.09343448681502954, -0.07849251761061869, -0.06339563395050733, -0.04876190402554907, -0.035190434458261914, -0.023236842807310545, -0.0133905105193723, -0.006054547595210294, -0.0015292892188787776, -4.866449831938302e-35],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0003265886845946169, 0.0006398067900693896, 0.0009268311303846159, 0.0011759108953238626, 0.0013768487300448871, 0.0015214182159850175, 0.0016037006614036719, 0.0016203274133176331, 0.001570617770549984, 0.0014566068517162661, 0.0012829622772292475, 0.0010567930763707959, 0.0007873586427978106, 0.000485689653877971, 0.00016413647346225706, -0.00016413647346221946, -0.00048568965387793544, -0.0007873586427977759, -0.0010567930763707633, -0.0012829622772292176, -0.00145660685171624, -0.0015706177705499605, -0.001620327413317614, -0.0016037006614036562, -0.0015214182159850053, -0.0013768487300448787, -0.001175910895323857, -0.0009268311303846129, -0.0006398067900693883, -0.0003265886845946183, -3.9737578502850895e-19],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0014143661568459076, 0.002770827997049577, 0.004013851813495571, 0.0050925480650894925, 0.005962754800549661, 0.006588845653881386, 0.0069451883919867145, 0.007017194301299119, 0.006801915451431019, 0.0063081653837902, 0.005556158284159053, 0.0045766814115535525, 0.003409834664221818, 0.0021033888850982814, 0.0007108301178811246, -0.0007108301178810836, -0.002103388885098247, -0.003409834664221782, -0.004576681411553518, -0.005556158284159023, -0.006308165383790175, -0.0068019154514309945, -0.0070171943012991, -0.006945188391986699, -0.006588845653881374, -0.005962754800549653, -0.005092548065089488, -0.004013851813495568, -0.002770827997049576, -0.0014143661568459145, -1.7209257038161997e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.003056738220897418, 0.012101809661674639, 0.02676490803479547, 0.04644572437002343, 0.07033852373419659, 0.09746513211406094, 0.1267149829753037, 0.15689058398695405, 0.18675654249811238, 0.2150901426582309, 0.24073140354769748, 0.26263056893297737, 0.27989108441015437, 0.2918063024475684] + [0.2978884126187891] * 2 + [0.2918063024475684, 0.27989108441015437, 0.2626305689329774, 0.2407314035476975, 0.21509014265823087, 0.18675654249811252, 0.15689058398695419, 0.1267149829753037, 0.09746513211406094, 0.07033852373419659, 0.04644572437002344, 0.02676490803479547, 0.012101809661674639, 0.0030567382208974508, 1.0537630773744575e-34],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.00032658868459461683, 0.0006398067900693892, 0.0009268311303846151, 0.0011759108953238613, 0.001376848730044885, 0.0015214182159850145, 0.001603700661403668, 0.0016203274133176284, 0.001570617770549978, 0.0014566068517162594, 0.0012829622772292401, 0.0010567930763707878, 0.000787358642797802, 0.00048568965387796206, 0.00016413647346224795, -0.00016413647346222857, -0.0004856896538779444, -0.0007873586427977844, -0.0010567930763707714, -0.001282962277229225, -0.0014566068517162466, -0.0015706177705499663, -0.0016203274133176188, -0.0016037006614036601, -0.0015214182159850084, -0.0013768487300448808, -0.0011759108953238583, -0.0009268311303846136, -0.0006398067900693887, -0.00032658868459461835, -3.9737578502850895e-19],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0015292892188787611, 0.006054547595210294, 0.0133905105193723, 0.023236842807310535, 0.035190434458261914, 0.04876190402554907, 0.06339563395050733, 0.07849251761061862, 0.09343448681502947, 0.10760981558890452, 0.12043816430671082, 0.1313943388646745, 0.14002979218909664, 0.14599098780625266] + [0.14903387366716553] * 2 + [0.14599098780625266, 0.14002979218909664, 0.13139433886467453, 0.12043816430671084, 0.1076098155889045, 0.09343448681502954, 0.07849251761061869, 0.06339563395050733, 0.04876190402554907, 0.035190434458261914, 0.023236842807310545, 0.0133905105193723, 0.006054547595210294, 0.0015292892188787776, 2.433224915969151e-35],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0003265886845946166, -0.0006398067900693885, -0.0009268311303846134, -0.0011759108953238583, -0.0013768487300448806, -0.0015214182159850084, -0.0016037006614036601, -0.0016203274133176188, -0.0015706177705499668, -0.0014566068517162464, -0.0012829622772292254, -0.0010567930763707718, -0.0007873586427977849, -0.00048568965387794417, -0.00016413647346222968, 0.00016413647346224684, 0.0004856896538779623, 0.0007873586427978016, 0.0010567930763707874, 0.0012829622772292397, 0.0014566068517162596, 0.0015706177705499776, 0.0016203274133176284, 0.001603700661403668, 0.0015214182159850145, 0.0013768487300448852, 0.0011759108953238613, 0.0009268311303846154, 0.0006398067900693894, 0.00032658868459461857, 3.9737578502850895e-19],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0015292892188787611, -0.006054547595210294, -0.0133905105193723, -0.023236842807310535, -0.035190434458261914, -0.04876190402554907, -0.06339563395050733, -0.07849251761061862, -0.09343448681502947, -0.10760981558890452, -0.12043816430671082, -0.1313943388646745, -0.14002979218909664, -0.14599098780625266] + [-0.14903387366716553] * 2 + [-0.14599098780625266, -0.14002979218909664, -0.13139433886467453, -0.12043816430671084, -0.1076098155889045, -0.09343448681502954, -0.07849251761061869, -0.06339563395050733, -0.04876190402554907, -0.035190434458261914, -0.023236842807310545, -0.0133905105193723, -0.006054547595210294, -0.0015292892188787776, 2.433224915969151e-35],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q2.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05,
        },
        "q2.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "q2.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "q2.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q2.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "q2.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q2.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "q2.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q2.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.5,
        },
        "q2.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.002445685357886733, 0.009682614785639218, 0.021414507542182167, 0.03716105855962382, 0.056277602189168996, 0.07798150488852554, 0.10138420633108153, 0.12552759717007808, 0.14942324414783864, 0.17209285666947452, 0.19260833813478337, 0.21012978232210897, 0.22393985924773974, 0.23347318273972528] + [0.23833945741418575] * 2 + [0.23347318273972528, 0.22393985924773974, 0.210129782322109, 0.1926083381347834, 0.1720928566694745, 0.14942324414783875, 0.1255275971700782, 0.10138420633108153, 0.07798150488852554, 0.056277602189168996, 0.03716105855962384, 0.021414507542182167, 0.009682614785639218, 0.0024456853578867593, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.004368349936092402, -0.008557859112541859, -0.012397008531418102, -0.01572862296443143, -0.01841630572524621, -0.020350022765845365, -0.021450607483384662, -0.02167300181021483, -0.021008100896465216, -0.01948312586366727, -0.017160509368840702, -0.01413533960417955, -0.010531467375291407, -0.006496435634663962, -0.002195439055854606, 0.0021954390558545918, 0.0064964356346639655, 0.010531467375291403, 0.014135339604179545, 0.0171605093688407, 0.01948312586366727, 0.021008100896465213, 0.02167300181021483, 0.021450607483384662, 0.020350022765845365, 0.018416305725246215, 0.01572862296443143, 0.012397008531418103, 0.008557859112541862, 0.0043683499360924255, 5.315176449816806e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.z.const.wf": {
            "type": "constant",
            "sample": 2.5,
        },
        "q2.z.Cz_unipolar.wf": {
            "type": "constant",
            "sample": 0.059,
        },
        "q2.z.Cz_flattop.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.005917563276925436, 0.021409945066494435, 0.040559582091781565, 0.056051963881350565] + [0.061969527158276] * 71 + [0.056051963881350565, 0.040559582091781565, 0.021409945066494438, 0.005917563276925439],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.0036215299041330576,
        },
        "q2.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "q2.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "c1.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.017087752868164427, 0.0653963837643907, 0.13657289153658733, 0.21831021137266807, 0.2964752181127355, 0.3575524719846385] + [0.39098116092449303] * 2 + [0.3575524719846385, 0.29647521811273564, 0.21831021137266823, 0.1365728915365874, 0.06539638376439066, 0.017087752868164406, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "c1.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "c1.xy.x180_Long.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 2.4922517611573513e-05, 9.962908086939387e-05, 0.00022393687029448296, 0.0003975416838956756, 0.0006200186816028761, 0.000890823424921892, 0.0012092932092666738, 0.0015746486857084786, 0.0019859957681732953, 0.0024423278214202905, 0.0029425281244469374, 0.0034853726032924256, 0.004069532826551766, 0.004693579256270052, 0.00535598474626138, 0.006055128279291481, 0.0067892989339785065, 0.007556700071704382, 0.008355453733290547, 0.009183605234678772, 0.010039127950370588, 0.010919928272919488, 0.011823850736339184, 0.012748683290890074, 0.01369216271633561, 0.014651980160421482, 0.015625786789023977, 0.01661119953414071, 0.017605806925657295, 0.018607174992618863, 0.019612853219564838, 0.0206203805433509, 0.021627291375782894, 0.022631121637324315, 0.02362941478711187, 0.02461972783452256, 0.025599637317581147, 0.02656674523357772, 0.027518684907382126, 0.028453126783094626, 0.029367784124859547, 0.03026041861289111, 0.031128845821017014, 0.03197094056233536, 0.03278464208990307, 0.033567959139729014, 0.0343189748037307, 0.03503585122072947, 0.03571683407400492, 0.036360256884401775, 0.03696454508848367, 0.03752821989175377, 0.03804990188751264, 0.03852831443249762, 0.03896228677104288, 0.03935075690011456, 0.03969277416821035, 0.03998750160176287, 0.040234217953354294, 0.040432319466729656, 0.04058132135428957, 0.04068085898344671] + [0.04073068876894286] * 2 + [0.04068085898344671, 0.04058132135428957, 0.040432319466729656, 0.040234217953354294, 0.03998750160176286, 0.039692774168210355, 0.03935075690011457, 0.038962286771042874, 0.038528314432497636, 0.03804990188751264, 0.03752821989175377, 0.03696454508848368, 0.03636025688440177, 0.03571683407400493, 0.035035851220729476, 0.034318974803730695, 0.03356795913972903, 0.032784642089903074, 0.03197094056233536, 0.03112884582101702, 0.030260418612891103, 0.029367784124859558, 0.028453126783094636, 0.027518684907382136, 0.02656674523357773, 0.025599637317581157, 0.02461972783452256, 0.023629414787111887, 0.022631121637324326, 0.021627291375782884, 0.020620380543350907, 0.01961285321956485, 0.01860717499261887, 0.0176058069256573, 0.01661119953414071, 0.015625786789023977, 0.014651980160421471, 0.013692162716335618, 0.012748683290890083, 0.011823850736339176, 0.010919928272919491, 0.010039127950370588, 0.00918360523467877, 0.008355453733290559, 0.007556700071704392, 0.006789298933978502, 0.0060551282792914865, 0.005355984746261386, 0.0046935792562700545, 0.004069532826551776, 0.0034853726032924234, 0.002942528124446935, 0.002442327821420297, 0.0019859957681732975, 0.0015746486857084808, 0.0012092932092666827, 0.000890823424921892, 0.0006200186816028739, 0.00039754168389567784, 0.00022393687029448523, 9.962908086939613e-05, 2.4922517611573513e-05, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "c1.xy.x180_Long.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 128,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "c1.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.008543876434082213, 0.03269819188219535, 0.06828644576829367, 0.10915510568633403, 0.14823760905636774, 0.17877623599231926] + [0.19549058046224652] * 2 + [0.17877623599231926, 0.14823760905636782, 0.10915510568633412, 0.0682864457682937, 0.03269819188219533, 0.008543876434082203, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "c1.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.030066968357884298, -0.05493508473685802, -0.07030442589525175, -0.07351749312069815, -0.06401871790011814, -0.04345052476281385, -0.015369341158393722, 0.015369341158393704, 0.04345052476281383, 0.06401871790011811, 0.07351749312069814, 0.07030442589525175, 0.054935084736858014, 0.030066968357884298, 8.376210093969753e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "c1.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.00854387643408221, -0.032698191882195346, -0.06828644576829365, -0.10915510568633402, -0.14823760905636774, -0.17877623599231926] + [-0.19549058046224652] * 2 + [-0.17877623599231926, -0.14823760905636782, -0.10915510568633413, -0.06828644576829371, -0.03269819188219534, -0.008543876434082207, -1.0257898880565809e-32],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "c1.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.030066968357884298, 0.05493508473685803, 0.07030442589525177, 0.07351749312069816, 0.06401871790011815, 0.04345052476281387, 0.015369341158393746, -0.01536934115839368, -0.04345052476281381, -0.0640187179001181, -0.07351749312069812, -0.07030442589525174, -0.05493508473685801, -0.030066968357884298, -8.376210093969753e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "c1.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 1.0463230927309285e-18, 4.00437360264365e-18, 8.362677723529016e-18, 1.3367645078936002e-17, 1.8153871344413744e-17, 2.1893774517160563e-17] + [2.394069136265483e-17] * 2 + [2.1893774517160563e-17, 1.8153871344413753e-17, 1.3367645078936013e-17, 8.36267772352902e-18, 4.004373602643648e-18, 1.0463230927309271e-18, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "c1.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.017087752868164427, 0.0653963837643907, 0.13657289153658733, 0.21831021137266807, 0.2964752181127355, 0.3575524719846385] + [0.39098116092449303] * 2 + [0.3575524719846385, 0.29647521811273564, 0.21831021137266823, 0.1365728915365874, 0.06539638376439066, 0.017087752868164406, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "c1.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.030066968357884298, 0.05493508473685802, 0.07030442589525175, 0.07351749312069815, 0.06401871790011815, 0.043450524762813865, 0.015369341158393734, -0.015369341158393692, -0.043450524762813816, -0.0640187179001181, -0.07351749312069814, -0.07030442589525175, -0.054935084736858014, -0.030066968357884298, -8.376210093969753e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "c1.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.008543876434082212, 0.03269819188219535, 0.06828644576829367, 0.10915510568633403, 0.14823760905636774, 0.17877623599231926] + [0.19549058046224652] * 2 + [0.17877623599231926, 0.14823760905636782, 0.10915510568633412, 0.0682864457682937, 0.03269819188219533, 0.008543876434082205, 5.1289494402829045e-33],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "c1.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.030066968357884298, -0.05493508473685802, -0.07030442589525175, -0.07351749312069815, -0.06401871790011812, -0.04345052476281384, -0.01536934115839371, 0.015369341158393716, 0.043450524762813844, 0.06401871790011812, 0.07351749312069814, 0.07030442589525175, 0.054935084736858014, 0.030066968357884298, 8.376210093969753e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "c1.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.008543876434082215, -0.03269819188219535, -0.06828644576829367, -0.10915510568633403, -0.14823760905636774, -0.17877623599231926] + [-0.19549058046224652] * 2 + [-0.17877623599231926, -0.14823760905636782, -0.10915510568633412, -0.0682864457682937, -0.03269819188219533, -0.008543876434082201, 5.1289494402829045e-33],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "c1.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "c1.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "c1.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05,
        },
        "c1.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "c1.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "c1.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "c1.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "c1.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "c1.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "c1.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "c1.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "c1.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "c1.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.15,
        },
        "c1.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "c1.z.const.wf": {
            "type": "constant",
            "sample": 2.5,
        },
        "c1.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.011674147072951396,
        },
        "c1.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "c1.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "c1.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "coupler_q1_q2.const.wf": {
            "type": "constant",
            "sample": 2.5,
        },
        "q2.z.Cz_unipolar.flux_pulse_control_q1_q2.wf": {
            "type": "constant",
            "sample": 0.059,
        },
        "coupler_q1_q2.Cz_unipolar.coupler_flux_pulse_q1_q2.wf": {
            "type": "constant",
            "sample": -0.225,
        },
        "q2.z.Cz_flattop.flux_pulse_control_q1_q2.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.009075227138867558, 0.030984763579137996, 0.05289430001940844] + [0.061969527158276] * 73 + [0.05289430001940844, 0.030984763579138, 0.009075227138867561],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "coupler_q1_q2.Cz_flattop.coupler_flux_pulse_q1_q2.wf": {
            "type": "arbitrary",
            "samples": [-0.0, -0.0025363429867735834, -0.010047901708510151, -0.02224601117606403, -0.038661904883375724, -0.0586647292414125, -0.08148578692780814, -0.10624807749387107, -0.13199999999999998, -0.15775192250612893, -0.18251421307219184, -0.20533527075858746, -0.22533809511662428, -0.241753988823936, -0.25395209829148985, -0.26146365701322644] + [-0.264] * 49 + [-0.26146365701322644, -0.25395209829148985, -0.241753988823936, -0.22533809511662428, -0.20533527075858748, -0.18251421307219187, -0.15775192250612896, -0.132, -0.10624807749387108, -0.08148578692780815, -0.05866472924141254, -0.03866190488337574, -0.022246011176064014, -0.010047901708510151, -0.0025363429867735834],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
    },
    "digital_waveforms": {
        "ON": {
            "samples": [(1, 0)],
        },
    },
    "integration_weights": {
        "q1.resonator.readout.iw1": {
            "cosine": [(0.3829858928526527, 1200)],
            "sine": [(0.9237541912629443, 1200)],
        },
        "q1.resonator.readout.iw2": {
            "cosine": [(-0.9237541912629443, 1200)],
            "sine": [(0.3829858928526527, 1200)],
        },
        "q1.resonator.readout.iw3": {
            "cosine": [(0.9237541912629443, 1200)],
            "sine": [(-0.3829858928526527, 1200)],
        },
        "q2.resonator.readout.iw1": {
            "cosine": [(-0.9709198669465744, 2000)],
            "sine": [(0.23940470331312666, 2000)],
        },
        "q2.resonator.readout.iw2": {
            "cosine": [(-0.23940470331312666, 2000)],
            "sine": [(-0.9709198669465744, 2000)],
        },
        "q2.resonator.readout.iw3": {
            "cosine": [(0.23940470331312666, 2000)],
            "sine": [(0.9709198669465744, 2000)],
        },
        "c1.resonator.readout.iw1": {
            "cosine": [(0.016549154040551798, 2000)],
            "sine": [(0.9998630533730817, 2000)],
        },
        "c1.resonator.readout.iw2": {
            "cosine": [(-0.9998630533730817, 2000)],
            "sine": [(0.016549154040551798, 2000)],
        },
        "c1.resonator.readout.iw3": {
            "cosine": [(0.9998630533730817, 2000)],
            "sine": [(-0.016549154040551798, 2000)],
        },
    },
    "mixers": {},
}


