
# Single QUA script generated at 2026-01-21 11:02:50.656439
# QUA library version: 1.2.4a1


from qm import CompilerOptionArguments
from qm.qua import *

with program() as prog:
    v1 = declare(int, )
    v2 = declare(fixed, )
    v3 = declare(fixed, )
    v4 = declare(fixed, )
    v5 = declare(int, value=192)
    v6 = declare(fixed, )
    v7 = declare(fixed, )
    v8 = declare(fixed, )
    v9 = declare(fixed, )
    set_dc_offset("q6.z", "single", 0.041916898056998705)
    set_dc_offset("q7.z", "single", 0.03836502756796534)
    set_dc_offset("q8.z", "single", 0.061267548650565454)
    set_dc_offset("q9.z", "single", -0.19519714747484346)
    set_dc_offset("q10.z", "single", 0.07793351478577137)
    set_dc_offset("q5.z", "single", 0.0)
    set_dc_offset("coupler_q6_q7", "single", -0.05)
    set_dc_offset("coupler_q7_q8", "single", -0.05)
    set_dc_offset("coupler_q8_q9", "single", -0.05)
    set_dc_offset("coupler_q9_q10", "single", -0.05)
    set_dc_offset("coupler_q5_q6", "single", -0.05)
    wait(24, "q10.z")
    wait(24, "q9.z")
    align("q10.xy", "q10.z", "q10.resonator", "q9.xy", "q9.z", "q9.resonator", "coupler_q9_q10")
    set_dc_offset("coupler_q9_q10", "single", -0.05)
    wait(1000, )
    with for_(v1,0,(v1<150),(v1+1)):
        r1 = declare_stream()
        save(v1, r1)
        with for_(v2,-0.5,(v2<0.5050000000000009),(v2+0.010000000000000009)):
            with for_(v3,-0.2,(v3<0.20400000000000035),(v3+0.008000000000000007)):
                wait(12500, )
                wait(8124, )
                align()
                assign(v4, v3)
                play("x180", "q10.xy")
                align()
                play("const"*amp((v4/1.25)), "q10.z", duration=v5)
                play("const"*amp((v2/1.25)), "coupler_q9_q10", duration=v5)
                align()
                wait(20, )
                measure("readout", "q10.resonator", dual_demod.full("iw1", "iw2", v6), dual_demod.full("iw3", "iw1", v7))
                measure("readout", "q9.resonator", dual_demod.full("iw1", "iw2", v8), dual_demod.full("iw3", "iw1", v9))
                r2 = declare_stream()
                save(v6, r2)
                r3 = declare_stream()
                save(v7, r3)
                r4 = declare_stream()
                save(v8, r4)
                r5 = declare_stream()
                save(v9, r5)
    align()
    with stream_processing():
        r1.save("n")
        r2.buffer(51).buffer(101).average().save("I_control1")
        r3.buffer(51).buffer(101).average().save("Q_control1")
        r4.buffer(51).buffer(101).average().save("I_target1")
        r5.buffer(51).buffer(101).average().save("Q_target1")

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "fems": {
                "2": {
                    "type": "LF",
                    "analog_outputs": {
                        "3": {
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                        "1": {
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                        "2": {
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                    },
                },
                "3": {
                    "type": "LF",
                    "analog_outputs": {
                        "1": {
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                        "2": {
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                        "3": {
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                        "4": {
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                        "5": {
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                        "6": {
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                        "7": {
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                        "8": {
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                    },
                },
                "7": {
                    "type": "MW",
                    "analog_outputs": {
                        "1": {
                            "band": 2,
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": -5,
                            "upconverter_frequency": 6025000000,
                        },
                        "7": {
                            "band": 1,
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 7,
                            "upconverter_frequency": 4100000000,
                        },
                        "2": {
                            "band": 1,
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 7,
                            "upconverter_frequency": 4250000000,
                        },
                        "3": {
                            "band": 1,
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 10,
                            "upconverter_frequency": 4150000000,
                        },
                        "4": {
                            "band": 1,
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 7,
                            "upconverter_frequency": 3900000000,
                        },
                        "5": {
                            "band": 1,
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 10,
                            "upconverter_frequency": 3800000000,
                        },
                        "6": {
                            "band": 1,
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 7,
                            "upconverter_frequency": 3800000000,
                        },
                    },
                    "analog_inputs": {
                        "1": {
                            "band": 2,
                            "downconverter_frequency": 6025000000,
                            "sampling_rate": 1000000000.0,
                            "shareable": False,
                        },
                    },
                },
            },
        },
    },
    "elements": {
        "q5.xy": {
            "operations": {
                "x180_DragCosine": "q5.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "q5.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "q5.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "q5.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "q5.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "q5.xy.-y90_DragCosine.pulse",
                "x180_Square": "q5.xy.x180_Square.pulse",
                "x90_Square": "q5.xy.x90_Square.pulse",
                "-x90_Square": "q5.xy.-x90_Square.pulse",
                "y180_Square": "q5.xy.y180_Square.pulse",
                "y90_Square": "q5.xy.y90_Square.pulse",
                "-y90_Square": "q5.xy.-y90_Square.pulse",
                "x180": "q5.xy.x180_DragCosine.pulse",
                "x90": "q5.xy.x90_DragCosine.pulse",
                "-x90": "q5.xy.-x90_DragCosine.pulse",
                "y180": "q5.xy.y180_DragCosine.pulse",
                "y90": "q5.xy.y90_DragCosine.pulse",
                "-y90": "q5.xy.-y90_DragCosine.pulse",
                "saturation": "q5.xy.saturation.pulse",
            },
            "intermediate_frequency": -250000000,
            "core": "a",
            "MWInput": {
                "port": ('con1', 7, 7),
                "upconverter": 1,
            },
        },
        "q5.z": {
            "operations": {
                "const": "q5.z.const.pulse",
            },
            "singleInput": {
                "port": ('con1', 3, 8),
            },
        },
        "q5.resonator": {
            "operations": {
                "readout": "q5.resonator.readout.pulse",
                "const": "q5.resonator.const.pulse",
            },
            "intermediate_frequency": -175000000,
            "core": "a",
            "MWOutput": {
                "port": ('con1', 7, 1),
            },
            "smearing": 0,
            "time_of_flight": 28,
            "MWInput": {
                "port": ('con1', 7, 1),
                "upconverter": 1,
            },
        },
        "q6.xy": {
            "operations": {
                "x180_DragCosine": "q6.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "q6.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "q6.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "q6.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "q6.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "q6.xy.-y90_DragCosine.pulse",
                "x180_Square": "q6.xy.x180_Square.pulse",
                "x90_Square": "q6.xy.x90_Square.pulse",
                "-x90_Square": "q6.xy.-x90_Square.pulse",
                "y180_Square": "q6.xy.y180_Square.pulse",
                "y90_Square": "q6.xy.y90_Square.pulse",
                "-y90_Square": "q6.xy.-y90_Square.pulse",
                "x180": "q6.xy.x180_DragCosine.pulse",
                "x90": "q6.xy.x90_DragCosine.pulse",
                "-x90": "q6.xy.-x90_DragCosine.pulse",
                "y180": "q6.xy.y180_DragCosine.pulse",
                "y90": "q6.xy.y90_DragCosine.pulse",
                "-y90": "q6.xy.-y90_DragCosine.pulse",
                "saturation": "q6.xy.saturation.pulse",
                "EF_x180": "q6.xy.EF_x180.pulse",
            },
            "intermediate_frequency": -50904333.24319054,
            "core": "b",
            "MWInput": {
                "port": ('con1', 7, 2),
                "upconverter": 1,
            },
        },
        "q6.z": {
            "operations": {
                "const": "q6.z.const.pulse",
            },
            "singleInput": {
                "port": ('con1', 2, 2),
            },
        },
        "q6.resonator": {
            "operations": {
                "readout": "q6.resonator.readout.pulse",
                "const": "q6.resonator.const.pulse",
            },
            "intermediate_frequency": -25632647.0,
            "core": "b",
            "MWOutput": {
                "port": ('con1', 7, 1),
            },
            "smearing": 0,
            "time_of_flight": 388,
            "MWInput": {
                "port": ('con1', 7, 1),
                "upconverter": 1,
            },
        },
        "q7.xy": {
            "operations": {
                "x180_DragCosine": "q7.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "q7.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "q7.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "q7.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "q7.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "q7.xy.-y90_DragCosine.pulse",
                "x180_Square": "q7.xy.x180_Square.pulse",
                "x90_Square": "q7.xy.x90_Square.pulse",
                "-x90_Square": "q7.xy.-x90_Square.pulse",
                "y180_Square": "q7.xy.y180_Square.pulse",
                "y90_Square": "q7.xy.y90_Square.pulse",
                "-y90_Square": "q7.xy.-y90_Square.pulse",
                "x180": "q7.xy.x180_DragCosine.pulse",
                "x90": "q7.xy.x90_DragCosine.pulse",
                "-x90": "q7.xy.-x90_DragCosine.pulse",
                "y180": "q7.xy.y180_DragCosine.pulse",
                "y90": "q7.xy.y90_DragCosine.pulse",
                "-y90": "q7.xy.-y90_DragCosine.pulse",
                "saturation": "q7.xy.saturation.pulse",
            },
            "intermediate_frequency": -256772753.10845205,
            "core": "c",
            "MWInput": {
                "port": ('con1', 7, 3),
                "upconverter": 1,
            },
        },
        "q7.z": {
            "operations": {
                "const": "q7.z.const.pulse",
            },
            "singleInput": {
                "port": ('con1', 3, 1),
            },
        },
        "q7.resonator": {
            "operations": {
                "readout": "q7.resonator.readout.pulse",
                "const": "q7.resonator.const.pulse",
            },
            "intermediate_frequency": 77254665.0,
            "core": "c",
            "MWOutput": {
                "port": ('con1', 7, 1),
            },
            "smearing": 0,
            "time_of_flight": 388,
            "MWInput": {
                "port": ('con1', 7, 1),
                "upconverter": 1,
            },
        },
        "q8.xy": {
            "operations": {
                "x180_DragCosine": "q8.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "q8.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "q8.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "q8.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "q8.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "q8.xy.-y90_DragCosine.pulse",
                "x180_Square": "q8.xy.x180_Square.pulse",
                "x90_Square": "q8.xy.x90_Square.pulse",
                "-x90_Square": "q8.xy.-x90_Square.pulse",
                "y180_Square": "q8.xy.y180_Square.pulse",
                "y90_Square": "q8.xy.y90_Square.pulse",
                "-y90_Square": "q8.xy.-y90_Square.pulse",
                "x180": "q8.xy.x180_DragCosine.pulse",
                "x90": "q8.xy.x90_DragCosine.pulse",
                "-x90": "q8.xy.-x90_DragCosine.pulse",
                "y180": "q8.xy.y180_DragCosine.pulse",
                "y90": "q8.xy.y90_DragCosine.pulse",
                "-y90": "q8.xy.-y90_DragCosine.pulse",
                "saturation": "q8.xy.saturation.pulse",
            },
            "intermediate_frequency": -79071692.25275058,
            "core": "d",
            "MWInput": {
                "port": ('con1', 7, 4),
                "upconverter": 1,
            },
        },
        "q8.z": {
            "operations": {
                "const": "q8.z.const.pulse",
            },
            "singleInput": {
                "port": ('con1', 3, 3),
            },
        },
        "q8.resonator": {
            "operations": {
                "readout": "q8.resonator.readout.pulse",
                "const": "q8.resonator.const.pulse",
            },
            "intermediate_frequency": -76041688.0,
            "core": "d",
            "MWOutput": {
                "port": ('con1', 7, 1),
            },
            "smearing": 0,
            "time_of_flight": 388,
            "MWInput": {
                "port": ('con1', 7, 1),
                "upconverter": 1,
            },
        },
        "q9.xy": {
            "operations": {
                "x180_DragCosine": "q9.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "q9.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "q9.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "q9.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "q9.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "q9.xy.-y90_DragCosine.pulse",
                "x180_Square": "q9.xy.x180_Square.pulse",
                "x90_Square": "q9.xy.x90_Square.pulse",
                "-x90_Square": "q9.xy.-x90_Square.pulse",
                "y180_Square": "q9.xy.y180_Square.pulse",
                "y90_Square": "q9.xy.y90_Square.pulse",
                "-y90_Square": "q9.xy.-y90_Square.pulse",
                "x180": "q9.xy.x180_DragCosine.pulse",
                "x90": "q9.xy.x90_DragCosine.pulse",
                "-x90": "q9.xy.-x90_DragCosine.pulse",
                "y180": "q9.xy.y180_DragCosine.pulse",
                "y90": "q9.xy.y90_DragCosine.pulse",
                "-y90": "q9.xy.-y90_DragCosine.pulse",
                "saturation": "q9.xy.saturation.pulse",
            },
            "intermediate_frequency": -165000000.0,
            "core": "e",
            "MWInput": {
                "port": ('con1', 7, 5),
                "upconverter": 1,
            },
        },
        "q9.z": {
            "operations": {
                "const": "q9.z.const.pulse",
            },
            "singleInput": {
                "port": ('con1', 3, 5),
            },
        },
        "q9.resonator": {
            "operations": {
                "readout": "q9.resonator.readout.pulse",
                "const": "q9.resonator.const.pulse",
            },
            "intermediate_frequency": 148633787.0,
            "core": "e",
            "MWOutput": {
                "port": ('con1', 7, 1),
            },
            "smearing": 0,
            "time_of_flight": 388,
            "MWInput": {
                "port": ('con1', 7, 1),
                "upconverter": 1,
            },
        },
        "q10.xy": {
            "operations": {
                "x180_DragCosine": "q10.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "q10.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "q10.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "q10.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "q10.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "q10.xy.-y90_DragCosine.pulse",
                "x180_Square": "q10.xy.x180_Square.pulse",
                "x90_Square": "q10.xy.x90_Square.pulse",
                "-x90_Square": "q10.xy.-x90_Square.pulse",
                "y180_Square": "q10.xy.y180_Square.pulse",
                "y90_Square": "q10.xy.y90_Square.pulse",
                "-y90_Square": "q10.xy.-y90_Square.pulse",
                "x180": "q10.xy.x180_DragCosine.pulse",
                "x90": "q10.xy.x90_DragCosine.pulse",
                "-x90": "q10.xy.-x90_DragCosine.pulse",
                "y180": "q10.xy.y180_DragCosine.pulse",
                "y90": "q10.xy.y90_DragCosine.pulse",
                "-y90": "q10.xy.-y90_DragCosine.pulse",
                "saturation": "q10.xy.saturation.pulse",
            },
            "intermediate_frequency": -198995429.8353858,
            "core": "f",
            "MWInput": {
                "port": ('con1', 7, 6),
                "upconverter": 1,
            },
        },
        "q10.z": {
            "operations": {
                "const": "q10.z.const.pulse",
                "SWAP_unipolar": "q10.z.SWAP_unipolar.pulse",
                "SWAP_flattop": "q10.z.SWAP_flattop.pulse",
                "SWAP_bipolar": "q10.z.SWAP_bipolar.pulse",
                "SWAP_unipolar.flux_pulse_control_q9_q10": "q10.z.SWAP_unipolar.flux_pulse_control_q9_q10.pulse",
                "SWAP_flattop.flux_pulse_control_q9_q10": "q10.z.SWAP_flattop.flux_pulse_control_q9_q10.pulse",
                "SWAP_bipolar.flux_pulse_control_q9_q10": "q10.z.SWAP_bipolar.flux_pulse_control_q9_q10.pulse",
            },
            "singleInput": {
                "port": ('con1', 3, 7),
            },
        },
        "q10.resonator": {
            "operations": {
                "readout": "q10.resonator.readout.pulse",
                "const": "q10.resonator.const.pulse",
            },
            "intermediate_frequency": 25616322.0,
            "core": "f",
            "MWOutput": {
                "port": ('con1', 7, 1),
            },
            "smearing": 0,
            "time_of_flight": 388,
            "MWInput": {
                "port": ('con1', 7, 1),
                "upconverter": 1,
            },
        },
        "coupler_q6_q7": {
            "operations": {
                "const": "coupler_q6_q7.const.pulse",
            },
            "singleInput": {
                "port": ('con1', 2, 3),
            },
        },
        "coupler_q7_q8": {
            "operations": {
                "const": "coupler_q7_q8.const.pulse",
            },
            "singleInput": {
                "port": ('con1', 3, 2),
            },
        },
        "coupler_q8_q9": {
            "operations": {
                "const": "coupler_q8_q9.const.pulse",
            },
            "singleInput": {
                "port": ('con1', 3, 4),
            },
        },
        "coupler_q9_q10": {
            "operations": {
                "const": "coupler_q9_q10.const.pulse",
                "SWAP_unipolar.coupler_pulse_control_q9_q10": "coupler_q9_q10.SWAP_unipolar.coupler_pulse_control_q9_q10.pulse",
                "SWAP_flattop.coupler_pulse_control_q9_q10": "coupler_q9_q10.SWAP_flattop.coupler_pulse_control_q9_q10.pulse",
                "SWAP_bipolar.coupler_pulse_control_q9_q10": "coupler_q9_q10.SWAP_bipolar.coupler_pulse_control_q9_q10.pulse",
            },
            "singleInput": {
                "port": ('con1', 3, 6),
            },
        },
        "coupler_q5_q6": {
            "operations": {
                "const": "coupler_q5_q6.const.pulse",
            },
            "singleInput": {
                "port": ('con1', 2, 1),
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
        "q5.xy.x180_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.xy.x180_DragCosine.wf.I",
                "Q": "q5.xy.x180_DragCosine.wf.Q",
            },
        },
        "q5.xy.x90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.xy.x90_DragCosine.wf.I",
                "Q": "q5.xy.x90_DragCosine.wf.Q",
            },
        },
        "q5.xy.-x90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.xy.-x90_DragCosine.wf.I",
                "Q": "q5.xy.-x90_DragCosine.wf.Q",
            },
        },
        "q5.xy.y180_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.xy.y180_DragCosine.wf.I",
                "Q": "q5.xy.y180_DragCosine.wf.Q",
            },
        },
        "q5.xy.y90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.xy.y90_DragCosine.wf.I",
                "Q": "q5.xy.y90_DragCosine.wf.Q",
            },
        },
        "q5.xy.-y90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.xy.-y90_DragCosine.wf.I",
                "Q": "q5.xy.-y90_DragCosine.wf.Q",
            },
        },
        "q5.xy.x180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.xy.x180_Square.wf.I",
                "Q": "q5.xy.x180_Square.wf.Q",
            },
        },
        "q5.xy.x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.xy.x90_Square.wf.I",
                "Q": "q5.xy.x90_Square.wf.Q",
            },
        },
        "q5.xy.-x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.xy.-x90_Square.wf.I",
                "Q": "q5.xy.-x90_Square.wf.Q",
            },
        },
        "q5.xy.y180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.xy.y180_Square.wf.I",
                "Q": "q5.xy.y180_Square.wf.Q",
            },
        },
        "q5.xy.y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.xy.y90_Square.wf.I",
                "Q": "q5.xy.y90_Square.wf.Q",
            },
        },
        "q5.xy.-y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.xy.-y90_Square.wf.I",
                "Q": "q5.xy.-y90_Square.wf.Q",
            },
        },
        "q5.xy.saturation.pulse": {
            "operation": "control",
            "length": 20000,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.xy.saturation.wf.I",
                "Q": "q5.xy.saturation.wf.Q",
            },
        },
        "q5.z.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "q5.z.const.wf",
            },
        },
        "q5.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 1500,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.resonator.readout.wf.I",
                "Q": "q5.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "q5.resonator.readout.iw1",
                "iw2": "q5.resonator.readout.iw2",
                "iw3": "q5.resonator.readout.iw3",
            },
        },
        "q5.resonator.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "I": "q5.resonator.const.wf.I",
                "Q": "q5.resonator.const.wf.Q",
            },
        },
        "q6.xy.x180_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q6.xy.x180_DragCosine.wf.I",
                "Q": "q6.xy.x180_DragCosine.wf.Q",
            },
        },
        "q6.xy.x90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q6.xy.x90_DragCosine.wf.I",
                "Q": "q6.xy.x90_DragCosine.wf.Q",
            },
        },
        "q6.xy.-x90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q6.xy.-x90_DragCosine.wf.I",
                "Q": "q6.xy.-x90_DragCosine.wf.Q",
            },
        },
        "q6.xy.y180_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q6.xy.y180_DragCosine.wf.I",
                "Q": "q6.xy.y180_DragCosine.wf.Q",
            },
        },
        "q6.xy.y90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q6.xy.y90_DragCosine.wf.I",
                "Q": "q6.xy.y90_DragCosine.wf.Q",
            },
        },
        "q6.xy.-y90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q6.xy.-y90_DragCosine.wf.I",
                "Q": "q6.xy.-y90_DragCosine.wf.Q",
            },
        },
        "q6.xy.x180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q6.xy.x180_Square.wf.I",
                "Q": "q6.xy.x180_Square.wf.Q",
            },
        },
        "q6.xy.x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q6.xy.x90_Square.wf.I",
                "Q": "q6.xy.x90_Square.wf.Q",
            },
        },
        "q6.xy.-x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q6.xy.-x90_Square.wf.I",
                "Q": "q6.xy.-x90_Square.wf.Q",
            },
        },
        "q6.xy.y180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q6.xy.y180_Square.wf.I",
                "Q": "q6.xy.y180_Square.wf.Q",
            },
        },
        "q6.xy.y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q6.xy.y90_Square.wf.I",
                "Q": "q6.xy.y90_Square.wf.Q",
            },
        },
        "q6.xy.-y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q6.xy.-y90_Square.wf.I",
                "Q": "q6.xy.-y90_Square.wf.Q",
            },
        },
        "q6.xy.saturation.pulse": {
            "operation": "control",
            "length": 20000,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q6.xy.saturation.wf.I",
                "Q": "q6.xy.saturation.wf.Q",
            },
        },
        "q6.xy.EF_x180.pulse": {
            "operation": "control",
            "length": 272,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q6.xy.EF_x180.wf.I",
                "Q": "q6.xy.EF_x180.wf.Q",
            },
        },
        "q6.z.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "q6.z.const.wf",
            },
        },
        "q6.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 3000,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q6.resonator.readout.wf.I",
                "Q": "q6.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "q6.resonator.readout.iw1",
                "iw2": "q6.resonator.readout.iw2",
                "iw3": "q6.resonator.readout.iw3",
            },
        },
        "q6.resonator.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "I": "q6.resonator.const.wf.I",
                "Q": "q6.resonator.const.wf.Q",
            },
        },
        "q7.xy.x180_DragCosine.pulse": {
            "operation": "control",
            "length": 64,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q7.xy.x180_DragCosine.wf.I",
                "Q": "q7.xy.x180_DragCosine.wf.Q",
            },
        },
        "q7.xy.x90_DragCosine.pulse": {
            "operation": "control",
            "length": 64,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q7.xy.x90_DragCosine.wf.I",
                "Q": "q7.xy.x90_DragCosine.wf.Q",
            },
        },
        "q7.xy.-x90_DragCosine.pulse": {
            "operation": "control",
            "length": 64,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q7.xy.-x90_DragCosine.wf.I",
                "Q": "q7.xy.-x90_DragCosine.wf.Q",
            },
        },
        "q7.xy.y180_DragCosine.pulse": {
            "operation": "control",
            "length": 64,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q7.xy.y180_DragCosine.wf.I",
                "Q": "q7.xy.y180_DragCosine.wf.Q",
            },
        },
        "q7.xy.y90_DragCosine.pulse": {
            "operation": "control",
            "length": 64,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q7.xy.y90_DragCosine.wf.I",
                "Q": "q7.xy.y90_DragCosine.wf.Q",
            },
        },
        "q7.xy.-y90_DragCosine.pulse": {
            "operation": "control",
            "length": 64,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q7.xy.-y90_DragCosine.wf.I",
                "Q": "q7.xy.-y90_DragCosine.wf.Q",
            },
        },
        "q7.xy.x180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q7.xy.x180_Square.wf.I",
                "Q": "q7.xy.x180_Square.wf.Q",
            },
        },
        "q7.xy.x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q7.xy.x90_Square.wf.I",
                "Q": "q7.xy.x90_Square.wf.Q",
            },
        },
        "q7.xy.-x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q7.xy.-x90_Square.wf.I",
                "Q": "q7.xy.-x90_Square.wf.Q",
            },
        },
        "q7.xy.y180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q7.xy.y180_Square.wf.I",
                "Q": "q7.xy.y180_Square.wf.Q",
            },
        },
        "q7.xy.y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q7.xy.y90_Square.wf.I",
                "Q": "q7.xy.y90_Square.wf.Q",
            },
        },
        "q7.xy.-y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q7.xy.-y90_Square.wf.I",
                "Q": "q7.xy.-y90_Square.wf.Q",
            },
        },
        "q7.xy.saturation.pulse": {
            "operation": "control",
            "length": 20000,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q7.xy.saturation.wf.I",
                "Q": "q7.xy.saturation.wf.Q",
            },
        },
        "q7.z.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "q7.z.const.wf",
            },
        },
        "q7.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 2000,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q7.resonator.readout.wf.I",
                "Q": "q7.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "q7.resonator.readout.iw1",
                "iw2": "q7.resonator.readout.iw2",
                "iw3": "q7.resonator.readout.iw3",
            },
        },
        "q7.resonator.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "I": "q7.resonator.const.wf.I",
                "Q": "q7.resonator.const.wf.Q",
            },
        },
        "q8.xy.x180_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q8.xy.x180_DragCosine.wf.I",
                "Q": "q8.xy.x180_DragCosine.wf.Q",
            },
        },
        "q8.xy.x90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q8.xy.x90_DragCosine.wf.I",
                "Q": "q8.xy.x90_DragCosine.wf.Q",
            },
        },
        "q8.xy.-x90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q8.xy.-x90_DragCosine.wf.I",
                "Q": "q8.xy.-x90_DragCosine.wf.Q",
            },
        },
        "q8.xy.y180_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q8.xy.y180_DragCosine.wf.I",
                "Q": "q8.xy.y180_DragCosine.wf.Q",
            },
        },
        "q8.xy.y90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q8.xy.y90_DragCosine.wf.I",
                "Q": "q8.xy.y90_DragCosine.wf.Q",
            },
        },
        "q8.xy.-y90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q8.xy.-y90_DragCosine.wf.I",
                "Q": "q8.xy.-y90_DragCosine.wf.Q",
            },
        },
        "q8.xy.x180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q8.xy.x180_Square.wf.I",
                "Q": "q8.xy.x180_Square.wf.Q",
            },
        },
        "q8.xy.x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q8.xy.x90_Square.wf.I",
                "Q": "q8.xy.x90_Square.wf.Q",
            },
        },
        "q8.xy.-x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q8.xy.-x90_Square.wf.I",
                "Q": "q8.xy.-x90_Square.wf.Q",
            },
        },
        "q8.xy.y180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q8.xy.y180_Square.wf.I",
                "Q": "q8.xy.y180_Square.wf.Q",
            },
        },
        "q8.xy.y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q8.xy.y90_Square.wf.I",
                "Q": "q8.xy.y90_Square.wf.Q",
            },
        },
        "q8.xy.-y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q8.xy.-y90_Square.wf.I",
                "Q": "q8.xy.-y90_Square.wf.Q",
            },
        },
        "q8.xy.saturation.pulse": {
            "operation": "control",
            "length": 20000,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q8.xy.saturation.wf.I",
                "Q": "q8.xy.saturation.wf.Q",
            },
        },
        "q8.z.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "q8.z.const.wf",
            },
        },
        "q8.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 2000,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q8.resonator.readout.wf.I",
                "Q": "q8.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "q8.resonator.readout.iw1",
                "iw2": "q8.resonator.readout.iw2",
                "iw3": "q8.resonator.readout.iw3",
            },
        },
        "q8.resonator.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "I": "q8.resonator.const.wf.I",
                "Q": "q8.resonator.const.wf.Q",
            },
        },
        "q9.xy.x180_DragCosine.pulse": {
            "operation": "control",
            "length": 64,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q9.xy.x180_DragCosine.wf.I",
                "Q": "q9.xy.x180_DragCosine.wf.Q",
            },
        },
        "q9.xy.x90_DragCosine.pulse": {
            "operation": "control",
            "length": 64,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q9.xy.x90_DragCosine.wf.I",
                "Q": "q9.xy.x90_DragCosine.wf.Q",
            },
        },
        "q9.xy.-x90_DragCosine.pulse": {
            "operation": "control",
            "length": 64,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q9.xy.-x90_DragCosine.wf.I",
                "Q": "q9.xy.-x90_DragCosine.wf.Q",
            },
        },
        "q9.xy.y180_DragCosine.pulse": {
            "operation": "control",
            "length": 64,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q9.xy.y180_DragCosine.wf.I",
                "Q": "q9.xy.y180_DragCosine.wf.Q",
            },
        },
        "q9.xy.y90_DragCosine.pulse": {
            "operation": "control",
            "length": 64,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q9.xy.y90_DragCosine.wf.I",
                "Q": "q9.xy.y90_DragCosine.wf.Q",
            },
        },
        "q9.xy.-y90_DragCosine.pulse": {
            "operation": "control",
            "length": 64,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q9.xy.-y90_DragCosine.wf.I",
                "Q": "q9.xy.-y90_DragCosine.wf.Q",
            },
        },
        "q9.xy.x180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q9.xy.x180_Square.wf.I",
                "Q": "q9.xy.x180_Square.wf.Q",
            },
        },
        "q9.xy.x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q9.xy.x90_Square.wf.I",
                "Q": "q9.xy.x90_Square.wf.Q",
            },
        },
        "q9.xy.-x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q9.xy.-x90_Square.wf.I",
                "Q": "q9.xy.-x90_Square.wf.Q",
            },
        },
        "q9.xy.y180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q9.xy.y180_Square.wf.I",
                "Q": "q9.xy.y180_Square.wf.Q",
            },
        },
        "q9.xy.y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q9.xy.y90_Square.wf.I",
                "Q": "q9.xy.y90_Square.wf.Q",
            },
        },
        "q9.xy.-y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q9.xy.-y90_Square.wf.I",
                "Q": "q9.xy.-y90_Square.wf.Q",
            },
        },
        "q9.xy.saturation.pulse": {
            "operation": "control",
            "length": 20000,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q9.xy.saturation.wf.I",
                "Q": "q9.xy.saturation.wf.Q",
            },
        },
        "q9.z.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "q9.z.const.wf",
            },
        },
        "q9.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 2000,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q9.resonator.readout.wf.I",
                "Q": "q9.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "q9.resonator.readout.iw1",
                "iw2": "q9.resonator.readout.iw2",
                "iw3": "q9.resonator.readout.iw3",
            },
        },
        "q9.resonator.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "I": "q9.resonator.const.wf.I",
                "Q": "q9.resonator.const.wf.Q",
            },
        },
        "q10.xy.x180_DragCosine.pulse": {
            "operation": "control",
            "length": 64,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q10.xy.x180_DragCosine.wf.I",
                "Q": "q10.xy.x180_DragCosine.wf.Q",
            },
        },
        "q10.xy.x90_DragCosine.pulse": {
            "operation": "control",
            "length": 64,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q10.xy.x90_DragCosine.wf.I",
                "Q": "q10.xy.x90_DragCosine.wf.Q",
            },
        },
        "q10.xy.-x90_DragCosine.pulse": {
            "operation": "control",
            "length": 64,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q10.xy.-x90_DragCosine.wf.I",
                "Q": "q10.xy.-x90_DragCosine.wf.Q",
            },
        },
        "q10.xy.y180_DragCosine.pulse": {
            "operation": "control",
            "length": 64,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q10.xy.y180_DragCosine.wf.I",
                "Q": "q10.xy.y180_DragCosine.wf.Q",
            },
        },
        "q10.xy.y90_DragCosine.pulse": {
            "operation": "control",
            "length": 64,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q10.xy.y90_DragCosine.wf.I",
                "Q": "q10.xy.y90_DragCosine.wf.Q",
            },
        },
        "q10.xy.-y90_DragCosine.pulse": {
            "operation": "control",
            "length": 64,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q10.xy.-y90_DragCosine.wf.I",
                "Q": "q10.xy.-y90_DragCosine.wf.Q",
            },
        },
        "q10.xy.x180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q10.xy.x180_Square.wf.I",
                "Q": "q10.xy.x180_Square.wf.Q",
            },
        },
        "q10.xy.x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q10.xy.x90_Square.wf.I",
                "Q": "q10.xy.x90_Square.wf.Q",
            },
        },
        "q10.xy.-x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q10.xy.-x90_Square.wf.I",
                "Q": "q10.xy.-x90_Square.wf.Q",
            },
        },
        "q10.xy.y180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q10.xy.y180_Square.wf.I",
                "Q": "q10.xy.y180_Square.wf.Q",
            },
        },
        "q10.xy.y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q10.xy.y90_Square.wf.I",
                "Q": "q10.xy.y90_Square.wf.Q",
            },
        },
        "q10.xy.-y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q10.xy.-y90_Square.wf.I",
                "Q": "q10.xy.-y90_Square.wf.Q",
            },
        },
        "q10.xy.saturation.pulse": {
            "operation": "control",
            "length": 20000,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q10.xy.saturation.wf.I",
                "Q": "q10.xy.saturation.wf.Q",
            },
        },
        "q10.z.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "q10.z.const.wf",
            },
        },
        "q10.z.SWAP_unipolar.pulse": {
            "operation": "control",
            "length": 128,
            "waveforms": {
                "single": "q10.z.SWAP_unipolar.wf",
            },
        },
        "q10.z.SWAP_flattop.pulse": {
            "operation": "control",
            "length": 128,
            "waveforms": {
                "single": "q10.z.SWAP_flattop.wf",
            },
        },
        "q10.z.SWAP_bipolar.pulse": {
            "operation": "control",
            "length": 128,
            "waveforms": {
                "single": "q10.z.SWAP_bipolar.wf",
            },
        },
        "q10.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 2000,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q10.resonator.readout.wf.I",
                "Q": "q10.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "q10.resonator.readout.iw1",
                "iw2": "q10.resonator.readout.iw2",
                "iw3": "q10.resonator.readout.iw3",
            },
        },
        "q10.resonator.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "I": "q10.resonator.const.wf.I",
                "Q": "q10.resonator.const.wf.Q",
            },
        },
        "coupler_q6_q7.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "coupler_q6_q7.const.wf",
            },
        },
        "coupler_q7_q8.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "coupler_q7_q8.const.wf",
            },
        },
        "coupler_q8_q9.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "coupler_q8_q9.const.wf",
            },
        },
        "coupler_q9_q10.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "coupler_q9_q10.const.wf",
            },
        },
        "q10.z.SWAP_unipolar.flux_pulse_control_q9_q10.pulse": {
            "operation": "control",
            "length": 128,
            "waveforms": {
                "single": "q10.z.SWAP_unipolar.flux_pulse_control_q9_q10.wf",
            },
        },
        "coupler_q9_q10.SWAP_unipolar.coupler_pulse_control_q9_q10.pulse": {
            "operation": "control",
            "length": 128,
            "waveforms": {
                "single": "coupler_q9_q10.SWAP_unipolar.coupler_pulse_control_q9_q10.wf",
            },
        },
        "q10.z.SWAP_flattop.flux_pulse_control_q9_q10.pulse": {
            "operation": "control",
            "length": 128,
            "waveforms": {
                "single": "q10.z.SWAP_flattop.flux_pulse_control_q9_q10.wf",
            },
        },
        "coupler_q9_q10.SWAP_flattop.coupler_pulse_control_q9_q10.pulse": {
            "operation": "control",
            "length": 128,
            "waveforms": {
                "single": "coupler_q9_q10.SWAP_flattop.coupler_pulse_control_q9_q10.wf",
            },
        },
        "q10.z.SWAP_bipolar.flux_pulse_control_q9_q10.pulse": {
            "operation": "control",
            "length": 128,
            "waveforms": {
                "single": "q10.z.SWAP_bipolar.flux_pulse_control_q9_q10.wf",
            },
        },
        "coupler_q9_q10.SWAP_bipolar.coupler_pulse_control_q9_q10.pulse": {
            "operation": "control",
            "length": 128,
            "waveforms": {
                "single": "coupler_q9_q10.SWAP_bipolar.coupler_pulse_control_q9_q10.wf",
            },
        },
        "coupler_q5_q6.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "coupler_q5_q6.const.wf",
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
        "q5.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.013669663395844208, 0.05231504459724201, 0.10925400611220526, 0.17464128422057051, 0.23717082451262841, 0.2860307014088422] + [0.3127725983158096] * 2 + [0.2860307014088422, 0.23717082451262853, 0.17464128422057065, 0.1092540061122053, 0.052315044597241976, 0.013669663395844191, 0.0],
        },
        "q5.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
        },
        "q5.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.006834831697922104, 0.026157522298621005, 0.05462700305610263, 0.08732064211028526, 0.11858541225631421, 0.1430153507044211] + [0.1563862991579048] * 2 + [0.1430153507044211, 0.11858541225631426, 0.08732064211028533, 0.05462700305610265, 0.026157522298620988, 0.0068348316979220955, 0.0],
        },
        "q5.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
        },
        "q5.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.006834831697922104, -0.026157522298621005, -0.05462700305610263, -0.08732064211028526, -0.11858541225631421, -0.1430153507044211] + [-0.1563862991579048] * 2 + [-0.1430153507044211, -0.11858541225631426, -0.08732064211028533, -0.05462700305610265, -0.026157522298620988, -0.0068348316979220955, 0.0],
        },
        "q5.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 8.370254761571174e-19, 3.203372595663173e-18, 6.689878443966877e-18, 1.0693694485985243e-17, 1.4522524554526452e-17, 1.7514329146910545e-17] + [1.9151798069422854e-17] * 2 + [1.7514329146910545e-17, 1.4522524554526458e-17, 1.069369448598525e-17, 6.689878443966879e-18, 3.203372595663171e-18, 8.370254761571164e-19, 0.0],
        },
        "q5.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 8.370254761571174e-19, 3.203372595663173e-18, 6.689878443966877e-18, 1.0693694485985243e-17, 1.4522524554526452e-17, 1.7514329146910545e-17] + [1.9151798069422854e-17] * 2 + [1.7514329146910545e-17, 1.4522524554526458e-17, 1.069369448598525e-17, 6.689878443966879e-18, 3.203372595663171e-18, 8.370254761571164e-19, 0.0],
        },
        "q5.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.013669663395844208, 0.05231504459724201, 0.10925400611220526, 0.17464128422057051, 0.23717082451262841, 0.2860307014088422] + [0.3127725983158096] * 2 + [0.2860307014088422, 0.23717082451262853, 0.17464128422057065, 0.1092540061122053, 0.052315044597241976, 0.013669663395844191, 0.0],
        },
        "q5.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 4.185127380785587e-19, 1.6016862978315865e-18, 3.3449392219834385e-18, 5.346847242992621e-18, 7.261262277263226e-18, 8.757164573455273e-18] + [9.575899034711427e-18] * 2 + [8.757164573455273e-18, 7.261262277263229e-18, 5.346847242992625e-18, 3.3449392219834396e-18, 1.6016862978315856e-18, 4.185127380785582e-19, 0.0],
        },
        "q5.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.006834831697922104, 0.026157522298621005, 0.05462700305610263, 0.08732064211028526, 0.11858541225631421, 0.1430153507044211] + [0.1563862991579048] * 2 + [0.1430153507044211, 0.11858541225631426, 0.08732064211028533, 0.05462700305610265, 0.026157522298620988, 0.0068348316979220955, 0.0],
        },
        "q5.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 4.185127380785587e-19, 1.6016862978315865e-18, 3.3449392219834385e-18, 5.346847242992621e-18, 7.261262277263226e-18, 8.757164573455273e-18] + [9.575899034711427e-18] * 2 + [8.757164573455273e-18, 7.261262277263229e-18, 5.346847242992625e-18, 3.3449392219834396e-18, 1.6016862978315856e-18, 4.185127380785582e-19, 0.0],
        },
        "q5.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.006834831697922104, -0.026157522298621005, -0.05462700305610263, -0.08732064211028526, -0.11858541225631421, -0.1430153507044211] + [-0.1563862991579048] * 2 + [-0.1430153507044211, -0.11858541225631426, -0.08732064211028533, -0.05462700305610265, -0.026157522298620988, -0.0068348316979220955, 0.0],
        },
        "q5.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.31622776601683794,
        },
        "q5.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q5.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.15811388300841897,
        },
        "q5.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q5.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "q5.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q5.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "q5.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q5.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "q5.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q5.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "q5.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q5.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.5,
        },
        "q5.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q5.z.const.wf": {
            "type": "constant",
            "sample": 1.25,
        },
        "q5.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q5.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q5.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "q5.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q6.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.010149304539422525, 0.03884231119928679, 0.08111773846032007, 0.12966578088889005, 0.17609204090397754, 0.21236899637964252] + [0.23222403214834825] * 2 + [0.21236899637964252, 0.1760920409039776, 0.12966578088889016, 0.0811177384603201, 0.038842311199286765, 0.010149304539422511, 0.0],
        },
        "q6.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.004244330997905204, -0.007754778607735973, -0.009924354546337218, -0.01037791942394615, -0.009037047752716626, -0.006133588426040946, -0.002169575938601245, 0.002169575938601242, 0.0061335884260409436, 0.009037047752716623, 0.010377919423946148, 0.00992435454633722, 0.007754778607735971, 0.004244330997905204, 1.182407475327649e-17],
        },
        "q6.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.005074652269711263, 0.019421155599643396, 0.040558869230160034, 0.06483289044444503, 0.08804602045198877, 0.10618449818982126] + [0.11611201607417412] * 2 + [0.10618449818982126, 0.0880460204519888, 0.06483289044444508, 0.04055886923016005, 0.019421155599643383, 0.005074652269711256, 0.0],
        },
        "q6.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
        },
        "q6.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.005074652269711263, -0.019421155599643396, -0.040558869230160034, -0.06483289044444503, -0.08804602045198877, -0.10618449818982126] + [-0.11611201607417412] * 2 + [-0.10618449818982126, -0.0880460204519888, -0.06483289044444508, -0.04055886923016005, -0.019421155599643383, -0.005074652269711256, 0.0],
        },
        "q6.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 6.214656658887749e-19, 2.378405604084598e-18, 4.967028937975156e-18, 7.939739176226062e-18, 1.0782527712419045e-17, 1.3003850582723253e-17] + [1.4219620882778337e-17] * 2 + [1.3003850582723253e-17, 1.0782527712419048e-17, 7.93973917622607e-18, 4.9670289379751576e-18, 2.3784056040845962e-18, 6.21465665888774e-19, 0.0],
        },
        "q6.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004244330997905205, 0.007754778607735976, 0.009924354546337224, 0.010377919423946159, 0.009037047752716637, 0.006133588426040959, 0.002169575938601259, -0.0021695759386012275, -0.0061335884260409305, -0.009037047752716612, -0.01037791942394614, -0.009924354546337215, -0.007754778607735969, -0.004244330997905203, -1.182407475327649e-17],
        },
        "q6.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.010149304539422525, 0.03884231119928679, 0.08111773846032007, 0.12966578088889005, 0.17609204090397754, 0.21236899637964252] + [0.23222403214834825] * 2 + [0.21236899637964252, 0.1760920409039776, 0.12966578088889016, 0.0811177384603201, 0.038842311199286765, 0.010149304539422511, 7.240157649739542e-34],
        },
        "q6.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 3.1073283294438743e-19, 1.189202802042299e-18, 2.483514468987578e-18, 3.969869588113031e-18, 5.3912638562095224e-18, 6.501925291361626e-18] + [7.109810441389169e-18] * 2 + [6.501925291361626e-18, 5.391263856209524e-18, 3.969869588113035e-18, 2.4835144689875788e-18, 1.1892028020422981e-18, 3.10732832944387e-19, 0.0],
        },
        "q6.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.005074652269711263, 0.019421155599643396, 0.040558869230160034, 0.06483289044444503, 0.08804602045198877, 0.10618449818982126] + [0.11611201607417412] * 2 + [0.10618449818982126, 0.0880460204519888, 0.06483289044444508, 0.04055886923016005, 0.019421155599643383, 0.005074652269711256, 0.0],
        },
        "q6.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 3.1073283294438743e-19, 1.189202802042299e-18, 2.483514468987578e-18, 3.969869588113031e-18, 5.3912638562095224e-18, 6.501925291361626e-18] + [7.109810441389169e-18] * 2 + [6.501925291361626e-18, 5.391263856209524e-18, 3.969869588113035e-18, 2.4835144689875788e-18, 1.1892028020422981e-18, 3.10732832944387e-19, 0.0],
        },
        "q6.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.005074652269711263, -0.019421155599643396, -0.040558869230160034, -0.06483289044444503, -0.08804602045198877, -0.10618449818982126] + [-0.11611201607417412] * 2 + [-0.10618449818982126, -0.0880460204519888, -0.06483289044444508, -0.04055886923016005, -0.019421155599643383, -0.005074652269711256, 0.0],
        },
        "q6.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.31622776601683794,
        },
        "q6.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q6.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.15811388300841897,
        },
        "q6.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q6.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "q6.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q6.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "q6.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q6.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "q6.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q6.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "q6.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q6.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.9,
        },
        "q6.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q6.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 3.168763005159296e-05, 0.00012673348719410162, 0.00028508648154643336, 0.0005066614938209973, 0.0007913394210777118, 0.0011389672407453107, 0.0015493580928753236, 0.002022291380584713, 0.0025575128886330266, 0.0031547349200703977, 0.0038136364508829356, 0.004533863302552358, 0.005315028332437103, 0.006156711641872697, 0.007058460801879262, 0.008019791096355216, 0.00904018578262603, 0.010119096369208278, 0.011255942910639652, 0.01245011431921621, 0.013700968693469729, 0.015007833663208025, 0.016370006750933464, 0.017786755749444636, 0.019257319115418888, 0.02078090637876371, 0.022356698567517187, 0.023983848648068896, 0.0256614819804649, 0.027388696788551832, 0.02916456464470753, 0.030988130968897586, 0.032858415541789476, 0.03477441303164879, 0.036735093534733795, 0.0387394031288984, 0.04078626444010559, 0.042874577221546896, 0.045003218945056836, 0.04717104540450389, 0.049376891330834324, 0.0516195710184378, 0.05389787896249817, 0.05621059050698702, 0.05855646250295156, 0.06093423397674275, 0.06334262680782511, 0.06578034641580306, 0.06824608245629535, 0.07073850952528254, 0.07325628787155, 0.07579806411684264, 0.0783624719833446, 0.08094813302809288, 0.08355365738393013, 0.08617764450659811, 0.08881868392757053, 0.09147535601222055, 0.09414623272291499, 0.09682987838662585, 0.09952485046664568, 0.10222970033799257, 0.10494297406608788, 0.10766321318828753, 0.11038895549784807, 0.11311873582990449, 0.1158510868490386, 0.11858453983801395, 0.1213176254872537, 0.12404887468463659, 0.12677681930518717, 0.12949999300023488, 0.13221693198561887, 0.13492617582851413, 0.13762626823245577, 0.14031575782014064, 0.1429931989135841, 0.14565715231121368, 0.14830618606148113, 0.15093887623257768, 0.15355380767783833, 0.15614957479642377, 0.15872478228887146, 0.16127804590710879, 0.16380799319852676, 0.16631326424371232, 0.16879251238744433, 0.17124440496255933, 0.17366762400629882, 0.17606086696875145, 0.17842284741301176, 0.1807522957066767, 0.18304795970430945, 0.18530860542050392, 0.18753301769318687, 0.1897200008368029, 0.1918683792850299, 0.19397699822267925, 0.19604472420644256, 0.19807044577415003, 0.20005307404221317, 0.20199154329093114, 0.2038848115373457, 0.20573186109533706, 0.20753169912265937, 0.20928335815462196, 0.21098589662412942, 0.21263839936780118, 0.21423997811789788, 0.2157897719797909, 0.2172869478947181, 0.21873070108757675, 0.22012025549951322, 0.22145486420507712, 0.22273380981371502, 0.2239564048553886, 0.2251219921501095, 0.22622994516119255, 0.22727966833203717, 0.22827059740625627, 0.2292021997309801, 0.2300739745431727, 0.23088545323880616, 0.2316361996247491, 0.2323258101532328, 0.23295391413876973, 0.23352017395740768, 0.2340242852282124, 0.2344659769768811, 0.23484501178139874, 0.23516118589965962, 0.23541432937898407, 0.23560430614747335, 0.23573101408715189] + [0.23579438508885894] * 2 + [0.23573101408715189, 0.23560430614747335, 0.23541432937898407, 0.23516118589965965, 0.23484501178139877, 0.2344659769768811, 0.2340242852282124, 0.23352017395740768, 0.23295391413876973, 0.23232581015323284, 0.23163619962474913, 0.23088545323880616, 0.2300739745431727, 0.2292021997309801, 0.22827059740625627, 0.2272796683320372, 0.22622994516119255, 0.2251219921501095, 0.2239564048553886, 0.22273380981371504, 0.22145486420507718, 0.22012025549951325, 0.21873070108757675, 0.21728694789471814, 0.21578977197979093, 0.21423997811789788, 0.21263839936780124, 0.21098589662412948, 0.20928335815462198, 0.20753169912265937, 0.20573186109533706, 0.20388481153734572, 0.20199154329093114, 0.20005307404221315, 0.19807044577415003, 0.1960447242064426, 0.19397699822267928, 0.19186837928502992, 0.189720000836803, 0.18753301769318695, 0.18530860542050387, 0.1830479597043095, 0.18075229570667675, 0.17842284741301173, 0.17606086696875142, 0.17366762400629882, 0.17124440496255944, 0.16879251238744436, 0.1663132642437124, 0.16380799319852687, 0.16127804590710873, 0.15872478228887144, 0.15614957479642383, 0.1535538076778383, 0.1509388762325777, 0.1483061860614812, 0.14565715231121368, 0.14299319891358417, 0.14031575782014072, 0.1376262682324558, 0.13492617582851418, 0.13221693198561893, 0.12949999300023482, 0.12677681930518714, 0.12404887468463664, 0.12131762548725367, 0.11858453983801395, 0.11585108684903864, 0.11311873582990449, 0.11038895549784812, 0.10766321318828762, 0.10494297406608788, 0.10222970033799254, 0.09952485046664568, 0.0968298783866259, 0.09414623272291499, 0.09147535601222058, 0.08881868392757061, 0.08617764450659812, 0.08355365738393018, 0.08094813302809298, 0.07836247198334463, 0.07579806411684263, 0.07325628787155003, 0.0707385095252825, 0.06824608245629535, 0.0657803464158031, 0.06334262680782508, 0.060934233976742774, 0.058556462502951614, 0.05621059050698704, 0.05389787896249822, 0.05161957101843788, 0.049376891330834366, 0.04717104540450386, 0.045003218945056836, 0.042874577221546945, 0.04078626444010557, 0.038739403128898425, 0.03673509353473386, 0.034774413031648806, 0.03285841554178953, 0.030988130968897652, 0.029164564644707573, 0.027388696788551832, 0.025661481980464914, 0.02398384864806887, 0.022356698567517187, 0.02078090637876375, 0.019257319115418888, 0.01778675574944466, 0.016370006750933502, 0.015007833663208037, 0.013700968693469755, 0.012450114319216225, 0.011255942910639626, 0.010119096369208278, 0.009040185782626041, 0.008019791096355242, 0.007058460801879262, 0.00615671164187271, 0.005315028332437129, 0.004533863302552371, 0.0038136364508829616, 0.0031547349200704237, 0.0025575128886330136, 0.002022291380584713, 0.0015493580928753236, 0.0011389672407453107, 0.0007913394210777118, 0.0005066614938209973, 0.00028508648154643336, 0.00012673348719410162, 3.168763005159296e-05, 0.0],
        },
        "q6.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 272,
        },
        "q6.z.const.wf": {
            "type": "constant",
            "sample": 1.25,
        },
        "q6.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.020839591836734693,
        },
        "q6.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q6.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "q6.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q7.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.00010369801696355364, 0.00041376147037330623, 0.0009271088103273131, 0.0016386381603299622, 0.0025412780220327082, 0.003626057554961664, 0.004882195732760553, 0.006297208489874216, 0.007857032793800813, 0.009546166409826912, 0.011347821969200736, 0.013244093809544888, 0.015216135929373494, 0.01724434928812192, 0.01930857659021714, 0.021388302617337186, 0.023462858117867694, 0.02551162522721051, 0.02751424237738503, 0.02945080665943862, 0.03130207162749756, 0.03304963857859294, 0.03467613940723678, 0.03616540921745729, 0.03750264697679746, 0.03867456261562523, 0.039669509109816314, 0.04047759823411264, 0.04109079883574741, 0.04150301665164941] + [0.041710154875967465] * 2 + [0.04150301665164941, 0.04109079883574741, 0.04047759823411264, 0.039669509109816314, 0.03867456261562523, 0.03750264697679747, 0.036165409217457305, 0.03467613940723679, 0.03304963857859296, 0.03130207162749755, 0.029450806659438625, 0.027514242377385045, 0.025511625227210514, 0.02346285811786771, 0.021388302617337193, 0.01930857659021714, 0.01724434928812193, 0.015216135929373495, 0.013244093809544897, 0.011347821969200741, 0.009546166409826907, 0.007857032793800818, 0.0062972084898742136, 0.004882195732760558, 0.003626057554961673, 0.002541278022032713, 0.0016386381603299622, 0.0009271088103273154, 0.00041376147037330623, 0.00010369801696355596, 0.0],
        },
        "q7.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 64,
        },
        "q7.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 5.184900848177682e-05, 0.00020688073518665312, 0.00046355440516365656, 0.0008193190801649811, 0.0012706390110163541, 0.001813028777480832, 0.0024410978663802767, 0.003148604244937108, 0.003928516396900407, 0.004773083204913456, 0.005673910984600368, 0.006622046904772444, 0.007608067964686747, 0.00862217464406096, 0.00965428829510857, 0.010694151308668593, 0.011731429058933847, 0.012755812613605255, 0.013757121188692515, 0.01472540332971931, 0.01565103581374878, 0.01652481928929647, 0.01733806970361839, 0.018082704608728645, 0.01875132348839873, 0.019337281307812614, 0.019834754554908157, 0.02023879911705632, 0.020545399417873703, 0.020751508325824706] + [0.020855077437983732] * 2 + [0.020751508325824706, 0.020545399417873703, 0.02023879911705632, 0.019834754554908157, 0.019337281307812614, 0.018751323488398735, 0.018082704608728652, 0.017338069703618394, 0.01652481928929648, 0.015651035813748774, 0.014725403329719312, 0.013757121188692522, 0.012755812613605257, 0.011731429058933854, 0.010694151308668597, 0.00965428829510857, 0.008622174644060966, 0.007608067964686748, 0.006622046904772448, 0.005673910984600371, 0.004773083204913453, 0.003928516396900409, 0.0031486042449371068, 0.002441097866380279, 0.0018130287774808366, 0.0012706390110163565, 0.0008193190801649811, 0.0004635544051636577, 0.00020688073518665312, 5.184900848177798e-05, 0.0],
        },
        "q7.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 64,
        },
        "q7.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -5.184900848177682e-05, -0.00020688073518665312, -0.00046355440516365656, -0.0008193190801649811, -0.0012706390110163541, -0.001813028777480832, -0.0024410978663802767, -0.003148604244937108, -0.003928516396900407, -0.004773083204913456, -0.005673910984600368, -0.006622046904772444, -0.007608067964686747, -0.00862217464406096, -0.00965428829510857, -0.010694151308668593, -0.011731429058933847, -0.012755812613605255, -0.013757121188692515, -0.01472540332971931, -0.01565103581374878, -0.01652481928929647, -0.01733806970361839, -0.018082704608728645, -0.01875132348839873, -0.019337281307812614, -0.019834754554908157, -0.02023879911705632, -0.020545399417873703, -0.020751508325824706] + [-0.020855077437983732] * 2 + [-0.020751508325824706, -0.020545399417873703, -0.02023879911705632, -0.019834754554908157, -0.019337281307812614, -0.018751323488398735, -0.018082704608728652, -0.017338069703618394, -0.01652481928929648, -0.015651035813748774, -0.014725403329719312, -0.013757121188692522, -0.012755812613605257, -0.011731429058933854, -0.010694151308668597, -0.00965428829510857, -0.008622174644060966, -0.007608067964686748, -0.006622046904772448, -0.005673910984600371, -0.004773083204913453, -0.003928516396900409, -0.0031486042449371068, -0.002441097866380279, -0.0018130287774808366, -0.0012706390110163565, -0.0008193190801649811, -0.0004635544051636577, -0.00020688073518665312, -5.184900848177798e-05, 0.0],
        },
        "q7.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 6.3496722276171956e-21, 2.5335583015158596e-20, 5.676904185143273e-20, 1.0033764890043977e-19, 1.5560839977129365e-19, 2.22031988910394e-19, 2.989482688468039e-19, 3.8559281103439985e-19, 4.811045030861976e-19, 5.845341068961254e-19, 6.948536925937848e-19, 8.109668545733211e-19, 9.317196080649142e-19, 1.0559118579538724e-18, 1.1823093258650467e-18, 1.309655816975847e-18, 1.4366857046447582e-18, 1.5621365087775109e-18, 1.6847614429214519e-18, 1.803341805389453e-18, 1.9166990912648035e-18, 2.023706704912536e-18, 2.123301156592996e-18, 2.214492631900663e-18, 2.296374828984409e-18, 2.368133965782466e-18, 2.429056867754166e-18, 2.4785380557289302e-18, 2.5160857634302926e-18, 2.5413268248700876e-18] + [2.554010383039696e-18] * 2 + [2.5413268248700876e-18, 2.5160857634302926e-18, 2.4785380557289302e-18, 2.429056867754166e-18, 2.368133965782466e-18, 2.2963748289844092e-18, 2.2144926319006636e-18, 2.1233011565929965e-18, 2.023706704912537e-18, 1.9166990912648027e-18, 1.803341805389453e-18, 1.6847614429214529e-18, 1.562136508777511e-18, 1.4366857046447591e-18, 1.3096558169758475e-18, 1.1823093258650467e-18, 1.0559118579538732e-18, 9.317196080649144e-19, 8.109668545733217e-19, 6.948536925937852e-19, 5.845341068961251e-19, 4.811045030861979e-19, 3.8559281103439966e-19, 2.989482688468042e-19, 2.2203198891039454e-19, 1.5560839977129394e-19, 1.0033764890043977e-19, 5.676904185143287e-20, 2.5335583015158596e-20, 6.349672227617337e-21, 0.0],
        },
        "q7.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 6.3496722276171956e-21, 2.5335583015158596e-20, 5.676904185143273e-20, 1.0033764890043977e-19, 1.5560839977129365e-19, 2.22031988910394e-19, 2.989482688468039e-19, 3.8559281103439985e-19, 4.811045030861976e-19, 5.845341068961254e-19, 6.948536925937848e-19, 8.109668545733211e-19, 9.317196080649142e-19, 1.0559118579538724e-18, 1.1823093258650467e-18, 1.309655816975847e-18, 1.4366857046447582e-18, 1.5621365087775109e-18, 1.6847614429214519e-18, 1.803341805389453e-18, 1.9166990912648035e-18, 2.023706704912536e-18, 2.123301156592996e-18, 2.214492631900663e-18, 2.296374828984409e-18, 2.368133965782466e-18, 2.429056867754166e-18, 2.4785380557289302e-18, 2.5160857634302926e-18, 2.5413268248700876e-18] + [2.554010383039696e-18] * 2 + [2.5413268248700876e-18, 2.5160857634302926e-18, 2.4785380557289302e-18, 2.429056867754166e-18, 2.368133965782466e-18, 2.2963748289844092e-18, 2.2144926319006636e-18, 2.1233011565929965e-18, 2.023706704912537e-18, 1.9166990912648027e-18, 1.803341805389453e-18, 1.6847614429214529e-18, 1.562136508777511e-18, 1.4366857046447591e-18, 1.3096558169758475e-18, 1.1823093258650467e-18, 1.0559118579538732e-18, 9.317196080649144e-19, 8.109668545733217e-19, 6.948536925937852e-19, 5.845341068961251e-19, 4.811045030861979e-19, 3.8559281103439966e-19, 2.989482688468042e-19, 2.2203198891039454e-19, 1.5560839977129394e-19, 1.0033764890043977e-19, 5.676904185143287e-20, 2.5335583015158596e-20, 6.349672227617337e-21, 0.0],
        },
        "q7.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.00010369801696355364, 0.00041376147037330623, 0.0009271088103273131, 0.0016386381603299622, 0.0025412780220327082, 0.003626057554961664, 0.004882195732760553, 0.006297208489874216, 0.007857032793800813, 0.009546166409826912, 0.011347821969200736, 0.013244093809544888, 0.015216135929373494, 0.01724434928812192, 0.01930857659021714, 0.021388302617337186, 0.023462858117867694, 0.02551162522721051, 0.02751424237738503, 0.02945080665943862, 0.03130207162749756, 0.03304963857859294, 0.03467613940723678, 0.03616540921745729, 0.03750264697679746, 0.03867456261562523, 0.039669509109816314, 0.04047759823411264, 0.04109079883574741, 0.04150301665164941] + [0.041710154875967465] * 2 + [0.04150301665164941, 0.04109079883574741, 0.04047759823411264, 0.039669509109816314, 0.03867456261562523, 0.03750264697679747, 0.036165409217457305, 0.03467613940723679, 0.03304963857859296, 0.03130207162749755, 0.029450806659438625, 0.027514242377385045, 0.025511625227210514, 0.02346285811786771, 0.021388302617337193, 0.01930857659021714, 0.01724434928812193, 0.015216135929373495, 0.013244093809544897, 0.011347821969200741, 0.009546166409826907, 0.007857032793800818, 0.0062972084898742136, 0.004882195732760558, 0.003626057554961673, 0.002541278022032713, 0.0016386381603299622, 0.0009271088103273154, 0.00041376147037330623, 0.00010369801696355596, 0.0],
        },
        "q7.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 3.1748361138085978e-21, 1.2667791507579298e-20, 2.8384520925716364e-20, 5.0168824450219887e-20, 7.780419988564683e-20, 1.11015994455197e-19, 1.4947413442340195e-19, 1.9279640551719992e-19, 2.405522515430988e-19, 2.922670534480627e-19, 3.474268462968924e-19, 4.0548342728666054e-19, 4.658598040324571e-19, 5.279559289769362e-19, 5.911546629325234e-19, 6.548279084879235e-19, 7.183428523223791e-19, 7.810682543887554e-19, 8.4238072146072595e-19, 9.016709026947266e-19, 9.583495456324017e-19, 1.011853352456268e-18, 1.061650578296498e-18, 1.1072463159503314e-18, 1.1481874144922044e-18, 1.184066982891233e-18, 1.214528433877083e-18, 1.2392690278644651e-18, 1.2580428817151463e-18, 1.2706634124350438e-18] + [1.277005191519848e-18] * 2 + [1.2706634124350438e-18, 1.2580428817151463e-18, 1.2392690278644651e-18, 1.214528433877083e-18, 1.184066982891233e-18, 1.1481874144922046e-18, 1.1072463159503318e-18, 1.0616505782964983e-18, 1.0118533524562686e-18, 9.583495456324014e-19, 9.016709026947266e-19, 8.423807214607264e-19, 7.810682543887555e-19, 7.183428523223796e-19, 6.548279084879237e-19, 5.911546629325234e-19, 5.279559289769366e-19, 4.658598040324572e-19, 4.0548342728666083e-19, 3.474268462968926e-19, 2.9226705344806255e-19, 2.4055225154309894e-19, 1.9279640551719983e-19, 1.494741344234021e-19, 1.1101599445519727e-19, 7.780419988564697e-20, 5.0168824450219887e-20, 2.8384520925716437e-20, 1.2667791507579298e-20, 3.1748361138086685e-21, 0.0],
        },
        "q7.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 5.184900848177682e-05, 0.00020688073518665312, 0.00046355440516365656, 0.0008193190801649811, 0.0012706390110163541, 0.001813028777480832, 0.0024410978663802767, 0.003148604244937108, 0.003928516396900407, 0.004773083204913456, 0.005673910984600368, 0.006622046904772444, 0.007608067964686747, 0.00862217464406096, 0.00965428829510857, 0.010694151308668593, 0.011731429058933847, 0.012755812613605255, 0.013757121188692515, 0.01472540332971931, 0.01565103581374878, 0.01652481928929647, 0.01733806970361839, 0.018082704608728645, 0.01875132348839873, 0.019337281307812614, 0.019834754554908157, 0.02023879911705632, 0.020545399417873703, 0.020751508325824706] + [0.020855077437983732] * 2 + [0.020751508325824706, 0.020545399417873703, 0.02023879911705632, 0.019834754554908157, 0.019337281307812614, 0.018751323488398735, 0.018082704608728652, 0.017338069703618394, 0.01652481928929648, 0.015651035813748774, 0.014725403329719312, 0.013757121188692522, 0.012755812613605257, 0.011731429058933854, 0.010694151308668597, 0.00965428829510857, 0.008622174644060966, 0.007608067964686748, 0.006622046904772448, 0.005673910984600371, 0.004773083204913453, 0.003928516396900409, 0.0031486042449371068, 0.002441097866380279, 0.0018130287774808366, 0.0012706390110163565, 0.0008193190801649811, 0.0004635544051636577, 0.00020688073518665312, 5.184900848177798e-05, 0.0],
        },
        "q7.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 3.1748361138085978e-21, 1.2667791507579298e-20, 2.8384520925716364e-20, 5.0168824450219887e-20, 7.780419988564683e-20, 1.11015994455197e-19, 1.4947413442340195e-19, 1.9279640551719992e-19, 2.405522515430988e-19, 2.922670534480627e-19, 3.474268462968924e-19, 4.0548342728666054e-19, 4.658598040324571e-19, 5.279559289769362e-19, 5.911546629325234e-19, 6.548279084879235e-19, 7.183428523223791e-19, 7.810682543887554e-19, 8.4238072146072595e-19, 9.016709026947266e-19, 9.583495456324017e-19, 1.011853352456268e-18, 1.061650578296498e-18, 1.1072463159503314e-18, 1.1481874144922044e-18, 1.184066982891233e-18, 1.214528433877083e-18, 1.2392690278644651e-18, 1.2580428817151463e-18, 1.2706634124350438e-18] + [1.277005191519848e-18] * 2 + [1.2706634124350438e-18, 1.2580428817151463e-18, 1.2392690278644651e-18, 1.214528433877083e-18, 1.184066982891233e-18, 1.1481874144922046e-18, 1.1072463159503318e-18, 1.0616505782964983e-18, 1.0118533524562686e-18, 9.583495456324014e-19, 9.016709026947266e-19, 8.423807214607264e-19, 7.810682543887555e-19, 7.183428523223796e-19, 6.548279084879237e-19, 5.911546629325234e-19, 5.279559289769366e-19, 4.658598040324572e-19, 4.0548342728666083e-19, 3.474268462968926e-19, 2.9226705344806255e-19, 2.4055225154309894e-19, 1.9279640551719983e-19, 1.494741344234021e-19, 1.1101599445519727e-19, 7.780419988564697e-20, 5.0168824450219887e-20, 2.8384520925716437e-20, 1.2667791507579298e-20, 3.1748361138086685e-21, 0.0],
        },
        "q7.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -5.184900848177682e-05, -0.00020688073518665312, -0.00046355440516365656, -0.0008193190801649811, -0.0012706390110163541, -0.001813028777480832, -0.0024410978663802767, -0.003148604244937108, -0.003928516396900407, -0.004773083204913456, -0.005673910984600368, -0.006622046904772444, -0.007608067964686747, -0.00862217464406096, -0.00965428829510857, -0.010694151308668593, -0.011731429058933847, -0.012755812613605255, -0.013757121188692515, -0.01472540332971931, -0.01565103581374878, -0.01652481928929647, -0.01733806970361839, -0.018082704608728645, -0.01875132348839873, -0.019337281307812614, -0.019834754554908157, -0.02023879911705632, -0.020545399417873703, -0.020751508325824706] + [-0.020855077437983732] * 2 + [-0.020751508325824706, -0.020545399417873703, -0.02023879911705632, -0.019834754554908157, -0.019337281307812614, -0.018751323488398735, -0.018082704608728652, -0.017338069703618394, -0.01652481928929648, -0.015651035813748774, -0.014725403329719312, -0.013757121188692522, -0.012755812613605257, -0.011731429058933854, -0.010694151308668597, -0.00965428829510857, -0.008622174644060966, -0.007608067964686748, -0.006622046904772448, -0.005673910984600371, -0.004773083204913453, -0.003928516396900409, -0.0031486042449371068, -0.002441097866380279, -0.0018130287774808366, -0.0012706390110163565, -0.0008193190801649811, -0.0004635544051636577, -0.00020688073518665312, -5.184900848177798e-05, 0.0],
        },
        "q7.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.31622776601683794,
        },
        "q7.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q7.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.15811388300841897,
        },
        "q7.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q7.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "q7.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q7.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "q7.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q7.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "q7.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q7.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "q7.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q7.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q7.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q7.z.const.wf": {
            "type": "constant",
            "sample": 1.25,
        },
        "q7.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.01681561085678195,
        },
        "q7.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q7.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "q7.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q8.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.006911554496302379, 0.026451147423297412, 0.05524020565227865, 0.0883008394750719, 0.11991656495728692, 0.14462073594319957] + [0.1581417768389978] * 2 + [0.14462073594319957, 0.11991656495728698, 0.08830083947507197, 0.05524020565227867, 0.026451147423297395, 0.006911554496302369, 0.0],
        },
        "q8.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
        },
        "q8.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0034557772481511894, 0.013225573711648706, 0.027620102826139324, 0.04415041973753595, 0.05995828247864346, 0.07231036797159979] + [0.0790708884194989] * 2 + [0.07231036797159979, 0.05995828247864349, 0.04415041973753599, 0.027620102826139335, 0.013225573711648697, 0.0034557772481511847, 0.0],
        },
        "q8.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
        },
        "q8.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0034557772481511894, -0.013225573711648706, -0.027620102826139324, -0.04415041973753595, -0.05995828247864346, -0.07231036797159979] + [-0.0790708884194989] * 2 + [-0.07231036797159979, -0.05995828247864349, -0.04415041973753599, -0.027620102826139335, -0.013225573711648697, -0.0034557772481511847, 0.0],
        },
        "q8.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 4.2321065455146026e-19, 1.6196656512857967e-18, 3.382487051815229e-18, 5.406867021258553e-18, 7.342771871984354e-18, 8.855466068158696e-18] + [9.683391040867685e-18] * 2 + [8.855466068158696e-18, 7.342771871984358e-18, 5.406867021258557e-18, 3.38248705181523e-18, 1.6196656512857958e-18, 4.232106545514597e-19, 0.0],
        },
        "q8.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 4.2321065455146026e-19, 1.6196656512857967e-18, 3.382487051815229e-18, 5.406867021258553e-18, 7.342771871984354e-18, 8.855466068158696e-18] + [9.683391040867685e-18] * 2 + [8.855466068158696e-18, 7.342771871984358e-18, 5.406867021258557e-18, 3.38248705181523e-18, 1.6196656512857958e-18, 4.232106545514597e-19, 0.0],
        },
        "q8.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.006911554496302379, 0.026451147423297412, 0.05524020565227865, 0.0883008394750719, 0.11991656495728692, 0.14462073594319957] + [0.1581417768389978] * 2 + [0.14462073594319957, 0.11991656495728698, 0.08830083947507197, 0.05524020565227867, 0.026451147423297395, 0.006911554496302369, 0.0],
        },
        "q8.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 2.1160532727573013e-19, 8.098328256428984e-19, 1.6912435259076145e-18, 2.7034335106292763e-18, 3.671385935992177e-18, 4.427733034079348e-18] + [4.8416955204338425e-18] * 2 + [4.427733034079348e-18, 3.671385935992179e-18, 2.7034335106292786e-18, 1.691243525907615e-18, 8.098328256428979e-19, 2.1160532727572984e-19, 0.0],
        },
        "q8.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0034557772481511894, 0.013225573711648706, 0.027620102826139324, 0.04415041973753595, 0.05995828247864346, 0.07231036797159979] + [0.0790708884194989] * 2 + [0.07231036797159979, 0.05995828247864349, 0.04415041973753599, 0.027620102826139335, 0.013225573711648697, 0.0034557772481511847, 0.0],
        },
        "q8.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 2.1160532727573013e-19, 8.098328256428984e-19, 1.6912435259076145e-18, 2.7034335106292763e-18, 3.671385935992177e-18, 4.427733034079348e-18] + [4.8416955204338425e-18] * 2 + [4.427733034079348e-18, 3.671385935992179e-18, 2.7034335106292786e-18, 1.691243525907615e-18, 8.098328256428979e-19, 2.1160532727572984e-19, 0.0],
        },
        "q8.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0034557772481511894, -0.013225573711648706, -0.027620102826139324, -0.04415041973753595, -0.05995828247864346, -0.07231036797159979] + [-0.0790708884194989] * 2 + [-0.07231036797159979, -0.05995828247864349, -0.04415041973753599, -0.027620102826139335, -0.013225573711648697, -0.0034557772481511847, 0.0],
        },
        "q8.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.31622776601683794,
        },
        "q8.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q8.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.15811388300841897,
        },
        "q8.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q8.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "q8.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q8.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "q8.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q8.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "q8.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q8.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "q8.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q8.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.95,
        },
        "q8.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q8.z.const.wf": {
            "type": "constant",
            "sample": 1.25,
        },
        "q8.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.01820124141779137,
        },
        "q8.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q8.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "q8.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q9.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.00044756053064166733, 0.0017857940649379934, 0.004001400636798037, 0.007072360552706858, 0.010968153233332464, 0.015650060540755153, 0.02107155157671702, 0.027178745127593742, 0.03391094516010276, 0.04120124404575092, 0.0489771875189059, 0.057161494759839615, 0.06567282644623451, 0.07442659313991205, 0.08333579597467274, 0.09231189129011431, 0.10126567061830875, 0.11010814727763525, 0.118751440762401, 0.1271096501387889, 0.1350997077669304, 0.1426422048644302, 0.14966218070652176, 0.15608986761942747, 0.16186138436284517, 0.16691937101041399, 0.17121355901843288, 0.1747012708172317, 0.1773478439600325, 0.17912697561391838] + [0.18002098396920393] * 2 + [0.17912697561391838, 0.1773478439600325, 0.1747012708172317, 0.17121355901843288, 0.16691937101041399, 0.1618613843628452, 0.15608986761942753, 0.1496621807065218, 0.14264220486443027, 0.13509970776693034, 0.12710965013878892, 0.11875144076240106, 0.11010814727763527, 0.10126567061830881, 0.09231189129011433, 0.08333579597467274, 0.0744265931399121, 0.06567282644623453, 0.05716149475983966, 0.048977187518905924, 0.0412012440457509, 0.03391094516010278, 0.02717874512759373, 0.02107155157671704, 0.01565006054075519, 0.010968153233332485, 0.007072360552706858, 0.004001400636798047, 0.0017857940649379934, 0.0004475605306416773, 0.0],
        },
        "q9.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 64,
        },
        "q9.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.00022378026532083366, 0.0008928970324689967, 0.0020007003183990183, 0.003536180276353429, 0.005484076616666232, 0.007825030270377576, 0.01053577578835851, 0.013589372563796871, 0.01695547258005138, 0.02060062202287546, 0.02448859375945295, 0.028580747379919808, 0.032836413223117256, 0.03721329656995603, 0.04166789798733637, 0.04615594564505716, 0.05063283530915438, 0.055054073638817626, 0.0593757203812005, 0.06355482506939444, 0.0675498538834652, 0.0713211024322151, 0.07483109035326088, 0.07804493380971374, 0.08093069218142258, 0.08345968550520699, 0.08560677950921644, 0.08735063540861585, 0.08867392198001625, 0.08956348780695919] + [0.09001049198460197] * 2 + [0.08956348780695919, 0.08867392198001625, 0.08735063540861585, 0.08560677950921644, 0.08345968550520699, 0.0809306921814226, 0.07804493380971377, 0.0748310903532609, 0.07132110243221514, 0.06754985388346517, 0.06355482506939446, 0.05937572038120053, 0.05505407363881763, 0.050632835309154405, 0.046155945645057164, 0.04166789798733637, 0.03721329656995605, 0.03283641322311726, 0.02858074737991983, 0.024488593759452962, 0.02060062202287545, 0.01695547258005139, 0.013589372563796866, 0.01053577578835852, 0.007825030270377595, 0.005484076616666242, 0.003536180276353429, 0.0020007003183990235, 0.0008928970324689967, 0.00022378026532083865, 0.0],
        },
        "q9.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 64,
        },
        "q9.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.00022378026532083366, -0.0008928970324689967, -0.0020007003183990183, -0.003536180276353429, -0.005484076616666232, -0.007825030270377576, -0.01053577578835851, -0.013589372563796871, -0.01695547258005138, -0.02060062202287546, -0.02448859375945295, -0.028580747379919808, -0.032836413223117256, -0.03721329656995603, -0.04166789798733637, -0.04615594564505716, -0.05063283530915438, -0.055054073638817626, -0.0593757203812005, -0.06355482506939444, -0.0675498538834652, -0.0713211024322151, -0.07483109035326088, -0.07804493380971374, -0.08093069218142258, -0.08345968550520699, -0.08560677950921644, -0.08735063540861585, -0.08867392198001625, -0.08956348780695919] + [-0.09001049198460197] * 2 + [-0.08956348780695919, -0.08867392198001625, -0.08735063540861585, -0.08560677950921644, -0.08345968550520699, -0.0809306921814226, -0.07804493380971377, -0.0748310903532609, -0.07132110243221514, -0.06754985388346517, -0.06355482506939446, -0.05937572038120053, -0.05505407363881763, -0.050632835309154405, -0.046155945645057164, -0.04166789798733637, -0.03721329656995605, -0.03283641322311726, -0.02858074737991983, -0.024488593759452962, -0.02060062202287545, -0.01695547258005139, -0.013589372563796866, -0.01053577578835852, -0.007825030270377595, -0.005484076616666242, -0.003536180276353429, -0.0020007003183990235, -0.0008928970324689967, -0.00022378026532083865, 0.0],
        },
        "q9.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 2.740517856375044e-20, 1.0934834927813272e-19, 2.450151240980448e-19, 4.3305718566442297e-19, 6.716056874879148e-19, 9.582898273849047e-19, 1.290260409574743e-18, 1.6642181612674699e-18, 2.076446522319064e-18, 2.5228485820758904e-18, 2.9989877963133907e-18, 3.500132079605789e-18, 4.021300834917037e-18, 4.5573144530117824e-18, 5.102845789738992e-18, 5.652473109583847e-18, 6.200733969311099e-18, 6.742179506180068e-18, 7.271428591190546e-18, 7.783221309160386e-18, 8.272471234125706e-18, 8.734315980527272e-18, 9.164165527782732e-18, 9.557747837973297e-18, 9.911151313275889e-18, 1.022086367117965e-17, 1.0483806851127514e-17, 1.0697367605664886e-17, 1.0859423472066903e-17, 1.0968363866326557e-17] + [1.102310608986213e-17] * 2 + [1.0968363866326557e-17, 1.0859423472066903e-17, 1.0697367605664886e-17, 1.0483806851127514e-17, 1.022086367117965e-17, 9.91115131327589e-18, 9.5577478379733e-18, 9.164165527782734e-18, 8.734315980527278e-18, 8.272471234125703e-18, 7.783221309160389e-18, 7.271428591190549e-18, 6.742179506180069e-18, 6.200733969311103e-18, 5.6524731095838475e-18, 5.102845789738992e-18, 4.5573144530117855e-18, 4.021300834917038e-18, 3.500132079605792e-18, 2.998987796313392e-18, 2.5228485820758893e-18, 2.076446522319065e-18, 1.6642181612674693e-18, 1.2902604095747443e-18, 9.58289827384907e-19, 6.71605687487916e-19, 4.3305718566442297e-19, 2.4501512409804543e-19, 1.0934834927813272e-19, 2.740517856375105e-20, 0.0],
        },
        "q9.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 2.740517856375044e-20, 1.0934834927813272e-19, 2.450151240980448e-19, 4.3305718566442297e-19, 6.716056874879148e-19, 9.582898273849047e-19, 1.290260409574743e-18, 1.6642181612674699e-18, 2.076446522319064e-18, 2.5228485820758904e-18, 2.9989877963133907e-18, 3.500132079605789e-18, 4.021300834917037e-18, 4.5573144530117824e-18, 5.102845789738992e-18, 5.652473109583847e-18, 6.200733969311099e-18, 6.742179506180068e-18, 7.271428591190546e-18, 7.783221309160386e-18, 8.272471234125706e-18, 8.734315980527272e-18, 9.164165527782732e-18, 9.557747837973297e-18, 9.911151313275889e-18, 1.022086367117965e-17, 1.0483806851127514e-17, 1.0697367605664886e-17, 1.0859423472066903e-17, 1.0968363866326557e-17] + [1.102310608986213e-17] * 2 + [1.0968363866326557e-17, 1.0859423472066903e-17, 1.0697367605664886e-17, 1.0483806851127514e-17, 1.022086367117965e-17, 9.91115131327589e-18, 9.5577478379733e-18, 9.164165527782734e-18, 8.734315980527278e-18, 8.272471234125703e-18, 7.783221309160389e-18, 7.271428591190549e-18, 6.742179506180069e-18, 6.200733969311103e-18, 5.6524731095838475e-18, 5.102845789738992e-18, 4.5573144530117855e-18, 4.021300834917038e-18, 3.500132079605792e-18, 2.998987796313392e-18, 2.5228485820758893e-18, 2.076446522319065e-18, 1.6642181612674693e-18, 1.2902604095747443e-18, 9.58289827384907e-19, 6.71605687487916e-19, 4.3305718566442297e-19, 2.4501512409804543e-19, 1.0934834927813272e-19, 2.740517856375105e-20, 0.0],
        },
        "q9.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.00044756053064166733, 0.0017857940649379934, 0.004001400636798037, 0.007072360552706858, 0.010968153233332464, 0.015650060540755153, 0.02107155157671702, 0.027178745127593742, 0.03391094516010276, 0.04120124404575092, 0.0489771875189059, 0.057161494759839615, 0.06567282644623451, 0.07442659313991205, 0.08333579597467274, 0.09231189129011431, 0.10126567061830875, 0.11010814727763525, 0.118751440762401, 0.1271096501387889, 0.1350997077669304, 0.1426422048644302, 0.14966218070652176, 0.15608986761942747, 0.16186138436284517, 0.16691937101041399, 0.17121355901843288, 0.1747012708172317, 0.1773478439600325, 0.17912697561391838] + [0.18002098396920393] * 2 + [0.17912697561391838, 0.1773478439600325, 0.1747012708172317, 0.17121355901843288, 0.16691937101041399, 0.1618613843628452, 0.15608986761942753, 0.1496621807065218, 0.14264220486443027, 0.13509970776693034, 0.12710965013878892, 0.11875144076240106, 0.11010814727763527, 0.10126567061830881, 0.09231189129011433, 0.08333579597467274, 0.0744265931399121, 0.06567282644623453, 0.05716149475983966, 0.048977187518905924, 0.0412012440457509, 0.03391094516010278, 0.02717874512759373, 0.02107155157671704, 0.01565006054075519, 0.010968153233332485, 0.007072360552706858, 0.004001400636798047, 0.0017857940649379934, 0.0004475605306416773, 0.0],
        },
        "q9.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 1.370258928187522e-20, 5.467417463906636e-20, 1.225075620490224e-19, 2.1652859283221149e-19, 3.358028437439574e-19, 4.791449136924524e-19, 6.451302047873715e-19, 8.321090806337349e-19, 1.038223261159532e-18, 1.2614242910379452e-18, 1.4994938981566953e-18, 1.7500660398028946e-18, 2.0106504174585186e-18, 2.2786572265058912e-18, 2.551422894869496e-18, 2.8262365547919234e-18, 3.1003669846555495e-18, 3.371089753090034e-18, 3.635714295595273e-18, 3.891610654580193e-18, 4.136235617062853e-18, 4.367157990263636e-18, 4.582082763891366e-18, 4.778873918986649e-18, 4.955575656637944e-18, 5.110431835589825e-18, 5.241903425563757e-18, 5.348683802832443e-18, 5.4297117360334516e-18, 5.484181933163278e-18] + [5.511553044931065e-18] * 2 + [5.484181933163278e-18, 5.4297117360334516e-18, 5.348683802832443e-18, 5.241903425563757e-18, 5.110431835589825e-18, 4.955575656637945e-18, 4.77887391898665e-18, 4.582082763891367e-18, 4.367157990263639e-18, 4.1362356170628515e-18, 3.8916106545801945e-18, 3.6357142955952746e-18, 3.3710897530900345e-18, 3.1003669846555515e-18, 2.8262365547919238e-18, 2.551422894869496e-18, 2.2786572265058927e-18, 2.010650417458519e-18, 1.750066039802896e-18, 1.499493898156696e-18, 1.2614242910379446e-18, 1.0382232611595326e-18, 8.3210908063373465e-19, 6.451302047873722e-19, 4.791449136924535e-19, 3.35802843743958e-19, 2.1652859283221149e-19, 1.2250756204902272e-19, 5.467417463906636e-20, 1.3702589281875524e-20, 0.0],
        },
        "q9.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.00022378026532083366, 0.0008928970324689967, 0.0020007003183990183, 0.003536180276353429, 0.005484076616666232, 0.007825030270377576, 0.01053577578835851, 0.013589372563796871, 0.01695547258005138, 0.02060062202287546, 0.02448859375945295, 0.028580747379919808, 0.032836413223117256, 0.03721329656995603, 0.04166789798733637, 0.04615594564505716, 0.05063283530915438, 0.055054073638817626, 0.0593757203812005, 0.06355482506939444, 0.0675498538834652, 0.0713211024322151, 0.07483109035326088, 0.07804493380971374, 0.08093069218142258, 0.08345968550520699, 0.08560677950921644, 0.08735063540861585, 0.08867392198001625, 0.08956348780695919] + [0.09001049198460197] * 2 + [0.08956348780695919, 0.08867392198001625, 0.08735063540861585, 0.08560677950921644, 0.08345968550520699, 0.0809306921814226, 0.07804493380971377, 0.0748310903532609, 0.07132110243221514, 0.06754985388346517, 0.06355482506939446, 0.05937572038120053, 0.05505407363881763, 0.050632835309154405, 0.046155945645057164, 0.04166789798733637, 0.03721329656995605, 0.03283641322311726, 0.02858074737991983, 0.024488593759452962, 0.02060062202287545, 0.01695547258005139, 0.013589372563796866, 0.01053577578835852, 0.007825030270377595, 0.005484076616666242, 0.003536180276353429, 0.0020007003183990235, 0.0008928970324689967, 0.00022378026532083865, 0.0],
        },
        "q9.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 1.370258928187522e-20, 5.467417463906636e-20, 1.225075620490224e-19, 2.1652859283221149e-19, 3.358028437439574e-19, 4.791449136924524e-19, 6.451302047873715e-19, 8.321090806337349e-19, 1.038223261159532e-18, 1.2614242910379452e-18, 1.4994938981566953e-18, 1.7500660398028946e-18, 2.0106504174585186e-18, 2.2786572265058912e-18, 2.551422894869496e-18, 2.8262365547919234e-18, 3.1003669846555495e-18, 3.371089753090034e-18, 3.635714295595273e-18, 3.891610654580193e-18, 4.136235617062853e-18, 4.367157990263636e-18, 4.582082763891366e-18, 4.778873918986649e-18, 4.955575656637944e-18, 5.110431835589825e-18, 5.241903425563757e-18, 5.348683802832443e-18, 5.4297117360334516e-18, 5.484181933163278e-18] + [5.511553044931065e-18] * 2 + [5.484181933163278e-18, 5.4297117360334516e-18, 5.348683802832443e-18, 5.241903425563757e-18, 5.110431835589825e-18, 4.955575656637945e-18, 4.77887391898665e-18, 4.582082763891367e-18, 4.367157990263639e-18, 4.1362356170628515e-18, 3.8916106545801945e-18, 3.6357142955952746e-18, 3.3710897530900345e-18, 3.1003669846555515e-18, 2.8262365547919238e-18, 2.551422894869496e-18, 2.2786572265058927e-18, 2.010650417458519e-18, 1.750066039802896e-18, 1.499493898156696e-18, 1.2614242910379446e-18, 1.0382232611595326e-18, 8.3210908063373465e-19, 6.451302047873722e-19, 4.791449136924535e-19, 3.35802843743958e-19, 2.1652859283221149e-19, 1.2250756204902272e-19, 5.467417463906636e-20, 1.3702589281875524e-20, 0.0],
        },
        "q9.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.00022378026532083366, -0.0008928970324689967, -0.0020007003183990183, -0.003536180276353429, -0.005484076616666232, -0.007825030270377576, -0.01053577578835851, -0.013589372563796871, -0.01695547258005138, -0.02060062202287546, -0.02448859375945295, -0.028580747379919808, -0.032836413223117256, -0.03721329656995603, -0.04166789798733637, -0.04615594564505716, -0.05063283530915438, -0.055054073638817626, -0.0593757203812005, -0.06355482506939444, -0.0675498538834652, -0.0713211024322151, -0.07483109035326088, -0.07804493380971374, -0.08093069218142258, -0.08345968550520699, -0.08560677950921644, -0.08735063540861585, -0.08867392198001625, -0.08956348780695919] + [-0.09001049198460197] * 2 + [-0.08956348780695919, -0.08867392198001625, -0.08735063540861585, -0.08560677950921644, -0.08345968550520699, -0.0809306921814226, -0.07804493380971377, -0.0748310903532609, -0.07132110243221514, -0.06754985388346517, -0.06355482506939446, -0.05937572038120053, -0.05505407363881763, -0.050632835309154405, -0.046155945645057164, -0.04166789798733637, -0.03721329656995605, -0.03283641322311726, -0.02858074737991983, -0.024488593759452962, -0.02060062202287545, -0.01695547258005139, -0.013589372563796866, -0.01053577578835852, -0.007825030270377595, -0.005484076616666242, -0.003536180276353429, -0.0020007003183990235, -0.0008928970324689967, -0.00022378026532083865, 0.0],
        },
        "q9.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.31622776601683794,
        },
        "q9.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q9.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.15811388300841897,
        },
        "q9.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q9.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "q9.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q9.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "q9.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q9.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "q9.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q9.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "q9.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q9.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.04,
        },
        "q9.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q9.z.const.wf": {
            "type": "constant",
            "sample": 1.25,
        },
        "q9.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.024,
        },
        "q9.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q9.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "q9.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q10.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 8.065723531015662e-05, 0.000321827333399303, 0.0007211134374823331, 0.0012745472628699207, 0.0019766285355645946, 0.002820379656454826, 0.0037974150478298193, 0.004898024493018182, 0.0061112696408837115, 0.007425092716073247, 0.008826436354605317, 0.010301373373819657, 0.011835245186975544, 0.013412807486872479, 0.015018381750623935, 0.016636011059861698, 0.01824961868775953, 0.01984316787676718, 0.021400821219111736, 0.0229070980560713, 0.0243470283317164, 0.025706301372051328, 0.026971408110921746, 0.028129775349183275, 0.02916989071280334, 0.0300814170680065, 0.030855295256354026, 0.031483834128730494, 0.03196078698343906, 0.03228141364872942] + [0.032442527592754226] * 2 + [0.03228141364872942, 0.031960786983439064, 0.031483834128730494, 0.030855295256354026, 0.0300814170680065, 0.029169890712803345, 0.028129775349183282, 0.026971408110921757, 0.025706301372051342, 0.024347028331716387, 0.022907098056071305, 0.021400821219111747, 0.019843167876767185, 0.01824961868775954, 0.0166360110598617, 0.015018381750623935, 0.013412807486872486, 0.011835245186975546, 0.010301373373819664, 0.00882643635460532, 0.0074250927160732436, 0.006111269640883715, 0.00489802449301818, 0.0037974150478298227, 0.002820379656454833, 0.001976628535564598, 0.0012745472628699207, 0.0007211134374823349, 0.000321827333399303, 8.065723531015843e-05, 0.0],
        },
        "q10.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 64,
        },
        "q10.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 4.254073592307837e-05, 0.0001697401547464678, 0.00038033409149912075, 0.0006722295690242197, 0.0010425255989264736, 0.0014875420128981146, 0.002002856037884925, 0.0025833462515918674, 0.00322324348147875, 0.003916188141388843, 0.004655293435973267, 0.005433213804757202, 0.006242217925620011, 0.00707426555214793, 0.007921087421215153, 0.008774267436635914, 0.009625326312109453, 0.010465805842176781, 0.011287352963666524, 0.012081802772189502, 0.01284125966862679, 0.013558175829140992, 0.01422542621884058, 0.014836379403577672, 0.015384963456119902, 0.015865726301691186, 0.016273889903140305, 0.01660539774722072, 0.016856955160041695, 0.017026062051017622] + [0.017111037759891465] * 2 + [0.017026062051017622, 0.016856955160041695, 0.01660539774722072, 0.016273889903140305, 0.015865726301691186, 0.015384963456119904, 0.014836379403577677, 0.014225426218840584, 0.013558175829141, 0.012841259668626783, 0.012081802772189504, 0.01128735296366653, 0.010465805842176783, 0.009625326312109458, 0.008774267436635915, 0.007921087421215153, 0.007074265552147934, 0.006242217925620012, 0.005433213804757205, 0.004655293435973269, 0.003916188141388841, 0.0032232434814787516, 0.0025833462515918665, 0.0020028560378849273, 0.0014875420128981182, 0.0010425255989264756, 0.0006722295690242197, 0.00038033409149912167, 0.0001697401547464678, 4.254073592307933e-05, 0.0],
        },
        "q10.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 64,
        },
        "q10.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -4.254073592307837e-05, -0.0001697401547464678, -0.00038033409149912075, -0.0006722295690242197, -0.0010425255989264736, -0.0014875420128981146, -0.002002856037884925, -0.0025833462515918674, -0.00322324348147875, -0.003916188141388843, -0.004655293435973267, -0.005433213804757202, -0.006242217925620011, -0.00707426555214793, -0.007921087421215153, -0.008774267436635914, -0.009625326312109453, -0.010465805842176781, -0.011287352963666524, -0.012081802772189502, -0.01284125966862679, -0.013558175829140992, -0.01422542621884058, -0.014836379403577672, -0.015384963456119902, -0.015865726301691186, -0.016273889903140305, -0.01660539774722072, -0.016856955160041695, -0.017026062051017622] + [-0.017111037759891465] * 2 + [-0.017026062051017622, -0.016856955160041695, -0.01660539774722072, -0.016273889903140305, -0.015865726301691186, -0.015384963456119904, -0.014836379403577677, -0.014225426218840584, -0.013558175829141, -0.012841259668626783, -0.012081802772189504, -0.01128735296366653, -0.010465805842176783, -0.009625326312109458, -0.008774267436635915, -0.007921087421215153, -0.007074265552147934, -0.006242217925620012, -0.005433213804757205, -0.004655293435973269, -0.003916188141388841, -0.0032232434814787516, -0.0025833462515918665, -0.0020028560378849273, -0.0014875420128981182, -0.0010425255989264756, -0.0006722295690242197, -0.00038033409149912167, -0.0001697401547464678, -4.254073592307933e-05, 0.0],
        },
        "q10.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 5.2097376081570756e-21, 2.078717371970382e-20, 4.657749277610148e-20, 8.232437899977153e-20, 1.2767256377544832e-19, 1.8217135646928868e-19, 2.4527912359487237e-19, 3.1636867181012935e-19, 3.947334812465522e-19, 4.795947272210669e-19, 5.701090205456345e-19, 6.65376789507912e-19, 7.644512202190777e-19, 8.663476664756347e-19, 9.7005343561575e-19, 1.0745378531139023e-18, 1.178762505887364e-18, 1.2816915625119464e-18, 1.38230206778006e-18, 1.4795941092891492e-18, 1.57260075502038e-18, 1.6603976631434528e-18, 1.7421122685409953e-18, 1.8169324547527114e-18, 1.884114625153624e-18, 1.9429910931514083e-18, 1.9929767179557204e-18, 2.0335747199702525e-18, 2.064381618011552e-18, 2.085091239286295e-18] + [2.095497762274058e-18] * 2 + [2.085091239286295e-18, 2.064381618011552e-18, 2.0335747199702525e-18, 1.9929767179557204e-18, 1.9429910931514083e-18, 1.884114625153624e-18, 1.8169324547527117e-18, 1.7421122685409957e-18, 1.6603976631434536e-18, 1.5726007550203792e-18, 1.4795941092891494e-18, 1.3823020677800608e-18, 1.2816915625119466e-18, 1.1787625058873645e-18, 1.0745378531139025e-18, 9.7005343561575e-19, 8.663476664756351e-19, 7.644512202190778e-19, 6.653767895079124e-19, 5.701090205456348e-19, 4.795947272210667e-19, 3.9473348124655243e-19, 3.1636867181012925e-19, 2.452791235948726e-19, 1.8217135646928914e-19, 1.2767256377544856e-19, 8.232437899977153e-20, 4.657749277610159e-20, 2.078717371970382e-20, 5.209737608157192e-21, 0.0],
        },
        "q10.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 4.9388312525329095e-21, 1.9706240686279225e-20, 4.415546315174421e-20, 7.804351129178343e-20, 1.2103359045912505e-19, 1.726984459328857e-19, 2.3252460916793905e-19, 2.999175008760027e-19, 3.742073402217316e-19, 4.546558014055714e-19, 5.404633514772617e-19, 6.307771964535007e-19, 7.246997567676859e-19, 8.212975878189018e-19, 9.196106569637313e-19, 1.0186618847519797e-18, 1.1174668555812214e-18, 1.2150436012613255e-18, 1.3104223602554973e-18, 1.4026552156061138e-18, 1.4908255157593207e-18, 1.5740569846599936e-18, 1.6515224305768638e-18, 1.7224519671055709e-18, 1.786140664645636e-18, 1.841955556307536e-18, 1.8893419286220233e-18, 1.9278288345317998e-18, 1.9570337738749517e-18, 1.9766664948434083e-18] + [1.9865318786358074e-18] * 2 + [1.9766664948434083e-18, 1.957033773874952e-18, 1.9278288345317998e-18, 1.8893419286220233e-18, 1.841955556307536e-18, 1.7861406646456362e-18, 1.7224519671055713e-18, 1.6515224305768644e-18, 1.5740569846599945e-18, 1.4908255157593197e-18, 1.402655215606114e-18, 1.310422360255498e-18, 1.2150436012613257e-18, 1.117466855581222e-18, 1.01866188475198e-18, 9.196106569637313e-19, 8.212975878189022e-19, 7.24699756767686e-19, 6.307771964535011e-19, 5.404633514772619e-19, 4.546558014055712e-19, 3.742073402217318e-19, 2.999175008760026e-19, 2.325246091679393e-19, 1.7269844593288614e-19, 1.2103359045912527e-19, 7.804351129178343e-20, 4.415546315174432e-20, 1.9706240686279225e-20, 4.938831252533019e-21, 0.0],
        },
        "q10.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 8.065723531015662e-05, 0.000321827333399303, 0.0007211134374823331, 0.0012745472628699207, 0.0019766285355645946, 0.002820379656454826, 0.0037974150478298193, 0.004898024493018182, 0.0061112696408837115, 0.007425092716073247, 0.008826436354605317, 0.010301373373819657, 0.011835245186975544, 0.013412807486872479, 0.015018381750623935, 0.016636011059861698, 0.01824961868775953, 0.01984316787676718, 0.021400821219111736, 0.0229070980560713, 0.0243470283317164, 0.025706301372051328, 0.026971408110921746, 0.028129775349183275, 0.02916989071280334, 0.0300814170680065, 0.030855295256354026, 0.031483834128730494, 0.03196078698343906, 0.03228141364872942] + [0.032442527592754226] * 2 + [0.03228141364872942, 0.031960786983439064, 0.031483834128730494, 0.030855295256354026, 0.0300814170680065, 0.029169890712803345, 0.028129775349183282, 0.026971408110921757, 0.025706301372051342, 0.024347028331716387, 0.022907098056071305, 0.021400821219111747, 0.019843167876767185, 0.01824961868775954, 0.0166360110598617, 0.015018381750623935, 0.013412807486872486, 0.011835245186975546, 0.010301373373819664, 0.00882643635460532, 0.0074250927160732436, 0.006111269640883715, 0.00489802449301818, 0.0037974150478298227, 0.002820379656454833, 0.001976628535564598, 0.0012745472628699207, 0.0007211134374823349, 0.000321827333399303, 8.065723531015843e-05, 0.0],
        },
        "q10.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 2.6048688040785378e-21, 1.039358685985191e-20, 2.328874638805074e-20, 4.1162189499885765e-20, 6.383628188772416e-20, 9.108567823464434e-20, 1.2263956179743618e-19, 1.5818433590506467e-19, 1.973667406232761e-19, 2.3979736361053343e-19, 2.8505451027281725e-19, 3.32688394753956e-19, 3.8222561010953886e-19, 4.3317383323781734e-19, 4.85026717807875e-19, 5.372689265569512e-19, 5.89381252943682e-19, 6.408457812559732e-19, 6.9115103389003e-19, 7.397970546445746e-19, 7.8630037751019e-19, 8.301988315717264e-19, 8.710561342704977e-19, 9.084662273763557e-19, 9.42057312576812e-19, 9.714955465757042e-19, 9.964883589778602e-19, 1.0167873599851263e-18, 1.032190809005776e-18, 1.0425456196431475e-18] + [1.047748881137029e-18] * 2 + [1.0425456196431475e-18, 1.032190809005776e-18, 1.0167873599851263e-18, 9.964883589778602e-19, 9.714955465757042e-19, 9.42057312576812e-19, 9.084662273763559e-19, 8.710561342704979e-19, 8.301988315717268e-19, 7.863003775101896e-19, 7.397970546445747e-19, 6.911510338900304e-19, 6.408457812559733e-19, 5.893812529436823e-19, 5.3726892655695125e-19, 4.85026717807875e-19, 4.3317383323781753e-19, 3.822256101095389e-19, 3.326883947539562e-19, 2.850545102728174e-19, 2.3979736361053333e-19, 1.9736674062327622e-19, 1.5818433590506463e-19, 1.226395617974363e-19, 9.108567823464457e-20, 6.383628188772428e-20, 4.1162189499885765e-20, 2.3288746388050794e-20, 1.039358685985191e-20, 2.604868804078596e-21, 0.0],
        },
        "q10.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 4.254073592307837e-05, 0.0001697401547464678, 0.00038033409149912075, 0.0006722295690242197, 0.0010425255989264736, 0.0014875420128981146, 0.002002856037884925, 0.0025833462515918674, 0.00322324348147875, 0.003916188141388843, 0.004655293435973267, 0.005433213804757202, 0.006242217925620011, 0.00707426555214793, 0.007921087421215153, 0.008774267436635914, 0.009625326312109453, 0.010465805842176781, 0.011287352963666524, 0.012081802772189502, 0.01284125966862679, 0.013558175829140992, 0.01422542621884058, 0.014836379403577672, 0.015384963456119902, 0.015865726301691186, 0.016273889903140305, 0.01660539774722072, 0.016856955160041695, 0.017026062051017622] + [0.017111037759891465] * 2 + [0.017026062051017622, 0.016856955160041695, 0.01660539774722072, 0.016273889903140305, 0.015865726301691186, 0.015384963456119904, 0.014836379403577677, 0.014225426218840584, 0.013558175829141, 0.012841259668626783, 0.012081802772189504, 0.01128735296366653, 0.010465805842176783, 0.009625326312109458, 0.008774267436635915, 0.007921087421215153, 0.007074265552147934, 0.006242217925620012, 0.005433213804757205, 0.004655293435973269, 0.003916188141388841, 0.0032232434814787516, 0.0025833462515918665, 0.0020028560378849273, 0.0014875420128981182, 0.0010425255989264756, 0.0006722295690242197, 0.00038033409149912167, 0.0001697401547464678, 4.254073592307933e-05, 0.0],
        },
        "q10.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 2.6048688040785378e-21, 1.039358685985191e-20, 2.328874638805074e-20, 4.1162189499885765e-20, 6.383628188772416e-20, 9.108567823464434e-20, 1.2263956179743618e-19, 1.5818433590506467e-19, 1.973667406232761e-19, 2.3979736361053343e-19, 2.8505451027281725e-19, 3.32688394753956e-19, 3.8222561010953886e-19, 4.3317383323781734e-19, 4.85026717807875e-19, 5.372689265569512e-19, 5.89381252943682e-19, 6.408457812559732e-19, 6.9115103389003e-19, 7.397970546445746e-19, 7.8630037751019e-19, 8.301988315717264e-19, 8.710561342704977e-19, 9.084662273763557e-19, 9.42057312576812e-19, 9.714955465757042e-19, 9.964883589778602e-19, 1.0167873599851263e-18, 1.032190809005776e-18, 1.0425456196431475e-18] + [1.047748881137029e-18] * 2 + [1.0425456196431475e-18, 1.032190809005776e-18, 1.0167873599851263e-18, 9.964883589778602e-19, 9.714955465757042e-19, 9.42057312576812e-19, 9.084662273763559e-19, 8.710561342704979e-19, 8.301988315717268e-19, 7.863003775101896e-19, 7.397970546445747e-19, 6.911510338900304e-19, 6.408457812559733e-19, 5.893812529436823e-19, 5.3726892655695125e-19, 4.85026717807875e-19, 4.3317383323781753e-19, 3.822256101095389e-19, 3.326883947539562e-19, 2.850545102728174e-19, 2.3979736361053333e-19, 1.9736674062327622e-19, 1.5818433590506463e-19, 1.226395617974363e-19, 9.108567823464457e-20, 6.383628188772428e-20, 4.1162189499885765e-20, 2.3288746388050794e-20, 1.039358685985191e-20, 2.604868804078596e-21, 0.0],
        },
        "q10.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -4.254073592307837e-05, -0.0001697401547464678, -0.00038033409149912075, -0.0006722295690242197, -0.0010425255989264736, -0.0014875420128981146, -0.002002856037884925, -0.0025833462515918674, -0.00322324348147875, -0.003916188141388843, -0.004655293435973267, -0.005433213804757202, -0.006242217925620011, -0.00707426555214793, -0.007921087421215153, -0.008774267436635914, -0.009625326312109453, -0.010465805842176781, -0.011287352963666524, -0.012081802772189502, -0.01284125966862679, -0.013558175829140992, -0.01422542621884058, -0.014836379403577672, -0.015384963456119902, -0.015865726301691186, -0.016273889903140305, -0.01660539774722072, -0.016856955160041695, -0.017026062051017622] + [-0.017111037759891465] * 2 + [-0.017026062051017622, -0.016856955160041695, -0.01660539774722072, -0.016273889903140305, -0.015865726301691186, -0.015384963456119904, -0.014836379403577677, -0.014225426218840584, -0.013558175829141, -0.012841259668626783, -0.012081802772189504, -0.01128735296366653, -0.010465805842176783, -0.009625326312109458, -0.008774267436635915, -0.007921087421215153, -0.007074265552147934, -0.006242217925620012, -0.005433213804757205, -0.004655293435973269, -0.003916188141388841, -0.0032232434814787516, -0.0025833462515918665, -0.0020028560378849273, -0.0014875420128981182, -0.0010425255989264756, -0.0006722295690242197, -0.00038033409149912167, -0.0001697401547464678, -4.254073592307933e-05, 0.0],
        },
        "q10.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.31622776601683794,
        },
        "q10.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q10.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.15811388300841897,
        },
        "q10.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q10.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "q10.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q10.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "q10.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q10.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "q10.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q10.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "q10.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q10.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.5,
        },
        "q10.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q10.z.const.wf": {
            "type": "constant",
            "sample": 1.25,
        },
        "q10.z.SWAP_unipolar.wf": {
            "type": "constant",
            "sample": 0.0495,
        },
        "q10.z.SWAP_flattop.wf": {
            "type": "arbitrary",
            "samples": [0.0, 8.025662367830151e-05, 0.0003205059997116571, 0.0007191900187052127, 0.0012737230618081494, 0.0019805087694441006, 0.0028349633650830547, 0.003831545382791081, 0.004963791605763346, 0.006224358982765249, 0.007605072250637413, 0.009096976954014412, 0.010690397518403894, 0.012374999999999997, 0.014139859105274408, 0.015973529045697245, 0.017864117768067796, 0.0197993640790429, 0.021766717163680754, 0.023753417982291976, 0.025746582017708023, 0.027733282836319244, 0.0297006359209571, 0.0316358822319322, 0.03352647095430276, 0.0353601408947256, 0.037125, 0.03880960248159611, 0.04040302304598559, 0.04189492774936259, 0.04327564101723475, 0.04453620839423665, 0.04566845461720892, 0.046665036634916945, 0.047519491230555905, 0.04822627693819185, 0.04878080998129479, 0.04917949400028834, 0.0494197433763217] + [0.0495] * 51 + [0.0494197433763217, 0.04917949400028834, 0.04878080998129479, 0.04822627693819185, 0.047519491230555905, 0.046665036634916945, 0.04566845461720893, 0.04453620839423666, 0.04327564101723475, 0.04189492774936259, 0.04040302304598559, 0.03880960248159611, 0.037125000000000005, 0.03536014089472559, 0.03352647095430276, 0.03163588223193221, 0.029700635920957102, 0.027733282836319244, 0.025746582017708027, 0.023753417982291983, 0.021766717163680758, 0.019799364079042904, 0.0178641177680678, 0.01597352904569724, 0.014139859105274405, 0.012375000000000006, 0.010690397518403897, 0.009096976954014416, 0.007605072250637415, 0.006224358982765246, 0.004963791605763351, 0.003831545382791084, 0.0028349633650830578, 0.0019805087694441006, 0.0012737230618081494, 0.0007191900187052127, 0.0003205059997116571, 8.025662367830424e-05],
        },
        "q10.z.SWAP_bipolar.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.00018045536607316418, 0.0007191900187052127, 0.0016083479935359829, 0.0028349633650830547, 0.0043811493191320055, 0.006224358982765249, 0.008337714208540317, 0.010690397518403894, 0.01324810149191673, 0.015973529045697245, 0.018826937308882945, 0.021766717163680754, 0.024749999999999998, 0.027733282836319244, 0.03067306269111705, 0.03352647095430276, 0.036251898508083275, 0.03880960248159611, 0.041162285791459675, 0.04327564101723475, 0.045118850680867996, 0.046665036634916945, 0.04789165200646402, 0.04878080998129479, 0.049319544633926835] + [0.0495] * 26 + [0.049139089267853674, 0.04806161996258958, 0.04628330401292804, 0.043830073269833895, 0.04073770136173599, 0.03705128203446951, 0.03282457158291937, 0.028119204963192215, 0.023003797016166547, 0.017552941908605513, 0.011846125382234113, 0.005966565672638489, 3.0310008278896994e-18, -0.005966565672638483, -0.011846125382234096, -0.017552941908605516, -0.023003797016166543, -0.02811920496319221, -0.032824571582919355, -0.03705128203446951, -0.04073770136173599, -0.04383007326983389, -0.04628330401292803, -0.04806161996258958, -0.049139089267853674] + [-0.0495] * 26 + [-0.049319544633926835, -0.04878080998129479, -0.047891652006464024, -0.046665036634916945, -0.045118850680867996, -0.04327564101723475, -0.04116228579145968, -0.03880960248159611, -0.036251898508083275, -0.03352647095430276, -0.030673062691117057, -0.027733282836319244, -0.02475, -0.021766717163680758, -0.018826937308882952, -0.01597352904569724, -0.01324810149191673, -0.010690397518403897, -0.008337714208540325, -0.006224358982765246, -0.0043811493191320055, -0.0028349633650830578, -0.0016083479935359857, -0.0007191900187052127, -0.00018045536607316418],
        },
        "q10.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.026,
        },
        "q10.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q10.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "q10.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "coupler_q6_q7.const.wf": {
            "type": "constant",
            "sample": 1.25,
        },
        "coupler_q7_q8.const.wf": {
            "type": "constant",
            "sample": 1.25,
        },
        "coupler_q8_q9.const.wf": {
            "type": "constant",
            "sample": 1.25,
        },
        "coupler_q9_q10.const.wf": {
            "type": "constant",
            "sample": 1.25,
        },
        "q10.z.SWAP_unipolar.flux_pulse_control_q9_q10.wf": {
            "type": "constant",
            "sample": 0.0495,
        },
        "coupler_q9_q10.SWAP_unipolar.coupler_pulse_control_q9_q10.wf": {
            "type": "constant",
            "sample": -0.12499999999999994,
        },
        "q10.z.SWAP_flattop.flux_pulse_control_q9_q10.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.0012113512216949502, 0.00472682938922005, 0.01020231500576129, 0.01710182938922005, 0.024749999999999998, 0.03239817061077995, 0.039297684994238705, 0.04477317061077995, 0.048288648778305056] + [0.0495] * 109 + [0.048288648778305056, 0.04477317061077995, 0.03929768499423871, 0.03239817061077995, 0.02475, 0.017101829389220054, 0.010202315005761292, 0.004726829389220054, 0.0012113512216949502],
        },
        "coupler_q9_q10.SWAP_flattop.coupler_pulse_control_q9_q10.wf": {
            "type": "constant",
            "sample": -0.12499999999999994,
        },
        "q10.z.SWAP_bipolar.flux_pulse_control_q9_q10.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.0024510205194151263, 0.009318627403996345, 0.01924260688458122, 0.030257393115418783, 0.040181372596003656, 0.047048979480584875] + [0.0495] * 55 + [0.04286825748732972, 0.024750000000000008, 3.0310008278896994e-18, -0.02474999999999999, -0.04286825748732972] + [-0.0495] * 55 + [-0.047048979480584875, -0.040181372596003656, -0.030257393115418783, -0.01924260688458122, -0.009318627403996346, -0.002451020519415129],
        },
        "coupler_q9_q10.SWAP_bipolar.coupler_pulse_control_q9_q10.wf": {
            "type": "constant",
            "sample": -0.12499999999999994,
        },
        "coupler_q5_q6.const.wf": {
            "type": "constant",
            "sample": 1.25,
        },
    },
    "digital_waveforms": {
        "ON": {
            "samples": [[1, 0]],
        },
    },
    "integration_weights": {
        "q5.resonator.readout.iw1": {
            "cosine": [(1.0, 1500)],
            "sine": [(-0.0, 1500)],
        },
        "q5.resonator.readout.iw2": {
            "cosine": [(0.0, 1500)],
            "sine": [(1.0, 1500)],
        },
        "q5.resonator.readout.iw3": {
            "cosine": [(-0.0, 1500)],
            "sine": [(-1.0, 1500)],
        },
        "q6.resonator.readout.iw1": {
            "cosine": [(0.9263896341509306, 3000)],
            "sine": [(-0.3765663895486492, 3000)],
        },
        "q6.resonator.readout.iw2": {
            "cosine": [(0.3765663895486492, 3000)],
            "sine": [(0.9263896341509306, 3000)],
        },
        "q6.resonator.readout.iw3": {
            "cosine": [(-0.3765663895486492, 3000)],
            "sine": [(-0.9263896341509306, 3000)],
        },
        "q7.resonator.readout.iw1": {
            "cosine": [(-0.9804339027423286, 2000)],
            "sine": [(0.19684857721976617, 2000)],
        },
        "q7.resonator.readout.iw2": {
            "cosine": [(-0.19684857721976617, 2000)],
            "sine": [(-0.9804339027423286, 2000)],
        },
        "q7.resonator.readout.iw3": {
            "cosine": [(0.19684857721976617, 2000)],
            "sine": [(0.9804339027423286, 2000)],
        },
        "q8.resonator.readout.iw1": {
            "cosine": [(-0.5569761778677528, 2000)],
            "sine": [(0.83052846867993, 2000)],
        },
        "q8.resonator.readout.iw2": {
            "cosine": [(-0.83052846867993, 2000)],
            "sine": [(-0.5569761778677528, 2000)],
        },
        "q8.resonator.readout.iw3": {
            "cosine": [(0.83052846867993, 2000)],
            "sine": [(0.5569761778677528, 2000)],
        },
        "q9.resonator.readout.iw1": {
            "cosine": [(0.03478521455234504, 2000)],
            "sine": [(-0.9993948112975909, 2000)],
        },
        "q9.resonator.readout.iw2": {
            "cosine": [(0.9993948112975909, 2000)],
            "sine": [(0.03478521455234504, 2000)],
        },
        "q9.resonator.readout.iw3": {
            "cosine": [(-0.9993948112975909, 2000)],
            "sine": [(-0.03478521455234504, 2000)],
        },
        "q10.resonator.readout.iw1": {
            "cosine": [(0.9835308516317905, 2000)],
            "sine": [(0.18074032170062312, 2000)],
        },
        "q10.resonator.readout.iw2": {
            "cosine": [(-0.18074032170062312, 2000)],
            "sine": [(0.9835308516317905, 2000)],
        },
        "q10.resonator.readout.iw3": {
            "cosine": [(0.18074032170062312, 2000)],
            "sine": [(-0.9835308516317905, 2000)],
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
                "2": {
                    "type": "LF",
                    "analog_outputs": {
                        "3": {
                            "offset": 0.0,
                            "delay": 0,
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
                        "1": {
                            "offset": 0.0,
                            "delay": 0,
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
                            "delay": 0,
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
                "3": {
                    "type": "LF",
                    "analog_outputs": {
                        "1": {
                            "offset": 0.0,
                            "delay": 0,
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
                            "delay": 0,
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
                            "delay": 0,
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
                        "4": {
                            "offset": 0.0,
                            "delay": 0,
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
                        "5": {
                            "offset": 0.0,
                            "delay": 0,
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
                        "6": {
                            "offset": 0.0,
                            "delay": 0,
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
                        "7": {
                            "offset": 0.0,
                            "delay": 0,
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
                        "8": {
                            "offset": 0.0,
                            "delay": 0,
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
                "7": {
                    "type": "MW",
                    "analog_outputs": {
                        "1": {
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": -5,
                            "band": 2,
                            "delay": 0,
                            "shareable": False,
                            "upconverters": {
                                "1": {
                                    "frequency": 6025000000.0,
                                },
                            },
                        },
                        "7": {
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 7,
                            "band": 1,
                            "delay": 0,
                            "shareable": False,
                            "upconverters": {
                                "1": {
                                    "frequency": 4100000000.0,
                                },
                            },
                        },
                        "2": {
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 7,
                            "band": 1,
                            "delay": 0,
                            "shareable": False,
                            "upconverters": {
                                "1": {
                                    "frequency": 4250000000.0,
                                },
                            },
                        },
                        "3": {
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 10,
                            "band": 1,
                            "delay": 0,
                            "shareable": False,
                            "upconverters": {
                                "1": {
                                    "frequency": 4150000000.0,
                                },
                            },
                        },
                        "4": {
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 7,
                            "band": 1,
                            "delay": 0,
                            "shareable": False,
                            "upconverters": {
                                "1": {
                                    "frequency": 3900000000.0,
                                },
                            },
                        },
                        "5": {
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 10,
                            "band": 1,
                            "delay": 0,
                            "shareable": False,
                            "upconverters": {
                                "1": {
                                    "frequency": 3800000000.0,
                                },
                            },
                        },
                        "6": {
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 7,
                            "band": 1,
                            "delay": 0,
                            "shareable": False,
                            "upconverters": {
                                "1": {
                                    "frequency": 3800000000.0,
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
                            "downconverter_frequency": 6025000000.0,
                        },
                    },
                },
            },
        },
    },
    "oscillators": {},
    "elements": {
        "q5.xy": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "x180_DragCosine": "q5.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "q5.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "q5.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "q5.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "q5.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "q5.xy.-y90_DragCosine.pulse",
                "x180_Square": "q5.xy.x180_Square.pulse",
                "x90_Square": "q5.xy.x90_Square.pulse",
                "-x90_Square": "q5.xy.-x90_Square.pulse",
                "y180_Square": "q5.xy.y180_Square.pulse",
                "y90_Square": "q5.xy.y90_Square.pulse",
                "-y90_Square": "q5.xy.-y90_Square.pulse",
                "x180": "q5.xy.x180_DragCosine.pulse",
                "x90": "q5.xy.x90_DragCosine.pulse",
                "-x90": "q5.xy.-x90_DragCosine.pulse",
                "y180": "q5.xy.y180_DragCosine.pulse",
                "y90": "q5.xy.y90_DragCosine.pulse",
                "-y90": "q5.xy.-y90_DragCosine.pulse",
                "saturation": "q5.xy.saturation.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "core": "a",
            "MWInput": {
                "port": ('con1', 7, 7),
                "upconverter": 1,
            },
            "intermediate_frequency": -250000000.0,
        },
        "q5.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "q5.z.const.pulse",
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
                "port": ('con1', 3, 8),
            },
        },
        "q5.resonator": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "readout": "q5.resonator.readout.pulse",
                "const": "q5.resonator.const.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "core": "a",
            "MWInput": {
                "port": ('con1', 7, 1),
                "upconverter": 1,
            },
            "MWOutput": {
                "port": ('con1', 7, 1),
            },
            "smearing": 0,
            "time_of_flight": 28,
            "intermediate_frequency": -175000000.0,
        },
        "q6.xy": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "x180_DragCosine": "q6.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "q6.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "q6.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "q6.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "q6.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "q6.xy.-y90_DragCosine.pulse",
                "x180_Square": "q6.xy.x180_Square.pulse",
                "x90_Square": "q6.xy.x90_Square.pulse",
                "-x90_Square": "q6.xy.-x90_Square.pulse",
                "y180_Square": "q6.xy.y180_Square.pulse",
                "y90_Square": "q6.xy.y90_Square.pulse",
                "-y90_Square": "q6.xy.-y90_Square.pulse",
                "x180": "q6.xy.x180_DragCosine.pulse",
                "x90": "q6.xy.x90_DragCosine.pulse",
                "-x90": "q6.xy.-x90_DragCosine.pulse",
                "y180": "q6.xy.y180_DragCosine.pulse",
                "y90": "q6.xy.y90_DragCosine.pulse",
                "-y90": "q6.xy.-y90_DragCosine.pulse",
                "saturation": "q6.xy.saturation.pulse",
                "EF_x180": "q6.xy.EF_x180.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "core": "b",
            "MWInput": {
                "port": ('con1', 7, 2),
                "upconverter": 1,
            },
            "intermediate_frequency": -50904333.24319054,
        },
        "q6.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "q6.z.const.pulse",
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
                "port": ('con1', 2, 2),
            },
        },
        "q6.resonator": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "readout": "q6.resonator.readout.pulse",
                "const": "q6.resonator.const.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "core": "b",
            "MWInput": {
                "port": ('con1', 7, 1),
                "upconverter": 1,
            },
            "MWOutput": {
                "port": ('con1', 7, 1),
            },
            "smearing": 0,
            "time_of_flight": 388,
            "intermediate_frequency": -25632647.0,
        },
        "q7.xy": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "x180_DragCosine": "q7.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "q7.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "q7.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "q7.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "q7.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "q7.xy.-y90_DragCosine.pulse",
                "x180_Square": "q7.xy.x180_Square.pulse",
                "x90_Square": "q7.xy.x90_Square.pulse",
                "-x90_Square": "q7.xy.-x90_Square.pulse",
                "y180_Square": "q7.xy.y180_Square.pulse",
                "y90_Square": "q7.xy.y90_Square.pulse",
                "-y90_Square": "q7.xy.-y90_Square.pulse",
                "x180": "q7.xy.x180_DragCosine.pulse",
                "x90": "q7.xy.x90_DragCosine.pulse",
                "-x90": "q7.xy.-x90_DragCosine.pulse",
                "y180": "q7.xy.y180_DragCosine.pulse",
                "y90": "q7.xy.y90_DragCosine.pulse",
                "-y90": "q7.xy.-y90_DragCosine.pulse",
                "saturation": "q7.xy.saturation.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "core": "c",
            "MWInput": {
                "port": ('con1', 7, 3),
                "upconverter": 1,
            },
            "intermediate_frequency": -256772753.10845205,
        },
        "q7.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "q7.z.const.pulse",
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
                "port": ('con1', 3, 1),
            },
        },
        "q7.resonator": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "readout": "q7.resonator.readout.pulse",
                "const": "q7.resonator.const.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "core": "c",
            "MWInput": {
                "port": ('con1', 7, 1),
                "upconverter": 1,
            },
            "MWOutput": {
                "port": ('con1', 7, 1),
            },
            "smearing": 0,
            "time_of_flight": 388,
            "intermediate_frequency": 77254665.0,
        },
        "q8.xy": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "x180_DragCosine": "q8.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "q8.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "q8.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "q8.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "q8.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "q8.xy.-y90_DragCosine.pulse",
                "x180_Square": "q8.xy.x180_Square.pulse",
                "x90_Square": "q8.xy.x90_Square.pulse",
                "-x90_Square": "q8.xy.-x90_Square.pulse",
                "y180_Square": "q8.xy.y180_Square.pulse",
                "y90_Square": "q8.xy.y90_Square.pulse",
                "-y90_Square": "q8.xy.-y90_Square.pulse",
                "x180": "q8.xy.x180_DragCosine.pulse",
                "x90": "q8.xy.x90_DragCosine.pulse",
                "-x90": "q8.xy.-x90_DragCosine.pulse",
                "y180": "q8.xy.y180_DragCosine.pulse",
                "y90": "q8.xy.y90_DragCosine.pulse",
                "-y90": "q8.xy.-y90_DragCosine.pulse",
                "saturation": "q8.xy.saturation.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "core": "d",
            "MWInput": {
                "port": ('con1', 7, 4),
                "upconverter": 1,
            },
            "intermediate_frequency": -79071692.25275058,
        },
        "q8.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "q8.z.const.pulse",
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
                "port": ('con1', 3, 3),
            },
        },
        "q8.resonator": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "readout": "q8.resonator.readout.pulse",
                "const": "q8.resonator.const.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "core": "d",
            "MWInput": {
                "port": ('con1', 7, 1),
                "upconverter": 1,
            },
            "MWOutput": {
                "port": ('con1', 7, 1),
            },
            "smearing": 0,
            "time_of_flight": 388,
            "intermediate_frequency": -76041688.0,
        },
        "q9.xy": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "x180_DragCosine": "q9.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "q9.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "q9.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "q9.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "q9.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "q9.xy.-y90_DragCosine.pulse",
                "x180_Square": "q9.xy.x180_Square.pulse",
                "x90_Square": "q9.xy.x90_Square.pulse",
                "-x90_Square": "q9.xy.-x90_Square.pulse",
                "y180_Square": "q9.xy.y180_Square.pulse",
                "y90_Square": "q9.xy.y90_Square.pulse",
                "-y90_Square": "q9.xy.-y90_Square.pulse",
                "x180": "q9.xy.x180_DragCosine.pulse",
                "x90": "q9.xy.x90_DragCosine.pulse",
                "-x90": "q9.xy.-x90_DragCosine.pulse",
                "y180": "q9.xy.y180_DragCosine.pulse",
                "y90": "q9.xy.y90_DragCosine.pulse",
                "-y90": "q9.xy.-y90_DragCosine.pulse",
                "saturation": "q9.xy.saturation.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "core": "e",
            "MWInput": {
                "port": ('con1', 7, 5),
                "upconverter": 1,
            },
            "intermediate_frequency": -165000000.0,
        },
        "q9.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "q9.z.const.pulse",
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
                "port": ('con1', 3, 5),
            },
        },
        "q9.resonator": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "readout": "q9.resonator.readout.pulse",
                "const": "q9.resonator.const.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "core": "e",
            "MWInput": {
                "port": ('con1', 7, 1),
                "upconverter": 1,
            },
            "MWOutput": {
                "port": ('con1', 7, 1),
            },
            "smearing": 0,
            "time_of_flight": 388,
            "intermediate_frequency": 148633787.0,
        },
        "q10.xy": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "x180_DragCosine": "q10.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "q10.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "q10.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "q10.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "q10.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "q10.xy.-y90_DragCosine.pulse",
                "x180_Square": "q10.xy.x180_Square.pulse",
                "x90_Square": "q10.xy.x90_Square.pulse",
                "-x90_Square": "q10.xy.-x90_Square.pulse",
                "y180_Square": "q10.xy.y180_Square.pulse",
                "y90_Square": "q10.xy.y90_Square.pulse",
                "-y90_Square": "q10.xy.-y90_Square.pulse",
                "x180": "q10.xy.x180_DragCosine.pulse",
                "x90": "q10.xy.x90_DragCosine.pulse",
                "-x90": "q10.xy.-x90_DragCosine.pulse",
                "y180": "q10.xy.y180_DragCosine.pulse",
                "y90": "q10.xy.y90_DragCosine.pulse",
                "-y90": "q10.xy.-y90_DragCosine.pulse",
                "saturation": "q10.xy.saturation.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "core": "f",
            "MWInput": {
                "port": ('con1', 7, 6),
                "upconverter": 1,
            },
            "intermediate_frequency": -198995429.8353858,
        },
        "q10.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "q10.z.const.pulse",
                "SWAP_unipolar": "q10.z.SWAP_unipolar.pulse",
                "SWAP_flattop": "q10.z.SWAP_flattop.pulse",
                "SWAP_bipolar": "q10.z.SWAP_bipolar.pulse",
                "SWAP_unipolar.flux_pulse_control_q9_q10": "q10.z.SWAP_unipolar.flux_pulse_control_q9_q10.pulse",
                "SWAP_flattop.flux_pulse_control_q9_q10": "q10.z.SWAP_flattop.flux_pulse_control_q9_q10.pulse",
                "SWAP_bipolar.flux_pulse_control_q9_q10": "q10.z.SWAP_bipolar.flux_pulse_control_q9_q10.pulse",
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
                "port": ('con1', 3, 7),
            },
        },
        "q10.resonator": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "readout": "q10.resonator.readout.pulse",
                "const": "q10.resonator.const.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "core": "f",
            "MWInput": {
                "port": ('con1', 7, 1),
                "upconverter": 1,
            },
            "MWOutput": {
                "port": ('con1', 7, 1),
            },
            "smearing": 0,
            "time_of_flight": 388,
            "intermediate_frequency": 25616322.0,
        },
        "coupler_q6_q7": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "coupler_q6_q7.const.pulse",
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
                "port": ('con1', 2, 3),
            },
        },
        "coupler_q7_q8": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "coupler_q7_q8.const.pulse",
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
                "port": ('con1', 3, 2),
            },
        },
        "coupler_q8_q9": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "coupler_q8_q9.const.pulse",
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
                "port": ('con1', 3, 4),
            },
        },
        "coupler_q9_q10": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "coupler_q9_q10.const.pulse",
                "SWAP_unipolar.coupler_pulse_control_q9_q10": "coupler_q9_q10.SWAP_unipolar.coupler_pulse_control_q9_q10.pulse",
                "SWAP_flattop.coupler_pulse_control_q9_q10": "coupler_q9_q10.SWAP_flattop.coupler_pulse_control_q9_q10.pulse",
                "SWAP_bipolar.coupler_pulse_control_q9_q10": "coupler_q9_q10.SWAP_bipolar.coupler_pulse_control_q9_q10.pulse",
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
                "port": ('con1', 3, 6),
            },
        },
        "coupler_q5_q6": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "coupler_q5_q6.const.pulse",
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
                "port": ('con1', 2, 1),
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
        "q5.xy.x180_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q5.xy.x180_DragCosine.wf.I",
                "Q": "q5.xy.x180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q5.xy.x90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q5.xy.x90_DragCosine.wf.I",
                "Q": "q5.xy.x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q5.xy.-x90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q5.xy.-x90_DragCosine.wf.I",
                "Q": "q5.xy.-x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q5.xy.y180_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q5.xy.y180_DragCosine.wf.I",
                "Q": "q5.xy.y180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q5.xy.y90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q5.xy.y90_DragCosine.wf.I",
                "Q": "q5.xy.y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q5.xy.-y90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q5.xy.-y90_DragCosine.wf.I",
                "Q": "q5.xy.-y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q5.xy.x180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q5.xy.x180_Square.wf.I",
                "Q": "q5.xy.x180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q5.xy.x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q5.xy.x90_Square.wf.I",
                "Q": "q5.xy.x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q5.xy.-x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q5.xy.-x90_Square.wf.I",
                "Q": "q5.xy.-x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q5.xy.y180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q5.xy.y180_Square.wf.I",
                "Q": "q5.xy.y180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q5.xy.y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q5.xy.y90_Square.wf.I",
                "Q": "q5.xy.y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q5.xy.-y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q5.xy.-y90_Square.wf.I",
                "Q": "q5.xy.-y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q5.xy.saturation.pulse": {
            "length": 20000,
            "waveforms": {
                "I": "q5.xy.saturation.wf.I",
                "Q": "q5.xy.saturation.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q5.z.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "q5.z.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q5.resonator.readout.pulse": {
            "length": 1500,
            "waveforms": {
                "I": "q5.resonator.readout.wf.I",
                "Q": "q5.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "q5.resonator.readout.iw1",
                "iw2": "q5.resonator.readout.iw2",
                "iw3": "q5.resonator.readout.iw3",
            },
            "operation": "measurement",
            "digital_marker": "ON",
        },
        "q5.resonator.const.pulse": {
            "length": 100,
            "waveforms": {
                "I": "q5.resonator.const.wf.I",
                "Q": "q5.resonator.const.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q6.xy.x180_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q6.xy.x180_DragCosine.wf.I",
                "Q": "q6.xy.x180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q6.xy.x90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q6.xy.x90_DragCosine.wf.I",
                "Q": "q6.xy.x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q6.xy.-x90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q6.xy.-x90_DragCosine.wf.I",
                "Q": "q6.xy.-x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q6.xy.y180_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q6.xy.y180_DragCosine.wf.I",
                "Q": "q6.xy.y180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q6.xy.y90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q6.xy.y90_DragCosine.wf.I",
                "Q": "q6.xy.y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q6.xy.-y90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q6.xy.-y90_DragCosine.wf.I",
                "Q": "q6.xy.-y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q6.xy.x180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q6.xy.x180_Square.wf.I",
                "Q": "q6.xy.x180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q6.xy.x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q6.xy.x90_Square.wf.I",
                "Q": "q6.xy.x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q6.xy.-x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q6.xy.-x90_Square.wf.I",
                "Q": "q6.xy.-x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q6.xy.y180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q6.xy.y180_Square.wf.I",
                "Q": "q6.xy.y180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q6.xy.y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q6.xy.y90_Square.wf.I",
                "Q": "q6.xy.y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q6.xy.-y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q6.xy.-y90_Square.wf.I",
                "Q": "q6.xy.-y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q6.xy.saturation.pulse": {
            "length": 20000,
            "waveforms": {
                "I": "q6.xy.saturation.wf.I",
                "Q": "q6.xy.saturation.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q6.xy.EF_x180.pulse": {
            "length": 272,
            "waveforms": {
                "I": "q6.xy.EF_x180.wf.I",
                "Q": "q6.xy.EF_x180.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q6.z.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "q6.z.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q6.resonator.readout.pulse": {
            "length": 3000,
            "waveforms": {
                "I": "q6.resonator.readout.wf.I",
                "Q": "q6.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "q6.resonator.readout.iw1",
                "iw2": "q6.resonator.readout.iw2",
                "iw3": "q6.resonator.readout.iw3",
            },
            "operation": "measurement",
            "digital_marker": "ON",
        },
        "q6.resonator.const.pulse": {
            "length": 100,
            "waveforms": {
                "I": "q6.resonator.const.wf.I",
                "Q": "q6.resonator.const.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q7.xy.x180_DragCosine.pulse": {
            "length": 64,
            "waveforms": {
                "I": "q7.xy.x180_DragCosine.wf.I",
                "Q": "q7.xy.x180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q7.xy.x90_DragCosine.pulse": {
            "length": 64,
            "waveforms": {
                "I": "q7.xy.x90_DragCosine.wf.I",
                "Q": "q7.xy.x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q7.xy.-x90_DragCosine.pulse": {
            "length": 64,
            "waveforms": {
                "I": "q7.xy.-x90_DragCosine.wf.I",
                "Q": "q7.xy.-x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q7.xy.y180_DragCosine.pulse": {
            "length": 64,
            "waveforms": {
                "I": "q7.xy.y180_DragCosine.wf.I",
                "Q": "q7.xy.y180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q7.xy.y90_DragCosine.pulse": {
            "length": 64,
            "waveforms": {
                "I": "q7.xy.y90_DragCosine.wf.I",
                "Q": "q7.xy.y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q7.xy.-y90_DragCosine.pulse": {
            "length": 64,
            "waveforms": {
                "I": "q7.xy.-y90_DragCosine.wf.I",
                "Q": "q7.xy.-y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q7.xy.x180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q7.xy.x180_Square.wf.I",
                "Q": "q7.xy.x180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q7.xy.x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q7.xy.x90_Square.wf.I",
                "Q": "q7.xy.x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q7.xy.-x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q7.xy.-x90_Square.wf.I",
                "Q": "q7.xy.-x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q7.xy.y180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q7.xy.y180_Square.wf.I",
                "Q": "q7.xy.y180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q7.xy.y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q7.xy.y90_Square.wf.I",
                "Q": "q7.xy.y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q7.xy.-y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q7.xy.-y90_Square.wf.I",
                "Q": "q7.xy.-y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q7.xy.saturation.pulse": {
            "length": 20000,
            "waveforms": {
                "I": "q7.xy.saturation.wf.I",
                "Q": "q7.xy.saturation.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q7.z.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "q7.z.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q7.resonator.readout.pulse": {
            "length": 2000,
            "waveforms": {
                "I": "q7.resonator.readout.wf.I",
                "Q": "q7.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "q7.resonator.readout.iw1",
                "iw2": "q7.resonator.readout.iw2",
                "iw3": "q7.resonator.readout.iw3",
            },
            "operation": "measurement",
            "digital_marker": "ON",
        },
        "q7.resonator.const.pulse": {
            "length": 100,
            "waveforms": {
                "I": "q7.resonator.const.wf.I",
                "Q": "q7.resonator.const.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q8.xy.x180_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q8.xy.x180_DragCosine.wf.I",
                "Q": "q8.xy.x180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q8.xy.x90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q8.xy.x90_DragCosine.wf.I",
                "Q": "q8.xy.x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q8.xy.-x90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q8.xy.-x90_DragCosine.wf.I",
                "Q": "q8.xy.-x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q8.xy.y180_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q8.xy.y180_DragCosine.wf.I",
                "Q": "q8.xy.y180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q8.xy.y90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q8.xy.y90_DragCosine.wf.I",
                "Q": "q8.xy.y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q8.xy.-y90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q8.xy.-y90_DragCosine.wf.I",
                "Q": "q8.xy.-y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q8.xy.x180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q8.xy.x180_Square.wf.I",
                "Q": "q8.xy.x180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q8.xy.x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q8.xy.x90_Square.wf.I",
                "Q": "q8.xy.x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q8.xy.-x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q8.xy.-x90_Square.wf.I",
                "Q": "q8.xy.-x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q8.xy.y180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q8.xy.y180_Square.wf.I",
                "Q": "q8.xy.y180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q8.xy.y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q8.xy.y90_Square.wf.I",
                "Q": "q8.xy.y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q8.xy.-y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q8.xy.-y90_Square.wf.I",
                "Q": "q8.xy.-y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q8.xy.saturation.pulse": {
            "length": 20000,
            "waveforms": {
                "I": "q8.xy.saturation.wf.I",
                "Q": "q8.xy.saturation.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q8.z.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "q8.z.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q8.resonator.readout.pulse": {
            "length": 2000,
            "waveforms": {
                "I": "q8.resonator.readout.wf.I",
                "Q": "q8.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "q8.resonator.readout.iw1",
                "iw2": "q8.resonator.readout.iw2",
                "iw3": "q8.resonator.readout.iw3",
            },
            "operation": "measurement",
            "digital_marker": "ON",
        },
        "q8.resonator.const.pulse": {
            "length": 100,
            "waveforms": {
                "I": "q8.resonator.const.wf.I",
                "Q": "q8.resonator.const.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q9.xy.x180_DragCosine.pulse": {
            "length": 64,
            "waveforms": {
                "I": "q9.xy.x180_DragCosine.wf.I",
                "Q": "q9.xy.x180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q9.xy.x90_DragCosine.pulse": {
            "length": 64,
            "waveforms": {
                "I": "q9.xy.x90_DragCosine.wf.I",
                "Q": "q9.xy.x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q9.xy.-x90_DragCosine.pulse": {
            "length": 64,
            "waveforms": {
                "I": "q9.xy.-x90_DragCosine.wf.I",
                "Q": "q9.xy.-x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q9.xy.y180_DragCosine.pulse": {
            "length": 64,
            "waveforms": {
                "I": "q9.xy.y180_DragCosine.wf.I",
                "Q": "q9.xy.y180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q9.xy.y90_DragCosine.pulse": {
            "length": 64,
            "waveforms": {
                "I": "q9.xy.y90_DragCosine.wf.I",
                "Q": "q9.xy.y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q9.xy.-y90_DragCosine.pulse": {
            "length": 64,
            "waveforms": {
                "I": "q9.xy.-y90_DragCosine.wf.I",
                "Q": "q9.xy.-y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q9.xy.x180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q9.xy.x180_Square.wf.I",
                "Q": "q9.xy.x180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q9.xy.x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q9.xy.x90_Square.wf.I",
                "Q": "q9.xy.x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q9.xy.-x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q9.xy.-x90_Square.wf.I",
                "Q": "q9.xy.-x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q9.xy.y180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q9.xy.y180_Square.wf.I",
                "Q": "q9.xy.y180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q9.xy.y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q9.xy.y90_Square.wf.I",
                "Q": "q9.xy.y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q9.xy.-y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q9.xy.-y90_Square.wf.I",
                "Q": "q9.xy.-y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q9.xy.saturation.pulse": {
            "length": 20000,
            "waveforms": {
                "I": "q9.xy.saturation.wf.I",
                "Q": "q9.xy.saturation.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q9.z.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "q9.z.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q9.resonator.readout.pulse": {
            "length": 2000,
            "waveforms": {
                "I": "q9.resonator.readout.wf.I",
                "Q": "q9.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "q9.resonator.readout.iw1",
                "iw2": "q9.resonator.readout.iw2",
                "iw3": "q9.resonator.readout.iw3",
            },
            "operation": "measurement",
            "digital_marker": "ON",
        },
        "q9.resonator.const.pulse": {
            "length": 100,
            "waveforms": {
                "I": "q9.resonator.const.wf.I",
                "Q": "q9.resonator.const.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q10.xy.x180_DragCosine.pulse": {
            "length": 64,
            "waveforms": {
                "I": "q10.xy.x180_DragCosine.wf.I",
                "Q": "q10.xy.x180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q10.xy.x90_DragCosine.pulse": {
            "length": 64,
            "waveforms": {
                "I": "q10.xy.x90_DragCosine.wf.I",
                "Q": "q10.xy.x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q10.xy.-x90_DragCosine.pulse": {
            "length": 64,
            "waveforms": {
                "I": "q10.xy.-x90_DragCosine.wf.I",
                "Q": "q10.xy.-x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q10.xy.y180_DragCosine.pulse": {
            "length": 64,
            "waveforms": {
                "I": "q10.xy.y180_DragCosine.wf.I",
                "Q": "q10.xy.y180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q10.xy.y90_DragCosine.pulse": {
            "length": 64,
            "waveforms": {
                "I": "q10.xy.y90_DragCosine.wf.I",
                "Q": "q10.xy.y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q10.xy.-y90_DragCosine.pulse": {
            "length": 64,
            "waveforms": {
                "I": "q10.xy.-y90_DragCosine.wf.I",
                "Q": "q10.xy.-y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q10.xy.x180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q10.xy.x180_Square.wf.I",
                "Q": "q10.xy.x180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q10.xy.x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q10.xy.x90_Square.wf.I",
                "Q": "q10.xy.x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q10.xy.-x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q10.xy.-x90_Square.wf.I",
                "Q": "q10.xy.-x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q10.xy.y180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q10.xy.y180_Square.wf.I",
                "Q": "q10.xy.y180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q10.xy.y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q10.xy.y90_Square.wf.I",
                "Q": "q10.xy.y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q10.xy.-y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q10.xy.-y90_Square.wf.I",
                "Q": "q10.xy.-y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q10.xy.saturation.pulse": {
            "length": 20000,
            "waveforms": {
                "I": "q10.xy.saturation.wf.I",
                "Q": "q10.xy.saturation.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q10.z.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "q10.z.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q10.z.SWAP_unipolar.pulse": {
            "length": 128,
            "waveforms": {
                "single": "q10.z.SWAP_unipolar.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q10.z.SWAP_flattop.pulse": {
            "length": 128,
            "waveforms": {
                "single": "q10.z.SWAP_flattop.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q10.z.SWAP_bipolar.pulse": {
            "length": 128,
            "waveforms": {
                "single": "q10.z.SWAP_bipolar.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q10.resonator.readout.pulse": {
            "length": 2000,
            "waveforms": {
                "I": "q10.resonator.readout.wf.I",
                "Q": "q10.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "q10.resonator.readout.iw1",
                "iw2": "q10.resonator.readout.iw2",
                "iw3": "q10.resonator.readout.iw3",
            },
            "operation": "measurement",
            "digital_marker": "ON",
        },
        "q10.resonator.const.pulse": {
            "length": 100,
            "waveforms": {
                "I": "q10.resonator.const.wf.I",
                "Q": "q10.resonator.const.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q6_q7.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "coupler_q6_q7.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q7_q8.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "coupler_q7_q8.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q8_q9.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "coupler_q8_q9.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q9_q10.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "coupler_q9_q10.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q10.z.SWAP_unipolar.flux_pulse_control_q9_q10.pulse": {
            "length": 128,
            "waveforms": {
                "single": "q10.z.SWAP_unipolar.flux_pulse_control_q9_q10.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q9_q10.SWAP_unipolar.coupler_pulse_control_q9_q10.pulse": {
            "length": 128,
            "waveforms": {
                "single": "coupler_q9_q10.SWAP_unipolar.coupler_pulse_control_q9_q10.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q10.z.SWAP_flattop.flux_pulse_control_q9_q10.pulse": {
            "length": 128,
            "waveforms": {
                "single": "q10.z.SWAP_flattop.flux_pulse_control_q9_q10.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q9_q10.SWAP_flattop.coupler_pulse_control_q9_q10.pulse": {
            "length": 128,
            "waveforms": {
                "single": "coupler_q9_q10.SWAP_flattop.coupler_pulse_control_q9_q10.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q10.z.SWAP_bipolar.flux_pulse_control_q9_q10.pulse": {
            "length": 128,
            "waveforms": {
                "single": "q10.z.SWAP_bipolar.flux_pulse_control_q9_q10.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q9_q10.SWAP_bipolar.coupler_pulse_control_q9_q10.pulse": {
            "length": 128,
            "waveforms": {
                "single": "coupler_q9_q10.SWAP_bipolar.coupler_pulse_control_q9_q10.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q5_q6.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "coupler_q5_q6.const.wf",
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
        "q5.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.013669663395844208, 0.05231504459724201, 0.10925400611220526, 0.17464128422057051, 0.23717082451262841, 0.2860307014088422] + [0.3127725983158096] * 2 + [0.2860307014088422, 0.23717082451262853, 0.17464128422057065, 0.1092540061122053, 0.052315044597241976, 0.013669663395844191, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.006834831697922104, 0.026157522298621005, 0.05462700305610263, 0.08732064211028526, 0.11858541225631421, 0.1430153507044211] + [0.1563862991579048] * 2 + [0.1430153507044211, 0.11858541225631426, 0.08732064211028533, 0.05462700305610265, 0.026157522298620988, 0.0068348316979220955, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.006834831697922104, -0.026157522298621005, -0.05462700305610263, -0.08732064211028526, -0.11858541225631421, -0.1430153507044211] + [-0.1563862991579048] * 2 + [-0.1430153507044211, -0.11858541225631426, -0.08732064211028533, -0.05462700305610265, -0.026157522298620988, -0.0068348316979220955, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 8.370254761571174e-19, 3.203372595663173e-18, 6.689878443966877e-18, 1.0693694485985243e-17, 1.4522524554526452e-17, 1.7514329146910545e-17] + [1.9151798069422854e-17] * 2 + [1.7514329146910545e-17, 1.4522524554526458e-17, 1.069369448598525e-17, 6.689878443966879e-18, 3.203372595663171e-18, 8.370254761571164e-19, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 8.370254761571174e-19, 3.203372595663173e-18, 6.689878443966877e-18, 1.0693694485985243e-17, 1.4522524554526452e-17, 1.7514329146910545e-17] + [1.9151798069422854e-17] * 2 + [1.7514329146910545e-17, 1.4522524554526458e-17, 1.069369448598525e-17, 6.689878443966879e-18, 3.203372595663171e-18, 8.370254761571164e-19, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.013669663395844208, 0.05231504459724201, 0.10925400611220526, 0.17464128422057051, 0.23717082451262841, 0.2860307014088422] + [0.3127725983158096] * 2 + [0.2860307014088422, 0.23717082451262853, 0.17464128422057065, 0.1092540061122053, 0.052315044597241976, 0.013669663395844191, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 4.185127380785587e-19, 1.6016862978315865e-18, 3.3449392219834385e-18, 5.346847242992621e-18, 7.261262277263226e-18, 8.757164573455273e-18] + [9.575899034711427e-18] * 2 + [8.757164573455273e-18, 7.261262277263229e-18, 5.346847242992625e-18, 3.3449392219834396e-18, 1.6016862978315856e-18, 4.185127380785582e-19, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.006834831697922104, 0.026157522298621005, 0.05462700305610263, 0.08732064211028526, 0.11858541225631421, 0.1430153507044211] + [0.1563862991579048] * 2 + [0.1430153507044211, 0.11858541225631426, 0.08732064211028533, 0.05462700305610265, 0.026157522298620988, 0.0068348316979220955, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 4.185127380785587e-19, 1.6016862978315865e-18, 3.3449392219834385e-18, 5.346847242992621e-18, 7.261262277263226e-18, 8.757164573455273e-18] + [9.575899034711427e-18] * 2 + [8.757164573455273e-18, 7.261262277263229e-18, 5.346847242992625e-18, 3.3449392219834396e-18, 1.6016862978315856e-18, 4.185127380785582e-19, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.006834831697922104, -0.026157522298621005, -0.05462700305610263, -0.08732064211028526, -0.11858541225631421, -0.1430153507044211] + [-0.1563862991579048] * 2 + [-0.1430153507044211, -0.11858541225631426, -0.08732064211028533, -0.05462700305610265, -0.026157522298620988, -0.0068348316979220955, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.31622776601683794,
        },
        "q5.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q5.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.15811388300841897,
        },
        "q5.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q5.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "q5.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q5.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "q5.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q5.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "q5.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q5.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "q5.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q5.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.5,
        },
        "q5.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q5.z.const.wf": {
            "type": "constant",
            "sample": 1.25,
        },
        "q5.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q5.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q5.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "q5.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q6.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.010149304539422525, 0.03884231119928679, 0.08111773846032007, 0.12966578088889005, 0.17609204090397754, 0.21236899637964252] + [0.23222403214834825] * 2 + [0.21236899637964252, 0.1760920409039776, 0.12966578088889016, 0.0811177384603201, 0.038842311199286765, 0.010149304539422511, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q6.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.004244330997905204, -0.007754778607735973, -0.009924354546337218, -0.01037791942394615, -0.009037047752716626, -0.006133588426040946, -0.002169575938601245, 0.002169575938601242, 0.0061335884260409436, 0.009037047752716623, 0.010377919423946148, 0.00992435454633722, 0.007754778607735971, 0.004244330997905204, 1.182407475327649e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q6.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.005074652269711263, 0.019421155599643396, 0.040558869230160034, 0.06483289044444503, 0.08804602045198877, 0.10618449818982126] + [0.11611201607417412] * 2 + [0.10618449818982126, 0.0880460204519888, 0.06483289044444508, 0.04055886923016005, 0.019421155599643383, 0.005074652269711256, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q6.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q6.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.005074652269711263, -0.019421155599643396, -0.040558869230160034, -0.06483289044444503, -0.08804602045198877, -0.10618449818982126] + [-0.11611201607417412] * 2 + [-0.10618449818982126, -0.0880460204519888, -0.06483289044444508, -0.04055886923016005, -0.019421155599643383, -0.005074652269711256, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q6.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 6.214656658887749e-19, 2.378405604084598e-18, 4.967028937975156e-18, 7.939739176226062e-18, 1.0782527712419045e-17, 1.3003850582723253e-17] + [1.4219620882778337e-17] * 2 + [1.3003850582723253e-17, 1.0782527712419048e-17, 7.93973917622607e-18, 4.9670289379751576e-18, 2.3784056040845962e-18, 6.21465665888774e-19, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q6.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004244330997905205, 0.007754778607735976, 0.009924354546337224, 0.010377919423946159, 0.009037047752716637, 0.006133588426040959, 0.002169575938601259, -0.0021695759386012275, -0.0061335884260409305, -0.009037047752716612, -0.01037791942394614, -0.009924354546337215, -0.007754778607735969, -0.004244330997905203, -1.182407475327649e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q6.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.010149304539422525, 0.03884231119928679, 0.08111773846032007, 0.12966578088889005, 0.17609204090397754, 0.21236899637964252] + [0.23222403214834825] * 2 + [0.21236899637964252, 0.1760920409039776, 0.12966578088889016, 0.0811177384603201, 0.038842311199286765, 0.010149304539422511, 7.240157649739542e-34],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q6.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 3.1073283294438743e-19, 1.189202802042299e-18, 2.483514468987578e-18, 3.969869588113031e-18, 5.3912638562095224e-18, 6.501925291361626e-18] + [7.109810441389169e-18] * 2 + [6.501925291361626e-18, 5.391263856209524e-18, 3.969869588113035e-18, 2.4835144689875788e-18, 1.1892028020422981e-18, 3.10732832944387e-19, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q6.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.005074652269711263, 0.019421155599643396, 0.040558869230160034, 0.06483289044444503, 0.08804602045198877, 0.10618449818982126] + [0.11611201607417412] * 2 + [0.10618449818982126, 0.0880460204519888, 0.06483289044444508, 0.04055886923016005, 0.019421155599643383, 0.005074652269711256, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q6.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 3.1073283294438743e-19, 1.189202802042299e-18, 2.483514468987578e-18, 3.969869588113031e-18, 5.3912638562095224e-18, 6.501925291361626e-18] + [7.109810441389169e-18] * 2 + [6.501925291361626e-18, 5.391263856209524e-18, 3.969869588113035e-18, 2.4835144689875788e-18, 1.1892028020422981e-18, 3.10732832944387e-19, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q6.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.005074652269711263, -0.019421155599643396, -0.040558869230160034, -0.06483289044444503, -0.08804602045198877, -0.10618449818982126] + [-0.11611201607417412] * 2 + [-0.10618449818982126, -0.0880460204519888, -0.06483289044444508, -0.04055886923016005, -0.019421155599643383, -0.005074652269711256, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q6.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.31622776601683794,
        },
        "q6.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q6.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.15811388300841897,
        },
        "q6.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q6.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "q6.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q6.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "q6.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q6.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "q6.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q6.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "q6.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q6.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.9,
        },
        "q6.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q6.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 3.168763005159296e-05, 0.00012673348719410162, 0.00028508648154643336, 0.0005066614938209973, 0.0007913394210777118, 0.0011389672407453107, 0.0015493580928753236, 0.002022291380584713, 0.0025575128886330266, 0.0031547349200703977, 0.0038136364508829356, 0.004533863302552358, 0.005315028332437103, 0.006156711641872697, 0.007058460801879262, 0.008019791096355216, 0.00904018578262603, 0.010119096369208278, 0.011255942910639652, 0.01245011431921621, 0.013700968693469729, 0.015007833663208025, 0.016370006750933464, 0.017786755749444636, 0.019257319115418888, 0.02078090637876371, 0.022356698567517187, 0.023983848648068896, 0.0256614819804649, 0.027388696788551832, 0.02916456464470753, 0.030988130968897586, 0.032858415541789476, 0.03477441303164879, 0.036735093534733795, 0.0387394031288984, 0.04078626444010559, 0.042874577221546896, 0.045003218945056836, 0.04717104540450389, 0.049376891330834324, 0.0516195710184378, 0.05389787896249817, 0.05621059050698702, 0.05855646250295156, 0.06093423397674275, 0.06334262680782511, 0.06578034641580306, 0.06824608245629535, 0.07073850952528254, 0.07325628787155, 0.07579806411684264, 0.0783624719833446, 0.08094813302809288, 0.08355365738393013, 0.08617764450659811, 0.08881868392757053, 0.09147535601222055, 0.09414623272291499, 0.09682987838662585, 0.09952485046664568, 0.10222970033799257, 0.10494297406608788, 0.10766321318828753, 0.11038895549784807, 0.11311873582990449, 0.1158510868490386, 0.11858453983801395, 0.1213176254872537, 0.12404887468463659, 0.12677681930518717, 0.12949999300023488, 0.13221693198561887, 0.13492617582851413, 0.13762626823245577, 0.14031575782014064, 0.1429931989135841, 0.14565715231121368, 0.14830618606148113, 0.15093887623257768, 0.15355380767783833, 0.15614957479642377, 0.15872478228887146, 0.16127804590710879, 0.16380799319852676, 0.16631326424371232, 0.16879251238744433, 0.17124440496255933, 0.17366762400629882, 0.17606086696875145, 0.17842284741301176, 0.1807522957066767, 0.18304795970430945, 0.18530860542050392, 0.18753301769318687, 0.1897200008368029, 0.1918683792850299, 0.19397699822267925, 0.19604472420644256, 0.19807044577415003, 0.20005307404221317, 0.20199154329093114, 0.2038848115373457, 0.20573186109533706, 0.20753169912265937, 0.20928335815462196, 0.21098589662412942, 0.21263839936780118, 0.21423997811789788, 0.2157897719797909, 0.2172869478947181, 0.21873070108757675, 0.22012025549951322, 0.22145486420507712, 0.22273380981371502, 0.2239564048553886, 0.2251219921501095, 0.22622994516119255, 0.22727966833203717, 0.22827059740625627, 0.2292021997309801, 0.2300739745431727, 0.23088545323880616, 0.2316361996247491, 0.2323258101532328, 0.23295391413876973, 0.23352017395740768, 0.2340242852282124, 0.2344659769768811, 0.23484501178139874, 0.23516118589965962, 0.23541432937898407, 0.23560430614747335, 0.23573101408715189] + [0.23579438508885894] * 2 + [0.23573101408715189, 0.23560430614747335, 0.23541432937898407, 0.23516118589965965, 0.23484501178139877, 0.2344659769768811, 0.2340242852282124, 0.23352017395740768, 0.23295391413876973, 0.23232581015323284, 0.23163619962474913, 0.23088545323880616, 0.2300739745431727, 0.2292021997309801, 0.22827059740625627, 0.2272796683320372, 0.22622994516119255, 0.2251219921501095, 0.2239564048553886, 0.22273380981371504, 0.22145486420507718, 0.22012025549951325, 0.21873070108757675, 0.21728694789471814, 0.21578977197979093, 0.21423997811789788, 0.21263839936780124, 0.21098589662412948, 0.20928335815462198, 0.20753169912265937, 0.20573186109533706, 0.20388481153734572, 0.20199154329093114, 0.20005307404221315, 0.19807044577415003, 0.1960447242064426, 0.19397699822267928, 0.19186837928502992, 0.189720000836803, 0.18753301769318695, 0.18530860542050387, 0.1830479597043095, 0.18075229570667675, 0.17842284741301173, 0.17606086696875142, 0.17366762400629882, 0.17124440496255944, 0.16879251238744436, 0.1663132642437124, 0.16380799319852687, 0.16127804590710873, 0.15872478228887144, 0.15614957479642383, 0.1535538076778383, 0.1509388762325777, 0.1483061860614812, 0.14565715231121368, 0.14299319891358417, 0.14031575782014072, 0.1376262682324558, 0.13492617582851418, 0.13221693198561893, 0.12949999300023482, 0.12677681930518714, 0.12404887468463664, 0.12131762548725367, 0.11858453983801395, 0.11585108684903864, 0.11311873582990449, 0.11038895549784812, 0.10766321318828762, 0.10494297406608788, 0.10222970033799254, 0.09952485046664568, 0.0968298783866259, 0.09414623272291499, 0.09147535601222058, 0.08881868392757061, 0.08617764450659812, 0.08355365738393018, 0.08094813302809298, 0.07836247198334463, 0.07579806411684263, 0.07325628787155003, 0.0707385095252825, 0.06824608245629535, 0.0657803464158031, 0.06334262680782508, 0.060934233976742774, 0.058556462502951614, 0.05621059050698704, 0.05389787896249822, 0.05161957101843788, 0.049376891330834366, 0.04717104540450386, 0.045003218945056836, 0.042874577221546945, 0.04078626444010557, 0.038739403128898425, 0.03673509353473386, 0.034774413031648806, 0.03285841554178953, 0.030988130968897652, 0.029164564644707573, 0.027388696788551832, 0.025661481980464914, 0.02398384864806887, 0.022356698567517187, 0.02078090637876375, 0.019257319115418888, 0.01778675574944466, 0.016370006750933502, 0.015007833663208037, 0.013700968693469755, 0.012450114319216225, 0.011255942910639626, 0.010119096369208278, 0.009040185782626041, 0.008019791096355242, 0.007058460801879262, 0.00615671164187271, 0.005315028332437129, 0.004533863302552371, 0.0038136364508829616, 0.0031547349200704237, 0.0025575128886330136, 0.002022291380584713, 0.0015493580928753236, 0.0011389672407453107, 0.0007913394210777118, 0.0005066614938209973, 0.00028508648154643336, 0.00012673348719410162, 3.168763005159296e-05, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q6.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 272,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q6.z.const.wf": {
            "type": "constant",
            "sample": 1.25,
        },
        "q6.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.020839591836734693,
        },
        "q6.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q6.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "q6.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q7.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.00010369801696355364, 0.00041376147037330623, 0.0009271088103273131, 0.0016386381603299622, 0.0025412780220327082, 0.003626057554961664, 0.004882195732760553, 0.006297208489874216, 0.007857032793800813, 0.009546166409826912, 0.011347821969200736, 0.013244093809544888, 0.015216135929373494, 0.01724434928812192, 0.01930857659021714, 0.021388302617337186, 0.023462858117867694, 0.02551162522721051, 0.02751424237738503, 0.02945080665943862, 0.03130207162749756, 0.03304963857859294, 0.03467613940723678, 0.03616540921745729, 0.03750264697679746, 0.03867456261562523, 0.039669509109816314, 0.04047759823411264, 0.04109079883574741, 0.04150301665164941] + [0.041710154875967465] * 2 + [0.04150301665164941, 0.04109079883574741, 0.04047759823411264, 0.039669509109816314, 0.03867456261562523, 0.03750264697679747, 0.036165409217457305, 0.03467613940723679, 0.03304963857859296, 0.03130207162749755, 0.029450806659438625, 0.027514242377385045, 0.025511625227210514, 0.02346285811786771, 0.021388302617337193, 0.01930857659021714, 0.01724434928812193, 0.015216135929373495, 0.013244093809544897, 0.011347821969200741, 0.009546166409826907, 0.007857032793800818, 0.0062972084898742136, 0.004882195732760558, 0.003626057554961673, 0.002541278022032713, 0.0016386381603299622, 0.0009271088103273154, 0.00041376147037330623, 0.00010369801696355596, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q7.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 64,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q7.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 5.184900848177682e-05, 0.00020688073518665312, 0.00046355440516365656, 0.0008193190801649811, 0.0012706390110163541, 0.001813028777480832, 0.0024410978663802767, 0.003148604244937108, 0.003928516396900407, 0.004773083204913456, 0.005673910984600368, 0.006622046904772444, 0.007608067964686747, 0.00862217464406096, 0.00965428829510857, 0.010694151308668593, 0.011731429058933847, 0.012755812613605255, 0.013757121188692515, 0.01472540332971931, 0.01565103581374878, 0.01652481928929647, 0.01733806970361839, 0.018082704608728645, 0.01875132348839873, 0.019337281307812614, 0.019834754554908157, 0.02023879911705632, 0.020545399417873703, 0.020751508325824706] + [0.020855077437983732] * 2 + [0.020751508325824706, 0.020545399417873703, 0.02023879911705632, 0.019834754554908157, 0.019337281307812614, 0.018751323488398735, 0.018082704608728652, 0.017338069703618394, 0.01652481928929648, 0.015651035813748774, 0.014725403329719312, 0.013757121188692522, 0.012755812613605257, 0.011731429058933854, 0.010694151308668597, 0.00965428829510857, 0.008622174644060966, 0.007608067964686748, 0.006622046904772448, 0.005673910984600371, 0.004773083204913453, 0.003928516396900409, 0.0031486042449371068, 0.002441097866380279, 0.0018130287774808366, 0.0012706390110163565, 0.0008193190801649811, 0.0004635544051636577, 0.00020688073518665312, 5.184900848177798e-05, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q7.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 64,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q7.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -5.184900848177682e-05, -0.00020688073518665312, -0.00046355440516365656, -0.0008193190801649811, -0.0012706390110163541, -0.001813028777480832, -0.0024410978663802767, -0.003148604244937108, -0.003928516396900407, -0.004773083204913456, -0.005673910984600368, -0.006622046904772444, -0.007608067964686747, -0.00862217464406096, -0.00965428829510857, -0.010694151308668593, -0.011731429058933847, -0.012755812613605255, -0.013757121188692515, -0.01472540332971931, -0.01565103581374878, -0.01652481928929647, -0.01733806970361839, -0.018082704608728645, -0.01875132348839873, -0.019337281307812614, -0.019834754554908157, -0.02023879911705632, -0.020545399417873703, -0.020751508325824706] + [-0.020855077437983732] * 2 + [-0.020751508325824706, -0.020545399417873703, -0.02023879911705632, -0.019834754554908157, -0.019337281307812614, -0.018751323488398735, -0.018082704608728652, -0.017338069703618394, -0.01652481928929648, -0.015651035813748774, -0.014725403329719312, -0.013757121188692522, -0.012755812613605257, -0.011731429058933854, -0.010694151308668597, -0.00965428829510857, -0.008622174644060966, -0.007608067964686748, -0.006622046904772448, -0.005673910984600371, -0.004773083204913453, -0.003928516396900409, -0.0031486042449371068, -0.002441097866380279, -0.0018130287774808366, -0.0012706390110163565, -0.0008193190801649811, -0.0004635544051636577, -0.00020688073518665312, -5.184900848177798e-05, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q7.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 6.3496722276171956e-21, 2.5335583015158596e-20, 5.676904185143273e-20, 1.0033764890043977e-19, 1.5560839977129365e-19, 2.22031988910394e-19, 2.989482688468039e-19, 3.8559281103439985e-19, 4.811045030861976e-19, 5.845341068961254e-19, 6.948536925937848e-19, 8.109668545733211e-19, 9.317196080649142e-19, 1.0559118579538724e-18, 1.1823093258650467e-18, 1.309655816975847e-18, 1.4366857046447582e-18, 1.5621365087775109e-18, 1.6847614429214519e-18, 1.803341805389453e-18, 1.9166990912648035e-18, 2.023706704912536e-18, 2.123301156592996e-18, 2.214492631900663e-18, 2.296374828984409e-18, 2.368133965782466e-18, 2.429056867754166e-18, 2.4785380557289302e-18, 2.5160857634302926e-18, 2.5413268248700876e-18] + [2.554010383039696e-18] * 2 + [2.5413268248700876e-18, 2.5160857634302926e-18, 2.4785380557289302e-18, 2.429056867754166e-18, 2.368133965782466e-18, 2.2963748289844092e-18, 2.2144926319006636e-18, 2.1233011565929965e-18, 2.023706704912537e-18, 1.9166990912648027e-18, 1.803341805389453e-18, 1.6847614429214529e-18, 1.562136508777511e-18, 1.4366857046447591e-18, 1.3096558169758475e-18, 1.1823093258650467e-18, 1.0559118579538732e-18, 9.317196080649144e-19, 8.109668545733217e-19, 6.948536925937852e-19, 5.845341068961251e-19, 4.811045030861979e-19, 3.8559281103439966e-19, 2.989482688468042e-19, 2.2203198891039454e-19, 1.5560839977129394e-19, 1.0033764890043977e-19, 5.676904185143287e-20, 2.5335583015158596e-20, 6.349672227617337e-21, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q7.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 6.3496722276171956e-21, 2.5335583015158596e-20, 5.676904185143273e-20, 1.0033764890043977e-19, 1.5560839977129365e-19, 2.22031988910394e-19, 2.989482688468039e-19, 3.8559281103439985e-19, 4.811045030861976e-19, 5.845341068961254e-19, 6.948536925937848e-19, 8.109668545733211e-19, 9.317196080649142e-19, 1.0559118579538724e-18, 1.1823093258650467e-18, 1.309655816975847e-18, 1.4366857046447582e-18, 1.5621365087775109e-18, 1.6847614429214519e-18, 1.803341805389453e-18, 1.9166990912648035e-18, 2.023706704912536e-18, 2.123301156592996e-18, 2.214492631900663e-18, 2.296374828984409e-18, 2.368133965782466e-18, 2.429056867754166e-18, 2.4785380557289302e-18, 2.5160857634302926e-18, 2.5413268248700876e-18] + [2.554010383039696e-18] * 2 + [2.5413268248700876e-18, 2.5160857634302926e-18, 2.4785380557289302e-18, 2.429056867754166e-18, 2.368133965782466e-18, 2.2963748289844092e-18, 2.2144926319006636e-18, 2.1233011565929965e-18, 2.023706704912537e-18, 1.9166990912648027e-18, 1.803341805389453e-18, 1.6847614429214529e-18, 1.562136508777511e-18, 1.4366857046447591e-18, 1.3096558169758475e-18, 1.1823093258650467e-18, 1.0559118579538732e-18, 9.317196080649144e-19, 8.109668545733217e-19, 6.948536925937852e-19, 5.845341068961251e-19, 4.811045030861979e-19, 3.8559281103439966e-19, 2.989482688468042e-19, 2.2203198891039454e-19, 1.5560839977129394e-19, 1.0033764890043977e-19, 5.676904185143287e-20, 2.5335583015158596e-20, 6.349672227617337e-21, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q7.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.00010369801696355364, 0.00041376147037330623, 0.0009271088103273131, 0.0016386381603299622, 0.0025412780220327082, 0.003626057554961664, 0.004882195732760553, 0.006297208489874216, 0.007857032793800813, 0.009546166409826912, 0.011347821969200736, 0.013244093809544888, 0.015216135929373494, 0.01724434928812192, 0.01930857659021714, 0.021388302617337186, 0.023462858117867694, 0.02551162522721051, 0.02751424237738503, 0.02945080665943862, 0.03130207162749756, 0.03304963857859294, 0.03467613940723678, 0.03616540921745729, 0.03750264697679746, 0.03867456261562523, 0.039669509109816314, 0.04047759823411264, 0.04109079883574741, 0.04150301665164941] + [0.041710154875967465] * 2 + [0.04150301665164941, 0.04109079883574741, 0.04047759823411264, 0.039669509109816314, 0.03867456261562523, 0.03750264697679747, 0.036165409217457305, 0.03467613940723679, 0.03304963857859296, 0.03130207162749755, 0.029450806659438625, 0.027514242377385045, 0.025511625227210514, 0.02346285811786771, 0.021388302617337193, 0.01930857659021714, 0.01724434928812193, 0.015216135929373495, 0.013244093809544897, 0.011347821969200741, 0.009546166409826907, 0.007857032793800818, 0.0062972084898742136, 0.004882195732760558, 0.003626057554961673, 0.002541278022032713, 0.0016386381603299622, 0.0009271088103273154, 0.00041376147037330623, 0.00010369801696355596, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q7.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 3.1748361138085978e-21, 1.2667791507579298e-20, 2.8384520925716364e-20, 5.0168824450219887e-20, 7.780419988564683e-20, 1.11015994455197e-19, 1.4947413442340195e-19, 1.9279640551719992e-19, 2.405522515430988e-19, 2.922670534480627e-19, 3.474268462968924e-19, 4.0548342728666054e-19, 4.658598040324571e-19, 5.279559289769362e-19, 5.911546629325234e-19, 6.548279084879235e-19, 7.183428523223791e-19, 7.810682543887554e-19, 8.4238072146072595e-19, 9.016709026947266e-19, 9.583495456324017e-19, 1.011853352456268e-18, 1.061650578296498e-18, 1.1072463159503314e-18, 1.1481874144922044e-18, 1.184066982891233e-18, 1.214528433877083e-18, 1.2392690278644651e-18, 1.2580428817151463e-18, 1.2706634124350438e-18] + [1.277005191519848e-18] * 2 + [1.2706634124350438e-18, 1.2580428817151463e-18, 1.2392690278644651e-18, 1.214528433877083e-18, 1.184066982891233e-18, 1.1481874144922046e-18, 1.1072463159503318e-18, 1.0616505782964983e-18, 1.0118533524562686e-18, 9.583495456324014e-19, 9.016709026947266e-19, 8.423807214607264e-19, 7.810682543887555e-19, 7.183428523223796e-19, 6.548279084879237e-19, 5.911546629325234e-19, 5.279559289769366e-19, 4.658598040324572e-19, 4.0548342728666083e-19, 3.474268462968926e-19, 2.9226705344806255e-19, 2.4055225154309894e-19, 1.9279640551719983e-19, 1.494741344234021e-19, 1.1101599445519727e-19, 7.780419988564697e-20, 5.0168824450219887e-20, 2.8384520925716437e-20, 1.2667791507579298e-20, 3.1748361138086685e-21, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q7.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 5.184900848177682e-05, 0.00020688073518665312, 0.00046355440516365656, 0.0008193190801649811, 0.0012706390110163541, 0.001813028777480832, 0.0024410978663802767, 0.003148604244937108, 0.003928516396900407, 0.004773083204913456, 0.005673910984600368, 0.006622046904772444, 0.007608067964686747, 0.00862217464406096, 0.00965428829510857, 0.010694151308668593, 0.011731429058933847, 0.012755812613605255, 0.013757121188692515, 0.01472540332971931, 0.01565103581374878, 0.01652481928929647, 0.01733806970361839, 0.018082704608728645, 0.01875132348839873, 0.019337281307812614, 0.019834754554908157, 0.02023879911705632, 0.020545399417873703, 0.020751508325824706] + [0.020855077437983732] * 2 + [0.020751508325824706, 0.020545399417873703, 0.02023879911705632, 0.019834754554908157, 0.019337281307812614, 0.018751323488398735, 0.018082704608728652, 0.017338069703618394, 0.01652481928929648, 0.015651035813748774, 0.014725403329719312, 0.013757121188692522, 0.012755812613605257, 0.011731429058933854, 0.010694151308668597, 0.00965428829510857, 0.008622174644060966, 0.007608067964686748, 0.006622046904772448, 0.005673910984600371, 0.004773083204913453, 0.003928516396900409, 0.0031486042449371068, 0.002441097866380279, 0.0018130287774808366, 0.0012706390110163565, 0.0008193190801649811, 0.0004635544051636577, 0.00020688073518665312, 5.184900848177798e-05, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q7.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 3.1748361138085978e-21, 1.2667791507579298e-20, 2.8384520925716364e-20, 5.0168824450219887e-20, 7.780419988564683e-20, 1.11015994455197e-19, 1.4947413442340195e-19, 1.9279640551719992e-19, 2.405522515430988e-19, 2.922670534480627e-19, 3.474268462968924e-19, 4.0548342728666054e-19, 4.658598040324571e-19, 5.279559289769362e-19, 5.911546629325234e-19, 6.548279084879235e-19, 7.183428523223791e-19, 7.810682543887554e-19, 8.4238072146072595e-19, 9.016709026947266e-19, 9.583495456324017e-19, 1.011853352456268e-18, 1.061650578296498e-18, 1.1072463159503314e-18, 1.1481874144922044e-18, 1.184066982891233e-18, 1.214528433877083e-18, 1.2392690278644651e-18, 1.2580428817151463e-18, 1.2706634124350438e-18] + [1.277005191519848e-18] * 2 + [1.2706634124350438e-18, 1.2580428817151463e-18, 1.2392690278644651e-18, 1.214528433877083e-18, 1.184066982891233e-18, 1.1481874144922046e-18, 1.1072463159503318e-18, 1.0616505782964983e-18, 1.0118533524562686e-18, 9.583495456324014e-19, 9.016709026947266e-19, 8.423807214607264e-19, 7.810682543887555e-19, 7.183428523223796e-19, 6.548279084879237e-19, 5.911546629325234e-19, 5.279559289769366e-19, 4.658598040324572e-19, 4.0548342728666083e-19, 3.474268462968926e-19, 2.9226705344806255e-19, 2.4055225154309894e-19, 1.9279640551719983e-19, 1.494741344234021e-19, 1.1101599445519727e-19, 7.780419988564697e-20, 5.0168824450219887e-20, 2.8384520925716437e-20, 1.2667791507579298e-20, 3.1748361138086685e-21, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q7.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -5.184900848177682e-05, -0.00020688073518665312, -0.00046355440516365656, -0.0008193190801649811, -0.0012706390110163541, -0.001813028777480832, -0.0024410978663802767, -0.003148604244937108, -0.003928516396900407, -0.004773083204913456, -0.005673910984600368, -0.006622046904772444, -0.007608067964686747, -0.00862217464406096, -0.00965428829510857, -0.010694151308668593, -0.011731429058933847, -0.012755812613605255, -0.013757121188692515, -0.01472540332971931, -0.01565103581374878, -0.01652481928929647, -0.01733806970361839, -0.018082704608728645, -0.01875132348839873, -0.019337281307812614, -0.019834754554908157, -0.02023879911705632, -0.020545399417873703, -0.020751508325824706] + [-0.020855077437983732] * 2 + [-0.020751508325824706, -0.020545399417873703, -0.02023879911705632, -0.019834754554908157, -0.019337281307812614, -0.018751323488398735, -0.018082704608728652, -0.017338069703618394, -0.01652481928929648, -0.015651035813748774, -0.014725403329719312, -0.013757121188692522, -0.012755812613605257, -0.011731429058933854, -0.010694151308668597, -0.00965428829510857, -0.008622174644060966, -0.007608067964686748, -0.006622046904772448, -0.005673910984600371, -0.004773083204913453, -0.003928516396900409, -0.0031486042449371068, -0.002441097866380279, -0.0018130287774808366, -0.0012706390110163565, -0.0008193190801649811, -0.0004635544051636577, -0.00020688073518665312, -5.184900848177798e-05, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q7.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.31622776601683794,
        },
        "q7.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q7.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.15811388300841897,
        },
        "q7.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q7.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "q7.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q7.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "q7.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q7.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "q7.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q7.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "q7.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q7.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q7.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q7.z.const.wf": {
            "type": "constant",
            "sample": 1.25,
        },
        "q7.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.01681561085678195,
        },
        "q7.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q7.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "q7.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q8.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.006911554496302379, 0.026451147423297412, 0.05524020565227865, 0.0883008394750719, 0.11991656495728692, 0.14462073594319957] + [0.1581417768389978] * 2 + [0.14462073594319957, 0.11991656495728698, 0.08830083947507197, 0.05524020565227867, 0.026451147423297395, 0.006911554496302369, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q8.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q8.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0034557772481511894, 0.013225573711648706, 0.027620102826139324, 0.04415041973753595, 0.05995828247864346, 0.07231036797159979] + [0.0790708884194989] * 2 + [0.07231036797159979, 0.05995828247864349, 0.04415041973753599, 0.027620102826139335, 0.013225573711648697, 0.0034557772481511847, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q8.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q8.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0034557772481511894, -0.013225573711648706, -0.027620102826139324, -0.04415041973753595, -0.05995828247864346, -0.07231036797159979] + [-0.0790708884194989] * 2 + [-0.07231036797159979, -0.05995828247864349, -0.04415041973753599, -0.027620102826139335, -0.013225573711648697, -0.0034557772481511847, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q8.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 4.2321065455146026e-19, 1.6196656512857967e-18, 3.382487051815229e-18, 5.406867021258553e-18, 7.342771871984354e-18, 8.855466068158696e-18] + [9.683391040867685e-18] * 2 + [8.855466068158696e-18, 7.342771871984358e-18, 5.406867021258557e-18, 3.38248705181523e-18, 1.6196656512857958e-18, 4.232106545514597e-19, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q8.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 4.2321065455146026e-19, 1.6196656512857967e-18, 3.382487051815229e-18, 5.406867021258553e-18, 7.342771871984354e-18, 8.855466068158696e-18] + [9.683391040867685e-18] * 2 + [8.855466068158696e-18, 7.342771871984358e-18, 5.406867021258557e-18, 3.38248705181523e-18, 1.6196656512857958e-18, 4.232106545514597e-19, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q8.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.006911554496302379, 0.026451147423297412, 0.05524020565227865, 0.0883008394750719, 0.11991656495728692, 0.14462073594319957] + [0.1581417768389978] * 2 + [0.14462073594319957, 0.11991656495728698, 0.08830083947507197, 0.05524020565227867, 0.026451147423297395, 0.006911554496302369, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q8.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 2.1160532727573013e-19, 8.098328256428984e-19, 1.6912435259076145e-18, 2.7034335106292763e-18, 3.671385935992177e-18, 4.427733034079348e-18] + [4.8416955204338425e-18] * 2 + [4.427733034079348e-18, 3.671385935992179e-18, 2.7034335106292786e-18, 1.691243525907615e-18, 8.098328256428979e-19, 2.1160532727572984e-19, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q8.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0034557772481511894, 0.013225573711648706, 0.027620102826139324, 0.04415041973753595, 0.05995828247864346, 0.07231036797159979] + [0.0790708884194989] * 2 + [0.07231036797159979, 0.05995828247864349, 0.04415041973753599, 0.027620102826139335, 0.013225573711648697, 0.0034557772481511847, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q8.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 2.1160532727573013e-19, 8.098328256428984e-19, 1.6912435259076145e-18, 2.7034335106292763e-18, 3.671385935992177e-18, 4.427733034079348e-18] + [4.8416955204338425e-18] * 2 + [4.427733034079348e-18, 3.671385935992179e-18, 2.7034335106292786e-18, 1.691243525907615e-18, 8.098328256428979e-19, 2.1160532727572984e-19, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q8.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0034557772481511894, -0.013225573711648706, -0.027620102826139324, -0.04415041973753595, -0.05995828247864346, -0.07231036797159979] + [-0.0790708884194989] * 2 + [-0.07231036797159979, -0.05995828247864349, -0.04415041973753599, -0.027620102826139335, -0.013225573711648697, -0.0034557772481511847, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q8.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.31622776601683794,
        },
        "q8.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q8.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.15811388300841897,
        },
        "q8.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q8.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "q8.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q8.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "q8.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q8.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "q8.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q8.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "q8.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q8.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.95,
        },
        "q8.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q8.z.const.wf": {
            "type": "constant",
            "sample": 1.25,
        },
        "q8.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.01820124141779137,
        },
        "q8.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q8.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "q8.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q9.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.00044756053064166733, 0.0017857940649379934, 0.004001400636798037, 0.007072360552706858, 0.010968153233332464, 0.015650060540755153, 0.02107155157671702, 0.027178745127593742, 0.03391094516010276, 0.04120124404575092, 0.0489771875189059, 0.057161494759839615, 0.06567282644623451, 0.07442659313991205, 0.08333579597467274, 0.09231189129011431, 0.10126567061830875, 0.11010814727763525, 0.118751440762401, 0.1271096501387889, 0.1350997077669304, 0.1426422048644302, 0.14966218070652176, 0.15608986761942747, 0.16186138436284517, 0.16691937101041399, 0.17121355901843288, 0.1747012708172317, 0.1773478439600325, 0.17912697561391838] + [0.18002098396920393] * 2 + [0.17912697561391838, 0.1773478439600325, 0.1747012708172317, 0.17121355901843288, 0.16691937101041399, 0.1618613843628452, 0.15608986761942753, 0.1496621807065218, 0.14264220486443027, 0.13509970776693034, 0.12710965013878892, 0.11875144076240106, 0.11010814727763527, 0.10126567061830881, 0.09231189129011433, 0.08333579597467274, 0.0744265931399121, 0.06567282644623453, 0.05716149475983966, 0.048977187518905924, 0.0412012440457509, 0.03391094516010278, 0.02717874512759373, 0.02107155157671704, 0.01565006054075519, 0.010968153233332485, 0.007072360552706858, 0.004001400636798047, 0.0017857940649379934, 0.0004475605306416773, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q9.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 64,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q9.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.00022378026532083366, 0.0008928970324689967, 0.0020007003183990183, 0.003536180276353429, 0.005484076616666232, 0.007825030270377576, 0.01053577578835851, 0.013589372563796871, 0.01695547258005138, 0.02060062202287546, 0.02448859375945295, 0.028580747379919808, 0.032836413223117256, 0.03721329656995603, 0.04166789798733637, 0.04615594564505716, 0.05063283530915438, 0.055054073638817626, 0.0593757203812005, 0.06355482506939444, 0.0675498538834652, 0.0713211024322151, 0.07483109035326088, 0.07804493380971374, 0.08093069218142258, 0.08345968550520699, 0.08560677950921644, 0.08735063540861585, 0.08867392198001625, 0.08956348780695919] + [0.09001049198460197] * 2 + [0.08956348780695919, 0.08867392198001625, 0.08735063540861585, 0.08560677950921644, 0.08345968550520699, 0.0809306921814226, 0.07804493380971377, 0.0748310903532609, 0.07132110243221514, 0.06754985388346517, 0.06355482506939446, 0.05937572038120053, 0.05505407363881763, 0.050632835309154405, 0.046155945645057164, 0.04166789798733637, 0.03721329656995605, 0.03283641322311726, 0.02858074737991983, 0.024488593759452962, 0.02060062202287545, 0.01695547258005139, 0.013589372563796866, 0.01053577578835852, 0.007825030270377595, 0.005484076616666242, 0.003536180276353429, 0.0020007003183990235, 0.0008928970324689967, 0.00022378026532083865, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q9.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 64,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q9.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.00022378026532083366, -0.0008928970324689967, -0.0020007003183990183, -0.003536180276353429, -0.005484076616666232, -0.007825030270377576, -0.01053577578835851, -0.013589372563796871, -0.01695547258005138, -0.02060062202287546, -0.02448859375945295, -0.028580747379919808, -0.032836413223117256, -0.03721329656995603, -0.04166789798733637, -0.04615594564505716, -0.05063283530915438, -0.055054073638817626, -0.0593757203812005, -0.06355482506939444, -0.0675498538834652, -0.0713211024322151, -0.07483109035326088, -0.07804493380971374, -0.08093069218142258, -0.08345968550520699, -0.08560677950921644, -0.08735063540861585, -0.08867392198001625, -0.08956348780695919] + [-0.09001049198460197] * 2 + [-0.08956348780695919, -0.08867392198001625, -0.08735063540861585, -0.08560677950921644, -0.08345968550520699, -0.0809306921814226, -0.07804493380971377, -0.0748310903532609, -0.07132110243221514, -0.06754985388346517, -0.06355482506939446, -0.05937572038120053, -0.05505407363881763, -0.050632835309154405, -0.046155945645057164, -0.04166789798733637, -0.03721329656995605, -0.03283641322311726, -0.02858074737991983, -0.024488593759452962, -0.02060062202287545, -0.01695547258005139, -0.013589372563796866, -0.01053577578835852, -0.007825030270377595, -0.005484076616666242, -0.003536180276353429, -0.0020007003183990235, -0.0008928970324689967, -0.00022378026532083865, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q9.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 2.740517856375044e-20, 1.0934834927813272e-19, 2.450151240980448e-19, 4.3305718566442297e-19, 6.716056874879148e-19, 9.582898273849047e-19, 1.290260409574743e-18, 1.6642181612674699e-18, 2.076446522319064e-18, 2.5228485820758904e-18, 2.9989877963133907e-18, 3.500132079605789e-18, 4.021300834917037e-18, 4.5573144530117824e-18, 5.102845789738992e-18, 5.652473109583847e-18, 6.200733969311099e-18, 6.742179506180068e-18, 7.271428591190546e-18, 7.783221309160386e-18, 8.272471234125706e-18, 8.734315980527272e-18, 9.164165527782732e-18, 9.557747837973297e-18, 9.911151313275889e-18, 1.022086367117965e-17, 1.0483806851127514e-17, 1.0697367605664886e-17, 1.0859423472066903e-17, 1.0968363866326557e-17] + [1.102310608986213e-17] * 2 + [1.0968363866326557e-17, 1.0859423472066903e-17, 1.0697367605664886e-17, 1.0483806851127514e-17, 1.022086367117965e-17, 9.91115131327589e-18, 9.5577478379733e-18, 9.164165527782734e-18, 8.734315980527278e-18, 8.272471234125703e-18, 7.783221309160389e-18, 7.271428591190549e-18, 6.742179506180069e-18, 6.200733969311103e-18, 5.6524731095838475e-18, 5.102845789738992e-18, 4.5573144530117855e-18, 4.021300834917038e-18, 3.500132079605792e-18, 2.998987796313392e-18, 2.5228485820758893e-18, 2.076446522319065e-18, 1.6642181612674693e-18, 1.2902604095747443e-18, 9.58289827384907e-19, 6.71605687487916e-19, 4.3305718566442297e-19, 2.4501512409804543e-19, 1.0934834927813272e-19, 2.740517856375105e-20, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q9.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 2.740517856375044e-20, 1.0934834927813272e-19, 2.450151240980448e-19, 4.3305718566442297e-19, 6.716056874879148e-19, 9.582898273849047e-19, 1.290260409574743e-18, 1.6642181612674699e-18, 2.076446522319064e-18, 2.5228485820758904e-18, 2.9989877963133907e-18, 3.500132079605789e-18, 4.021300834917037e-18, 4.5573144530117824e-18, 5.102845789738992e-18, 5.652473109583847e-18, 6.200733969311099e-18, 6.742179506180068e-18, 7.271428591190546e-18, 7.783221309160386e-18, 8.272471234125706e-18, 8.734315980527272e-18, 9.164165527782732e-18, 9.557747837973297e-18, 9.911151313275889e-18, 1.022086367117965e-17, 1.0483806851127514e-17, 1.0697367605664886e-17, 1.0859423472066903e-17, 1.0968363866326557e-17] + [1.102310608986213e-17] * 2 + [1.0968363866326557e-17, 1.0859423472066903e-17, 1.0697367605664886e-17, 1.0483806851127514e-17, 1.022086367117965e-17, 9.91115131327589e-18, 9.5577478379733e-18, 9.164165527782734e-18, 8.734315980527278e-18, 8.272471234125703e-18, 7.783221309160389e-18, 7.271428591190549e-18, 6.742179506180069e-18, 6.200733969311103e-18, 5.6524731095838475e-18, 5.102845789738992e-18, 4.5573144530117855e-18, 4.021300834917038e-18, 3.500132079605792e-18, 2.998987796313392e-18, 2.5228485820758893e-18, 2.076446522319065e-18, 1.6642181612674693e-18, 1.2902604095747443e-18, 9.58289827384907e-19, 6.71605687487916e-19, 4.3305718566442297e-19, 2.4501512409804543e-19, 1.0934834927813272e-19, 2.740517856375105e-20, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q9.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.00044756053064166733, 0.0017857940649379934, 0.004001400636798037, 0.007072360552706858, 0.010968153233332464, 0.015650060540755153, 0.02107155157671702, 0.027178745127593742, 0.03391094516010276, 0.04120124404575092, 0.0489771875189059, 0.057161494759839615, 0.06567282644623451, 0.07442659313991205, 0.08333579597467274, 0.09231189129011431, 0.10126567061830875, 0.11010814727763525, 0.118751440762401, 0.1271096501387889, 0.1350997077669304, 0.1426422048644302, 0.14966218070652176, 0.15608986761942747, 0.16186138436284517, 0.16691937101041399, 0.17121355901843288, 0.1747012708172317, 0.1773478439600325, 0.17912697561391838] + [0.18002098396920393] * 2 + [0.17912697561391838, 0.1773478439600325, 0.1747012708172317, 0.17121355901843288, 0.16691937101041399, 0.1618613843628452, 0.15608986761942753, 0.1496621807065218, 0.14264220486443027, 0.13509970776693034, 0.12710965013878892, 0.11875144076240106, 0.11010814727763527, 0.10126567061830881, 0.09231189129011433, 0.08333579597467274, 0.0744265931399121, 0.06567282644623453, 0.05716149475983966, 0.048977187518905924, 0.0412012440457509, 0.03391094516010278, 0.02717874512759373, 0.02107155157671704, 0.01565006054075519, 0.010968153233332485, 0.007072360552706858, 0.004001400636798047, 0.0017857940649379934, 0.0004475605306416773, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q9.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 1.370258928187522e-20, 5.467417463906636e-20, 1.225075620490224e-19, 2.1652859283221149e-19, 3.358028437439574e-19, 4.791449136924524e-19, 6.451302047873715e-19, 8.321090806337349e-19, 1.038223261159532e-18, 1.2614242910379452e-18, 1.4994938981566953e-18, 1.7500660398028946e-18, 2.0106504174585186e-18, 2.2786572265058912e-18, 2.551422894869496e-18, 2.8262365547919234e-18, 3.1003669846555495e-18, 3.371089753090034e-18, 3.635714295595273e-18, 3.891610654580193e-18, 4.136235617062853e-18, 4.367157990263636e-18, 4.582082763891366e-18, 4.778873918986649e-18, 4.955575656637944e-18, 5.110431835589825e-18, 5.241903425563757e-18, 5.348683802832443e-18, 5.4297117360334516e-18, 5.484181933163278e-18] + [5.511553044931065e-18] * 2 + [5.484181933163278e-18, 5.4297117360334516e-18, 5.348683802832443e-18, 5.241903425563757e-18, 5.110431835589825e-18, 4.955575656637945e-18, 4.77887391898665e-18, 4.582082763891367e-18, 4.367157990263639e-18, 4.1362356170628515e-18, 3.8916106545801945e-18, 3.6357142955952746e-18, 3.3710897530900345e-18, 3.1003669846555515e-18, 2.8262365547919238e-18, 2.551422894869496e-18, 2.2786572265058927e-18, 2.010650417458519e-18, 1.750066039802896e-18, 1.499493898156696e-18, 1.2614242910379446e-18, 1.0382232611595326e-18, 8.3210908063373465e-19, 6.451302047873722e-19, 4.791449136924535e-19, 3.35802843743958e-19, 2.1652859283221149e-19, 1.2250756204902272e-19, 5.467417463906636e-20, 1.3702589281875524e-20, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q9.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.00022378026532083366, 0.0008928970324689967, 0.0020007003183990183, 0.003536180276353429, 0.005484076616666232, 0.007825030270377576, 0.01053577578835851, 0.013589372563796871, 0.01695547258005138, 0.02060062202287546, 0.02448859375945295, 0.028580747379919808, 0.032836413223117256, 0.03721329656995603, 0.04166789798733637, 0.04615594564505716, 0.05063283530915438, 0.055054073638817626, 0.0593757203812005, 0.06355482506939444, 0.0675498538834652, 0.0713211024322151, 0.07483109035326088, 0.07804493380971374, 0.08093069218142258, 0.08345968550520699, 0.08560677950921644, 0.08735063540861585, 0.08867392198001625, 0.08956348780695919] + [0.09001049198460197] * 2 + [0.08956348780695919, 0.08867392198001625, 0.08735063540861585, 0.08560677950921644, 0.08345968550520699, 0.0809306921814226, 0.07804493380971377, 0.0748310903532609, 0.07132110243221514, 0.06754985388346517, 0.06355482506939446, 0.05937572038120053, 0.05505407363881763, 0.050632835309154405, 0.046155945645057164, 0.04166789798733637, 0.03721329656995605, 0.03283641322311726, 0.02858074737991983, 0.024488593759452962, 0.02060062202287545, 0.01695547258005139, 0.013589372563796866, 0.01053577578835852, 0.007825030270377595, 0.005484076616666242, 0.003536180276353429, 0.0020007003183990235, 0.0008928970324689967, 0.00022378026532083865, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q9.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 1.370258928187522e-20, 5.467417463906636e-20, 1.225075620490224e-19, 2.1652859283221149e-19, 3.358028437439574e-19, 4.791449136924524e-19, 6.451302047873715e-19, 8.321090806337349e-19, 1.038223261159532e-18, 1.2614242910379452e-18, 1.4994938981566953e-18, 1.7500660398028946e-18, 2.0106504174585186e-18, 2.2786572265058912e-18, 2.551422894869496e-18, 2.8262365547919234e-18, 3.1003669846555495e-18, 3.371089753090034e-18, 3.635714295595273e-18, 3.891610654580193e-18, 4.136235617062853e-18, 4.367157990263636e-18, 4.582082763891366e-18, 4.778873918986649e-18, 4.955575656637944e-18, 5.110431835589825e-18, 5.241903425563757e-18, 5.348683802832443e-18, 5.4297117360334516e-18, 5.484181933163278e-18] + [5.511553044931065e-18] * 2 + [5.484181933163278e-18, 5.4297117360334516e-18, 5.348683802832443e-18, 5.241903425563757e-18, 5.110431835589825e-18, 4.955575656637945e-18, 4.77887391898665e-18, 4.582082763891367e-18, 4.367157990263639e-18, 4.1362356170628515e-18, 3.8916106545801945e-18, 3.6357142955952746e-18, 3.3710897530900345e-18, 3.1003669846555515e-18, 2.8262365547919238e-18, 2.551422894869496e-18, 2.2786572265058927e-18, 2.010650417458519e-18, 1.750066039802896e-18, 1.499493898156696e-18, 1.2614242910379446e-18, 1.0382232611595326e-18, 8.3210908063373465e-19, 6.451302047873722e-19, 4.791449136924535e-19, 3.35802843743958e-19, 2.1652859283221149e-19, 1.2250756204902272e-19, 5.467417463906636e-20, 1.3702589281875524e-20, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q9.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.00022378026532083366, -0.0008928970324689967, -0.0020007003183990183, -0.003536180276353429, -0.005484076616666232, -0.007825030270377576, -0.01053577578835851, -0.013589372563796871, -0.01695547258005138, -0.02060062202287546, -0.02448859375945295, -0.028580747379919808, -0.032836413223117256, -0.03721329656995603, -0.04166789798733637, -0.04615594564505716, -0.05063283530915438, -0.055054073638817626, -0.0593757203812005, -0.06355482506939444, -0.0675498538834652, -0.0713211024322151, -0.07483109035326088, -0.07804493380971374, -0.08093069218142258, -0.08345968550520699, -0.08560677950921644, -0.08735063540861585, -0.08867392198001625, -0.08956348780695919] + [-0.09001049198460197] * 2 + [-0.08956348780695919, -0.08867392198001625, -0.08735063540861585, -0.08560677950921644, -0.08345968550520699, -0.0809306921814226, -0.07804493380971377, -0.0748310903532609, -0.07132110243221514, -0.06754985388346517, -0.06355482506939446, -0.05937572038120053, -0.05505407363881763, -0.050632835309154405, -0.046155945645057164, -0.04166789798733637, -0.03721329656995605, -0.03283641322311726, -0.02858074737991983, -0.024488593759452962, -0.02060062202287545, -0.01695547258005139, -0.013589372563796866, -0.01053577578835852, -0.007825030270377595, -0.005484076616666242, -0.003536180276353429, -0.0020007003183990235, -0.0008928970324689967, -0.00022378026532083865, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q9.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.31622776601683794,
        },
        "q9.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q9.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.15811388300841897,
        },
        "q9.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q9.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "q9.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q9.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "q9.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q9.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "q9.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q9.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "q9.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q9.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.04,
        },
        "q9.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q9.z.const.wf": {
            "type": "constant",
            "sample": 1.25,
        },
        "q9.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.024,
        },
        "q9.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q9.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "q9.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q10.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 8.065723531015662e-05, 0.000321827333399303, 0.0007211134374823331, 0.0012745472628699207, 0.0019766285355645946, 0.002820379656454826, 0.0037974150478298193, 0.004898024493018182, 0.0061112696408837115, 0.007425092716073247, 0.008826436354605317, 0.010301373373819657, 0.011835245186975544, 0.013412807486872479, 0.015018381750623935, 0.016636011059861698, 0.01824961868775953, 0.01984316787676718, 0.021400821219111736, 0.0229070980560713, 0.0243470283317164, 0.025706301372051328, 0.026971408110921746, 0.028129775349183275, 0.02916989071280334, 0.0300814170680065, 0.030855295256354026, 0.031483834128730494, 0.03196078698343906, 0.03228141364872942] + [0.032442527592754226] * 2 + [0.03228141364872942, 0.031960786983439064, 0.031483834128730494, 0.030855295256354026, 0.0300814170680065, 0.029169890712803345, 0.028129775349183282, 0.026971408110921757, 0.025706301372051342, 0.024347028331716387, 0.022907098056071305, 0.021400821219111747, 0.019843167876767185, 0.01824961868775954, 0.0166360110598617, 0.015018381750623935, 0.013412807486872486, 0.011835245186975546, 0.010301373373819664, 0.00882643635460532, 0.0074250927160732436, 0.006111269640883715, 0.00489802449301818, 0.0037974150478298227, 0.002820379656454833, 0.001976628535564598, 0.0012745472628699207, 0.0007211134374823349, 0.000321827333399303, 8.065723531015843e-05, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q10.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 64,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q10.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 4.254073592307837e-05, 0.0001697401547464678, 0.00038033409149912075, 0.0006722295690242197, 0.0010425255989264736, 0.0014875420128981146, 0.002002856037884925, 0.0025833462515918674, 0.00322324348147875, 0.003916188141388843, 0.004655293435973267, 0.005433213804757202, 0.006242217925620011, 0.00707426555214793, 0.007921087421215153, 0.008774267436635914, 0.009625326312109453, 0.010465805842176781, 0.011287352963666524, 0.012081802772189502, 0.01284125966862679, 0.013558175829140992, 0.01422542621884058, 0.014836379403577672, 0.015384963456119902, 0.015865726301691186, 0.016273889903140305, 0.01660539774722072, 0.016856955160041695, 0.017026062051017622] + [0.017111037759891465] * 2 + [0.017026062051017622, 0.016856955160041695, 0.01660539774722072, 0.016273889903140305, 0.015865726301691186, 0.015384963456119904, 0.014836379403577677, 0.014225426218840584, 0.013558175829141, 0.012841259668626783, 0.012081802772189504, 0.01128735296366653, 0.010465805842176783, 0.009625326312109458, 0.008774267436635915, 0.007921087421215153, 0.007074265552147934, 0.006242217925620012, 0.005433213804757205, 0.004655293435973269, 0.003916188141388841, 0.0032232434814787516, 0.0025833462515918665, 0.0020028560378849273, 0.0014875420128981182, 0.0010425255989264756, 0.0006722295690242197, 0.00038033409149912167, 0.0001697401547464678, 4.254073592307933e-05, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q10.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 64,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q10.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -4.254073592307837e-05, -0.0001697401547464678, -0.00038033409149912075, -0.0006722295690242197, -0.0010425255989264736, -0.0014875420128981146, -0.002002856037884925, -0.0025833462515918674, -0.00322324348147875, -0.003916188141388843, -0.004655293435973267, -0.005433213804757202, -0.006242217925620011, -0.00707426555214793, -0.007921087421215153, -0.008774267436635914, -0.009625326312109453, -0.010465805842176781, -0.011287352963666524, -0.012081802772189502, -0.01284125966862679, -0.013558175829140992, -0.01422542621884058, -0.014836379403577672, -0.015384963456119902, -0.015865726301691186, -0.016273889903140305, -0.01660539774722072, -0.016856955160041695, -0.017026062051017622] + [-0.017111037759891465] * 2 + [-0.017026062051017622, -0.016856955160041695, -0.01660539774722072, -0.016273889903140305, -0.015865726301691186, -0.015384963456119904, -0.014836379403577677, -0.014225426218840584, -0.013558175829141, -0.012841259668626783, -0.012081802772189504, -0.01128735296366653, -0.010465805842176783, -0.009625326312109458, -0.008774267436635915, -0.007921087421215153, -0.007074265552147934, -0.006242217925620012, -0.005433213804757205, -0.004655293435973269, -0.003916188141388841, -0.0032232434814787516, -0.0025833462515918665, -0.0020028560378849273, -0.0014875420128981182, -0.0010425255989264756, -0.0006722295690242197, -0.00038033409149912167, -0.0001697401547464678, -4.254073592307933e-05, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q10.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 5.2097376081570756e-21, 2.078717371970382e-20, 4.657749277610148e-20, 8.232437899977153e-20, 1.2767256377544832e-19, 1.8217135646928868e-19, 2.4527912359487237e-19, 3.1636867181012935e-19, 3.947334812465522e-19, 4.795947272210669e-19, 5.701090205456345e-19, 6.65376789507912e-19, 7.644512202190777e-19, 8.663476664756347e-19, 9.7005343561575e-19, 1.0745378531139023e-18, 1.178762505887364e-18, 1.2816915625119464e-18, 1.38230206778006e-18, 1.4795941092891492e-18, 1.57260075502038e-18, 1.6603976631434528e-18, 1.7421122685409953e-18, 1.8169324547527114e-18, 1.884114625153624e-18, 1.9429910931514083e-18, 1.9929767179557204e-18, 2.0335747199702525e-18, 2.064381618011552e-18, 2.085091239286295e-18] + [2.095497762274058e-18] * 2 + [2.085091239286295e-18, 2.064381618011552e-18, 2.0335747199702525e-18, 1.9929767179557204e-18, 1.9429910931514083e-18, 1.884114625153624e-18, 1.8169324547527117e-18, 1.7421122685409957e-18, 1.6603976631434536e-18, 1.5726007550203792e-18, 1.4795941092891494e-18, 1.3823020677800608e-18, 1.2816915625119466e-18, 1.1787625058873645e-18, 1.0745378531139025e-18, 9.7005343561575e-19, 8.663476664756351e-19, 7.644512202190778e-19, 6.653767895079124e-19, 5.701090205456348e-19, 4.795947272210667e-19, 3.9473348124655243e-19, 3.1636867181012925e-19, 2.452791235948726e-19, 1.8217135646928914e-19, 1.2767256377544856e-19, 8.232437899977153e-20, 4.657749277610159e-20, 2.078717371970382e-20, 5.209737608157192e-21, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q10.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 4.9388312525329095e-21, 1.9706240686279225e-20, 4.415546315174421e-20, 7.804351129178343e-20, 1.2103359045912505e-19, 1.726984459328857e-19, 2.3252460916793905e-19, 2.999175008760027e-19, 3.742073402217316e-19, 4.546558014055714e-19, 5.404633514772617e-19, 6.307771964535007e-19, 7.246997567676859e-19, 8.212975878189018e-19, 9.196106569637313e-19, 1.0186618847519797e-18, 1.1174668555812214e-18, 1.2150436012613255e-18, 1.3104223602554973e-18, 1.4026552156061138e-18, 1.4908255157593207e-18, 1.5740569846599936e-18, 1.6515224305768638e-18, 1.7224519671055709e-18, 1.786140664645636e-18, 1.841955556307536e-18, 1.8893419286220233e-18, 1.9278288345317998e-18, 1.9570337738749517e-18, 1.9766664948434083e-18] + [1.9865318786358074e-18] * 2 + [1.9766664948434083e-18, 1.957033773874952e-18, 1.9278288345317998e-18, 1.8893419286220233e-18, 1.841955556307536e-18, 1.7861406646456362e-18, 1.7224519671055713e-18, 1.6515224305768644e-18, 1.5740569846599945e-18, 1.4908255157593197e-18, 1.402655215606114e-18, 1.310422360255498e-18, 1.2150436012613257e-18, 1.117466855581222e-18, 1.01866188475198e-18, 9.196106569637313e-19, 8.212975878189022e-19, 7.24699756767686e-19, 6.307771964535011e-19, 5.404633514772619e-19, 4.546558014055712e-19, 3.742073402217318e-19, 2.999175008760026e-19, 2.325246091679393e-19, 1.7269844593288614e-19, 1.2103359045912527e-19, 7.804351129178343e-20, 4.415546315174432e-20, 1.9706240686279225e-20, 4.938831252533019e-21, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q10.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 8.065723531015662e-05, 0.000321827333399303, 0.0007211134374823331, 0.0012745472628699207, 0.0019766285355645946, 0.002820379656454826, 0.0037974150478298193, 0.004898024493018182, 0.0061112696408837115, 0.007425092716073247, 0.008826436354605317, 0.010301373373819657, 0.011835245186975544, 0.013412807486872479, 0.015018381750623935, 0.016636011059861698, 0.01824961868775953, 0.01984316787676718, 0.021400821219111736, 0.0229070980560713, 0.0243470283317164, 0.025706301372051328, 0.026971408110921746, 0.028129775349183275, 0.02916989071280334, 0.0300814170680065, 0.030855295256354026, 0.031483834128730494, 0.03196078698343906, 0.03228141364872942] + [0.032442527592754226] * 2 + [0.03228141364872942, 0.031960786983439064, 0.031483834128730494, 0.030855295256354026, 0.0300814170680065, 0.029169890712803345, 0.028129775349183282, 0.026971408110921757, 0.025706301372051342, 0.024347028331716387, 0.022907098056071305, 0.021400821219111747, 0.019843167876767185, 0.01824961868775954, 0.0166360110598617, 0.015018381750623935, 0.013412807486872486, 0.011835245186975546, 0.010301373373819664, 0.00882643635460532, 0.0074250927160732436, 0.006111269640883715, 0.00489802449301818, 0.0037974150478298227, 0.002820379656454833, 0.001976628535564598, 0.0012745472628699207, 0.0007211134374823349, 0.000321827333399303, 8.065723531015843e-05, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q10.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 2.6048688040785378e-21, 1.039358685985191e-20, 2.328874638805074e-20, 4.1162189499885765e-20, 6.383628188772416e-20, 9.108567823464434e-20, 1.2263956179743618e-19, 1.5818433590506467e-19, 1.973667406232761e-19, 2.3979736361053343e-19, 2.8505451027281725e-19, 3.32688394753956e-19, 3.8222561010953886e-19, 4.3317383323781734e-19, 4.85026717807875e-19, 5.372689265569512e-19, 5.89381252943682e-19, 6.408457812559732e-19, 6.9115103389003e-19, 7.397970546445746e-19, 7.8630037751019e-19, 8.301988315717264e-19, 8.710561342704977e-19, 9.084662273763557e-19, 9.42057312576812e-19, 9.714955465757042e-19, 9.964883589778602e-19, 1.0167873599851263e-18, 1.032190809005776e-18, 1.0425456196431475e-18] + [1.047748881137029e-18] * 2 + [1.0425456196431475e-18, 1.032190809005776e-18, 1.0167873599851263e-18, 9.964883589778602e-19, 9.714955465757042e-19, 9.42057312576812e-19, 9.084662273763559e-19, 8.710561342704979e-19, 8.301988315717268e-19, 7.863003775101896e-19, 7.397970546445747e-19, 6.911510338900304e-19, 6.408457812559733e-19, 5.893812529436823e-19, 5.3726892655695125e-19, 4.85026717807875e-19, 4.3317383323781753e-19, 3.822256101095389e-19, 3.326883947539562e-19, 2.850545102728174e-19, 2.3979736361053333e-19, 1.9736674062327622e-19, 1.5818433590506463e-19, 1.226395617974363e-19, 9.108567823464457e-20, 6.383628188772428e-20, 4.1162189499885765e-20, 2.3288746388050794e-20, 1.039358685985191e-20, 2.604868804078596e-21, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q10.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 4.254073592307837e-05, 0.0001697401547464678, 0.00038033409149912075, 0.0006722295690242197, 0.0010425255989264736, 0.0014875420128981146, 0.002002856037884925, 0.0025833462515918674, 0.00322324348147875, 0.003916188141388843, 0.004655293435973267, 0.005433213804757202, 0.006242217925620011, 0.00707426555214793, 0.007921087421215153, 0.008774267436635914, 0.009625326312109453, 0.010465805842176781, 0.011287352963666524, 0.012081802772189502, 0.01284125966862679, 0.013558175829140992, 0.01422542621884058, 0.014836379403577672, 0.015384963456119902, 0.015865726301691186, 0.016273889903140305, 0.01660539774722072, 0.016856955160041695, 0.017026062051017622] + [0.017111037759891465] * 2 + [0.017026062051017622, 0.016856955160041695, 0.01660539774722072, 0.016273889903140305, 0.015865726301691186, 0.015384963456119904, 0.014836379403577677, 0.014225426218840584, 0.013558175829141, 0.012841259668626783, 0.012081802772189504, 0.01128735296366653, 0.010465805842176783, 0.009625326312109458, 0.008774267436635915, 0.007921087421215153, 0.007074265552147934, 0.006242217925620012, 0.005433213804757205, 0.004655293435973269, 0.003916188141388841, 0.0032232434814787516, 0.0025833462515918665, 0.0020028560378849273, 0.0014875420128981182, 0.0010425255989264756, 0.0006722295690242197, 0.00038033409149912167, 0.0001697401547464678, 4.254073592307933e-05, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q10.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 2.6048688040785378e-21, 1.039358685985191e-20, 2.328874638805074e-20, 4.1162189499885765e-20, 6.383628188772416e-20, 9.108567823464434e-20, 1.2263956179743618e-19, 1.5818433590506467e-19, 1.973667406232761e-19, 2.3979736361053343e-19, 2.8505451027281725e-19, 3.32688394753956e-19, 3.8222561010953886e-19, 4.3317383323781734e-19, 4.85026717807875e-19, 5.372689265569512e-19, 5.89381252943682e-19, 6.408457812559732e-19, 6.9115103389003e-19, 7.397970546445746e-19, 7.8630037751019e-19, 8.301988315717264e-19, 8.710561342704977e-19, 9.084662273763557e-19, 9.42057312576812e-19, 9.714955465757042e-19, 9.964883589778602e-19, 1.0167873599851263e-18, 1.032190809005776e-18, 1.0425456196431475e-18] + [1.047748881137029e-18] * 2 + [1.0425456196431475e-18, 1.032190809005776e-18, 1.0167873599851263e-18, 9.964883589778602e-19, 9.714955465757042e-19, 9.42057312576812e-19, 9.084662273763559e-19, 8.710561342704979e-19, 8.301988315717268e-19, 7.863003775101896e-19, 7.397970546445747e-19, 6.911510338900304e-19, 6.408457812559733e-19, 5.893812529436823e-19, 5.3726892655695125e-19, 4.85026717807875e-19, 4.3317383323781753e-19, 3.822256101095389e-19, 3.326883947539562e-19, 2.850545102728174e-19, 2.3979736361053333e-19, 1.9736674062327622e-19, 1.5818433590506463e-19, 1.226395617974363e-19, 9.108567823464457e-20, 6.383628188772428e-20, 4.1162189499885765e-20, 2.3288746388050794e-20, 1.039358685985191e-20, 2.604868804078596e-21, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q10.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -4.254073592307837e-05, -0.0001697401547464678, -0.00038033409149912075, -0.0006722295690242197, -0.0010425255989264736, -0.0014875420128981146, -0.002002856037884925, -0.0025833462515918674, -0.00322324348147875, -0.003916188141388843, -0.004655293435973267, -0.005433213804757202, -0.006242217925620011, -0.00707426555214793, -0.007921087421215153, -0.008774267436635914, -0.009625326312109453, -0.010465805842176781, -0.011287352963666524, -0.012081802772189502, -0.01284125966862679, -0.013558175829140992, -0.01422542621884058, -0.014836379403577672, -0.015384963456119902, -0.015865726301691186, -0.016273889903140305, -0.01660539774722072, -0.016856955160041695, -0.017026062051017622] + [-0.017111037759891465] * 2 + [-0.017026062051017622, -0.016856955160041695, -0.01660539774722072, -0.016273889903140305, -0.015865726301691186, -0.015384963456119904, -0.014836379403577677, -0.014225426218840584, -0.013558175829141, -0.012841259668626783, -0.012081802772189504, -0.01128735296366653, -0.010465805842176783, -0.009625326312109458, -0.008774267436635915, -0.007921087421215153, -0.007074265552147934, -0.006242217925620012, -0.005433213804757205, -0.004655293435973269, -0.003916188141388841, -0.0032232434814787516, -0.0025833462515918665, -0.0020028560378849273, -0.0014875420128981182, -0.0010425255989264756, -0.0006722295690242197, -0.00038033409149912167, -0.0001697401547464678, -4.254073592307933e-05, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q10.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.31622776601683794,
        },
        "q10.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q10.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.15811388300841897,
        },
        "q10.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q10.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "q10.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q10.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "q10.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q10.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "q10.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q10.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "q10.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q10.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.5,
        },
        "q10.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q10.z.const.wf": {
            "type": "constant",
            "sample": 1.25,
        },
        "q10.z.SWAP_unipolar.wf": {
            "type": "constant",
            "sample": 0.0495,
        },
        "q10.z.SWAP_flattop.wf": {
            "type": "arbitrary",
            "samples": [0.0, 8.025662367830151e-05, 0.0003205059997116571, 0.0007191900187052127, 0.0012737230618081494, 0.0019805087694441006, 0.0028349633650830547, 0.003831545382791081, 0.004963791605763346, 0.006224358982765249, 0.007605072250637413, 0.009096976954014412, 0.010690397518403894, 0.012374999999999997, 0.014139859105274408, 0.015973529045697245, 0.017864117768067796, 0.0197993640790429, 0.021766717163680754, 0.023753417982291976, 0.025746582017708023, 0.027733282836319244, 0.0297006359209571, 0.0316358822319322, 0.03352647095430276, 0.0353601408947256, 0.037125, 0.03880960248159611, 0.04040302304598559, 0.04189492774936259, 0.04327564101723475, 0.04453620839423665, 0.04566845461720892, 0.046665036634916945, 0.047519491230555905, 0.04822627693819185, 0.04878080998129479, 0.04917949400028834, 0.0494197433763217] + [0.0495] * 51 + [0.0494197433763217, 0.04917949400028834, 0.04878080998129479, 0.04822627693819185, 0.047519491230555905, 0.046665036634916945, 0.04566845461720893, 0.04453620839423666, 0.04327564101723475, 0.04189492774936259, 0.04040302304598559, 0.03880960248159611, 0.037125000000000005, 0.03536014089472559, 0.03352647095430276, 0.03163588223193221, 0.029700635920957102, 0.027733282836319244, 0.025746582017708027, 0.023753417982291983, 0.021766717163680758, 0.019799364079042904, 0.0178641177680678, 0.01597352904569724, 0.014139859105274405, 0.012375000000000006, 0.010690397518403897, 0.009096976954014416, 0.007605072250637415, 0.006224358982765246, 0.004963791605763351, 0.003831545382791084, 0.0028349633650830578, 0.0019805087694441006, 0.0012737230618081494, 0.0007191900187052127, 0.0003205059997116571, 8.025662367830424e-05],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q10.z.SWAP_bipolar.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.00018045536607316418, 0.0007191900187052127, 0.0016083479935359829, 0.0028349633650830547, 0.0043811493191320055, 0.006224358982765249, 0.008337714208540317, 0.010690397518403894, 0.01324810149191673, 0.015973529045697245, 0.018826937308882945, 0.021766717163680754, 0.024749999999999998, 0.027733282836319244, 0.03067306269111705, 0.03352647095430276, 0.036251898508083275, 0.03880960248159611, 0.041162285791459675, 0.04327564101723475, 0.045118850680867996, 0.046665036634916945, 0.04789165200646402, 0.04878080998129479, 0.049319544633926835] + [0.0495] * 26 + [0.049139089267853674, 0.04806161996258958, 0.04628330401292804, 0.043830073269833895, 0.04073770136173599, 0.03705128203446951, 0.03282457158291937, 0.028119204963192215, 0.023003797016166547, 0.017552941908605513, 0.011846125382234113, 0.005966565672638489, 3.0310008278896994e-18, -0.005966565672638483, -0.011846125382234096, -0.017552941908605516, -0.023003797016166543, -0.02811920496319221, -0.032824571582919355, -0.03705128203446951, -0.04073770136173599, -0.04383007326983389, -0.04628330401292803, -0.04806161996258958, -0.049139089267853674] + [-0.0495] * 26 + [-0.049319544633926835, -0.04878080998129479, -0.047891652006464024, -0.046665036634916945, -0.045118850680867996, -0.04327564101723475, -0.04116228579145968, -0.03880960248159611, -0.036251898508083275, -0.03352647095430276, -0.030673062691117057, -0.027733282836319244, -0.02475, -0.021766717163680758, -0.018826937308882952, -0.01597352904569724, -0.01324810149191673, -0.010690397518403897, -0.008337714208540325, -0.006224358982765246, -0.0043811493191320055, -0.0028349633650830578, -0.0016083479935359857, -0.0007191900187052127, -0.00018045536607316418],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q10.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.026,
        },
        "q10.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q10.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "q10.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "coupler_q6_q7.const.wf": {
            "type": "constant",
            "sample": 1.25,
        },
        "coupler_q7_q8.const.wf": {
            "type": "constant",
            "sample": 1.25,
        },
        "coupler_q8_q9.const.wf": {
            "type": "constant",
            "sample": 1.25,
        },
        "coupler_q9_q10.const.wf": {
            "type": "constant",
            "sample": 1.25,
        },
        "q10.z.SWAP_unipolar.flux_pulse_control_q9_q10.wf": {
            "type": "constant",
            "sample": 0.0495,
        },
        "coupler_q9_q10.SWAP_unipolar.coupler_pulse_control_q9_q10.wf": {
            "type": "constant",
            "sample": -0.12499999999999994,
        },
        "q10.z.SWAP_flattop.flux_pulse_control_q9_q10.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.0012113512216949502, 0.00472682938922005, 0.01020231500576129, 0.01710182938922005, 0.024749999999999998, 0.03239817061077995, 0.039297684994238705, 0.04477317061077995, 0.048288648778305056] + [0.0495] * 109 + [0.048288648778305056, 0.04477317061077995, 0.03929768499423871, 0.03239817061077995, 0.02475, 0.017101829389220054, 0.010202315005761292, 0.004726829389220054, 0.0012113512216949502],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "coupler_q9_q10.SWAP_flattop.coupler_pulse_control_q9_q10.wf": {
            "type": "constant",
            "sample": -0.12499999999999994,
        },
        "q10.z.SWAP_bipolar.flux_pulse_control_q9_q10.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.0024510205194151263, 0.009318627403996345, 0.01924260688458122, 0.030257393115418783, 0.040181372596003656, 0.047048979480584875] + [0.0495] * 55 + [0.04286825748732972, 0.024750000000000008, 3.0310008278896994e-18, -0.02474999999999999, -0.04286825748732972] + [-0.0495] * 55 + [-0.047048979480584875, -0.040181372596003656, -0.030257393115418783, -0.01924260688458122, -0.009318627403996346, -0.002451020519415129],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "coupler_q9_q10.SWAP_bipolar.coupler_pulse_control_q9_q10.wf": {
            "type": "constant",
            "sample": -0.12499999999999994,
        },
        "coupler_q5_q6.const.wf": {
            "type": "constant",
            "sample": 1.25,
        },
    },
    "digital_waveforms": {
        "ON": {
            "samples": [(1, 0)],
        },
    },
    "integration_weights": {
        "q5.resonator.readout.iw1": {
            "cosine": [(1.0, 1500)],
            "sine": [(-0.0, 1500)],
        },
        "q5.resonator.readout.iw2": {
            "cosine": [(0.0, 1500)],
            "sine": [(1.0, 1500)],
        },
        "q5.resonator.readout.iw3": {
            "cosine": [(-0.0, 1500)],
            "sine": [(-1.0, 1500)],
        },
        "q6.resonator.readout.iw1": {
            "cosine": [(0.9263896341509306, 3000)],
            "sine": [(-0.3765663895486492, 3000)],
        },
        "q6.resonator.readout.iw2": {
            "cosine": [(0.3765663895486492, 3000)],
            "sine": [(0.9263896341509306, 3000)],
        },
        "q6.resonator.readout.iw3": {
            "cosine": [(-0.3765663895486492, 3000)],
            "sine": [(-0.9263896341509306, 3000)],
        },
        "q7.resonator.readout.iw1": {
            "cosine": [(-0.9804339027423286, 2000)],
            "sine": [(0.19684857721976617, 2000)],
        },
        "q7.resonator.readout.iw2": {
            "cosine": [(-0.19684857721976617, 2000)],
            "sine": [(-0.9804339027423286, 2000)],
        },
        "q7.resonator.readout.iw3": {
            "cosine": [(0.19684857721976617, 2000)],
            "sine": [(0.9804339027423286, 2000)],
        },
        "q8.resonator.readout.iw1": {
            "cosine": [(-0.5569761778677528, 2000)],
            "sine": [(0.83052846867993, 2000)],
        },
        "q8.resonator.readout.iw2": {
            "cosine": [(-0.83052846867993, 2000)],
            "sine": [(-0.5569761778677528, 2000)],
        },
        "q8.resonator.readout.iw3": {
            "cosine": [(0.83052846867993, 2000)],
            "sine": [(0.5569761778677528, 2000)],
        },
        "q9.resonator.readout.iw1": {
            "cosine": [(0.03478521455234504, 2000)],
            "sine": [(-0.9993948112975909, 2000)],
        },
        "q9.resonator.readout.iw2": {
            "cosine": [(0.9993948112975909, 2000)],
            "sine": [(0.03478521455234504, 2000)],
        },
        "q9.resonator.readout.iw3": {
            "cosine": [(-0.9993948112975909, 2000)],
            "sine": [(-0.03478521455234504, 2000)],
        },
        "q10.resonator.readout.iw1": {
            "cosine": [(0.9835308516317905, 2000)],
            "sine": [(0.18074032170062312, 2000)],
        },
        "q10.resonator.readout.iw2": {
            "cosine": [(-0.18074032170062312, 2000)],
            "sine": [(0.9835308516317905, 2000)],
        },
        "q10.resonator.readout.iw3": {
            "cosine": [(0.18074032170062312, 2000)],
            "sine": [(-0.9835308516317905, 2000)],
        },
    },
    "mixers": {},
}

