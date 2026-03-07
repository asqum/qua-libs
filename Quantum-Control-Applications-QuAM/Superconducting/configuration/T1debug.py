
# Single QUA script generated at 2026-01-20 12:59:37.874269
# QUA library version: 1.2.4a1


from qm import CompilerOptionArguments
from qm.qua import *

with program() as prog:
    v1 = declare(int, )
    v2 = declare(fixed, )
    v3 = declare(fixed, )
    v4 = declare(int, )
    set_dc_offset("coupler_q6_q7", "single", -0.2)
    set_dc_offset("coupler_q7_q8", "single", -0.2)
    set_dc_offset("coupler_q8_q9", "single", -0.2)
    set_dc_offset("coupler_q9_q10", "single", -0.05)
    set_dc_offset("coupler_q5_q6", "single", -0.1)
    set_dc_offset("q5.z", "single", 0.0)
    set_dc_offset("q6.z", "single", 0.11609780464027812)
    set_dc_offset("q7.z", "single", 0.10446088339413744)
    set_dc_offset("q8.z", "single", 0.13498012411922578)
    set_dc_offset("q9.z", "single", 0.07803690434214312)
    set_dc_offset("q10.z", "single", 0.10859151550357235)
    set_dc_offset("coupler_q6_q7", "single", -0.2)
    set_dc_offset("coupler_q7_q8", "single", -0.2)
    set_dc_offset("coupler_q8_q9", "single", -0.2)
    set_dc_offset("coupler_q9_q10", "single", -0.05)
    set_dc_offset("coupler_q5_q6", "single", -0.1)
    set_dc_offset("q9.z", "single", 0.07803690434214312)
    wait(24, "q9.z")
    align("q9.xy", "q9.resonator", "q9.z")
    wait(24, "q9.z")
    align("q9.xy", "q9.resonator", "q9.z")
    with for_(v1,0,(v1<900),(v1+1)):
        r1 = declare_stream()
        save(v1, r1)
        with for_(v4,4,(v4<=22279),(v4+225)):
            wait(8124, "q9.resonator")
            align("q9.xy", "q9.resonator", "q9.z")
            play("x180", "q9.xy")
            align("q9.xy", "q9.resonator", "q9.z")
            wait(v4, "q9.xy", "q9.z", "q9.resonator")
            measure("readout", "q9.resonator", dual_demod.full("iw1", "iw2", v2), dual_demod.full("iw3", "iw1", v3))
            r2 = declare_stream()
            save(v2, r2)
            r3 = declare_stream()
            save(v3, r3)
    with stream_processing():
        r1.save("n")
        r2.buffer(100).average().save("I1")
        r3.buffer(100).average().save("Q1")

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
                            "full_scale_power_dbm": 7,
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
                            "full_scale_power_dbm": 7,
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
            "intermediate_frequency": -260000000,
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
            "intermediate_frequency": -75341688.0,
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
            "intermediate_frequency": -203000000.0,
            "core": "f",
            "MWInput": {
                "port": ('con1', 7, 6),
                "upconverter": 1,
            },
        },
        "q10.z": {
            "operations": {
                "const": "q10.z.const.pulse",
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
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q9.xy.x180_DragCosine.wf.I",
                "Q": "q9.xy.x180_DragCosine.wf.Q",
            },
        },
        "q9.xy.x90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q9.xy.x90_DragCosine.wf.I",
                "Q": "q9.xy.x90_DragCosine.wf.Q",
            },
        },
        "q9.xy.-x90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q9.xy.-x90_DragCosine.wf.I",
                "Q": "q9.xy.-x90_DragCosine.wf.Q",
            },
        },
        "q9.xy.y180_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q9.xy.y180_DragCosine.wf.I",
                "Q": "q9.xy.y180_DragCosine.wf.Q",
            },
        },
        "q9.xy.y90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q9.xy.y90_DragCosine.wf.I",
                "Q": "q9.xy.y90_DragCosine.wf.Q",
            },
        },
        "q9.xy.-y90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
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
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q10.xy.x180_DragCosine.wf.I",
                "Q": "q10.xy.x180_DragCosine.wf.Q",
            },
        },
        "q10.xy.x90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q10.xy.x90_DragCosine.wf.I",
                "Q": "q10.xy.x90_DragCosine.wf.Q",
            },
        },
        "q10.xy.-x90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q10.xy.-x90_DragCosine.wf.I",
                "Q": "q10.xy.-x90_DragCosine.wf.Q",
            },
        },
        "q10.xy.y180_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q10.xy.y180_DragCosine.wf.I",
                "Q": "q10.xy.y180_DragCosine.wf.Q",
            },
        },
        "q10.xy.y90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q10.xy.y90_DragCosine.wf.I",
                "Q": "q10.xy.y90_DragCosine.wf.Q",
            },
        },
        "q10.xy.-y90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
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
            "samples": [0.0, 0.025050092881180982, 0.0958689828926182, 0.20021144058191612, 0.3200357070931256, 0.4346230781771715, 0.5241603563800364] + [0.5731657328798088] * 2 + [0.5241603563800364, 0.43462307817717166, 0.3200357070931259, 0.2002114405819162, 0.09586898289261814, 0.02505009288118095, 0.0],
        },
        "q8.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
        },
        "q8.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.012525046440590491, 0.0479344914463091, 0.10010572029095806, 0.1600178535465628, 0.21731153908858575, 0.2620801781900182] + [0.2865828664399044] * 2 + [0.2620801781900182, 0.21731153908858583, 0.16001785354656295, 0.1001057202909581, 0.04793449144630907, 0.012525046440590475, 0.0],
        },
        "q8.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
        },
        "q8.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.012525046440590491, -0.0479344914463091, -0.10010572029095806, -0.1600178535465628, -0.21731153908858575, -0.2620801781900182] + [-0.2865828664399044] * 2 + [-0.2620801781900182, -0.21731153908858583, -0.16001785354656295, -0.1001057202909581, -0.04793449144630907, -0.012525046440590475, 0.0],
        },
        "q8.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 1.5338758032641095e-18, 5.8702821518478624e-18, 1.2259414993066204e-17, 1.959653521522281e-17, 2.6612988076262146e-17, 3.2095565134037375e-17] + [3.509627900761023e-17] * 2 + [3.2095565134037375e-17, 2.6612988076262156e-17, 1.9596535215222827e-17, 1.2259414993066208e-17, 5.8702821518478586e-18, 1.5338758032641076e-18, 0.0],
        },
        "q8.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 1.5338758032641095e-18, 5.8702821518478624e-18, 1.2259414993066204e-17, 1.959653521522281e-17, 2.6612988076262146e-17, 3.2095565134037375e-17] + [3.509627900761023e-17] * 2 + [3.2095565134037375e-17, 2.6612988076262156e-17, 1.9596535215222827e-17, 1.2259414993066208e-17, 5.8702821518478586e-18, 1.5338758032641076e-18, 0.0],
        },
        "q8.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.025050092881180982, 0.0958689828926182, 0.20021144058191612, 0.3200357070931256, 0.4346230781771715, 0.5241603563800364] + [0.5731657328798088] * 2 + [0.5241603563800364, 0.43462307817717166, 0.3200357070931259, 0.2002114405819162, 0.09586898289261814, 0.02505009288118095, 0.0],
        },
        "q8.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 7.669379016320547e-19, 2.9351410759239312e-18, 6.129707496533102e-18, 9.798267607611404e-18, 1.3306494038131073e-17, 1.6047782567018688e-17] + [1.7548139503805115e-17] * 2 + [1.6047782567018688e-17, 1.3306494038131078e-17, 9.798267607611414e-18, 6.129707496533104e-18, 2.9351410759239293e-18, 7.669379016320538e-19, 0.0],
        },
        "q8.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.012525046440590491, 0.0479344914463091, 0.10010572029095806, 0.1600178535465628, 0.21731153908858575, 0.2620801781900182] + [0.2865828664399044] * 2 + [0.2620801781900182, 0.21731153908858583, 0.16001785354656295, 0.1001057202909581, 0.04793449144630907, 0.012525046440590475, 0.0],
        },
        "q8.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 7.669379016320547e-19, 2.9351410759239312e-18, 6.129707496533102e-18, 9.798267607611404e-18, 1.3306494038131073e-17, 1.6047782567018688e-17] + [1.7548139503805115e-17] * 2 + [1.6047782567018688e-17, 1.3306494038131078e-17, 9.798267607611414e-18, 6.129707496533104e-18, 2.9351410759239293e-18, 7.669379016320538e-19, 0.0],
        },
        "q8.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.012525046440590491, -0.0479344914463091, -0.10010572029095806, -0.1600178535465628, -0.21731153908858575, -0.2620801781900182] + [-0.2865828664399044] * 2 + [-0.2620801781900182, -0.21731153908858583, -0.16001785354656295, -0.1001057202909581, -0.04793449144630907, -0.012525046440590475, 0.0],
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
            "samples": [0.0, 0.012246214874223275, 0.04686737769177039, 0.09787717487809729, 0.15645554909022866, 0.21247376725906397, 0.25624577055367603] + [0.2802029819482603] * 2 + [0.25624577055367603, 0.21247376725906406, 0.1564555490902288, 0.09787717487809731, 0.04686737769177036, 0.01224621487422326, 0.0],
        },
        "q9.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
        },
        "q9.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.006123107437111638, 0.023433688845885196, 0.04893858743904864, 0.07822777454511433, 0.10623688362953199, 0.12812288527683802] + [0.14010149097413016] * 2 + [0.12812288527683802, 0.10623688362953203, 0.0782277745451144, 0.04893858743904866, 0.02343368884588518, 0.00612310743711163, 0.0],
        },
        "q9.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
        },
        "q9.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.006123107437111638, -0.023433688845885196, -0.04893858743904864, -0.07822777454511433, -0.10623688362953199, -0.12812288527683802] + [-0.14010149097413016] * 2 + [-0.12812288527683802, -0.10623688362953203, -0.0782277745451144, -0.04893858743904866, -0.02343368884588518, -0.00612310743711163, 0.0],
        },
        "q9.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 7.49864392369412e-19, 2.869799203732834e-18, 5.993248446202379e-18, 9.580139370109506e-18, 1.301026594882962e-17, 1.5690528135180323e-17] + [1.715748424772403e-17] * 2 + [1.5690528135180323e-17, 1.3010265948829624e-17, 9.580139370109515e-18, 5.99324844620238e-18, 2.8697992037328317e-18, 7.498643923694111e-19, 0.0],
        },
        "q9.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 7.49864392369412e-19, 2.869799203732834e-18, 5.993248446202379e-18, 9.580139370109506e-18, 1.301026594882962e-17, 1.5690528135180323e-17] + [1.715748424772403e-17] * 2 + [1.5690528135180323e-17, 1.3010265948829624e-17, 9.580139370109515e-18, 5.99324844620238e-18, 2.8697992037328317e-18, 7.498643923694111e-19, 0.0],
        },
        "q9.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.012246214874223275, 0.04686737769177039, 0.09787717487809729, 0.15645554909022866, 0.21247376725906397, 0.25624577055367603] + [0.2802029819482603] * 2 + [0.25624577055367603, 0.21247376725906406, 0.1564555490902288, 0.09787717487809731, 0.04686737769177036, 0.01224621487422326, 0.0],
        },
        "q9.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 3.74932196184706e-19, 1.434899601866417e-18, 2.9966242231011894e-18, 4.790069685054753e-18, 6.50513297441481e-18, 7.845264067590162e-18] + [8.578742123862015e-18] * 2 + [7.845264067590162e-18, 6.505132974414812e-18, 4.7900696850547574e-18, 2.99662422310119e-18, 1.4348996018664159e-18, 3.7493219618470554e-19, 0.0],
        },
        "q9.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.006123107437111638, 0.023433688845885196, 0.04893858743904864, 0.07822777454511433, 0.10623688362953199, 0.12812288527683802] + [0.14010149097413016] * 2 + [0.12812288527683802, 0.10623688362953203, 0.0782277745451144, 0.04893858743904866, 0.02343368884588518, 0.00612310743711163, 0.0],
        },
        "q9.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 3.74932196184706e-19, 1.434899601866417e-18, 2.9966242231011894e-18, 4.790069685054753e-18, 6.50513297441481e-18, 7.845264067590162e-18] + [8.578742123862015e-18] * 2 + [7.845264067590162e-18, 6.505132974414812e-18, 4.7900696850547574e-18, 2.99662422310119e-18, 1.4348996018664159e-18, 3.7493219618470554e-19, 0.0],
        },
        "q9.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.006123107437111638, -0.023433688845885196, -0.04893858743904864, -0.07822777454511433, -0.10623688362953199, -0.12812288527683802] + [-0.14010149097413016] * 2 + [-0.12812288527683802, -0.10623688362953203, -0.0782277745451144, -0.04893858743904866, -0.02343368884588518, -0.00612310743711163, 0.0],
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
            "samples": [0.0, 0.006347364654110417, 0.024291941603748365, 0.05073095047295769, 0.08109284643739696, 0.11012778185564907, 0.1328153526197908] + [0.14523267163459214] * 2 + [0.1328153526197908, 0.11012778185564913, 0.08109284643739703, 0.05073095047295771, 0.02429194160374835, 0.006347364654110409, 0.0],
        },
        "q10.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
        },
        "q10.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0031736823270552086, 0.012145970801874182, 0.025365475236478845, 0.04054642321869848, 0.05506389092782454, 0.0664076763098954] + [0.07261633581729607] * 2 + [0.0664076763098954, 0.055063890927824564, 0.040546423218698516, 0.025365475236478856, 0.012145970801874175, 0.0031736823270552047, 0.0],
        },
        "q10.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
        },
        "q10.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0031736823270552086, -0.012145970801874182, -0.025365475236478845, -0.04054642321869848, -0.05506389092782454, -0.0664076763098954] + [-0.07261633581729607] * 2 + [-0.0664076763098954, -0.055063890927824564, -0.040546423218698516, -0.025365475236478856, -0.012145970801874175, -0.0031736823270552047, 0.0],
        },
        "q10.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 3.8866399033386845e-19, 1.4874524265052428e-18, 3.106374805720527e-18, 4.965504741165302e-18, 6.74338177733593e-18, 8.132594823172691e-18] + [8.892936322446093e-18] * 2 + [8.132594823172691e-18, 6.743381777335933e-18, 4.965504741165306e-18, 3.106374805720528e-18, 1.487452426505242e-18, 3.8866399033386797e-19, 0.0],
        },
        "q10.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 3.8866399033386845e-19, 1.4874524265052428e-18, 3.106374805720527e-18, 4.965504741165302e-18, 6.74338177733593e-18, 8.132594823172691e-18] + [8.892936322446093e-18] * 2 + [8.132594823172691e-18, 6.743381777335933e-18, 4.965504741165306e-18, 3.106374805720528e-18, 1.487452426505242e-18, 3.8866399033386797e-19, 0.0],
        },
        "q10.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.006347364654110417, 0.024291941603748365, 0.05073095047295769, 0.08109284643739696, 0.11012778185564907, 0.1328153526197908] + [0.14523267163459214] * 2 + [0.1328153526197908, 0.11012778185564913, 0.08109284643739703, 0.05073095047295771, 0.02429194160374835, 0.006347364654110409, 0.0],
        },
        "q10.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 1.9433199516693422e-19, 7.437262132526214e-19, 1.5531874028602635e-18, 2.482752370582651e-18, 3.371690888667965e-18, 4.066297411586346e-18] + [4.4464681612230465e-18] * 2 + [4.066297411586346e-18, 3.3716908886679666e-18, 2.482752370582653e-18, 1.553187402860264e-18, 7.43726213252621e-19, 1.9433199516693398e-19, 0.0],
        },
        "q10.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0031736823270552086, 0.012145970801874182, 0.025365475236478845, 0.04054642321869848, 0.05506389092782454, 0.0664076763098954] + [0.07261633581729607] * 2 + [0.0664076763098954, 0.055063890927824564, 0.040546423218698516, 0.025365475236478856, 0.012145970801874175, 0.0031736823270552047, 0.0],
        },
        "q10.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 1.9433199516693422e-19, 7.437262132526214e-19, 1.5531874028602635e-18, 2.482752370582651e-18, 3.371690888667965e-18, 4.066297411586346e-18] + [4.4464681612230465e-18] * 2 + [4.066297411586346e-18, 3.3716908886679666e-18, 2.482752370582653e-18, 1.553187402860264e-18, 7.43726213252621e-19, 1.9433199516693398e-19, 0.0],
        },
        "q10.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0031736823270552086, -0.012145970801874182, -0.025365475236478845, -0.04054642321869848, -0.05506389092782454, -0.0664076763098954] + [-0.07261633581729607] * 2 + [-0.0664076763098954, -0.055063890927824564, -0.040546423218698516, -0.025365475236478856, -0.012145970801874175, -0.0031736823270552047, 0.0],
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
            "cosine": [(0.14654841787421538, 2000)],
            "sine": [(0.9892034983857287, 2000)],
        },
        "q8.resonator.readout.iw2": {
            "cosine": [(-0.9892034983857287, 2000)],
            "sine": [(0.14654841787421538, 2000)],
        },
        "q8.resonator.readout.iw3": {
            "cosine": [(0.9892034983857287, 2000)],
            "sine": [(-0.14654841787421538, 2000)],
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
                            "full_scale_power_dbm": 7,
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
            "intermediate_frequency": -260000000.0,
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
            "intermediate_frequency": -75341688.0,
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
            "intermediate_frequency": -203000000.0,
        },
        "q10.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "q10.z.const.pulse",
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
            "length": 16,
            "waveforms": {
                "I": "q9.xy.x180_DragCosine.wf.I",
                "Q": "q9.xy.x180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q9.xy.x90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q9.xy.x90_DragCosine.wf.I",
                "Q": "q9.xy.x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q9.xy.-x90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q9.xy.-x90_DragCosine.wf.I",
                "Q": "q9.xy.-x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q9.xy.y180_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q9.xy.y180_DragCosine.wf.I",
                "Q": "q9.xy.y180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q9.xy.y90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q9.xy.y90_DragCosine.wf.I",
                "Q": "q9.xy.y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q9.xy.-y90_DragCosine.pulse": {
            "length": 16,
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
            "length": 16,
            "waveforms": {
                "I": "q10.xy.x180_DragCosine.wf.I",
                "Q": "q10.xy.x180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q10.xy.x90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q10.xy.x90_DragCosine.wf.I",
                "Q": "q10.xy.x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q10.xy.-x90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q10.xy.-x90_DragCosine.wf.I",
                "Q": "q10.xy.-x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q10.xy.y180_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q10.xy.y180_DragCosine.wf.I",
                "Q": "q10.xy.y180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q10.xy.y90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q10.xy.y90_DragCosine.wf.I",
                "Q": "q10.xy.y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q10.xy.-y90_DragCosine.pulse": {
            "length": 16,
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
            "samples": [0.0, 0.025050092881180982, 0.0958689828926182, 0.20021144058191612, 0.3200357070931256, 0.4346230781771715, 0.5241603563800364] + [0.5731657328798088] * 2 + [0.5241603563800364, 0.43462307817717166, 0.3200357070931259, 0.2002114405819162, 0.09586898289261814, 0.02505009288118095, 0.0],
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
            "samples": [0.0, 0.012525046440590491, 0.0479344914463091, 0.10010572029095806, 0.1600178535465628, 0.21731153908858575, 0.2620801781900182] + [0.2865828664399044] * 2 + [0.2620801781900182, 0.21731153908858583, 0.16001785354656295, 0.1001057202909581, 0.04793449144630907, 0.012525046440590475, 0.0],
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
            "samples": [0.0, -0.012525046440590491, -0.0479344914463091, -0.10010572029095806, -0.1600178535465628, -0.21731153908858575, -0.2620801781900182] + [-0.2865828664399044] * 2 + [-0.2620801781900182, -0.21731153908858583, -0.16001785354656295, -0.1001057202909581, -0.04793449144630907, -0.012525046440590475, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q8.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 1.5338758032641095e-18, 5.8702821518478624e-18, 1.2259414993066204e-17, 1.959653521522281e-17, 2.6612988076262146e-17, 3.2095565134037375e-17] + [3.509627900761023e-17] * 2 + [3.2095565134037375e-17, 2.6612988076262156e-17, 1.9596535215222827e-17, 1.2259414993066208e-17, 5.8702821518478586e-18, 1.5338758032641076e-18, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q8.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 1.5338758032641095e-18, 5.8702821518478624e-18, 1.2259414993066204e-17, 1.959653521522281e-17, 2.6612988076262146e-17, 3.2095565134037375e-17] + [3.509627900761023e-17] * 2 + [3.2095565134037375e-17, 2.6612988076262156e-17, 1.9596535215222827e-17, 1.2259414993066208e-17, 5.8702821518478586e-18, 1.5338758032641076e-18, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q8.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.025050092881180982, 0.0958689828926182, 0.20021144058191612, 0.3200357070931256, 0.4346230781771715, 0.5241603563800364] + [0.5731657328798088] * 2 + [0.5241603563800364, 0.43462307817717166, 0.3200357070931259, 0.2002114405819162, 0.09586898289261814, 0.02505009288118095, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q8.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 7.669379016320547e-19, 2.9351410759239312e-18, 6.129707496533102e-18, 9.798267607611404e-18, 1.3306494038131073e-17, 1.6047782567018688e-17] + [1.7548139503805115e-17] * 2 + [1.6047782567018688e-17, 1.3306494038131078e-17, 9.798267607611414e-18, 6.129707496533104e-18, 2.9351410759239293e-18, 7.669379016320538e-19, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q8.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.012525046440590491, 0.0479344914463091, 0.10010572029095806, 0.1600178535465628, 0.21731153908858575, 0.2620801781900182] + [0.2865828664399044] * 2 + [0.2620801781900182, 0.21731153908858583, 0.16001785354656295, 0.1001057202909581, 0.04793449144630907, 0.012525046440590475, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q8.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 7.669379016320547e-19, 2.9351410759239312e-18, 6.129707496533102e-18, 9.798267607611404e-18, 1.3306494038131073e-17, 1.6047782567018688e-17] + [1.7548139503805115e-17] * 2 + [1.6047782567018688e-17, 1.3306494038131078e-17, 9.798267607611414e-18, 6.129707496533104e-18, 2.9351410759239293e-18, 7.669379016320538e-19, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q8.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.012525046440590491, -0.0479344914463091, -0.10010572029095806, -0.1600178535465628, -0.21731153908858575, -0.2620801781900182] + [-0.2865828664399044] * 2 + [-0.2620801781900182, -0.21731153908858583, -0.16001785354656295, -0.1001057202909581, -0.04793449144630907, -0.012525046440590475, 0.0],
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
            "samples": [0.0, 0.012246214874223275, 0.04686737769177039, 0.09787717487809729, 0.15645554909022866, 0.21247376725906397, 0.25624577055367603] + [0.2802029819482603] * 2 + [0.25624577055367603, 0.21247376725906406, 0.1564555490902288, 0.09787717487809731, 0.04686737769177036, 0.01224621487422326, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q9.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q9.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.006123107437111638, 0.023433688845885196, 0.04893858743904864, 0.07822777454511433, 0.10623688362953199, 0.12812288527683802] + [0.14010149097413016] * 2 + [0.12812288527683802, 0.10623688362953203, 0.0782277745451144, 0.04893858743904866, 0.02343368884588518, 0.00612310743711163, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q9.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q9.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.006123107437111638, -0.023433688845885196, -0.04893858743904864, -0.07822777454511433, -0.10623688362953199, -0.12812288527683802] + [-0.14010149097413016] * 2 + [-0.12812288527683802, -0.10623688362953203, -0.0782277745451144, -0.04893858743904866, -0.02343368884588518, -0.00612310743711163, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q9.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 7.49864392369412e-19, 2.869799203732834e-18, 5.993248446202379e-18, 9.580139370109506e-18, 1.301026594882962e-17, 1.5690528135180323e-17] + [1.715748424772403e-17] * 2 + [1.5690528135180323e-17, 1.3010265948829624e-17, 9.580139370109515e-18, 5.99324844620238e-18, 2.8697992037328317e-18, 7.498643923694111e-19, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q9.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 7.49864392369412e-19, 2.869799203732834e-18, 5.993248446202379e-18, 9.580139370109506e-18, 1.301026594882962e-17, 1.5690528135180323e-17] + [1.715748424772403e-17] * 2 + [1.5690528135180323e-17, 1.3010265948829624e-17, 9.580139370109515e-18, 5.99324844620238e-18, 2.8697992037328317e-18, 7.498643923694111e-19, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q9.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.012246214874223275, 0.04686737769177039, 0.09787717487809729, 0.15645554909022866, 0.21247376725906397, 0.25624577055367603] + [0.2802029819482603] * 2 + [0.25624577055367603, 0.21247376725906406, 0.1564555490902288, 0.09787717487809731, 0.04686737769177036, 0.01224621487422326, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q9.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 3.74932196184706e-19, 1.434899601866417e-18, 2.9966242231011894e-18, 4.790069685054753e-18, 6.50513297441481e-18, 7.845264067590162e-18] + [8.578742123862015e-18] * 2 + [7.845264067590162e-18, 6.505132974414812e-18, 4.7900696850547574e-18, 2.99662422310119e-18, 1.4348996018664159e-18, 3.7493219618470554e-19, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q9.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.006123107437111638, 0.023433688845885196, 0.04893858743904864, 0.07822777454511433, 0.10623688362953199, 0.12812288527683802] + [0.14010149097413016] * 2 + [0.12812288527683802, 0.10623688362953203, 0.0782277745451144, 0.04893858743904866, 0.02343368884588518, 0.00612310743711163, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q9.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 3.74932196184706e-19, 1.434899601866417e-18, 2.9966242231011894e-18, 4.790069685054753e-18, 6.50513297441481e-18, 7.845264067590162e-18] + [8.578742123862015e-18] * 2 + [7.845264067590162e-18, 6.505132974414812e-18, 4.7900696850547574e-18, 2.99662422310119e-18, 1.4348996018664159e-18, 3.7493219618470554e-19, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q9.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.006123107437111638, -0.023433688845885196, -0.04893858743904864, -0.07822777454511433, -0.10623688362953199, -0.12812288527683802] + [-0.14010149097413016] * 2 + [-0.12812288527683802, -0.10623688362953203, -0.0782277745451144, -0.04893858743904866, -0.02343368884588518, -0.00612310743711163, 0.0],
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
            "samples": [0.0, 0.006347364654110417, 0.024291941603748365, 0.05073095047295769, 0.08109284643739696, 0.11012778185564907, 0.1328153526197908] + [0.14523267163459214] * 2 + [0.1328153526197908, 0.11012778185564913, 0.08109284643739703, 0.05073095047295771, 0.02429194160374835, 0.006347364654110409, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q10.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q10.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0031736823270552086, 0.012145970801874182, 0.025365475236478845, 0.04054642321869848, 0.05506389092782454, 0.0664076763098954] + [0.07261633581729607] * 2 + [0.0664076763098954, 0.055063890927824564, 0.040546423218698516, 0.025365475236478856, 0.012145970801874175, 0.0031736823270552047, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q10.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q10.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0031736823270552086, -0.012145970801874182, -0.025365475236478845, -0.04054642321869848, -0.05506389092782454, -0.0664076763098954] + [-0.07261633581729607] * 2 + [-0.0664076763098954, -0.055063890927824564, -0.040546423218698516, -0.025365475236478856, -0.012145970801874175, -0.0031736823270552047, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q10.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 3.8866399033386845e-19, 1.4874524265052428e-18, 3.106374805720527e-18, 4.965504741165302e-18, 6.74338177733593e-18, 8.132594823172691e-18] + [8.892936322446093e-18] * 2 + [8.132594823172691e-18, 6.743381777335933e-18, 4.965504741165306e-18, 3.106374805720528e-18, 1.487452426505242e-18, 3.8866399033386797e-19, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q10.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 3.8866399033386845e-19, 1.4874524265052428e-18, 3.106374805720527e-18, 4.965504741165302e-18, 6.74338177733593e-18, 8.132594823172691e-18] + [8.892936322446093e-18] * 2 + [8.132594823172691e-18, 6.743381777335933e-18, 4.965504741165306e-18, 3.106374805720528e-18, 1.487452426505242e-18, 3.8866399033386797e-19, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q10.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.006347364654110417, 0.024291941603748365, 0.05073095047295769, 0.08109284643739696, 0.11012778185564907, 0.1328153526197908] + [0.14523267163459214] * 2 + [0.1328153526197908, 0.11012778185564913, 0.08109284643739703, 0.05073095047295771, 0.02429194160374835, 0.006347364654110409, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q10.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 1.9433199516693422e-19, 7.437262132526214e-19, 1.5531874028602635e-18, 2.482752370582651e-18, 3.371690888667965e-18, 4.066297411586346e-18] + [4.4464681612230465e-18] * 2 + [4.066297411586346e-18, 3.3716908886679666e-18, 2.482752370582653e-18, 1.553187402860264e-18, 7.43726213252621e-19, 1.9433199516693398e-19, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q10.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0031736823270552086, 0.012145970801874182, 0.025365475236478845, 0.04054642321869848, 0.05506389092782454, 0.0664076763098954] + [0.07261633581729607] * 2 + [0.0664076763098954, 0.055063890927824564, 0.040546423218698516, 0.025365475236478856, 0.012145970801874175, 0.0031736823270552047, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q10.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 1.9433199516693422e-19, 7.437262132526214e-19, 1.5531874028602635e-18, 2.482752370582651e-18, 3.371690888667965e-18, 4.066297411586346e-18] + [4.4464681612230465e-18] * 2 + [4.066297411586346e-18, 3.3716908886679666e-18, 2.482752370582653e-18, 1.553187402860264e-18, 7.43726213252621e-19, 1.9433199516693398e-19, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q10.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0031736823270552086, -0.012145970801874182, -0.025365475236478845, -0.04054642321869848, -0.05506389092782454, -0.0664076763098954] + [-0.07261633581729607] * 2 + [-0.0664076763098954, -0.055063890927824564, -0.040546423218698516, -0.025365475236478856, -0.012145970801874175, -0.0031736823270552047, 0.0],
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
            "cosine": [(0.14654841787421538, 2000)],
            "sine": [(0.9892034983857287, 2000)],
        },
        "q8.resonator.readout.iw2": {
            "cosine": [(-0.9892034983857287, 2000)],
            "sine": [(0.14654841787421538, 2000)],
        },
        "q8.resonator.readout.iw3": {
            "cosine": [(0.9892034983857287, 2000)],
            "sine": [(-0.14654841787421538, 2000)],
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

