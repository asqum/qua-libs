
# Single QUA script generated at 2025-08-14 04:33:35.374726
# QUA library version: 1.2.3a1

from qm import CompilerOptionArguments
from qm.qua import *

with program() as prog:
    v1 = declare(int, )
    v2 = declare(int, )
    v3 = declare(fixed, )
    a1 = declare(fixed, size=8)
    a2 = declare(fixed, size=8)
    a3 = declare(fixed, size=8)
    a4 = declare(fixed, size=8)
    a5 = declare(fixed, size=8)
    a6 = declare(fixed, size=8)
    a7 = declare(fixed, size=8)
    a8 = declare(fixed, size=8)
    a9 = declare(fixed, size=8)
    a10 = declare(fixed, size=8)
    a11 = declare(fixed, size=8)
    a12 = declare(fixed, size=8)
    a13 = declare(fixed, size=8)
    a14 = declare(fixed, size=8)
    a15 = declare(fixed, size=8)
    a16 = declare(fixed, size=8)
    a17 = declare(fixed, size=8)
    a18 = declare(fixed, size=8)
    a19 = declare(fixed, size=8)
    a20 = declare(fixed, size=8)
    a21 = declare(fixed, size=8)
    a22 = declare(fixed, size=8)
    a23 = declare(fixed, size=8)
    a24 = declare(fixed, size=8)
    v4 = declare(int, )
    v5 = declare(int, )
    set_dc_offset("q1.z", "single", 0.03459990521970473)
    set_dc_offset("q2.z", "single", 0.3452040552564276)
    set_dc_offset("c1.z", "single", 0.37153883357993694)
    set_dc_offset("coupler_q1_q2", "single", -0.018755978745310042)
    wait(24, "q1.z")
    align("q1.xy", "q1.resonator", "q1.z")
    with for_(v1,0,(v1<40.0),(v1+1)):
        r9 = declare_stream()
        save(v1, r9)
        with for_(v2,-3000000,(v2<=2800000),(v2+200000)):
            with for_(v3,0.2,(v3<2.0539285714285715),(v3+0.12785714285714284)):
                update_frequency("q1.resonator", (-231320485.0+v2), "Hz", False)
                update_frequency("q2.resonator", (-149374975.0+v2), "Hz", False)
                wait(31102, )
                align()
                measure("readout"*amp(v3), "q1.resonator", demod.accumulated("iw1", a1, 65, "out1"), demod.accumulated("iw2", a2, 65, "out2"), demod.accumulated("iw3", a3, 65, "out1"), demod.accumulated("iw1", a4, 65, "out2"))
                measure("readout"*amp(v3), "q2.resonator", demod.accumulated("iw1", a13, 65, "out1"), demod.accumulated("iw2", a14, 65, "out2"), demod.accumulated("iw3", a15, 65, "out1"), demod.accumulated("iw1", a16, 65, "out2"))
                align()
                wait(31102, )
                play("x180", "q1.xy")
                play("x180", "q2.xy")
                align()
                measure("readout"*amp(v3), "q1.resonator", demod.accumulated("iw1", a7, 65, "out1"), demod.accumulated("iw2", a8, 65, "out2"), demod.accumulated("iw3", a9, 65, "out1"), demod.accumulated("iw1", a10, 65, "out2"))
                measure("readout"*amp(v3), "q2.resonator", demod.accumulated("iw1", a19, 65, "out1"), demod.accumulated("iw2", a20, 65, "out2"), demod.accumulated("iw3", a21, 65, "out1"), demod.accumulated("iw1", a22, 65, "out2"))
                with for_(v4,0,(v4<8),(v4+1)):
                    assign(a5[v4], (a1[v4]+a2[v4]))
                    r1 = declare_stream()
                    save(a5[v4], r1)
                    assign(a6[v4], (a3[v4]+a4[v4]))
                    r2 = declare_stream()
                    save(a6[v4], r2)
                    assign(a11[v4], (a7[v4]+a8[v4]))
                    r3 = declare_stream()
                    save(a11[v4], r3)
                    assign(a12[v4], (a9[v4]+a10[v4]))
                    r4 = declare_stream()
                    save(a12[v4], r4)
                with for_(v5,0,(v5<8),(v5+1)):
                    assign(a17[v5], (a13[v5]+a14[v5]))
                    r5 = declare_stream()
                    save(a17[v5], r5)
                    assign(a18[v5], (a15[v5]+a16[v5]))
                    r6 = declare_stream()
                    save(a18[v5], r6)
                    assign(a23[v5], (a19[v5]+a20[v5]))
                    r7 = declare_stream()
                    save(a23[v5], r7)
                    assign(a24[v5], (a21[v5]+a22[v5]))
                    r8 = declare_stream()
                    save(a24[v5], r8)
    with stream_processing():
        r9.save("n")
        r1.buffer(8).buffer(15).buffer(30).buffer(40).save("I_g1")
        r2.buffer(8).buffer(15).buffer(30).buffer(40).save("Q_g1")
        r3.buffer(8).buffer(15).buffer(30).buffer(40).save("I_e1")
        r4.buffer(8).buffer(15).buffer(30).buffer(40).save("Q_e1")
        r5.buffer(8).buffer(15).buffer(30).buffer(40).save("I_g2")
        r6.buffer(8).buffer(15).buffer(30).buffer(40).save("Q_g2")
        r7.buffer(8).buffer(15).buffer(30).buffer(40).save("I_e2")
        r8.buffer(8).buffer(15).buffer(30).buffer(40).save("Q_e2")


config = {
    "version": 1,
    "controllers": {
        "con1": {
            "fems": {
                "1": {
                    "type": "LF",
                    "analog_outputs": {
                        "1": {
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "direct",
                            "offset": 0.0,
                        },
                        "2": {
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "direct",
                            "offset": 0.0,
                        },
                        "3": {
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "direct",
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
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 7,
                            "upconverter_frequency": 4400000000.0,
                        },
                        "3": {
                            "band": 1,
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 10,
                            "upconverter_frequency": 4700000000.0,
                        },
                    },
                    "analog_inputs": {
                        "1": {
                            "band": 2,
                            "downconverter_frequency": 6060000000,
                            "sampling_rate": 1000000000.0,
                            "shareable": False,
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
            },
            "intermediate_frequency": 277191481.2810402,
            "MWInput": {
                "port": ('con1', 6, 2),
                "upconverter": 1,
            },
        },
        "q1.z": {
            "operations": {
                "const": "q1.z.const.pulse",
            },
            "singleInput": {
                "port": ('con1', 1, 1),
            },
        },
        "q1.resonator": {
            "operations": {
                "readout": "q1.resonator.readout.pulse",
                "const": "q1.resonator.const.pulse",
            },
            "intermediate_frequency": -231320485.0,
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
            },
            "intermediate_frequency": 262899707.49565393,
            "MWInput": {
                "port": ('con1', 6, 3),
                "upconverter": 1,
            },
        },
        "q2.z": {
            "operations": {
                "const": "q2.z.const.pulse",
            },
            "singleInput": {
                "port": ('con1', 1, 2),
            },
        },
        "q2.resonator": {
            "operations": {
                "readout": "q2.resonator.readout.pulse",
                "const": "q2.resonator.const.pulse",
            },
            "intermediate_frequency": -149374975.0,
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
                "x180": "c1.xy.x180_DragCosine.pulse",
                "x90": "c1.xy.x90_DragCosine.pulse",
                "-x90": "c1.xy.-x90_DragCosine.pulse",
                "y180": "c1.xy.y180_DragCosine.pulse",
                "y90": "c1.xy.y90_DragCosine.pulse",
                "-y90": "c1.xy.-y90_DragCosine.pulse",
                "saturation": "c1.xy.saturation.pulse",
            },
            "intermediate_frequency": -5749984.452069147,
            "MWInput": {
                "port": ('con1', 6, 2),
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
            "intermediate_frequency": -65211388.0,
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
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.x180_DragCosine.wf.I",
                "Q": "q1.xy.x180_DragCosine.wf.Q",
            },
        },
        "q1.xy.x90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.x90_DragCosine.wf.I",
                "Q": "q1.xy.x90_DragCosine.wf.Q",
            },
        },
        "q1.xy.-x90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.-x90_DragCosine.wf.I",
                "Q": "q1.xy.-x90_DragCosine.wf.Q",
            },
        },
        "q1.xy.y180_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.y180_DragCosine.wf.I",
                "Q": "q1.xy.y180_DragCosine.wf.Q",
            },
        },
        "q1.xy.y90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.y90_DragCosine.wf.I",
                "Q": "q1.xy.y90_DragCosine.wf.Q",
            },
        },
        "q1.xy.-y90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
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
        "q1.z.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "q1.z.const.wf",
            },
        },
        "q1.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 2080,
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
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.x180_DragCosine.wf.I",
                "Q": "q2.xy.x180_DragCosine.wf.Q",
            },
        },
        "q2.xy.x90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.x90_DragCosine.wf.I",
                "Q": "q2.xy.x90_DragCosine.wf.Q",
            },
        },
        "q2.xy.-x90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.-x90_DragCosine.wf.I",
                "Q": "q2.xy.-x90_DragCosine.wf.Q",
            },
        },
        "q2.xy.y180_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.y180_DragCosine.wf.I",
                "Q": "q2.xy.y180_DragCosine.wf.Q",
            },
        },
        "q2.xy.y90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.y90_DragCosine.wf.I",
                "Q": "q2.xy.y90_DragCosine.wf.Q",
            },
        },
        "q2.xy.-y90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
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
        "q2.z.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "q2.z.const.wf",
            },
        },
        "q2.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 2080,
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
            "samples": [0.0, 0.010807759991626099, 0.041362280078535096, 0.08638040615859965, 0.1380780952557404, 0.18751634726630054, 0.22614683928523466] + [0.2472900082954664] * 2 + [0.22614683928523466, 0.18751634726630065, 0.13807809525574052, 0.08638040615859968, 0.04136228007853507, 0.010807759991626085, 0.0],
        },
        "q1.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
        },
        "q1.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0054038799958130495, 0.020681140039267548, 0.04319020307929983, 0.0690390476278702, 0.09375817363315027, 0.11307341964261733] + [0.1236450041477332] * 2 + [0.11307341964261733, 0.09375817363315032, 0.06903904762787026, 0.04319020307929984, 0.020681140039267534, 0.005403879995813043, 0.0],
        },
        "q1.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
        },
        "q1.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0054038799958130495, -0.020681140039267548, -0.04319020307929983, -0.0690390476278702, -0.09375817363315027, -0.11307341964261733] + [-0.1236450041477332] * 2 + [-0.11307341964261733, -0.09375817363315032, -0.06903904762787026, -0.04319020307929984, -0.020681140039267534, -0.005403879995813043, 0.0],
        },
        "q1.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 6.617844339848864e-19, 2.532709195180717e-18, 5.289274395558869e-18, 8.454844869365292e-18, 1.1482064723373924e-17, 1.3847500143397676e-17] + [1.5142145856008268e-17] * 2 + [1.3847500143397676e-17, 1.148206472337393e-17, 8.454844869365298e-18, 5.289274395558871e-18, 2.5327091951807154e-18, 6.617844339848855e-19, 0.0],
        },
        "q1.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 6.617844339848864e-19, 2.532709195180717e-18, 5.289274395558869e-18, 8.454844869365292e-18, 1.1482064723373924e-17, 1.3847500143397676e-17] + [1.5142145856008268e-17] * 2 + [1.3847500143397676e-17, 1.148206472337393e-17, 8.454844869365298e-18, 5.289274395558871e-18, 2.5327091951807154e-18, 6.617844339848855e-19, 0.0],
        },
        "q1.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.010807759991626099, 0.041362280078535096, 0.08638040615859965, 0.1380780952557404, 0.18751634726630054, 0.22614683928523466] + [0.2472900082954664] * 2 + [0.22614683928523466, 0.18751634726630065, 0.13807809525574052, 0.08638040615859968, 0.04136228007853507, 0.010807759991626085, 0.0],
        },
        "q1.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 3.308922169924432e-19, 1.2663545975903585e-18, 2.6446371977794346e-18, 4.227422434682646e-18, 5.741032361686962e-18, 6.923750071698838e-18] + [7.571072928004134e-18] * 2 + [6.923750071698838e-18, 5.741032361686965e-18, 4.227422434682649e-18, 2.6446371977794354e-18, 1.2663545975903577e-18, 3.3089221699244275e-19, 0.0],
        },
        "q1.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0054038799958130495, 0.020681140039267548, 0.04319020307929983, 0.0690390476278702, 0.09375817363315027, 0.11307341964261733] + [0.1236450041477332] * 2 + [0.11307341964261733, 0.09375817363315032, 0.06903904762787026, 0.04319020307929984, 0.020681140039267534, 0.005403879995813043, 0.0],
        },
        "q1.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 3.308922169924432e-19, 1.2663545975903585e-18, 2.6446371977794346e-18, 4.227422434682646e-18, 5.741032361686962e-18, 6.923750071698838e-18] + [7.571072928004134e-18] * 2 + [6.923750071698838e-18, 5.741032361686965e-18, 4.227422434682649e-18, 2.6446371977794354e-18, 1.2663545975903577e-18, 3.3089221699244275e-19, 0.0],
        },
        "q1.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0054038799958130495, -0.020681140039267548, -0.04319020307929983, -0.0690390476278702, -0.09375817363315027, -0.11307341964261733] + [-0.1236450041477332] * 2 + [-0.11307341964261733, -0.09375817363315032, -0.06903904762787026, -0.04319020307929984, -0.020681140039267534, -0.005403879995813043, 0.0],
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
            "sample": 0.0647339785650522,
        },
        "q1.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q1.z.const.wf": {
            "type": "constant",
            "sample": 0.1,
        },
        "q1.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.01215479592441148,
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
            "samples": [0.0, 0.022248548322258737, 0.08514721716240019, 0.17782025526149633, 0.2842436524240928, 0.3860158364545656, 0.46553947216277974] + [0.5090642004852348] * 2 + [0.46553947216277974, 0.3860158364545658, 0.28424365242409305, 0.1778202552614964, 0.08514721716240013, 0.022248548322258706, 0.0],
        },
        "q2.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
        },
        "q2.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.011124274161129368, 0.042573608581200094, 0.08891012763074817, 0.1421218262120464, 0.1930079182272828, 0.23276973608138987] + [0.2545321002426174] * 2 + [0.23276973608138987, 0.1930079182272829, 0.14212182621204653, 0.0889101276307482, 0.042573608581200066, 0.011124274161129353, 0.0],
        },
        "q2.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
        },
        "q2.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.011124274161129368, -0.042573608581200094, -0.08891012763074817, -0.1421218262120464, -0.1930079182272828, -0.23276973608138987] + [-0.2545321002426174] * 2 + [-0.23276973608138987, -0.1930079182272829, -0.14212182621204653, -0.0889101276307482, -0.042573608581200066, -0.011124274161129353, 0.0],
        },
        "q2.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 1.3623306744264688e-18, 5.213763347711898e-18, 1.0888350321477839e-17, 1.7404903955955902e-17, 2.3636652926713598e-17, 2.8506071223044825e-17] + [3.1171192184237466e-17] * 2 + [2.8506071223044825e-17, 2.3636652926713608e-17, 1.7404903955955917e-17, 1.0888350321477842e-17, 5.213763347711895e-18, 1.3623306744264669e-18, 0.0],
        },
        "q2.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 1.3623306744264688e-18, 5.213763347711898e-18, 1.0888350321477839e-17, 1.7404903955955902e-17, 2.3636652926713598e-17, 2.8506071223044825e-17] + [3.1171192184237466e-17] * 2 + [2.8506071223044825e-17, 2.3636652926713608e-17, 1.7404903955955917e-17, 1.0888350321477842e-17, 5.213763347711895e-18, 1.3623306744264669e-18, 0.0],
        },
        "q2.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.022248548322258737, 0.08514721716240019, 0.17782025526149633, 0.2842436524240928, 0.3860158364545656, 0.46553947216277974] + [0.5090642004852348] * 2 + [0.46553947216277974, 0.3860158364545658, 0.28424365242409305, 0.1778202552614964, 0.08514721716240013, 0.022248548322258706, 0.0],
        },
        "q2.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 6.811653372132344e-19, 2.606881673855949e-18, 5.4441751607389195e-18, 8.702451977977951e-18, 1.1818326463356799e-17, 1.4253035611522412e-17] + [1.5585596092118733e-17] * 2 + [1.4253035611522412e-17, 1.1818326463356804e-17, 8.702451977977959e-18, 5.444175160738921e-18, 2.6068816738559476e-18, 6.811653372132334e-19, 0.0],
        },
        "q2.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.011124274161129368, 0.042573608581200094, 0.08891012763074817, 0.1421218262120464, 0.1930079182272828, 0.23276973608138987] + [0.2545321002426174] * 2 + [0.23276973608138987, 0.1930079182272829, 0.14212182621204653, 0.0889101276307482, 0.042573608581200066, 0.011124274161129353, 0.0],
        },
        "q2.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 6.811653372132344e-19, 2.606881673855949e-18, 5.4441751607389195e-18, 8.702451977977951e-18, 1.1818326463356799e-17, 1.4253035611522412e-17] + [1.5585596092118733e-17] * 2 + [1.4253035611522412e-17, 1.1818326463356804e-17, 8.702451977977959e-18, 5.444175160738921e-18, 2.6068816738559476e-18, 6.811653372132334e-19, 0.0],
        },
        "q2.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.011124274161129368, -0.042573608581200094, -0.08891012763074817, -0.1421218262120464, -0.1930079182272828, -0.23276973608138987] + [-0.2545321002426174] * 2 + [-0.23276973608138987, -0.1930079182272829, -0.14212182621204653, -0.0889101276307482, -0.042573608581200066, -0.011124274161129353, 0.0],
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
            "sample": 0.16881819035760665,
        },
        "q2.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.z.const.wf": {
            "type": "constant",
            "sample": 0.1,
        },
        "q2.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.00704349717480433,
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
            "samples": [0.0, 0.024202882788326358, 0.09262663284291633, 0.1934401621690767, 0.3092118956841525, 0.41992384891020657, 0.5064329193479344] + [0.5537809028084203] * 2 + [0.5064329193479344, 0.4199238489102068, 0.30921189568415275, 0.19344016216907675, 0.09262663284291628, 0.024202882788326326, 0.0],
        },
        "c1.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
        },
        "c1.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.010627439357262167, 0.04067217661692243, 0.08493920375891634, 0.1357743496370766, 0.18438775570626728, 0.2223737224181958] + [0.24316413103669582] * 2 + [0.2223737224181958, 0.18438775570626734, 0.1357743496370767, 0.08493920375891637, 0.0406721766169224, 0.010627439357262153, 0.0],
        },
        "c1.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
        },
        "c1.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.010627439357262167, -0.04067217661692243, -0.08493920375891634, -0.1357743496370766, -0.18438775570626728, -0.2223737224181958] + [-0.24316413103669582] * 2 + [-0.2223737224181958, -0.18438775570626734, -0.1357743496370767, -0.08493920375891637, -0.0406721766169224, -0.010627439357262153, 0.0],
        },
        "c1.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 1.3014859592003717e-18, 4.980905090826987e-18, 1.0402052400548172e-17, 1.6627562268935945e-17, 2.2580987482764433e-17, 2.7232926737392553e-17] + [2.977901747415371e-17] * 2 + [2.7232926737392553e-17, 2.258098748276444e-17, 1.662756226893596e-17, 1.0402052400548175e-17, 4.980905090826984e-18, 1.30148595920037e-18, 0.0],
        },
        "c1.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 1.481999146843122e-18, 5.671745471343729e-18, 1.1844793771345236e-17, 1.8933767915394132e-17, 2.571291987267606e-17, 3.1010072683114877e-17] + [3.390930050266317e-17] * 2 + [3.1010072683114877e-17, 2.5712919872676076e-17, 1.8933767915394147e-17, 1.1844793771345239e-17, 5.671745471343726e-18, 1.48199914684312e-18, 0.0],
        },
        "c1.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.024202882788326358, 0.09262663284291633, 0.1934401621690767, 0.3092118956841525, 0.41992384891020657, 0.5064329193479344] + [0.5537809028084203] * 2 + [0.5064329193479344, 0.4199238489102068, 0.30921189568415275, 0.19344016216907675, 0.09262663284291628, 0.024202882788326326, 0.0],
        },
        "c1.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 6.507429796001859e-19, 2.4904525454134936e-18, 5.201026200274086e-18, 8.313781134467972e-18, 1.1290493741382217e-17, 1.3616463368696276e-17] + [1.4889508737076855e-17] * 2 + [1.3616463368696276e-17, 1.129049374138222e-17, 8.31378113446798e-18, 5.2010262002740875e-18, 2.490452545413492e-18, 6.50742979600185e-19, 0.0],
        },
        "c1.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.010627439357262167, 0.04067217661692243, 0.08493920375891634, 0.1357743496370766, 0.18438775570626728, 0.2223737224181958] + [0.24316413103669582] * 2 + [0.2223737224181958, 0.18438775570626734, 0.1357743496370767, 0.08493920375891637, 0.0406721766169224, 0.010627439357262153, 0.0],
        },
        "c1.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 6.507429796001859e-19, 2.4904525454134936e-18, 5.201026200274086e-18, 8.313781134467972e-18, 1.1290493741382217e-17, 1.3616463368696276e-17] + [1.4889508737076855e-17] * 2 + [1.3616463368696276e-17, 1.129049374138222e-17, 8.31378113446798e-18, 5.2010262002740875e-18, 2.490452545413492e-18, 6.50742979600185e-19, 0.0],
        },
        "c1.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.010627439357262167, -0.04067217661692243, -0.08493920375891634, -0.1357743496370766, -0.18438775570626728, -0.2223737224181958] + [-0.24316413103669582] * 2 + [-0.2223737224181958, -0.18438775570626734, -0.1357743496370767, -0.08493920375891637, -0.0406721766169224, -0.010627439357262153, 0.0],
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
            "sample": 0.14257697338929837,
        },
        "c1.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "c1.z.const.wf": {
            "type": "constant",
            "sample": 0.1,
        },
        "c1.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.005078372159941411,
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
            "sample": 0.1,
        },
    },
    "digital_waveforms": {
        "ON": {
            "samples": [[1, 0]],
        },
    },
    "integration_weights": {
        "q1.resonator.readout.iw1": {
            "cosine": [(-0.38671647471408993, 2080)],
            "sine": [(-0.9221986598259112, 2080)],
        },
        "q1.resonator.readout.iw2": {
            "cosine": [(0.9221986598259112, 2080)],
            "sine": [(-0.38671647471408993, 2080)],
        },
        "q1.resonator.readout.iw3": {
            "cosine": [(-0.9221986598259112, 2080)],
            "sine": [(0.38671647471408993, 2080)],
        },
        "q2.resonator.readout.iw1": {
            "cosine": [(-0.48930065127109346, 2080)],
            "sine": [(0.8721151716749823, 2080)],
        },
        "q2.resonator.readout.iw2": {
            "cosine": [(-0.8721151716749823, 2080)],
            "sine": [(-0.48930065127109346, 2080)],
        },
        "q2.resonator.readout.iw3": {
            "cosine": [(0.8721151716749823, 2080)],
            "sine": [(0.48930065127109346, 2080)],
        },
        "c1.resonator.readout.iw1": {
            "cosine": [(0.8749748875773569, 2000)],
            "sine": [(-0.4841683034947577, 2000)],
        },
        "c1.resonator.readout.iw2": {
            "cosine": [(0.4841683034947577, 2000)],
            "sine": [(0.8749748875773569, 2000)],
        },
        "c1.resonator.readout.iw3": {
            "cosine": [(-0.4841683034947577, 2000)],
            "sine": [(-0.8749748875773569, 2000)],
        },
    },
    "mixers": {},
    "oscillators": {},
}

loaded_config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1000",
            "fems": {
                "1": {
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
                            },
                            "crosstalk": {},
                            "output_mode": "direct",
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                        },
                        "2": {
                            "offset": 0.0,
                            "delay": 0,
                            "shareable": False,
                            "filter": {
                                "feedforward": [],
                                "exponential": [],
                                "high_pass": None,
                            },
                            "crosstalk": {},
                            "output_mode": "direct",
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                        },
                        "3": {
                            "offset": 0.0,
                            "delay": 0,
                            "shareable": False,
                            "filter": {
                                "feedforward": [],
                                "exponential": [],
                                "high_pass": None,
                            },
                            "crosstalk": {},
                            "output_mode": "direct",
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
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
                            "full_scale_power_dbm": 7,
                            "band": 1,
                            "delay": 0,
                            "shareable": False,
                            "upconverters": {
                                "1": {
                                    "frequency": 4400000000.0,
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
                                    "frequency": 4700000000.0,
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
            "intermediate_frequency": 277191481.2810402,
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
                "port": ('con1', 1, 1),
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
            "intermediate_frequency": -231320485.0,
        },
        "q2.xy": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "x180_DragCosine": "q2.xy.x180_DragCosine.pulse",
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
            "intermediate_frequency": 262899707.49565393,
        },
        "q2.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "q2.z.const.pulse",
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
            "intermediate_frequency": -149374975.0,
        },
        "c1.xy": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "x180_DragCosine": "c1.xy.x180_DragCosine.pulse",
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
                "x180": "c1.xy.x180_DragCosine.pulse",
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
                "port": ('con1', 6, 2),
                "upconverter": 1,
            },
            "intermediate_frequency": -5749984.452069147,
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
            "intermediate_frequency": -65211388.0,
        },
        "coupler_q1_q2": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "coupler_q1_q2.const.pulse",
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
            "length": 16,
            "waveforms": {
                "I": "q1.xy.x180_DragCosine.wf.I",
                "Q": "q1.xy.x180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.x90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q1.xy.x90_DragCosine.wf.I",
                "Q": "q1.xy.x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.-x90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q1.xy.-x90_DragCosine.wf.I",
                "Q": "q1.xy.-x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.y180_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q1.xy.y180_DragCosine.wf.I",
                "Q": "q1.xy.y180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.y90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q1.xy.y90_DragCosine.wf.I",
                "Q": "q1.xy.y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.-y90_DragCosine.pulse": {
            "length": 16,
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
        "q1.z.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "q1.z.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q1.resonator.readout.pulse": {
            "length": 2080,
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
            "length": 16,
            "waveforms": {
                "I": "q2.xy.x180_DragCosine.wf.I",
                "Q": "q2.xy.x180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.x90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q2.xy.x90_DragCosine.wf.I",
                "Q": "q2.xy.x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.-x90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q2.xy.-x90_DragCosine.wf.I",
                "Q": "q2.xy.-x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.y180_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q2.xy.y180_DragCosine.wf.I",
                "Q": "q2.xy.y180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.y90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q2.xy.y90_DragCosine.wf.I",
                "Q": "q2.xy.y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.-y90_DragCosine.pulse": {
            "length": 16,
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
        "q2.z.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "q2.z.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q2.resonator.readout.pulse": {
            "length": 2080,
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
            "samples": [0.0, 0.010807759991626099, 0.041362280078535096, 0.08638040615859965, 0.1380780952557404, 0.18751634726630054, 0.22614683928523466] + [0.2472900082954664] * 2 + [0.22614683928523466, 0.18751634726630065, 0.13807809525574052, 0.08638040615859968, 0.04136228007853507, 0.010807759991626085, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0054038799958130495, 0.020681140039267548, 0.04319020307929983, 0.0690390476278702, 0.09375817363315027, 0.11307341964261733] + [0.1236450041477332] * 2 + [0.11307341964261733, 0.09375817363315032, 0.06903904762787026, 0.04319020307929984, 0.020681140039267534, 0.005403879995813043, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0054038799958130495, -0.020681140039267548, -0.04319020307929983, -0.0690390476278702, -0.09375817363315027, -0.11307341964261733] + [-0.1236450041477332] * 2 + [-0.11307341964261733, -0.09375817363315032, -0.06903904762787026, -0.04319020307929984, -0.020681140039267534, -0.005403879995813043, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 6.617844339848864e-19, 2.532709195180717e-18, 5.289274395558869e-18, 8.454844869365292e-18, 1.1482064723373924e-17, 1.3847500143397676e-17] + [1.5142145856008268e-17] * 2 + [1.3847500143397676e-17, 1.148206472337393e-17, 8.454844869365298e-18, 5.289274395558871e-18, 2.5327091951807154e-18, 6.617844339848855e-19, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 6.617844339848864e-19, 2.532709195180717e-18, 5.289274395558869e-18, 8.454844869365292e-18, 1.1482064723373924e-17, 1.3847500143397676e-17] + [1.5142145856008268e-17] * 2 + [1.3847500143397676e-17, 1.148206472337393e-17, 8.454844869365298e-18, 5.289274395558871e-18, 2.5327091951807154e-18, 6.617844339848855e-19, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.010807759991626099, 0.041362280078535096, 0.08638040615859965, 0.1380780952557404, 0.18751634726630054, 0.22614683928523466] + [0.2472900082954664] * 2 + [0.22614683928523466, 0.18751634726630065, 0.13807809525574052, 0.08638040615859968, 0.04136228007853507, 0.010807759991626085, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 3.308922169924432e-19, 1.2663545975903585e-18, 2.6446371977794346e-18, 4.227422434682646e-18, 5.741032361686962e-18, 6.923750071698838e-18] + [7.571072928004134e-18] * 2 + [6.923750071698838e-18, 5.741032361686965e-18, 4.227422434682649e-18, 2.6446371977794354e-18, 1.2663545975903577e-18, 3.3089221699244275e-19, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0054038799958130495, 0.020681140039267548, 0.04319020307929983, 0.0690390476278702, 0.09375817363315027, 0.11307341964261733] + [0.1236450041477332] * 2 + [0.11307341964261733, 0.09375817363315032, 0.06903904762787026, 0.04319020307929984, 0.020681140039267534, 0.005403879995813043, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 3.308922169924432e-19, 1.2663545975903585e-18, 2.6446371977794346e-18, 4.227422434682646e-18, 5.741032361686962e-18, 6.923750071698838e-18] + [7.571072928004134e-18] * 2 + [6.923750071698838e-18, 5.741032361686965e-18, 4.227422434682649e-18, 2.6446371977794354e-18, 1.2663545975903577e-18, 3.3089221699244275e-19, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0054038799958130495, -0.020681140039267548, -0.04319020307929983, -0.0690390476278702, -0.09375817363315027, -0.11307341964261733] + [-0.1236450041477332] * 2 + [-0.11307341964261733, -0.09375817363315032, -0.06903904762787026, -0.04319020307929984, -0.020681140039267534, -0.005403879995813043, 0.0],
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
            "sample": 0.0647339785650522,
        },
        "q1.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q1.z.const.wf": {
            "type": "constant",
            "sample": 0.1,
        },
        "q1.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.01215479592441148,
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
            "samples": [0.0, 0.022248548322258737, 0.08514721716240019, 0.17782025526149633, 0.2842436524240928, 0.3860158364545656, 0.46553947216277974] + [0.5090642004852348] * 2 + [0.46553947216277974, 0.3860158364545658, 0.28424365242409305, 0.1778202552614964, 0.08514721716240013, 0.022248548322258706, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.011124274161129368, 0.042573608581200094, 0.08891012763074817, 0.1421218262120464, 0.1930079182272828, 0.23276973608138987] + [0.2545321002426174] * 2 + [0.23276973608138987, 0.1930079182272829, 0.14212182621204653, 0.0889101276307482, 0.042573608581200066, 0.011124274161129353, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.011124274161129368, -0.042573608581200094, -0.08891012763074817, -0.1421218262120464, -0.1930079182272828, -0.23276973608138987] + [-0.2545321002426174] * 2 + [-0.23276973608138987, -0.1930079182272829, -0.14212182621204653, -0.0889101276307482, -0.042573608581200066, -0.011124274161129353, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 1.3623306744264688e-18, 5.213763347711898e-18, 1.0888350321477839e-17, 1.7404903955955902e-17, 2.3636652926713598e-17, 2.8506071223044825e-17] + [3.1171192184237466e-17] * 2 + [2.8506071223044825e-17, 2.3636652926713608e-17, 1.7404903955955917e-17, 1.0888350321477842e-17, 5.213763347711895e-18, 1.3623306744264669e-18, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 1.3623306744264688e-18, 5.213763347711898e-18, 1.0888350321477839e-17, 1.7404903955955902e-17, 2.3636652926713598e-17, 2.8506071223044825e-17] + [3.1171192184237466e-17] * 2 + [2.8506071223044825e-17, 2.3636652926713608e-17, 1.7404903955955917e-17, 1.0888350321477842e-17, 5.213763347711895e-18, 1.3623306744264669e-18, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.022248548322258737, 0.08514721716240019, 0.17782025526149633, 0.2842436524240928, 0.3860158364545656, 0.46553947216277974] + [0.5090642004852348] * 2 + [0.46553947216277974, 0.3860158364545658, 0.28424365242409305, 0.1778202552614964, 0.08514721716240013, 0.022248548322258706, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 6.811653372132344e-19, 2.606881673855949e-18, 5.4441751607389195e-18, 8.702451977977951e-18, 1.1818326463356799e-17, 1.4253035611522412e-17] + [1.5585596092118733e-17] * 2 + [1.4253035611522412e-17, 1.1818326463356804e-17, 8.702451977977959e-18, 5.444175160738921e-18, 2.6068816738559476e-18, 6.811653372132334e-19, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.011124274161129368, 0.042573608581200094, 0.08891012763074817, 0.1421218262120464, 0.1930079182272828, 0.23276973608138987] + [0.2545321002426174] * 2 + [0.23276973608138987, 0.1930079182272829, 0.14212182621204653, 0.0889101276307482, 0.042573608581200066, 0.011124274161129353, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 6.811653372132344e-19, 2.606881673855949e-18, 5.4441751607389195e-18, 8.702451977977951e-18, 1.1818326463356799e-17, 1.4253035611522412e-17] + [1.5585596092118733e-17] * 2 + [1.4253035611522412e-17, 1.1818326463356804e-17, 8.702451977977959e-18, 5.444175160738921e-18, 2.6068816738559476e-18, 6.811653372132334e-19, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.011124274161129368, -0.042573608581200094, -0.08891012763074817, -0.1421218262120464, -0.1930079182272828, -0.23276973608138987] + [-0.2545321002426174] * 2 + [-0.23276973608138987, -0.1930079182272829, -0.14212182621204653, -0.0889101276307482, -0.042573608581200066, -0.011124274161129353, 0.0],
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
            "sample": 0.16881819035760665,
        },
        "q2.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.z.const.wf": {
            "type": "constant",
            "sample": 0.1,
        },
        "q2.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.00704349717480433,
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
            "samples": [0.0, 0.024202882788326358, 0.09262663284291633, 0.1934401621690767, 0.3092118956841525, 0.41992384891020657, 0.5064329193479344] + [0.5537809028084203] * 2 + [0.5064329193479344, 0.4199238489102068, 0.30921189568415275, 0.19344016216907675, 0.09262663284291628, 0.024202882788326326, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "c1.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "c1.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.010627439357262167, 0.04067217661692243, 0.08493920375891634, 0.1357743496370766, 0.18438775570626728, 0.2223737224181958] + [0.24316413103669582] * 2 + [0.2223737224181958, 0.18438775570626734, 0.1357743496370767, 0.08493920375891637, 0.0406721766169224, 0.010627439357262153, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "c1.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "c1.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.010627439357262167, -0.04067217661692243, -0.08493920375891634, -0.1357743496370766, -0.18438775570626728, -0.2223737224181958] + [-0.24316413103669582] * 2 + [-0.2223737224181958, -0.18438775570626734, -0.1357743496370767, -0.08493920375891637, -0.0406721766169224, -0.010627439357262153, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "c1.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 1.3014859592003717e-18, 4.980905090826987e-18, 1.0402052400548172e-17, 1.6627562268935945e-17, 2.2580987482764433e-17, 2.7232926737392553e-17] + [2.977901747415371e-17] * 2 + [2.7232926737392553e-17, 2.258098748276444e-17, 1.662756226893596e-17, 1.0402052400548175e-17, 4.980905090826984e-18, 1.30148595920037e-18, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "c1.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 1.481999146843122e-18, 5.671745471343729e-18, 1.1844793771345236e-17, 1.8933767915394132e-17, 2.571291987267606e-17, 3.1010072683114877e-17] + [3.390930050266317e-17] * 2 + [3.1010072683114877e-17, 2.5712919872676076e-17, 1.8933767915394147e-17, 1.1844793771345239e-17, 5.671745471343726e-18, 1.48199914684312e-18, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "c1.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.024202882788326358, 0.09262663284291633, 0.1934401621690767, 0.3092118956841525, 0.41992384891020657, 0.5064329193479344] + [0.5537809028084203] * 2 + [0.5064329193479344, 0.4199238489102068, 0.30921189568415275, 0.19344016216907675, 0.09262663284291628, 0.024202882788326326, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "c1.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 6.507429796001859e-19, 2.4904525454134936e-18, 5.201026200274086e-18, 8.313781134467972e-18, 1.1290493741382217e-17, 1.3616463368696276e-17] + [1.4889508737076855e-17] * 2 + [1.3616463368696276e-17, 1.129049374138222e-17, 8.31378113446798e-18, 5.2010262002740875e-18, 2.490452545413492e-18, 6.50742979600185e-19, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "c1.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.010627439357262167, 0.04067217661692243, 0.08493920375891634, 0.1357743496370766, 0.18438775570626728, 0.2223737224181958] + [0.24316413103669582] * 2 + [0.2223737224181958, 0.18438775570626734, 0.1357743496370767, 0.08493920375891637, 0.0406721766169224, 0.010627439357262153, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "c1.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 6.507429796001859e-19, 2.4904525454134936e-18, 5.201026200274086e-18, 8.313781134467972e-18, 1.1290493741382217e-17, 1.3616463368696276e-17] + [1.4889508737076855e-17] * 2 + [1.3616463368696276e-17, 1.129049374138222e-17, 8.31378113446798e-18, 5.2010262002740875e-18, 2.490452545413492e-18, 6.50742979600185e-19, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "c1.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.010627439357262167, -0.04067217661692243, -0.08493920375891634, -0.1357743496370766, -0.18438775570626728, -0.2223737224181958] + [-0.24316413103669582] * 2 + [-0.2223737224181958, -0.18438775570626734, -0.1357743496370767, -0.08493920375891637, -0.0406721766169224, -0.010627439357262153, 0.0],
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
            "sample": 0.14257697338929837,
        },
        "c1.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "c1.z.const.wf": {
            "type": "constant",
            "sample": 0.1,
        },
        "c1.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.005078372159941411,
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
            "sample": 0.1,
        },
    },
    "digital_waveforms": {
        "ON": {
            "samples": [(1, 0)],
        },
    },
    "integration_weights": {
        "q1.resonator.readout.iw1": {
            "cosine": [(-0.38671647471408993, 2080)],
            "sine": [(-0.9221986598259112, 2080)],
        },
        "q1.resonator.readout.iw2": {
            "cosine": [(0.9221986598259112, 2080)],
            "sine": [(-0.38671647471408993, 2080)],
        },
        "q1.resonator.readout.iw3": {
            "cosine": [(-0.9221986598259112, 2080)],
            "sine": [(0.38671647471408993, 2080)],
        },
        "q2.resonator.readout.iw1": {
            "cosine": [(-0.48930065127109346, 2080)],
            "sine": [(0.8721151716749823, 2080)],
        },
        "q2.resonator.readout.iw2": {
            "cosine": [(-0.8721151716749823, 2080)],
            "sine": [(-0.48930065127109346, 2080)],
        },
        "q2.resonator.readout.iw3": {
            "cosine": [(0.8721151716749823, 2080)],
            "sine": [(0.48930065127109346, 2080)],
        },
        "c1.resonator.readout.iw1": {
            "cosine": [(0.8749748875773569, 2000)],
            "sine": [(-0.4841683034947577, 2000)],
        },
        "c1.resonator.readout.iw2": {
            "cosine": [(0.4841683034947577, 2000)],
            "sine": [(0.8749748875773569, 2000)],
        },
        "c1.resonator.readout.iw3": {
            "cosine": [(-0.4841683034947577, 2000)],
            "sine": [(-0.8749748875773569, 2000)],
        },
    },
    "mixers": {},
}

