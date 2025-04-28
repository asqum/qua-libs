
# Single QUA script generated at 2025-04-20 15:59:25.697027
# QUA library version: 1.2.1

from qm import CompilerOptionArguments
from qm.qua import *

with program() as prog:
    v1 = declare(int, )
    v2 = declare(int, )
    v3 = declare(fixed, )
    a1 = declare(fixed, size=4)
    a2 = declare(fixed, size=4)
    a3 = declare(fixed, size=4)
    a4 = declare(fixed, size=4)
    a5 = declare(fixed, size=4)
    a6 = declare(fixed, size=4)
    a7 = declare(fixed, size=4)
    a8 = declare(fixed, size=4)
    a9 = declare(fixed, size=4)
    a10 = declare(fixed, size=4)
    a11 = declare(fixed, size=4)
    a12 = declare(fixed, size=4)
    a13 = declare(fixed, size=4)
    a14 = declare(fixed, size=4)
    a15 = declare(fixed, size=4)
    a16 = declare(fixed, size=4)
    a17 = declare(fixed, size=4)
    a18 = declare(fixed, size=4)
    a19 = declare(fixed, size=4)
    a20 = declare(fixed, size=4)
    a21 = declare(fixed, size=4)
    a22 = declare(fixed, size=4)
    a23 = declare(fixed, size=4)
    a24 = declare(fixed, size=4)
    a25 = declare(fixed, size=4)
    a26 = declare(fixed, size=4)
    a27 = declare(fixed, size=4)
    a28 = declare(fixed, size=4)
    a29 = declare(fixed, size=4)
    a30 = declare(fixed, size=4)
    a31 = declare(fixed, size=4)
    a32 = declare(fixed, size=4)
    a33 = declare(fixed, size=4)
    a34 = declare(fixed, size=4)
    a35 = declare(fixed, size=4)
    a36 = declare(fixed, size=4)
    v4 = declare(int, )
    v5 = declare(int, )
    v6 = declare(int, )
    set_dc_offset("q1.z", "single", 0.12188185683987682)
    set_dc_offset("q2.z", "single", 0.12505899308843357)
    set_dc_offset("q3.z", "single", 0.09251480237587392)
    set_dc_offset("q4.z", "single", 0.1135297239791084)
    set_dc_offset("q5.z", "single", 0.11280221415397873)
    set_dc_offset("coupler_q1_q2", "single", 0.128)
    set_dc_offset("coupler_q2_q3", "single", 0.14887)
    set_dc_offset("coupler_q3_q4", "single", 0.1266)
    set_dc_offset("coupler_q4_q5", "single", 0.1276)
    wait(24, "q1.z")
    align("q1.xy", "q1.resonator", "q1.z")
    with for_(v1,0,(v1<6.0),(v1+1)):
        r13 = declare_stream()
        save(v1, r13)
        with for_(v2,-2500000,(v2<=2300000),(v2+200000)):
            with for_(v3,0.2,(v3<2.0539285714285715),(v3+0.12785714285714284)):
                update_frequency("q1.resonator", (-16582781.0+v2), "Hz", False)
                update_frequency("q2.resonator", (74083957.0+v2), "Hz", False)
                update_frequency("q3.resonator", (-89526496.0+v2), "Hz", False)
                update_frequency("q4.resonator", (129555949.0+v2), "Hz", False)
                update_frequency("q5.resonator", (19883495.0+v2), "Hz", False)
                wait(85155, )
                align()
                measure("readout"*amp(v3), "q1.resonator", None, demod.accumulated("iw1", a1, 65, "out1"), demod.accumulated("iw2", a2, 65, "out2"), demod.accumulated("iw3", a3, 65, "out1"), demod.accumulated("iw1", a4, 65, "out2"))
                measure("readout"*amp(v3), "q2.resonator", None, demod.accumulated("iw1", a13, 65, "out1"), demod.accumulated("iw2", a14, 65, "out2"), demod.accumulated("iw3", a15, 65, "out1"), demod.accumulated("iw1", a16, 65, "out2"))
                measure("readout"*amp(v3), "q3.resonator", None, demod.accumulated("iw1", a25, 65, "out1"), demod.accumulated("iw2", a26, 65, "out2"), demod.accumulated("iw3", a27, 65, "out1"), demod.accumulated("iw1", a28, 65, "out2"))
                play("readout"*amp(v3), "q4.resonator")
                play("readout"*amp(v3), "q5.resonator")
                align()
                wait(85155, )
                play("x180", "q1.xy")
                play("x180", "q2.xy")
                play("x180", "q3.xy")
                play("x180", "q4.xy")
                play("x180", "q5.xy")
                align()
                measure("readout"*amp(v3), "q1.resonator", None, demod.accumulated("iw1", a7, 65, "out1"), demod.accumulated("iw2", a8, 65, "out2"), demod.accumulated("iw3", a9, 65, "out1"), demod.accumulated("iw1", a10, 65, "out2"))
                measure("readout"*amp(v3), "q2.resonator", None, demod.accumulated("iw1", a19, 65, "out1"), demod.accumulated("iw2", a20, 65, "out2"), demod.accumulated("iw3", a21, 65, "out1"), demod.accumulated("iw1", a22, 65, "out2"))
                measure("readout"*amp(v3), "q3.resonator", None, demod.accumulated("iw1", a31, 65, "out1"), demod.accumulated("iw2", a32, 65, "out2"), demod.accumulated("iw3", a33, 65, "out1"), demod.accumulated("iw1", a34, 65, "out2"))
                play("readout"*amp(v3), "q4.resonator")
                play("readout"*amp(v3), "q5.resonator")
                with for_(v4,0,(v4<4),(v4+1)):
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
                with for_(v5,0,(v5<4),(v5+1)):
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
                with for_(v6,0,(v6<4),(v6+1)):
                    assign(a29[v6], (a25[v6]+a26[v6]))
                    r9 = declare_stream()
                    save(a29[v6], r9)
                    assign(a30[v6], (a27[v6]+a28[v6]))
                    r10 = declare_stream()
                    save(a30[v6], r10)
                    assign(a35[v6], (a31[v6]+a32[v6]))
                    r11 = declare_stream()
                    save(a35[v6], r11)
                    assign(a36[v6], (a33[v6]+a34[v6]))
                    r12 = declare_stream()
                    save(a36[v6], r12)
    with stream_processing():
        r13.save("n")
        r1.buffer(4).buffer(15).buffer(25).buffer(6).save("I_g1")
        r2.buffer(4).buffer(15).buffer(25).buffer(6).save("Q_g1")
        r3.buffer(4).buffer(15).buffer(25).buffer(6).save("I_e1")
        r4.buffer(4).buffer(15).buffer(25).buffer(6).save("Q_e1")
        r5.buffer(4).buffer(15).buffer(25).buffer(6).save("I_g2")
        r6.buffer(4).buffer(15).buffer(25).buffer(6).save("Q_g2")
        r7.buffer(4).buffer(15).buffer(25).buffer(6).save("I_e2")
        r8.buffer(4).buffer(15).buffer(25).buffer(6).save("Q_e2")
        r9.buffer(4).buffer(15).buffer(25).buffer(6).save("I_g3")
        r10.buffer(4).buffer(15).buffer(25).buffer(6).save("Q_g3")
        r11.buffer(4).buffer(15).buffer(25).buffer(6).save("I_e3")
        r12.buffer(4).buffer(15).buffer(25).buffer(6).save("Q_e3")


config = {
    "version": 1,
    "controllers": {
        "con1": {
            "fems": {
                "1": {
                    "type": "LF",
                    "analog_outputs": {
                        "7": {
                            "delay": 168,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                        "1": {
                            "delay": 180,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                        "2": {
                            "delay": 172,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                        "3": {
                            "delay": 176,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                        "4": {
                            "delay": 177,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                        "5": {
                            "delay": 172,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                        "6": {
                            "delay": 172,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                        "8": {
                            "delay": 172,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                    },
                },
                "2": {
                    "type": "LF",
                    "analog_outputs": {
                        "8": {
                            "delay": 172,
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
                            "full_scale_power_dbm": -8,
                            "upconverter_frequency": 5950000000,
                        },
                        "2": {
                            "band": 1,
                            "delay": 20,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 4,
                            "upconverter_frequency": 4900000000.0,
                        },
                        "3": {
                            "band": 1,
                            "delay": 20,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 4,
                            "upconverter_frequency": 4900000000.0,
                        },
                        "4": {
                            "band": 1,
                            "delay": 20,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 4,
                            "upconverter_frequency": 5000000000.0,
                        },
                        "5": {
                            "band": 1,
                            "delay": 20,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 4,
                            "upconverter_frequency": 5000000000.0,
                        },
                        "6": {
                            "band": 1,
                            "delay": 20,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 4,
                            "upconverter_frequency": 4900000000.0,
                        },
                    },
                    "analog_inputs": {
                        "1": {
                            "band": 2,
                            "downconverter_frequency": 5950000000,
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
                "EF_x180": "q1.xy.EF_x180.pulse",
                "EF_x90": "q1.xy.EF_x90.pulse",
            },
            "intermediate_frequency": 213409441.43405086,
            "thread": "a",
            "MWInput": {
                "port": ('con1', 6, 2),
                "upconverter": 1,
            },
        },
        "q1.z": {
            "operations": {
                "const": "q1.z.const.pulse",
                "flux_pulse": "q1.z.flux_pulse.pulse",
                "cz1_2": "q1.z.cz1_2.pulse",
                "SWAP_Coupler.flux_pulse_control_q2": "q1.z.SWAP_Coupler.flux_pulse_control_q2.pulse",
                "Cz_unipolar.flux_pulse_control_q2_q1": "q1.z.Cz_unipolar.flux_pulse_control_q2_q1.pulse",
            },
            "singleInput": {
                "port": ('con1', 1, 7),
            },
        },
        "q1.resonator": {
            "operations": {
                "readout": "q1.resonator.readout.pulse",
                "const": "q1.resonator.const.pulse",
            },
            "intermediate_frequency": -16582781.0,
            "thread": "a",
            "MWOutput": {
                "port": ('con1', 6, 1),
            },
            "smearing": 0,
            "time_of_flight": 384,
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
                "EF_x180": "q2.xy.EF_x180.pulse",
                "EF_x90": "q2.xy.EF_x90.pulse",
            },
            "intermediate_frequency": -62236279.16832265,
            "thread": "b",
            "MWInput": {
                "port": ('con1', 6, 3),
                "upconverter": 1,
            },
        },
        "q2.z": {
            "operations": {
                "const": "q2.z.const.pulse",
                "flux_pulse": "q2.z.flux_pulse.pulse",
                "cz": "q2.z.cz.pulse",
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
            "intermediate_frequency": 74083957.0,
            "thread": "b",
            "MWOutput": {
                "port": ('con1', 6, 1),
            },
            "smearing": 0,
            "time_of_flight": 376,
            "MWInput": {
                "port": ('con1', 6, 1),
                "upconverter": 1,
            },
        },
        "q3.xy": {
            "operations": {
                "x180_DragCosine": "q3.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "q3.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "q3.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "q3.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "q3.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "q3.xy.-y90_DragCosine.pulse",
                "x180_Square": "q3.xy.x180_Square.pulse",
                "x90_Square": "q3.xy.x90_Square.pulse",
                "-x90_Square": "q3.xy.-x90_Square.pulse",
                "y180_Square": "q3.xy.y180_Square.pulse",
                "y90_Square": "q3.xy.y90_Square.pulse",
                "-y90_Square": "q3.xy.-y90_Square.pulse",
                "x180": "q3.xy.x180_DragCosine.pulse",
                "x90": "q3.xy.x90_DragCosine.pulse",
                "-x90": "q3.xy.-x90_DragCosine.pulse",
                "y180": "q3.xy.y180_DragCosine.pulse",
                "y90": "q3.xy.y90_DragCosine.pulse",
                "-y90": "q3.xy.-y90_DragCosine.pulse",
                "saturation": "q3.xy.saturation.pulse",
                "EF_x180": "q3.xy.EF_x180.pulse",
                "EF_x90": "q3.xy.EF_x90.pulse",
            },
            "intermediate_frequency": 145743108.35321197,
            "thread": "c",
            "MWInput": {
                "port": ('con1', 6, 4),
                "upconverter": 1,
            },
        },
        "q3.z": {
            "operations": {
                "const": "q3.z.const.pulse",
                "flux_pulse": "q3.z.flux_pulse.pulse",
                "cz3_2": "q3.z.cz3_2.pulse",
                "cz3_4": "q3.z.cz3_4.pulse",
                "SWAP_Coupler.flux_pulse_control_q2_q3": "q3.z.SWAP_Coupler.flux_pulse_control_q2_q3.pulse",
                "Cz_unipolar.flux_pulse_control_q2_q3": "q3.z.Cz_unipolar.flux_pulse_control_q2_q3.pulse",
                "SWAP_Coupler.flux_pulse_control_q3_q4": "q3.z.SWAP_Coupler.flux_pulse_control_q3_q4.pulse",
                "Cz_unipolar.flux_pulse_control_q4_q3": "q3.z.Cz_unipolar.flux_pulse_control_q4_q3.pulse",
            },
            "singleInput": {
                "port": ('con1', 1, 2),
            },
        },
        "q3.resonator": {
            "operations": {
                "readout": "q3.resonator.readout.pulse",
                "const": "q3.resonator.const.pulse",
            },
            "intermediate_frequency": -89526496.0,
            "thread": "c",
            "MWOutput": {
                "port": ('con1', 6, 1),
            },
            "smearing": 0,
            "time_of_flight": 384,
            "MWInput": {
                "port": ('con1', 6, 1),
                "upconverter": 1,
            },
        },
        "q4.xy": {
            "operations": {
                "x180_DragCosine": "q4.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "q4.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "q4.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "q4.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "q4.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "q4.xy.-y90_DragCosine.pulse",
                "x180_Square": "q4.xy.x180_Square.pulse",
                "x90_Square": "q4.xy.x90_Square.pulse",
                "-x90_Square": "q4.xy.-x90_Square.pulse",
                "y180_Square": "q4.xy.y180_Square.pulse",
                "y90_Square": "q4.xy.y90_Square.pulse",
                "-y90_Square": "q4.xy.-y90_Square.pulse",
                "x180": "q4.xy.x180_DragCosine.pulse",
                "x90": "q4.xy.x90_DragCosine.pulse",
                "-x90": "q4.xy.-x90_DragCosine.pulse",
                "y180": "q4.xy.y180_DragCosine.pulse",
                "y90": "q4.xy.y90_DragCosine.pulse",
                "-y90": "q4.xy.-y90_DragCosine.pulse",
                "saturation": "q4.xy.saturation.pulse",
                "EF_x180": "q4.xy.EF_x180.pulse",
                "EF_x90": "q4.xy.EF_x90.pulse",
            },
            "intermediate_frequency": -323280460.768938,
            "thread": "d",
            "MWInput": {
                "port": ('con1', 6, 5),
                "upconverter": 1,
            },
        },
        "q4.z": {
            "operations": {
                "const": "q4.z.const.pulse",
                "flux_pulse": "q4.z.flux_pulse.pulse",
                "cz": "q4.z.cz.pulse",
            },
            "singleInput": {
                "port": ('con1', 1, 3),
            },
        },
        "q4.resonator": {
            "operations": {
                "readout": "q4.resonator.readout.pulse",
                "const": "q4.resonator.const.pulse",
            },
            "intermediate_frequency": 129555949.0,
            "thread": "d",
            "MWOutput": {
                "port": ('con1', 6, 1),
            },
            "smearing": 0,
            "time_of_flight": 384,
            "MWInput": {
                "port": ('con1', 6, 1),
                "upconverter": 1,
            },
        },
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
                "EF_x180": "q5.xy.EF_x180.pulse",
                "EF_x90": "q5.xy.EF_x90.pulse",
            },
            "intermediate_frequency": -14023978.739179434,
            "thread": "e",
            "MWInput": {
                "port": ('con1', 6, 6),
                "upconverter": 1,
            },
        },
        "q5.z": {
            "operations": {
                "const": "q5.z.const.pulse",
                "flux_pulse": "q5.z.flux_pulse.pulse",
                "cz5_4": "q5.z.cz5_4.pulse",
            },
            "singleInput": {
                "port": ('con1', 1, 4),
            },
        },
        "q5.resonator": {
            "operations": {
                "readout": "q5.resonator.readout.pulse",
                "const": "q5.resonator.const.pulse",
            },
            "intermediate_frequency": 19883495.0,
            "thread": "e",
            "MWOutput": {
                "port": ('con1', 6, 1),
            },
            "smearing": 0,
            "time_of_flight": 384,
            "MWInput": {
                "port": ('con1', 6, 1),
                "upconverter": 1,
            },
        },
        "coupler_q1_q2": {
            "operations": {
                "const": "coupler_q1_q2.const.pulse",
                "flux_pulse": "coupler_q1_q2.flux_pulse.pulse",
                "cz": "coupler_q1_q2.cz.pulse",
                "SWAP_Coupler.coupler_pulse_control_q2": "coupler_q1_q2.SWAP_Coupler.coupler_pulse_control_q2.pulse",
                "Cz_unipolar.coupler_flux_pulse_q2_q1": "coupler_q1_q2.Cz_unipolar.coupler_flux_pulse_q2_q1.pulse",
            },
            "singleInput": {
                "port": ('con1', 1, 5),
            },
        },
        "coupler_q2_q3": {
            "operations": {
                "const": "coupler_q2_q3.const.pulse",
                "flux_pulse": "coupler_q2_q3.flux_pulse.pulse",
                "cz": "coupler_q2_q3.cz.pulse",
                "SWAP_Coupler.coupler_pulse_control_q2_q3": "coupler_q2_q3.SWAP_Coupler.coupler_pulse_control_q2_q3.pulse",
                "Cz_unipolar.coupler_flux_pulse_q2_q3": "coupler_q2_q3.Cz_unipolar.coupler_flux_pulse_q2_q3.pulse",
            },
            "singleInput": {
                "port": ('con1', 1, 6),
            },
        },
        "coupler_q3_q4": {
            "operations": {
                "const": "coupler_q3_q4.const.pulse",
                "flux_pulse": "coupler_q3_q4.flux_pulse.pulse",
                "cz": "coupler_q3_q4.cz.pulse",
                "SWAP_Coupler.coupler_pulse_control_q3_q4": "coupler_q3_q4.SWAP_Coupler.coupler_pulse_control_q3_q4.pulse",
                "Cz_unipolar.coupler_flux_pulse_q4_q3": "coupler_q3_q4.Cz_unipolar.coupler_flux_pulse_q4_q3.pulse",
            },
            "singleInput": {
                "port": ('con1', 2, 8),
            },
        },
        "coupler_q4_q5": {
            "operations": {
                "const": "coupler_q4_q5.const.pulse",
                "flux_pulse": "coupler_q4_q5.flux_pulse.pulse",
                "cz": "coupler_q4_q5.cz.pulse",
            },
            "singleInput": {
                "port": ('con1', 1, 8),
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
        "q1.xy.EF_x180.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.EF_x180.wf.I",
                "Q": "q1.xy.EF_x180.wf.Q",
            },
        },
        "q1.xy.EF_x90.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.EF_x90.wf.I",
                "Q": "q1.xy.EF_x90.wf.Q",
            },
        },
        "q1.z.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "q1.z.const.wf",
            },
        },
        "q1.z.flux_pulse.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "q1.z.flux_pulse.wf",
            },
        },
        "q1.z.cz1_2.pulse": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "single": "q1.z.cz1_2.wf",
            },
        },
        "q1.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 1040,
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
        "q2.xy.EF_x180.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.EF_x180.wf.I",
                "Q": "q2.xy.EF_x180.wf.Q",
            },
        },
        "q2.xy.EF_x90.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.EF_x90.wf.I",
                "Q": "q2.xy.EF_x90.wf.Q",
            },
        },
        "q2.z.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "q2.z.const.wf",
            },
        },
        "q2.z.flux_pulse.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "q2.z.flux_pulse.wf",
            },
        },
        "q2.z.cz.pulse": {
            "operation": "control",
            "length": 40,
            "waveforms": {
                "single": "q2.z.cz.wf",
            },
        },
        "q2.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 1040,
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
        "q3.xy.x180_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.x180_DragCosine.wf.I",
                "Q": "q3.xy.x180_DragCosine.wf.Q",
            },
        },
        "q3.xy.x90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.x90_DragCosine.wf.I",
                "Q": "q3.xy.x90_DragCosine.wf.Q",
            },
        },
        "q3.xy.-x90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.-x90_DragCosine.wf.I",
                "Q": "q3.xy.-x90_DragCosine.wf.Q",
            },
        },
        "q3.xy.y180_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.y180_DragCosine.wf.I",
                "Q": "q3.xy.y180_DragCosine.wf.Q",
            },
        },
        "q3.xy.y90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.y90_DragCosine.wf.I",
                "Q": "q3.xy.y90_DragCosine.wf.Q",
            },
        },
        "q3.xy.-y90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.-y90_DragCosine.wf.I",
                "Q": "q3.xy.-y90_DragCosine.wf.Q",
            },
        },
        "q3.xy.x180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.x180_Square.wf.I",
                "Q": "q3.xy.x180_Square.wf.Q",
            },
        },
        "q3.xy.x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.x90_Square.wf.I",
                "Q": "q3.xy.x90_Square.wf.Q",
            },
        },
        "q3.xy.-x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.-x90_Square.wf.I",
                "Q": "q3.xy.-x90_Square.wf.Q",
            },
        },
        "q3.xy.y180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.y180_Square.wf.I",
                "Q": "q3.xy.y180_Square.wf.Q",
            },
        },
        "q3.xy.y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.y90_Square.wf.I",
                "Q": "q3.xy.y90_Square.wf.Q",
            },
        },
        "q3.xy.-y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.-y90_Square.wf.I",
                "Q": "q3.xy.-y90_Square.wf.Q",
            },
        },
        "q3.xy.saturation.pulse": {
            "operation": "control",
            "length": 20000,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.saturation.wf.I",
                "Q": "q3.xy.saturation.wf.Q",
            },
        },
        "q3.xy.EF_x180.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.EF_x180.wf.I",
                "Q": "q3.xy.EF_x180.wf.Q",
            },
        },
        "q3.xy.EF_x90.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.EF_x90.wf.I",
                "Q": "q3.xy.EF_x90.wf.Q",
            },
        },
        "q3.z.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "q3.z.const.wf",
            },
        },
        "q3.z.flux_pulse.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "q3.z.flux_pulse.wf",
            },
        },
        "q3.z.cz3_2.pulse": {
            "operation": "control",
            "length": 88,
            "waveforms": {
                "single": "q3.z.cz3_2.wf",
            },
        },
        "q3.z.cz3_4.pulse": {
            "operation": "control",
            "length": 60,
            "waveforms": {
                "single": "q3.z.cz3_4.wf",
            },
        },
        "q3.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 1040,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.resonator.readout.wf.I",
                "Q": "q3.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "q3.resonator.readout.iw1",
                "iw2": "q3.resonator.readout.iw2",
                "iw3": "q3.resonator.readout.iw3",
            },
        },
        "q3.resonator.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "I": "q3.resonator.const.wf.I",
                "Q": "q3.resonator.const.wf.Q",
            },
        },
        "q4.xy.x180_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.xy.x180_DragCosine.wf.I",
                "Q": "q4.xy.x180_DragCosine.wf.Q",
            },
        },
        "q4.xy.x90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.xy.x90_DragCosine.wf.I",
                "Q": "q4.xy.x90_DragCosine.wf.Q",
            },
        },
        "q4.xy.-x90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.xy.-x90_DragCosine.wf.I",
                "Q": "q4.xy.-x90_DragCosine.wf.Q",
            },
        },
        "q4.xy.y180_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.xy.y180_DragCosine.wf.I",
                "Q": "q4.xy.y180_DragCosine.wf.Q",
            },
        },
        "q4.xy.y90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.xy.y90_DragCosine.wf.I",
                "Q": "q4.xy.y90_DragCosine.wf.Q",
            },
        },
        "q4.xy.-y90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.xy.-y90_DragCosine.wf.I",
                "Q": "q4.xy.-y90_DragCosine.wf.Q",
            },
        },
        "q4.xy.x180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.xy.x180_Square.wf.I",
                "Q": "q4.xy.x180_Square.wf.Q",
            },
        },
        "q4.xy.x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.xy.x90_Square.wf.I",
                "Q": "q4.xy.x90_Square.wf.Q",
            },
        },
        "q4.xy.-x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.xy.-x90_Square.wf.I",
                "Q": "q4.xy.-x90_Square.wf.Q",
            },
        },
        "q4.xy.y180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.xy.y180_Square.wf.I",
                "Q": "q4.xy.y180_Square.wf.Q",
            },
        },
        "q4.xy.y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.xy.y90_Square.wf.I",
                "Q": "q4.xy.y90_Square.wf.Q",
            },
        },
        "q4.xy.-y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.xy.-y90_Square.wf.I",
                "Q": "q4.xy.-y90_Square.wf.Q",
            },
        },
        "q4.xy.saturation.pulse": {
            "operation": "control",
            "length": 20000,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.xy.saturation.wf.I",
                "Q": "q4.xy.saturation.wf.Q",
            },
        },
        "q4.xy.EF_x180.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.xy.EF_x180.wf.I",
                "Q": "q4.xy.EF_x180.wf.Q",
            },
        },
        "q4.xy.EF_x90.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.xy.EF_x90.wf.I",
                "Q": "q4.xy.EF_x90.wf.Q",
            },
        },
        "q4.z.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "q4.z.const.wf",
            },
        },
        "q4.z.flux_pulse.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "q4.z.flux_pulse.wf",
            },
        },
        "q4.z.cz.pulse": {
            "operation": "control",
            "length": 40,
            "waveforms": {
                "single": "q4.z.cz.wf",
            },
        },
        "q4.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 1040,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.resonator.readout.wf.I",
                "Q": "q4.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "q4.resonator.readout.iw1",
                "iw2": "q4.resonator.readout.iw2",
                "iw3": "q4.resonator.readout.iw3",
            },
        },
        "q4.resonator.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "I": "q4.resonator.const.wf.I",
                "Q": "q4.resonator.const.wf.Q",
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
        "q5.xy.EF_x180.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.xy.EF_x180.wf.I",
                "Q": "q5.xy.EF_x180.wf.Q",
            },
        },
        "q5.xy.EF_x90.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.xy.EF_x90.wf.I",
                "Q": "q5.xy.EF_x90.wf.Q",
            },
        },
        "q5.z.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "q5.z.const.wf",
            },
        },
        "q5.z.flux_pulse.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "q5.z.flux_pulse.wf",
            },
        },
        "q5.z.cz5_4.pulse": {
            "operation": "control",
            "length": 88,
            "waveforms": {
                "single": "q5.z.cz5_4.wf",
            },
        },
        "q5.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 1040,
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
        "coupler_q1_q2.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "coupler_q1_q2.const.wf",
            },
        },
        "coupler_q1_q2.flux_pulse.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "coupler_q1_q2.flux_pulse.wf",
            },
        },
        "coupler_q1_q2.cz.pulse": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "single": "coupler_q1_q2.cz.wf",
            },
        },
        "q1.z.SWAP_Coupler.flux_pulse_control_q2.pulse": {
            "operation": "control",
            "length": 16,
            "waveforms": {
                "single": "q1.z.SWAP_Coupler.flux_pulse_control_q2.wf",
            },
        },
        "coupler_q1_q2.SWAP_Coupler.coupler_pulse_control_q2.pulse": {
            "operation": "control",
            "length": 16,
            "waveforms": {
                "single": "coupler_q1_q2.SWAP_Coupler.coupler_pulse_control_q2.wf",
            },
        },
        "q1.z.Cz_unipolar.flux_pulse_control_q2_q1.pulse": {
            "operation": "control",
            "length": 92,
            "waveforms": {
                "single": "q1.z.Cz_unipolar.flux_pulse_control_q2_q1.wf",
            },
        },
        "coupler_q1_q2.Cz_unipolar.coupler_flux_pulse_q2_q1.pulse": {
            "operation": "control",
            "length": 92,
            "waveforms": {
                "single": "coupler_q1_q2.Cz_unipolar.coupler_flux_pulse_q2_q1.wf",
            },
        },
        "coupler_q2_q3.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "coupler_q2_q3.const.wf",
            },
        },
        "coupler_q2_q3.flux_pulse.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "coupler_q2_q3.flux_pulse.wf",
            },
        },
        "coupler_q2_q3.cz.pulse": {
            "operation": "control",
            "length": 88,
            "waveforms": {
                "single": "coupler_q2_q3.cz.wf",
            },
        },
        "q3.z.SWAP_Coupler.flux_pulse_control_q2_q3.pulse": {
            "operation": "control",
            "length": 16,
            "waveforms": {
                "single": "q3.z.SWAP_Coupler.flux_pulse_control_q2_q3.wf",
            },
        },
        "coupler_q2_q3.SWAP_Coupler.coupler_pulse_control_q2_q3.pulse": {
            "operation": "control",
            "length": 16,
            "waveforms": {
                "single": "coupler_q2_q3.SWAP_Coupler.coupler_pulse_control_q2_q3.wf",
            },
        },
        "q3.z.Cz_unipolar.flux_pulse_control_q2_q3.pulse": {
            "operation": "control",
            "length": 88,
            "waveforms": {
                "single": "q3.z.Cz_unipolar.flux_pulse_control_q2_q3.wf",
            },
        },
        "coupler_q2_q3.Cz_unipolar.coupler_flux_pulse_q2_q3.pulse": {
            "operation": "control",
            "length": 88,
            "waveforms": {
                "single": "coupler_q2_q3.Cz_unipolar.coupler_flux_pulse_q2_q3.wf",
            },
        },
        "coupler_q3_q4.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "coupler_q3_q4.const.wf",
            },
        },
        "coupler_q3_q4.flux_pulse.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "coupler_q3_q4.flux_pulse.wf",
            },
        },
        "coupler_q3_q4.cz.pulse": {
            "operation": "control",
            "length": 60,
            "waveforms": {
                "single": "coupler_q3_q4.cz.wf",
            },
        },
        "q3.z.SWAP_Coupler.flux_pulse_control_q3_q4.pulse": {
            "operation": "control",
            "length": 16,
            "waveforms": {
                "single": "q3.z.SWAP_Coupler.flux_pulse_control_q3_q4.wf",
            },
        },
        "coupler_q3_q4.SWAP_Coupler.coupler_pulse_control_q3_q4.pulse": {
            "operation": "control",
            "length": 16,
            "waveforms": {
                "single": "coupler_q3_q4.SWAP_Coupler.coupler_pulse_control_q3_q4.wf",
            },
        },
        "q3.z.Cz_unipolar.flux_pulse_control_q4_q3.pulse": {
            "operation": "control",
            "length": 88,
            "waveforms": {
                "single": "q3.z.Cz_unipolar.flux_pulse_control_q4_q3.wf",
            },
        },
        "coupler_q3_q4.Cz_unipolar.coupler_flux_pulse_q4_q3.pulse": {
            "operation": "control",
            "length": 88,
            "waveforms": {
                "single": "coupler_q3_q4.Cz_unipolar.coupler_flux_pulse_q4_q3.wf",
            },
        },
        "coupler_q4_q5.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "coupler_q4_q5.const.wf",
            },
        },
        "coupler_q4_q5.flux_pulse.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "coupler_q4_q5.flux_pulse.wf",
            },
        },
        "coupler_q4_q5.cz.pulse": {
            "operation": "control",
            "length": 88,
            "waveforms": {
                "single": "coupler_q4_q5.cz.wf",
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
            "samples": [0.0, 0.006888790537259149, 0.026364027682447142, 0.055058266006185894, 0.08801001102332943, 0.11952160666320809, 0.1441444117658276] + [0.15762091963778316] * 2 + [0.1441444117658276, 0.11952160666320814, 0.0880100110233295, 0.055058266006185914, 0.026364027682447125, 0.006888790537259149, 0.0],
        },
        "q1.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.00887372008666669, -0.016213093355132522, -0.020749075491166814, -0.0216973539753452, -0.01889396284291366, -0.012823633888678514, -0.0045359821360342915, 0.004535982136034285, 0.012823633888678508, 0.018893962842913652, 0.0216973539753452, 0.020749075491166818, 0.01621309335513252, 0.00887372008666669, 2.472086406460364e-17],
        },
        "q1.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0034304512481714193, 0.013128648807205615, 0.027417686213786697, 0.04382685908848721, 0.059518872368615805, 0.071780434400573] + [0.07849140971623932] * 2 + [0.071780434400573, 0.05951887236861583, 0.04382685908848725, 0.027417686213786707, 0.013128648807205606, 0.0034304512481714193, 0.0],
        },
        "q1.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.002611251415316288, -0.004770993738450017, -0.006105787901088318, -0.006384835866286751, -0.005559887705791325, -0.0037735844509704623, -0.001334794162638302, 0.0013347941626383002, 0.003773584450970461, 0.005559887705791323, 0.00638483586628675, 0.006105787901088318, 0.004770993738450015, 0.002611251415316288, 7.274557980877916e-18],
        },
        "q1.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.003430451248171419, -0.013128648807205615, -0.027417686213786697, -0.04382685908848721, -0.059518872368615805, -0.071780434400573] + [-0.07849140971623932] * 2 + [-0.071780434400573, -0.05951887236861583, -0.04382685908848725, -0.027417686213786707, -0.013128648807205606, -0.0034304512481714198, -8.908764146493973e-34],
        },
        "q1.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0026112514153162885, 0.004770993738450018, 0.006105787901088322, 0.006384835866286756, 0.005559887705791332, 0.003773584450970471, 0.0013347941626383115, -0.0013347941626382907, -0.0037735844509704524, -0.005559887705791316, -0.006384835866286745, -0.006105787901088315, -0.004770993738450013, -0.0026112514153162876, -7.274557980877916e-18],
        },
        "q1.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.00887372008666669, 0.016213093355132522, 0.020749075491166818, 0.021697353975345206, 0.018893962842913666, 0.012823633888678522, 0.004535982136034301, -0.004535982136034276, -0.0128236338886785, -0.018893962842913645, -0.021697353975345193, -0.020749075491166814, -0.01621309335513252, -0.00887372008666669, -2.472086406460364e-17],
        },
        "q1.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.006888790537259148, 0.026364027682447142, 0.055058266006185894, 0.08801001102332943, 0.11952160666320809, 0.1441444117658276] + [0.15762091963778316] * 2 + [0.1441444117658276, 0.11952160666320814, 0.0880100110233295, 0.055058266006185914, 0.026364027682447125, 0.00688879053725915, 1.5137163524436836e-33],
        },
        "q1.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.002611251415316288, 0.0047709937384500176, 0.00610578790108832, 0.006384835866286753, 0.0055598877057913285, 0.0037735844509704667, 0.0013347941626383067, -0.0013347941626382955, -0.0037735844509704567, -0.00555988770579132, -0.006384835866286747, -0.0061057879010883165, -0.004770993738450014, -0.002611251415316288, -7.274557980877916e-18],
        },
        "q1.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0034304512481714193, 0.013128648807205615, 0.027417686213786697, 0.04382685908848721, 0.059518872368615805, 0.071780434400573] + [0.07849140971623932] * 2 + [0.071780434400573, 0.05951887236861583, 0.04382685908848725, 0.027417686213786707, 0.013128648807205606, 0.0034304512481714193, 4.454382073246986e-34],
        },
        "q1.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.002611251415316288, -0.004770993738450016, -0.0061057879010883165, -0.006384835866286748, -0.0055598877057913216, -0.003773584450970458, -0.0013347941626382972, 0.001334794162638305, 0.0037735844509704654, 0.005559887705791327, 0.0063848358662867526, 0.00610578790108832, 0.004770993738450016, 0.002611251415316288, 7.274557980877916e-18],
        },
        "q1.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0034304512481714193, -0.013128648807205615, -0.027417686213786697, -0.04382685908848721, -0.059518872368615805, -0.071780434400573] + [-0.07849140971623932] * 2 + [-0.071780434400573, -0.05951887236861583, -0.04382685908848725, -0.027417686213786707, -0.013128648807205606, -0.0034304512481714193, 4.454382073246986e-34],
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
            "sample": -0.11201840403229255,
        },
        "q1.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q1.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.05600920201614627,
        },
        "q1.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q1.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05600920201614627,
        },
        "q1.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q1.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.2089466375000419,
        },
        "q1.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q1.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004780244659962626, 0.018294430910983556, 0.038205833177733454, 0.061071589118880436, 0.08293800180332882, 0.10002416982781459] + [0.10937573951794065] * 2 + [0.10002416982781459, 0.08293800180332886, 0.061071589118880484, 0.03820583317773347, 0.018294430910983542, 0.004780244659962626, 0.0],
        },
        "q1.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0072780850996549355, -0.013297723166252124, -0.01701806409138189, -0.01779582713085304, -0.015496529989385619, -0.010517742031198108, -0.003720340925129766, 0.003720340925129762, 0.010517742031198103, 0.015496529989385612, 0.01779582713085304, 0.017018064091381895, 0.013297723166252122, 0.0072780850996549355, 2.027566236504672e-17],
        },
        "q1.xy.EF_x90.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004593670635837815, 0.01758039515822729, 0.036714650916558844, 0.0586879512609605, 0.07970091294071842, 0.09612020398463854] + [0.10510677980665072] * 2 + [0.09612020398463854, 0.07970091294071845, 0.05868795126096055, 0.03671465091655885, 0.017580395158227277, 0.004593670635837815, 0.0],
        },
        "q1.xy.EF_x90.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0026711344749756555, -0.0048804055327331276, -0.006245810136788678, -0.006531257426789689, -0.005687390973087767, -0.003860122951814034, -0.0013654046040555497, 0.0013654046040555482, 0.0038601229518140322, 0.005687390973087765, 0.006531257426789689, 0.006245810136788678, 0.004880405532733127, 0.0026711344749756555, 7.441383276599842e-18],
        },
        "q1.z.const.wf": {
            "type": "constant",
            "sample": 0.15,
        },
        "q1.z.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q1.z.cz1_2.wf": {
            "type": "constant",
            "sample": -0.07009506167631502,
        },
        "q1.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.056010324243378824,
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
            "samples": [0.0, 0.010433661187228306, 0.03993056994280524, 0.0833904427718881, 0.13329867284177863, 0.18102567377135648, 0.21831901351370606] + [0.23873033482801967] * 2 + [0.21831901351370606, 0.18102567377135656, 0.13329867284177874, 0.08339044277188813, 0.039930569942805215, 0.010433661187228306, 0.0],
        },
        "q2.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.01328862256854909, -0.02427952157164995, -0.031072270722483617, -0.03249234198268222, -0.02829419213041503, -0.01920371941413313, -0.006792749150833666, 0.006792749150833657, 0.019203719414133125, 0.02829419213041502, 0.032492341982682214, 0.03107227072248362, 0.024279521571649944, 0.01328862256854909, 3.7020125597214523e-17],
        },
        "q2.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.005174928339850182, 0.019804921236692558, 0.04136032000935012, 0.06611400038538015, 0.08978582614764065, 0.10828272357005098] + [0.11840641104923864] * 2 + [0.10828272357005098, 0.08978582614764068, 0.06611400038538019, 0.04136032000935013, 0.019804921236692544, 0.005174928339850182, 0.0],
        },
        "q2.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.004315701371891247, -0.00788517876966638, -0.01009123712356594, -0.010552428902789722, -0.00918900985891415, -0.0062367275308984746, -0.00220605835389956, 0.0022060583538995576, 0.006236727530898473, 0.009189009858914146, 0.01055242890278972, 0.010091237123565942, 0.007885178769666379, 0.004315701371891247, 1.2022901997804962e-17],
        },
        "q2.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0051749283398501815, -0.019804921236692558, -0.04136032000935012, -0.06611400038538015, -0.08978582614764065, -0.10828272357005098] + [-0.11840641104923864] * 2 + [-0.10828272357005098, -0.08978582614764068, -0.06611400038538019, -0.04136032000935013, -0.019804921236692544, -0.005174928339850183, -1.4723808448074165e-33],
        },
        "q2.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.004315701371891248, 0.007885178769666382, 0.010091237123565945, 0.010552428902789731, 0.00918900985891416, 0.006236727530898488, 0.0022060583538995745, -0.0022060583538995432, -0.00623672753089846, -0.009189009858914136, -0.010552428902789712, -0.010091237123565936, -0.007885178769666377, -0.004315701371891246, -1.2022901997804962e-17],
        },
        "q2.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.01328862256854909, 0.024279521571649955, 0.03107227072248362, 0.03249234198268223, 0.02829419213041504, 0.019203719414133145, 0.0067927491508336805, -0.006792749150833642, -0.01920371941413311, -0.028294192130415008, -0.03249234198268221, -0.031072270722483617, -0.02427952157164994, -0.01328862256854909, -3.7020125597214523e-17],
        },
        "q2.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.010433661187228306, 0.03993056994280524, 0.0833904427718881, 0.13329867284177863, 0.18102567377135648, 0.21831901351370606] + [0.23873033482801967] * 2 + [0.21831901351370606, 0.18102567377135656, 0.13329867284177874, 0.08339044277188813, 0.039930569942805215, 0.010433661187228306, 2.2668289158330883e-33],
        },
        "q2.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004315701371891247, 0.007885178769666382, 0.010091237123565942, 0.010552428902789726, 0.009189009858914155, 0.0062367275308984815, 0.0022060583538995675, -0.00220605835389955, -0.006236727530898466, -0.009189009858914141, -0.010552428902789717, -0.01009123712356594, -0.007885178769666377, -0.004315701371891247, -1.2022901997804962e-17],
        },
        "q2.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.005174928339850182, 0.019804921236692558, 0.04136032000935012, 0.06611400038538015, 0.08978582614764065, 0.10828272357005098] + [0.11840641104923864] * 2 + [0.10828272357005098, 0.08978582614764068, 0.06611400038538019, 0.04136032000935013, 0.019804921236692544, 0.005174928339850182, 7.361904224037082e-34],
        },
        "q2.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.004315701371891247, -0.007885178769666379, -0.010091237123565938, -0.010552428902789719, -0.009189009858914144, -0.006236727530898468, -0.0022060583538995528, 0.002206058353899565, 0.00623672753089848, 0.009189009858914151, 0.010552428902789724, 0.010091237123565943, 0.00788517876966638, 0.004315701371891247, 1.2022901997804962e-17],
        },
        "q2.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.005174928339850182, -0.019804921236692558, -0.04136032000935012, -0.06611400038538015, -0.08978582614764065, -0.10828272357005098] + [-0.11840641104923864] * 2 + [-0.10828272357005098, -0.08978582614764068, -0.06611400038538019, -0.04136032000935013, -0.019804921236692544, -0.005174928339850182, 7.361904224037082e-34],
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
            "sample": -0.11201840403229255,
        },
        "q2.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q2.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.05600920201614627,
        },
        "q2.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q2.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05600920201614627,
        },
        "q2.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q2.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.24275630005786697,
        },
        "q2.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0055190851200461535, 0.021122040523614365, 0.044110973473027844, 0.07051088861343363, 0.09575700078135582, 0.1154840278292319] + [0.12628098756606948] * 2 + [0.1154840278292319, 0.09575700078135588, 0.07051088861343369, 0.04411097347302786, 0.021122040523614348, 0.0055190851200461535, 0.0],
        },
        "q2.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.008203871732241273, -0.014989219512143104, -0.01918279506561107, -0.020059491082012687, -0.01746771885557883, -0.011855619349771416, -0.00419357555346797, 0.004193575553467966, 0.011855619349771411, 0.017467718855578823, 0.020059491082012687, 0.019182795065611075, 0.014989219512143097, 0.008203871732241273, 2.2854766199004924e-17],
        },
        "q2.xy.EF_x90.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.005559686906605402, 0.021277427252099614, 0.044435480939558375, 0.07102961010210704, 0.09646144820746218, 0.11633359940621195] + [0.1272099882232664] * 2 + [0.11633359940621195, 0.09646144820746223, 0.0710296101021071, 0.044435480939558396, 0.0212774272520996, 0.005559686906605402, 0.0],
        },
        "q2.xy.EF_x90.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0036246827501982067, -0.006622624923678125, -0.008475455083195447, -0.008862802061736062, -0.007717690047773467, -0.005238119311537855, -0.0018528301595173225, 0.0018528301595173203, 0.005238119311537853, 0.007717690047773464, 0.008862802061736062, 0.008475455083195449, 0.006622624923678123, 0.0036246827501982067, 1.0097826916988402e-17],
        },
        "q2.z.const.wf": {
            "type": "constant",
            "sample": 0.15,
        },
        "q2.z.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q2.z.cz.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q2.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.056010324243378824,
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
        "q3.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.007781272983842789, 0.0297796391458189, 0.06219138112764007, 0.09941221428921737, 0.1350063184362603, 0.16281914959946048] + [0.17804161659906168] * 2 + [0.16281914959946048, 0.13500631843626035, 0.09941221428921745, 0.06219138112764009, 0.02977963914581888, 0.007781272983842789, 0.0],
        },
        "q3.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.012389634822228572, -0.022636989227395012, -0.028970202544553833, -0.030294204655731537, -0.026380063567723915, -0.017904569833502963, -0.006333213317158819, 0.006333213317158812, 0.017904569833502956, 0.026380063567723905, 0.03029420465573153, 0.028970202544553833, 0.022636989227395005, 0.012389634822228572, 3.451567947366297e-17],
        },
        "q3.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.003859386399616804, 0.014770222628548728, 0.030845925980978097, 0.049306861304089726, 0.06696096516818531, 0.08075568263266411] + [0.08830578172684382] * 2 + [0.08075568263266411, 0.06696096516818534, 0.04930686130408977, 0.030845925980978108, 0.014770222628548718, 0.003859386399616804, 0.0],
        },
        "q3.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0066889491438483295, -0.012221318213530013, -0.015640510336901926, -0.016355316133447793, -0.014242139187138025, -0.009666366989599462, -0.003419192123371915, 0.003419192123371911, 0.009666366989599459, 0.014242139187138018, 0.01635531613344779, 0.01564051033690193, 0.01222131821353001, 0.0066889491438483295, 1.8634417234839303e-17],
        },
        "q3.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.003859386399616803, -0.014770222628548726, -0.030845925980978094, -0.049306861304089726, -0.06696096516818531, -0.08075568263266411] + [-0.08830578172684382] * 2 + [-0.08075568263266411, -0.06696096516818534, -0.04930686130408977, -0.03084592598097811, -0.01477022262854872, -0.0038593863996168047, -2.2820579420622225e-33],
        },
        "q3.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.00668894914384833, 0.012221318213530015, 0.01564051033690193, 0.0163553161334478, 0.014242139187138033, 0.009666366989599473, 0.0034191921233719257, -0.0034191921233719, -0.009666366989599448, -0.014242139187138009, -0.016355316133447782, -0.015640510336901926, -0.012221318213530008, -0.006688949143848329, -1.8634417234839303e-17],
        },
        "q3.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.012389634822228572, 0.022636989227395016, 0.028970202544553837, 0.030294204655731544, 0.026380063567723922, 0.017904569833502974, 0.00633321331715883, -0.006333213317158801, -0.017904569833502946, -0.026380063567723898, -0.030294204655731523, -0.02897020254455383, -0.022636989227395002, -0.012389634822228572, -3.451567947366297e-17],
        },
        "q3.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.007781272983842788, 0.0297796391458189, 0.06219138112764007, 0.09941221428921737, 0.1350063184362603, 0.16281914959946048] + [0.17804161659906168] * 2 + [0.16281914959946048, 0.13500631843626035, 0.09941221428921745, 0.06219138112764009, 0.02977963914581888, 0.00778127298384279, 2.113475819390868e-33],
        },
        "q3.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0066889491438483295, 0.012221318213530015, 0.01564051033690193, 0.016355316133447796, 0.014242139187138028, 0.009666366989599467, 0.00341919212337192, -0.0034191921233719057, -0.009666366989599454, -0.014242139187138014, -0.016355316133447786, -0.015640510336901926, -0.012221318213530008, -0.0066889491438483295, -1.8634417234839303e-17],
        },
        "q3.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0038593863996168034, 0.014770222628548728, 0.030845925980978097, 0.049306861304089726, 0.06696096516818531, 0.08075568263266411] + [0.08830578172684382] * 2 + [0.08075568263266411, 0.06696096516818534, 0.04930686130408977, 0.030845925980978108, 0.014770222628548718, 0.0038593863996168043, 1.1410289710311113e-33],
        },
        "q3.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0066889491438483295, -0.012221318213530011, -0.015640510336901923, -0.01635531613344779, -0.014242139187138021, -0.009666366989599457, -0.0034191921233719096, 0.003419192123371916, 0.009666366989599464, 0.014242139187138021, 0.016355316133447793, 0.015640510336901933, 0.012221318213530011, 0.0066889491438483295, 1.8634417234839303e-17],
        },
        "q3.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0038593863996168043, -0.014770222628548728, -0.030845925980978097, -0.049306861304089726, -0.06696096516818531, -0.08075568263266411] + [-0.08830578172684382] * 2 + [-0.08075568263266411, -0.06696096516818534, -0.04930686130408977, -0.030845925980978108, -0.014770222628548718, -0.0038593863996168034, 1.1410289710311113e-33],
        },
        "q3.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q3.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q3.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05,
        },
        "q3.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q3.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "q3.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q3.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229255,
        },
        "q3.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q3.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.05600920201614627,
        },
        "q3.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q3.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05600920201614627,
        },
        "q3.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q3.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.1274923951483565,
        },
        "q3.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q3.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0039058085470254585, 0.01494788440716442, 0.031216952350088895, 0.049899942728624934, 0.06776639677668451, 0.08172704227771864] + [0.08936795679611573] * 2 + [0.08172704227771864, 0.06776639677668454, 0.049899942728624976, 0.031216952350088905, 0.01494788440716441, 0.0039058085470254585, 0.0],
        },
        "q3.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
        },
        "q3.xy.EF_x90.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0038310543317542738, 0.014661793229020795, 0.030619483542288252, 0.048944895645327356, 0.06646939930438014, 0.08016284863167868] + [0.08765752183745126] * 2 + [0.08016284863167868, 0.06646939930438017, 0.0489448956453274, 0.030619483542288262, 0.014661793229020784, 0.0038310543317542738, 0.0],
        },
        "q3.xy.EF_x90.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0063272797074245945, -0.011560515271903889, -0.014794832721886832, -0.01547098918741978, -0.013472071072923433, -0.009143709479995186, -0.0032343174499829446, 0.0032343174499829407, 0.009143709479995182, 0.013472071072923426, 0.01547098918741978, 0.014794832721886836, 0.011560515271903885, 0.0063272797074245945, 1.7626859988630127e-17],
        },
        "q3.z.const.wf": {
            "type": "constant",
            "sample": 0.15,
        },
        "q3.z.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q3.z.cz3_2.wf": {
            "type": "constant",
            "sample": -0.07709397834356477,
        },
        "q3.z.cz3_4.wf": {
            "type": "constant",
            "sample": -0.23305216356732317,
        },
        "q3.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.09613408379227202,
        },
        "q3.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q3.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "q3.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q4.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.009874576618942933, 0.03779090247064414, 0.07892199120348593, 0.12615590391134945, 0.1713254679804194, 0.2066204554305464] + [0.22593804228670875] * 2 + [0.2066204554305464, 0.17132546798041945, 0.12615590391134957, 0.07892199120348596, 0.03779090247064412, 0.009874576618942933, 0.0],
        },
        "q4.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0059785901638416345, -0.010923427774568516, -0.01397950548684655, -0.01461839970062845, -0.012729639802179603, -0.008639809536786816, -0.0030560777122780342, 0.0030560777122780308, 0.008639809536786815, 0.012729639802179598, 0.014618399700628448, 0.013979505486846552, 0.010923427774568515, 0.0059785901638416345, 1.6655462792924684e-17],
        },
        "q4.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004917459842365154, 0.018819565888995474, 0.039302517707760054, 0.06282462684741899, 0.08531870694607228, 0.10289532720236044] + [0.11251533029538907] * 2 + [0.10289532720236044, 0.08531870694607233, 0.06282462684741905, 0.03930251770776007, 0.01881956588899546, 0.004917459842365154, 0.0],
        },
        "q4.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0004533402092001452, -0.0008282937777630782, -0.0010600278273380347, -0.0011084734355157558, -0.0009652539165277795, -0.0006551332263156106, -0.00023173404957495662, 0.00023173404957495632, 0.0006551332263156105, 0.0009652539165277791, 0.0011084734355157556, 0.001060027827338035, 0.000828293777763078, 0.0004533402092001452, 1.262938381783635e-18],
        },
        "q4.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.004917459842365154, -0.018819565888995474, -0.039302517707760054, -0.06282462684741899, -0.08531870694607228, -0.10289532720236044] + [-0.11251533029538907] * 2 + [-0.10289532720236044, -0.08531870694607233, -0.06282462684741905, -0.03930251770776007, -0.01881956588899546, -0.004917459842365154, -1.5466534467716666e-34],
        },
        "q4.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0004533402092001458, 0.0008282937777630805, 0.0010600278273380395, 0.0011084734355157634, 0.0009652539165277899, 0.0006551332263156232, 0.0002317340495749704, -0.00023173404957494255, -0.0006551332263155979, -0.0009652539165277687, -0.001108473435515748, -0.0010600278273380302, -0.0008282937777630758, -0.0004533402092001446, -1.262938381783635e-18],
        },
        "q4.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.005978590163841635, 0.010923427774568518, 0.013979505486846556, 0.014618399700628457, 0.012729639802179614, 0.008639809536786829, 0.003056077712278048, -0.003056077712278017, -0.008639809536786803, -0.012729639802179588, -0.014618399700628441, -0.013979505486846547, -0.010923427774568513, -0.005978590163841634, -1.6655462792924684e-17],
        },
        "q4.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.009874576618942933, 0.03779090247064414, 0.07892199120348593, 0.12615590391134945, 0.1713254679804194, 0.2066204554305464] + [0.22593804228670875] * 2 + [0.2066204554305464, 0.17132546798041945, 0.12615590391134957, 0.07892199120348596, 0.03779090247064412, 0.009874576618942933, 1.0198529598836525e-33],
        },
        "q4.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.00045334020920014554, 0.0008282937777630794, 0.0010600278273380371, 0.0011084734355157598, 0.0009652539165277847, 0.0006551332263156169, 0.0002317340495749635, -0.00023173404957494944, -0.0006551332263156042, -0.0009652539165277739, -0.0011084734355157517, -0.0010600278273380326, -0.0008282937777630768, -0.0004533402092001449, -1.262938381783635e-18],
        },
        "q4.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.004917459842365154, 0.018819565888995474, 0.039302517707760054, 0.06282462684741899, 0.08531870694607228, 0.10289532720236044] + [0.11251533029538907] * 2 + [0.10289532720236044, 0.08531870694607233, 0.06282462684741905, 0.03930251770776007, 0.01881956588899546, 0.004917459842365154, 7.733267233858333e-35],
        },
        "q4.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0004533402092001449, -0.000828293777763077, -0.0010600278273380323, -0.001108473435515752, -0.0009652539165277743, -0.0006551332263156043, -0.00023173404957494974, 0.0002317340495749632, 0.0006551332263156168, 0.0009652539165277843, 0.0011084734355157595, 0.0010600278273380373, 0.0008282937777630792, 0.00045334020920014554, 1.262938381783635e-18],
        },
        "q4.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.004917459842365154, -0.018819565888995474, -0.039302517707760054, -0.06282462684741899, -0.08531870694607228, -0.10289532720236044] + [-0.11251533029538907] * 2 + [-0.10289532720236044, -0.08531870694607233, -0.06282462684741905, -0.03930251770776007, -0.01881956588899546, -0.004917459842365154, 7.733267233858333e-35],
        },
        "q4.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q4.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q4.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05,
        },
        "q4.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q4.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "q4.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q4.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229255,
        },
        "q4.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q4.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.05600920201614627,
        },
        "q4.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q4.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05600920201614627,
        },
        "q4.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q4.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.1785114534008873,
        },
        "q4.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q4.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.007119076296882763, 0.02724535222102223, 0.056898811823657976, 0.090952102546271, 0.12351710105848734, 0.14896302327382097] + [0.16289003807229452] * 2 + [0.14896302327382097, 0.1235171010584874, 0.09095210254627106, 0.05689881182365799, 0.02724535222102221, 0.007119076296882763, 0.0],
        },
        "q4.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
        },
        "q4.xy.EF_x90.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.01296818135360987, 0.04963040904617126, 0.10364745084375789, 0.165679269490148, 0.22499999999999995, 0.27135254915624213] + [0.29672214011007086] * 2 + [0.27135254915624213, 0.22500000000000006, 0.16567926949014813, 0.10364745084375791, 0.04963040904617123, 0.01296818135360987, 0.0],
        },
        "q4.xy.EF_x90.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.00017685586774239802, -0.0003231317492670167, -0.00041353521578358374, -0.0004324346866416748, -0.0003765622716336233, -0.0002555788188992768, -9.040346651656707e-05, 9.040346651656696e-05, 0.0002555788188992767, 0.00037656227163362314, 0.00043243468664167477, 0.0004135352157835838, 0.0003231317492670166, 0.00017685586774239802, 4.926941376093873e-19],
        },
        "q4.z.const.wf": {
            "type": "constant",
            "sample": 0.15,
        },
        "q4.z.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q4.z.cz.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q4.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.056010324243378824,
        },
        "q4.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q4.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "q4.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q5.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.007400940096490705, 0.028324070607849552, 0.059151592187989224, 0.09455314629646207, 0.12840748261489066, 0.15486087883682856] + [0.16933930243394257] * 2 + [0.15486087883682856, 0.1284074826148907, 0.09455314629646214, 0.059151592187989245, 0.028324070607849534, 0.007400940096490705, 0.0],
        },
        "q5.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.00887630031746843, -0.016217807671389722, -0.02075510874477039, -0.0217036629619367, -0.01889945668139607, -0.012827362644468267, -0.0045373010733806615, 0.004537301073380656, 0.012827362644468264, 0.01889945668139606, 0.021703662961936696, 0.02075510874477039, 0.01621780767138972, 0.00887630031746843, 2.4728052203769877e-17],
        },
        "q5.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.003685309583474932, 0.014104014826930397, 0.029454626942807247, 0.047082885640337316, 0.063940704843015, 0.07711321446221778] + [0.0843227679162924] * 2 + [0.07711321446221778, 0.06394070484301503, 0.04708288564033735, 0.029454626942807257, 0.014104014826930387, 0.003685309583474932, 0.0],
        },
        "q5.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0022535285797811557, -0.004117401595453714, -0.005269338469853118, -0.005510158852377748, -0.004798222711104596, -0.003256630272596592, -0.0011519368743994056, 0.0011519368743994043, 0.00325663027259659, 0.004798222711104594, 0.005510158852377746, 0.00526933846985312, 0.004117401595453712, 0.0022535285797811557, 6.277995377629245e-18],
        },
        "q5.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0036853095834749315, -0.014104014826930397, -0.029454626942807247, -0.047082885640337316, -0.063940704843015, -0.07711321446221778] + [-0.0843227679162924] * 2 + [-0.07711321446221778, -0.06394070484301503, -0.04708288564033735, -0.029454626942807257, -0.014104014826930387, -0.0036853095834749323, -7.688326944275534e-34],
        },
        "q5.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.002253528579781156, 0.0041174015954537155, 0.005269338469853122, 0.005510158852377754, 0.004798222711104604, 0.0032566302725966015, 0.001151936874399416, -0.001151936874399394, -0.0032566302725965807, -0.004798222711104586, -0.00551015885237774, -0.005269338469853117, -0.00411740159545371, -0.0022535285797811552, -6.277995377629245e-18],
        },
        "q5.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.00887630031746843, 0.016217807671389722, 0.020755108744770392, 0.021703662961936706, 0.018899456681396077, 0.012827362644468276, 0.004537301073380672, -0.004537301073380646, -0.012827362644468255, -0.018899456681396053, -0.02170366296193669, -0.020755108744770386, -0.01621780767138972, -0.00887630031746843, -2.4728052203769877e-17],
        },
        "q5.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.007400940096490704, 0.028324070607849552, 0.059151592187989224, 0.09455314629646207, 0.12840748261489066, 0.15486087883682856] + [0.16933930243394257] * 2 + [0.15486087883682856, 0.1284074826148907, 0.09455314629646214, 0.059151592187989245, 0.028324070607849534, 0.007400940096490706, 1.5141564990247717e-33],
        },
        "q5.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.002253528579781156, 0.004117401595453715, 0.00526933846985312, 0.00551015885237775, 0.0047982227111046, 0.0032566302725965967, 0.0011519368743994108, -0.0011519368743993991, -0.0032566302725965854, -0.00479822271110459, -0.005510158852377743, -0.005269338469853118, -0.004117401595453711, -0.0022535285797811552, -6.277995377629245e-18],
        },
        "q5.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.003685309583474932, 0.014104014826930397, 0.029454626942807247, 0.047082885640337316, 0.063940704843015, 0.07711321446221778] + [0.0843227679162924] * 2 + [0.07711321446221778, 0.06394070484301503, 0.04708288564033735, 0.029454626942807257, 0.014104014826930387, 0.003685309583474932, 3.844163472137767e-34],
        },
        "q5.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0022535285797811552, -0.004117401595453713, -0.005269338469853117, -0.005510158852377745, -0.0047982227111045914, -0.003256630272596587, -0.0011519368743994004, 0.0011519368743994095, 0.003256630272596595, 0.004798222711104598, 0.0055101588523777485, 0.005269338469853122, 0.004117401595453713, 0.002253528579781156, 6.277995377629245e-18],
        },
        "q5.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.003685309583474932, -0.014104014826930397, -0.029454626942807247, -0.047082885640337316, -0.063940704843015, -0.07711321446221778] + [-0.0843227679162924] * 2 + [-0.07711321446221778, -0.06394070484301503, -0.04708288564033735, -0.029454626942807257, -0.014104014826930387, -0.003685309583474932, 3.844163472137767e-34],
        },
        "q5.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q5.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q5.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05,
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
            "sample": -0.11201840403229255,
        },
        "q5.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q5.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.05600920201614627,
        },
        "q5.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q5.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05600920201614627,
        },
        "q5.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q5.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.10287614322831423,
        },
        "q5.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q5.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.01296818135360987, 0.04963040904617126, 0.10364745084375789, 0.165679269490148, 0.22499999999999995, 0.27135254915624213] + [0.29672214011007086] * 2 + [0.27135254915624213, 0.22500000000000006, 0.16567926949014813, 0.10364745084375791, 0.04963040904617123, 0.01296818135360987, 0.0],
        },
        "q5.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
        },
        "q5.xy.EF_x90.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004618343160591265, 0.017674819153568368, 0.03691184466538491, 0.05900316322215746, 0.08012898515209145, 0.09663646392143418] + [0.10567130648522964] * 2 + [0.09663646392143418, 0.08012898515209148, 0.059003163222157505, 0.03691184466538492, 0.017674819153568357, 0.004618343160591265, 0.0],
        },
        "q5.xy.EF_x90.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0008065355175237464, -0.001473613716922486, -0.0018858907173049866, -0.001972080079886148, -0.0017172788808699087, -0.0011655445623624013, -0.0004122770003825005, 0.00041227700038250003, 0.001165544562362401, 0.001717278880869908, 0.001972080079886148, 0.0018858907173049866, 0.0014736137169224855, 0.0008065355175237464, 2.246887967757484e-18],
        },
        "q5.z.const.wf": {
            "type": "constant",
            "sample": 0.15,
        },
        "q5.z.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q5.z.cz5_4.wf": {
            "type": "constant",
            "sample": -0.1732650364352627,
        },
        "q5.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.056010324243378824,
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
        "coupler_q1_q2.const.wf": {
            "type": "constant",
            "sample": 0.15,
        },
        "coupler_q1_q2.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "coupler_q1_q2.cz.wf": {
            "type": "constant",
            "sample": -0.04662292425886875,
        },
        "q1.z.SWAP_Coupler.flux_pulse_control_q2.wf": {
            "type": "arbitrary",
            "samples": [0.1379194010900913] * 15 + [0.0],
        },
        "coupler_q1_q2.SWAP_Coupler.coupler_pulse_control_q2.wf": {
            "type": "arbitrary",
            "samples": [0.04600000000000054] * 15 + [0.0],
        },
        "q1.z.Cz_unipolar.flux_pulse_control_q2_q1.wf": {
            "type": "arbitrary",
            "samples": [0.07534407924424193] * 88 + [0.0] * 4,
        },
        "coupler_q1_q2.Cz_unipolar.coupler_flux_pulse_q2_q1.wf": {
            "type": "arbitrary",
            "samples": [0.1641] * 88 + [0.0] * 4,
        },
        "coupler_q2_q3.const.wf": {
            "type": "constant",
            "sample": 0.15,
        },
        "coupler_q2_q3.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "coupler_q2_q3.cz.wf": {
            "type": "constant",
            "sample": -0.10043328056510244,
        },
        "q3.z.SWAP_Coupler.flux_pulse_control_q2_q3.wf": {
            "type": "arbitrary",
            "samples": [0.1379194010900913] * 15 + [0.0],
        },
        "coupler_q2_q3.SWAP_Coupler.coupler_pulse_control_q2_q3.wf": {
            "type": "arbitrary",
            "samples": [0.04600000000000054] * 15 + [0.0],
        },
        "q3.z.Cz_unipolar.flux_pulse_control_q2_q3.wf": {
            "type": "arbitrary",
            "samples": [0.07410126789587144] * 86 + [0.0] * 2,
        },
        "coupler_q2_q3.Cz_unipolar.coupler_flux_pulse_q2_q3.wf": {
            "type": "arbitrary",
            "samples": [0.1795] * 86 + [0.0] * 2,
        },
        "coupler_q3_q4.const.wf": {
            "type": "constant",
            "sample": 0.15,
        },
        "coupler_q3_q4.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "coupler_q3_q4.cz.wf": {
            "type": "constant",
            "sample": -0.11048567109,
        },
        "q3.z.SWAP_Coupler.flux_pulse_control_q3_q4.wf": {
            "type": "arbitrary",
            "samples": [0.1379194010900913] * 15 + [0.0],
        },
        "coupler_q3_q4.SWAP_Coupler.coupler_pulse_control_q3_q4.wf": {
            "type": "arbitrary",
            "samples": [0.04600000000000054] * 15 + [0.0],
        },
        "q3.z.Cz_unipolar.flux_pulse_control_q4_q3.wf": {
            "type": "arbitrary",
            "samples": [0.13154871783227662] * 84 + [0.0] * 4,
        },
        "coupler_q3_q4.Cz_unipolar.coupler_flux_pulse_q4_q3.wf": {
            "type": "arbitrary",
            "samples": [0.158] * 84 + [0.0] * 4,
        },
        "coupler_q4_q5.const.wf": {
            "type": "constant",
            "sample": 0.15,
        },
        "coupler_q4_q5.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "coupler_q4_q5.cz.wf": {
            "type": "constant",
            "sample": -0.09938541439500001,
        },
    },
    "digital_waveforms": {
        "ON": {
            "samples": [[1, 0]],
        },
    },
    "integration_weights": {
        "q1.resonator.readout.iw1": {
            "cosine": [(-0.3970892049209327, 1040)],
            "sine": [(0.9177800190324812, 1040)],
        },
        "q1.resonator.readout.iw2": {
            "cosine": [(-0.9177800190324812, 1040)],
            "sine": [(-0.3970892049209327, 1040)],
        },
        "q1.resonator.readout.iw3": {
            "cosine": [(0.9177800190324812, 1040)],
            "sine": [(0.3970892049209327, 1040)],
        },
        "q2.resonator.readout.iw1": {
            "cosine": [(0.9413034281447746, 1040)],
            "sine": [(0.3375616331322253, 1040)],
        },
        "q2.resonator.readout.iw2": {
            "cosine": [(-0.3375616331322253, 1040)],
            "sine": [(0.9413034281447746, 1040)],
        },
        "q2.resonator.readout.iw3": {
            "cosine": [(0.3375616331322253, 1040)],
            "sine": [(-0.9413034281447746, 1040)],
        },
        "q3.resonator.readout.iw1": {
            "cosine": [(0.7913782651308222, 1040)],
            "sine": [(0.6113267877972713, 1040)],
        },
        "q3.resonator.readout.iw2": {
            "cosine": [(-0.6113267877972713, 1040)],
            "sine": [(0.7913782651308222, 1040)],
        },
        "q3.resonator.readout.iw3": {
            "cosine": [(0.6113267877972713, 1040)],
            "sine": [(-0.7913782651308222, 1040)],
        },
        "q4.resonator.readout.iw1": {
            "cosine": [(-0.999408409473025, 1040)],
            "sine": [(-0.034392311271538724, 1040)],
        },
        "q4.resonator.readout.iw2": {
            "cosine": [(0.034392311271538724, 1040)],
            "sine": [(-0.999408409473025, 1040)],
        },
        "q4.resonator.readout.iw3": {
            "cosine": [(-0.034392311271538724, 1040)],
            "sine": [(0.999408409473025, 1040)],
        },
        "q5.resonator.readout.iw1": {
            "cosine": [(0.8065831195035316, 1040)],
            "sine": [(0.591120691003074, 1040)],
        },
        "q5.resonator.readout.iw2": {
            "cosine": [(-0.591120691003074, 1040)],
            "sine": [(0.8065831195035316, 1040)],
        },
        "q5.resonator.readout.iw3": {
            "cosine": [(0.591120691003074, 1040)],
            "sine": [(-0.8065831195035316, 1040)],
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
                        "7": {
                            "offset": 0.0,
                            "delay": 168,
                            "shareable": False,
                            "filter": {
                                "feedforward": [],
                                "feedback": [],
                            },
                            "crosstalk": {},
                            "output_mode": "amplified",
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                        },
                        "1": {
                            "offset": 0.0,
                            "delay": 180,
                            "shareable": False,
                            "filter": {
                                "feedforward": [],
                                "feedback": [],
                            },
                            "crosstalk": {},
                            "output_mode": "amplified",
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                        },
                        "2": {
                            "offset": 0.0,
                            "delay": 172,
                            "shareable": False,
                            "filter": {
                                "feedforward": [],
                                "feedback": [],
                            },
                            "crosstalk": {},
                            "output_mode": "amplified",
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                        },
                        "3": {
                            "offset": 0.0,
                            "delay": 176,
                            "shareable": False,
                            "filter": {
                                "feedforward": [],
                                "feedback": [],
                            },
                            "crosstalk": {},
                            "output_mode": "amplified",
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                        },
                        "4": {
                            "offset": 0.0,
                            "delay": 177,
                            "shareable": False,
                            "filter": {
                                "feedforward": [],
                                "feedback": [],
                            },
                            "crosstalk": {},
                            "output_mode": "amplified",
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                        },
                        "5": {
                            "offset": 0.0,
                            "delay": 172,
                            "shareable": False,
                            "filter": {
                                "feedforward": [],
                                "feedback": [],
                            },
                            "crosstalk": {},
                            "output_mode": "amplified",
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                        },
                        "6": {
                            "offset": 0.0,
                            "delay": 172,
                            "shareable": False,
                            "filter": {
                                "feedforward": [],
                                "feedback": [],
                            },
                            "crosstalk": {},
                            "output_mode": "amplified",
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                        },
                        "8": {
                            "offset": 0.0,
                            "delay": 172,
                            "shareable": False,
                            "filter": {
                                "feedforward": [],
                                "feedback": [],
                            },
                            "crosstalk": {},
                            "output_mode": "amplified",
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                        },
                    },
                },
                "2": {
                    "type": "LF",
                    "analog_outputs": {
                        "8": {
                            "offset": 0.0,
                            "delay": 172,
                            "shareable": False,
                            "filter": {
                                "feedforward": [],
                                "feedback": [],
                            },
                            "crosstalk": {},
                            "output_mode": "amplified",
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
                            "full_scale_power_dbm": -8,
                            "band": 2,
                            "delay": 0,
                            "shareable": False,
                            "upconverters": {
                                "1": {
                                    "frequency": 5950000000.0,
                                },
                            },
                        },
                        "2": {
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 4,
                            "band": 1,
                            "delay": 20,
                            "shareable": False,
                            "upconverters": {
                                "1": {
                                    "frequency": 4900000000.0,
                                },
                            },
                        },
                        "3": {
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 4,
                            "band": 1,
                            "delay": 20,
                            "shareable": False,
                            "upconverters": {
                                "1": {
                                    "frequency": 4900000000.0,
                                },
                            },
                        },
                        "4": {
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 4,
                            "band": 1,
                            "delay": 20,
                            "shareable": False,
                            "upconverters": {
                                "1": {
                                    "frequency": 5000000000.0,
                                },
                            },
                        },
                        "5": {
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 4,
                            "band": 1,
                            "delay": 20,
                            "shareable": False,
                            "upconverters": {
                                "1": {
                                    "frequency": 5000000000.0,
                                },
                            },
                        },
                        "6": {
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 4,
                            "band": 1,
                            "delay": 20,
                            "shareable": False,
                            "upconverters": {
                                "1": {
                                    "frequency": 4900000000.0,
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
                            "downconverter_frequency": 5950000000.0,
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
                "EF_x90": "q1.xy.EF_x90.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "a",
            "MWInput": {
                "port": ('con1', 6, 2),
                "upconverter": 1,
            },
            "intermediate_frequency": 213409441.43405086,
        },
        "q1.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "q1.z.const.pulse",
                "flux_pulse": "q1.z.flux_pulse.pulse",
                "cz1_2": "q1.z.cz1_2.pulse",
                "SWAP_Coupler.flux_pulse_control_q2": "q1.z.SWAP_Coupler.flux_pulse_control_q2.pulse",
                "Cz_unipolar.flux_pulse_control_q2_q1": "q1.z.Cz_unipolar.flux_pulse_control_q2_q1.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "",
            "singleInput": {
                "port": ('con1', 1, 7),
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
            "thread": "a",
            "MWInput": {
                "port": ('con1', 6, 1),
                "upconverter": 1,
            },
            "MWOutput": {
                "port": ('con1', 6, 1),
            },
            "smearing": 0,
            "time_of_flight": 384,
            "intermediate_frequency": -16582781.0,
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
                "EF_x180": "q2.xy.EF_x180.pulse",
                "EF_x90": "q2.xy.EF_x90.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "b",
            "MWInput": {
                "port": ('con1', 6, 3),
                "upconverter": 1,
            },
            "intermediate_frequency": -62236279.16832265,
        },
        "q2.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "q2.z.const.pulse",
                "flux_pulse": "q2.z.flux_pulse.pulse",
                "cz": "q2.z.cz.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "",
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
            "thread": "b",
            "MWInput": {
                "port": ('con1', 6, 1),
                "upconverter": 1,
            },
            "MWOutput": {
                "port": ('con1', 6, 1),
            },
            "smearing": 0,
            "time_of_flight": 376,
            "intermediate_frequency": 74083957.0,
        },
        "q3.xy": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "x180_DragCosine": "q3.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "q3.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "q3.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "q3.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "q3.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "q3.xy.-y90_DragCosine.pulse",
                "x180_Square": "q3.xy.x180_Square.pulse",
                "x90_Square": "q3.xy.x90_Square.pulse",
                "-x90_Square": "q3.xy.-x90_Square.pulse",
                "y180_Square": "q3.xy.y180_Square.pulse",
                "y90_Square": "q3.xy.y90_Square.pulse",
                "-y90_Square": "q3.xy.-y90_Square.pulse",
                "x180": "q3.xy.x180_DragCosine.pulse",
                "x90": "q3.xy.x90_DragCosine.pulse",
                "-x90": "q3.xy.-x90_DragCosine.pulse",
                "y180": "q3.xy.y180_DragCosine.pulse",
                "y90": "q3.xy.y90_DragCosine.pulse",
                "-y90": "q3.xy.-y90_DragCosine.pulse",
                "saturation": "q3.xy.saturation.pulse",
                "EF_x180": "q3.xy.EF_x180.pulse",
                "EF_x90": "q3.xy.EF_x90.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "c",
            "MWInput": {
                "port": ('con1', 6, 4),
                "upconverter": 1,
            },
            "intermediate_frequency": 145743108.35321197,
        },
        "q3.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "q3.z.const.pulse",
                "flux_pulse": "q3.z.flux_pulse.pulse",
                "cz3_2": "q3.z.cz3_2.pulse",
                "cz3_4": "q3.z.cz3_4.pulse",
                "SWAP_Coupler.flux_pulse_control_q2_q3": "q3.z.SWAP_Coupler.flux_pulse_control_q2_q3.pulse",
                "Cz_unipolar.flux_pulse_control_q2_q3": "q3.z.Cz_unipolar.flux_pulse_control_q2_q3.pulse",
                "SWAP_Coupler.flux_pulse_control_q3_q4": "q3.z.SWAP_Coupler.flux_pulse_control_q3_q4.pulse",
                "Cz_unipolar.flux_pulse_control_q4_q3": "q3.z.Cz_unipolar.flux_pulse_control_q4_q3.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "",
            "singleInput": {
                "port": ('con1', 1, 2),
            },
        },
        "q3.resonator": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "readout": "q3.resonator.readout.pulse",
                "const": "q3.resonator.const.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "c",
            "MWInput": {
                "port": ('con1', 6, 1),
                "upconverter": 1,
            },
            "MWOutput": {
                "port": ('con1', 6, 1),
            },
            "smearing": 0,
            "time_of_flight": 384,
            "intermediate_frequency": -89526496.0,
        },
        "q4.xy": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "x180_DragCosine": "q4.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "q4.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "q4.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "q4.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "q4.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "q4.xy.-y90_DragCosine.pulse",
                "x180_Square": "q4.xy.x180_Square.pulse",
                "x90_Square": "q4.xy.x90_Square.pulse",
                "-x90_Square": "q4.xy.-x90_Square.pulse",
                "y180_Square": "q4.xy.y180_Square.pulse",
                "y90_Square": "q4.xy.y90_Square.pulse",
                "-y90_Square": "q4.xy.-y90_Square.pulse",
                "x180": "q4.xy.x180_DragCosine.pulse",
                "x90": "q4.xy.x90_DragCosine.pulse",
                "-x90": "q4.xy.-x90_DragCosine.pulse",
                "y180": "q4.xy.y180_DragCosine.pulse",
                "y90": "q4.xy.y90_DragCosine.pulse",
                "-y90": "q4.xy.-y90_DragCosine.pulse",
                "saturation": "q4.xy.saturation.pulse",
                "EF_x180": "q4.xy.EF_x180.pulse",
                "EF_x90": "q4.xy.EF_x90.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "d",
            "MWInput": {
                "port": ('con1', 6, 5),
                "upconverter": 1,
            },
            "intermediate_frequency": -323280460.768938,
        },
        "q4.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "q4.z.const.pulse",
                "flux_pulse": "q4.z.flux_pulse.pulse",
                "cz": "q4.z.cz.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "",
            "singleInput": {
                "port": ('con1', 1, 3),
            },
        },
        "q4.resonator": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "readout": "q4.resonator.readout.pulse",
                "const": "q4.resonator.const.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "d",
            "MWInput": {
                "port": ('con1', 6, 1),
                "upconverter": 1,
            },
            "MWOutput": {
                "port": ('con1', 6, 1),
            },
            "smearing": 0,
            "time_of_flight": 384,
            "intermediate_frequency": 129555949.0,
        },
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
                "EF_x180": "q5.xy.EF_x180.pulse",
                "EF_x90": "q5.xy.EF_x90.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "e",
            "MWInput": {
                "port": ('con1', 6, 6),
                "upconverter": 1,
            },
            "intermediate_frequency": -14023978.739179434,
        },
        "q5.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "q5.z.const.pulse",
                "flux_pulse": "q5.z.flux_pulse.pulse",
                "cz5_4": "q5.z.cz5_4.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "",
            "singleInput": {
                "port": ('con1', 1, 4),
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
            "thread": "e",
            "MWInput": {
                "port": ('con1', 6, 1),
                "upconverter": 1,
            },
            "MWOutput": {
                "port": ('con1', 6, 1),
            },
            "smearing": 0,
            "time_of_flight": 384,
            "intermediate_frequency": 19883495.0,
        },
        "coupler_q1_q2": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "coupler_q1_q2.const.pulse",
                "flux_pulse": "coupler_q1_q2.flux_pulse.pulse",
                "cz": "coupler_q1_q2.cz.pulse",
                "SWAP_Coupler.coupler_pulse_control_q2": "coupler_q1_q2.SWAP_Coupler.coupler_pulse_control_q2.pulse",
                "Cz_unipolar.coupler_flux_pulse_q2_q1": "coupler_q1_q2.Cz_unipolar.coupler_flux_pulse_q2_q1.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "",
            "singleInput": {
                "port": ('con1', 1, 5),
            },
        },
        "coupler_q2_q3": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "coupler_q2_q3.const.pulse",
                "flux_pulse": "coupler_q2_q3.flux_pulse.pulse",
                "cz": "coupler_q2_q3.cz.pulse",
                "SWAP_Coupler.coupler_pulse_control_q2_q3": "coupler_q2_q3.SWAP_Coupler.coupler_pulse_control_q2_q3.pulse",
                "Cz_unipolar.coupler_flux_pulse_q2_q3": "coupler_q2_q3.Cz_unipolar.coupler_flux_pulse_q2_q3.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "",
            "singleInput": {
                "port": ('con1', 1, 6),
            },
        },
        "coupler_q3_q4": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "coupler_q3_q4.const.pulse",
                "flux_pulse": "coupler_q3_q4.flux_pulse.pulse",
                "cz": "coupler_q3_q4.cz.pulse",
                "SWAP_Coupler.coupler_pulse_control_q3_q4": "coupler_q3_q4.SWAP_Coupler.coupler_pulse_control_q3_q4.pulse",
                "Cz_unipolar.coupler_flux_pulse_q4_q3": "coupler_q3_q4.Cz_unipolar.coupler_flux_pulse_q4_q3.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "",
            "singleInput": {
                "port": ('con1', 2, 8),
            },
        },
        "coupler_q4_q5": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "coupler_q4_q5.const.pulse",
                "flux_pulse": "coupler_q4_q5.flux_pulse.pulse",
                "cz": "coupler_q4_q5.cz.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "",
            "singleInput": {
                "port": ('con1', 1, 8),
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
        "q1.xy.EF_x90.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q1.xy.EF_x90.wf.I",
                "Q": "q1.xy.EF_x90.wf.Q",
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
        "q1.z.flux_pulse.pulse": {
            "length": 100,
            "waveforms": {
                "single": "q1.z.flux_pulse.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q1.z.cz1_2.pulse": {
            "length": 80,
            "waveforms": {
                "single": "q1.z.cz1_2.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q1.resonator.readout.pulse": {
            "length": 1040,
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
        "q2.xy.EF_x180.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q2.xy.EF_x180.wf.I",
                "Q": "q2.xy.EF_x180.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.EF_x90.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q2.xy.EF_x90.wf.I",
                "Q": "q2.xy.EF_x90.wf.Q",
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
        "q2.z.flux_pulse.pulse": {
            "length": 100,
            "waveforms": {
                "single": "q2.z.flux_pulse.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q2.z.cz.pulse": {
            "length": 40,
            "waveforms": {
                "single": "q2.z.cz.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q2.resonator.readout.pulse": {
            "length": 1040,
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
        "q3.xy.x180_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q3.xy.x180_DragCosine.wf.I",
                "Q": "q3.xy.x180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q3.xy.x90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q3.xy.x90_DragCosine.wf.I",
                "Q": "q3.xy.x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q3.xy.-x90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q3.xy.-x90_DragCosine.wf.I",
                "Q": "q3.xy.-x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q3.xy.y180_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q3.xy.y180_DragCosine.wf.I",
                "Q": "q3.xy.y180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q3.xy.y90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q3.xy.y90_DragCosine.wf.I",
                "Q": "q3.xy.y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q3.xy.-y90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q3.xy.-y90_DragCosine.wf.I",
                "Q": "q3.xy.-y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q3.xy.x180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q3.xy.x180_Square.wf.I",
                "Q": "q3.xy.x180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q3.xy.x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q3.xy.x90_Square.wf.I",
                "Q": "q3.xy.x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q3.xy.-x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q3.xy.-x90_Square.wf.I",
                "Q": "q3.xy.-x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q3.xy.y180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q3.xy.y180_Square.wf.I",
                "Q": "q3.xy.y180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q3.xy.y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q3.xy.y90_Square.wf.I",
                "Q": "q3.xy.y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q3.xy.-y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q3.xy.-y90_Square.wf.I",
                "Q": "q3.xy.-y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q3.xy.saturation.pulse": {
            "length": 20000,
            "waveforms": {
                "I": "q3.xy.saturation.wf.I",
                "Q": "q3.xy.saturation.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q3.xy.EF_x180.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q3.xy.EF_x180.wf.I",
                "Q": "q3.xy.EF_x180.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q3.xy.EF_x90.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q3.xy.EF_x90.wf.I",
                "Q": "q3.xy.EF_x90.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q3.z.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "q3.z.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q3.z.flux_pulse.pulse": {
            "length": 100,
            "waveforms": {
                "single": "q3.z.flux_pulse.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q3.z.cz3_2.pulse": {
            "length": 88,
            "waveforms": {
                "single": "q3.z.cz3_2.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q3.z.cz3_4.pulse": {
            "length": 60,
            "waveforms": {
                "single": "q3.z.cz3_4.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q3.resonator.readout.pulse": {
            "length": 1040,
            "waveforms": {
                "I": "q3.resonator.readout.wf.I",
                "Q": "q3.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "q3.resonator.readout.iw1",
                "iw2": "q3.resonator.readout.iw2",
                "iw3": "q3.resonator.readout.iw3",
            },
            "operation": "measurement",
            "digital_marker": "ON",
        },
        "q3.resonator.const.pulse": {
            "length": 100,
            "waveforms": {
                "I": "q3.resonator.const.wf.I",
                "Q": "q3.resonator.const.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q4.xy.x180_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q4.xy.x180_DragCosine.wf.I",
                "Q": "q4.xy.x180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q4.xy.x90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q4.xy.x90_DragCosine.wf.I",
                "Q": "q4.xy.x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q4.xy.-x90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q4.xy.-x90_DragCosine.wf.I",
                "Q": "q4.xy.-x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q4.xy.y180_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q4.xy.y180_DragCosine.wf.I",
                "Q": "q4.xy.y180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q4.xy.y90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q4.xy.y90_DragCosine.wf.I",
                "Q": "q4.xy.y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q4.xy.-y90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q4.xy.-y90_DragCosine.wf.I",
                "Q": "q4.xy.-y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q4.xy.x180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q4.xy.x180_Square.wf.I",
                "Q": "q4.xy.x180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q4.xy.x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q4.xy.x90_Square.wf.I",
                "Q": "q4.xy.x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q4.xy.-x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q4.xy.-x90_Square.wf.I",
                "Q": "q4.xy.-x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q4.xy.y180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q4.xy.y180_Square.wf.I",
                "Q": "q4.xy.y180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q4.xy.y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q4.xy.y90_Square.wf.I",
                "Q": "q4.xy.y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q4.xy.-y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q4.xy.-y90_Square.wf.I",
                "Q": "q4.xy.-y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q4.xy.saturation.pulse": {
            "length": 20000,
            "waveforms": {
                "I": "q4.xy.saturation.wf.I",
                "Q": "q4.xy.saturation.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q4.xy.EF_x180.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q4.xy.EF_x180.wf.I",
                "Q": "q4.xy.EF_x180.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q4.xy.EF_x90.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q4.xy.EF_x90.wf.I",
                "Q": "q4.xy.EF_x90.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q4.z.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "q4.z.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q4.z.flux_pulse.pulse": {
            "length": 100,
            "waveforms": {
                "single": "q4.z.flux_pulse.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q4.z.cz.pulse": {
            "length": 40,
            "waveforms": {
                "single": "q4.z.cz.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q4.resonator.readout.pulse": {
            "length": 1040,
            "waveforms": {
                "I": "q4.resonator.readout.wf.I",
                "Q": "q4.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "q4.resonator.readout.iw1",
                "iw2": "q4.resonator.readout.iw2",
                "iw3": "q4.resonator.readout.iw3",
            },
            "operation": "measurement",
            "digital_marker": "ON",
        },
        "q4.resonator.const.pulse": {
            "length": 100,
            "waveforms": {
                "I": "q4.resonator.const.wf.I",
                "Q": "q4.resonator.const.wf.Q",
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
        "q5.xy.EF_x180.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q5.xy.EF_x180.wf.I",
                "Q": "q5.xy.EF_x180.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q5.xy.EF_x90.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q5.xy.EF_x90.wf.I",
                "Q": "q5.xy.EF_x90.wf.Q",
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
        "q5.z.flux_pulse.pulse": {
            "length": 100,
            "waveforms": {
                "single": "q5.z.flux_pulse.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q5.z.cz5_4.pulse": {
            "length": 88,
            "waveforms": {
                "single": "q5.z.cz5_4.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q5.resonator.readout.pulse": {
            "length": 1040,
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
        "coupler_q1_q2.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "coupler_q1_q2.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q1_q2.flux_pulse.pulse": {
            "length": 100,
            "waveforms": {
                "single": "coupler_q1_q2.flux_pulse.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q1_q2.cz.pulse": {
            "length": 80,
            "waveforms": {
                "single": "coupler_q1_q2.cz.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q1.z.SWAP_Coupler.flux_pulse_control_q2.pulse": {
            "length": 16,
            "waveforms": {
                "single": "q1.z.SWAP_Coupler.flux_pulse_control_q2.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q1_q2.SWAP_Coupler.coupler_pulse_control_q2.pulse": {
            "length": 16,
            "waveforms": {
                "single": "coupler_q1_q2.SWAP_Coupler.coupler_pulse_control_q2.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q1.z.Cz_unipolar.flux_pulse_control_q2_q1.pulse": {
            "length": 92,
            "waveforms": {
                "single": "q1.z.Cz_unipolar.flux_pulse_control_q2_q1.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q1_q2.Cz_unipolar.coupler_flux_pulse_q2_q1.pulse": {
            "length": 92,
            "waveforms": {
                "single": "coupler_q1_q2.Cz_unipolar.coupler_flux_pulse_q2_q1.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q2_q3.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "coupler_q2_q3.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q2_q3.flux_pulse.pulse": {
            "length": 100,
            "waveforms": {
                "single": "coupler_q2_q3.flux_pulse.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q2_q3.cz.pulse": {
            "length": 88,
            "waveforms": {
                "single": "coupler_q2_q3.cz.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q3.z.SWAP_Coupler.flux_pulse_control_q2_q3.pulse": {
            "length": 16,
            "waveforms": {
                "single": "q3.z.SWAP_Coupler.flux_pulse_control_q2_q3.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q2_q3.SWAP_Coupler.coupler_pulse_control_q2_q3.pulse": {
            "length": 16,
            "waveforms": {
                "single": "coupler_q2_q3.SWAP_Coupler.coupler_pulse_control_q2_q3.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q3.z.Cz_unipolar.flux_pulse_control_q2_q3.pulse": {
            "length": 88,
            "waveforms": {
                "single": "q3.z.Cz_unipolar.flux_pulse_control_q2_q3.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q2_q3.Cz_unipolar.coupler_flux_pulse_q2_q3.pulse": {
            "length": 88,
            "waveforms": {
                "single": "coupler_q2_q3.Cz_unipolar.coupler_flux_pulse_q2_q3.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q3_q4.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "coupler_q3_q4.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q3_q4.flux_pulse.pulse": {
            "length": 100,
            "waveforms": {
                "single": "coupler_q3_q4.flux_pulse.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q3_q4.cz.pulse": {
            "length": 60,
            "waveforms": {
                "single": "coupler_q3_q4.cz.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q3.z.SWAP_Coupler.flux_pulse_control_q3_q4.pulse": {
            "length": 16,
            "waveforms": {
                "single": "q3.z.SWAP_Coupler.flux_pulse_control_q3_q4.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q3_q4.SWAP_Coupler.coupler_pulse_control_q3_q4.pulse": {
            "length": 16,
            "waveforms": {
                "single": "coupler_q3_q4.SWAP_Coupler.coupler_pulse_control_q3_q4.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q3.z.Cz_unipolar.flux_pulse_control_q4_q3.pulse": {
            "length": 88,
            "waveforms": {
                "single": "q3.z.Cz_unipolar.flux_pulse_control_q4_q3.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q3_q4.Cz_unipolar.coupler_flux_pulse_q4_q3.pulse": {
            "length": 88,
            "waveforms": {
                "single": "coupler_q3_q4.Cz_unipolar.coupler_flux_pulse_q4_q3.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q4_q5.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "coupler_q4_q5.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q4_q5.flux_pulse.pulse": {
            "length": 100,
            "waveforms": {
                "single": "coupler_q4_q5.flux_pulse.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q4_q5.cz.pulse": {
            "length": 88,
            "waveforms": {
                "single": "coupler_q4_q5.cz.wf",
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
            "samples": [0.0, 0.006888790537259149, 0.026364027682447142, 0.055058266006185894, 0.08801001102332943, 0.11952160666320809, 0.1441444117658276] + [0.15762091963778316] * 2 + [0.1441444117658276, 0.11952160666320814, 0.0880100110233295, 0.055058266006185914, 0.026364027682447125, 0.006888790537259149, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.00887372008666669, -0.016213093355132522, -0.020749075491166814, -0.0216973539753452, -0.01889396284291366, -0.012823633888678514, -0.0045359821360342915, 0.004535982136034285, 0.012823633888678508, 0.018893962842913652, 0.0216973539753452, 0.020749075491166818, 0.01621309335513252, 0.00887372008666669, 2.472086406460364e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0034304512481714193, 0.013128648807205615, 0.027417686213786697, 0.04382685908848721, 0.059518872368615805, 0.071780434400573] + [0.07849140971623932] * 2 + [0.071780434400573, 0.05951887236861583, 0.04382685908848725, 0.027417686213786707, 0.013128648807205606, 0.0034304512481714193, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.002611251415316288, -0.004770993738450017, -0.006105787901088318, -0.006384835866286751, -0.005559887705791325, -0.0037735844509704623, -0.001334794162638302, 0.0013347941626383002, 0.003773584450970461, 0.005559887705791323, 0.00638483586628675, 0.006105787901088318, 0.004770993738450015, 0.002611251415316288, 7.274557980877916e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.003430451248171419, -0.013128648807205615, -0.027417686213786697, -0.04382685908848721, -0.059518872368615805, -0.071780434400573] + [-0.07849140971623932] * 2 + [-0.071780434400573, -0.05951887236861583, -0.04382685908848725, -0.027417686213786707, -0.013128648807205606, -0.0034304512481714198, -8.908764146493973e-34],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0026112514153162885, 0.004770993738450018, 0.006105787901088322, 0.006384835866286756, 0.005559887705791332, 0.003773584450970471, 0.0013347941626383115, -0.0013347941626382907, -0.0037735844509704524, -0.005559887705791316, -0.006384835866286745, -0.006105787901088315, -0.004770993738450013, -0.0026112514153162876, -7.274557980877916e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.00887372008666669, 0.016213093355132522, 0.020749075491166818, 0.021697353975345206, 0.018893962842913666, 0.012823633888678522, 0.004535982136034301, -0.004535982136034276, -0.0128236338886785, -0.018893962842913645, -0.021697353975345193, -0.020749075491166814, -0.01621309335513252, -0.00887372008666669, -2.472086406460364e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.006888790537259148, 0.026364027682447142, 0.055058266006185894, 0.08801001102332943, 0.11952160666320809, 0.1441444117658276] + [0.15762091963778316] * 2 + [0.1441444117658276, 0.11952160666320814, 0.0880100110233295, 0.055058266006185914, 0.026364027682447125, 0.00688879053725915, 1.5137163524436836e-33],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.002611251415316288, 0.0047709937384500176, 0.00610578790108832, 0.006384835866286753, 0.0055598877057913285, 0.0037735844509704667, 0.0013347941626383067, -0.0013347941626382955, -0.0037735844509704567, -0.00555988770579132, -0.006384835866286747, -0.0061057879010883165, -0.004770993738450014, -0.002611251415316288, -7.274557980877916e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0034304512481714193, 0.013128648807205615, 0.027417686213786697, 0.04382685908848721, 0.059518872368615805, 0.071780434400573] + [0.07849140971623932] * 2 + [0.071780434400573, 0.05951887236861583, 0.04382685908848725, 0.027417686213786707, 0.013128648807205606, 0.0034304512481714193, 4.454382073246986e-34],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.002611251415316288, -0.004770993738450016, -0.0061057879010883165, -0.006384835866286748, -0.0055598877057913216, -0.003773584450970458, -0.0013347941626382972, 0.001334794162638305, 0.0037735844509704654, 0.005559887705791327, 0.0063848358662867526, 0.00610578790108832, 0.004770993738450016, 0.002611251415316288, 7.274557980877916e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0034304512481714193, -0.013128648807205615, -0.027417686213786697, -0.04382685908848721, -0.059518872368615805, -0.071780434400573] + [-0.07849140971623932] * 2 + [-0.071780434400573, -0.05951887236861583, -0.04382685908848725, -0.027417686213786707, -0.013128648807205606, -0.0034304512481714193, 4.454382073246986e-34],
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
            "sample": -0.11201840403229255,
        },
        "q1.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q1.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.05600920201614627,
        },
        "q1.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q1.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05600920201614627,
        },
        "q1.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q1.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.2089466375000419,
        },
        "q1.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q1.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004780244659962626, 0.018294430910983556, 0.038205833177733454, 0.061071589118880436, 0.08293800180332882, 0.10002416982781459] + [0.10937573951794065] * 2 + [0.10002416982781459, 0.08293800180332886, 0.061071589118880484, 0.03820583317773347, 0.018294430910983542, 0.004780244659962626, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0072780850996549355, -0.013297723166252124, -0.01701806409138189, -0.01779582713085304, -0.015496529989385619, -0.010517742031198108, -0.003720340925129766, 0.003720340925129762, 0.010517742031198103, 0.015496529989385612, 0.01779582713085304, 0.017018064091381895, 0.013297723166252122, 0.0072780850996549355, 2.027566236504672e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.EF_x90.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004593670635837815, 0.01758039515822729, 0.036714650916558844, 0.0586879512609605, 0.07970091294071842, 0.09612020398463854] + [0.10510677980665072] * 2 + [0.09612020398463854, 0.07970091294071845, 0.05868795126096055, 0.03671465091655885, 0.017580395158227277, 0.004593670635837815, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.EF_x90.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0026711344749756555, -0.0048804055327331276, -0.006245810136788678, -0.006531257426789689, -0.005687390973087767, -0.003860122951814034, -0.0013654046040555497, 0.0013654046040555482, 0.0038601229518140322, 0.005687390973087765, 0.006531257426789689, 0.006245810136788678, 0.004880405532733127, 0.0026711344749756555, 7.441383276599842e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.z.const.wf": {
            "type": "constant",
            "sample": 0.15,
        },
        "q1.z.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q1.z.cz1_2.wf": {
            "type": "constant",
            "sample": -0.07009506167631502,
        },
        "q1.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.056010324243378824,
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
            "samples": [0.0, 0.010433661187228306, 0.03993056994280524, 0.0833904427718881, 0.13329867284177863, 0.18102567377135648, 0.21831901351370606] + [0.23873033482801967] * 2 + [0.21831901351370606, 0.18102567377135656, 0.13329867284177874, 0.08339044277188813, 0.039930569942805215, 0.010433661187228306, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.01328862256854909, -0.02427952157164995, -0.031072270722483617, -0.03249234198268222, -0.02829419213041503, -0.01920371941413313, -0.006792749150833666, 0.006792749150833657, 0.019203719414133125, 0.02829419213041502, 0.032492341982682214, 0.03107227072248362, 0.024279521571649944, 0.01328862256854909, 3.7020125597214523e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.005174928339850182, 0.019804921236692558, 0.04136032000935012, 0.06611400038538015, 0.08978582614764065, 0.10828272357005098] + [0.11840641104923864] * 2 + [0.10828272357005098, 0.08978582614764068, 0.06611400038538019, 0.04136032000935013, 0.019804921236692544, 0.005174928339850182, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.004315701371891247, -0.00788517876966638, -0.01009123712356594, -0.010552428902789722, -0.00918900985891415, -0.0062367275308984746, -0.00220605835389956, 0.0022060583538995576, 0.006236727530898473, 0.009189009858914146, 0.01055242890278972, 0.010091237123565942, 0.007885178769666379, 0.004315701371891247, 1.2022901997804962e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0051749283398501815, -0.019804921236692558, -0.04136032000935012, -0.06611400038538015, -0.08978582614764065, -0.10828272357005098] + [-0.11840641104923864] * 2 + [-0.10828272357005098, -0.08978582614764068, -0.06611400038538019, -0.04136032000935013, -0.019804921236692544, -0.005174928339850183, -1.4723808448074165e-33],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.004315701371891248, 0.007885178769666382, 0.010091237123565945, 0.010552428902789731, 0.00918900985891416, 0.006236727530898488, 0.0022060583538995745, -0.0022060583538995432, -0.00623672753089846, -0.009189009858914136, -0.010552428902789712, -0.010091237123565936, -0.007885178769666377, -0.004315701371891246, -1.2022901997804962e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.01328862256854909, 0.024279521571649955, 0.03107227072248362, 0.03249234198268223, 0.02829419213041504, 0.019203719414133145, 0.0067927491508336805, -0.006792749150833642, -0.01920371941413311, -0.028294192130415008, -0.03249234198268221, -0.031072270722483617, -0.02427952157164994, -0.01328862256854909, -3.7020125597214523e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.010433661187228306, 0.03993056994280524, 0.0833904427718881, 0.13329867284177863, 0.18102567377135648, 0.21831901351370606] + [0.23873033482801967] * 2 + [0.21831901351370606, 0.18102567377135656, 0.13329867284177874, 0.08339044277188813, 0.039930569942805215, 0.010433661187228306, 2.2668289158330883e-33],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004315701371891247, 0.007885178769666382, 0.010091237123565942, 0.010552428902789726, 0.009189009858914155, 0.0062367275308984815, 0.0022060583538995675, -0.00220605835389955, -0.006236727530898466, -0.009189009858914141, -0.010552428902789717, -0.01009123712356594, -0.007885178769666377, -0.004315701371891247, -1.2022901997804962e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.005174928339850182, 0.019804921236692558, 0.04136032000935012, 0.06611400038538015, 0.08978582614764065, 0.10828272357005098] + [0.11840641104923864] * 2 + [0.10828272357005098, 0.08978582614764068, 0.06611400038538019, 0.04136032000935013, 0.019804921236692544, 0.005174928339850182, 7.361904224037082e-34],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.004315701371891247, -0.007885178769666379, -0.010091237123565938, -0.010552428902789719, -0.009189009858914144, -0.006236727530898468, -0.0022060583538995528, 0.002206058353899565, 0.00623672753089848, 0.009189009858914151, 0.010552428902789724, 0.010091237123565943, 0.00788517876966638, 0.004315701371891247, 1.2022901997804962e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.005174928339850182, -0.019804921236692558, -0.04136032000935012, -0.06611400038538015, -0.08978582614764065, -0.10828272357005098] + [-0.11840641104923864] * 2 + [-0.10828272357005098, -0.08978582614764068, -0.06611400038538019, -0.04136032000935013, -0.019804921236692544, -0.005174928339850182, 7.361904224037082e-34],
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
            "sample": -0.11201840403229255,
        },
        "q2.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q2.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.05600920201614627,
        },
        "q2.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q2.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05600920201614627,
        },
        "q2.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q2.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.24275630005786697,
        },
        "q2.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0055190851200461535, 0.021122040523614365, 0.044110973473027844, 0.07051088861343363, 0.09575700078135582, 0.1154840278292319] + [0.12628098756606948] * 2 + [0.1154840278292319, 0.09575700078135588, 0.07051088861343369, 0.04411097347302786, 0.021122040523614348, 0.0055190851200461535, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.008203871732241273, -0.014989219512143104, -0.01918279506561107, -0.020059491082012687, -0.01746771885557883, -0.011855619349771416, -0.00419357555346797, 0.004193575553467966, 0.011855619349771411, 0.017467718855578823, 0.020059491082012687, 0.019182795065611075, 0.014989219512143097, 0.008203871732241273, 2.2854766199004924e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.EF_x90.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.005559686906605402, 0.021277427252099614, 0.044435480939558375, 0.07102961010210704, 0.09646144820746218, 0.11633359940621195] + [0.1272099882232664] * 2 + [0.11633359940621195, 0.09646144820746223, 0.0710296101021071, 0.044435480939558396, 0.0212774272520996, 0.005559686906605402, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.EF_x90.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0036246827501982067, -0.006622624923678125, -0.008475455083195447, -0.008862802061736062, -0.007717690047773467, -0.005238119311537855, -0.0018528301595173225, 0.0018528301595173203, 0.005238119311537853, 0.007717690047773464, 0.008862802061736062, 0.008475455083195449, 0.006622624923678123, 0.0036246827501982067, 1.0097826916988402e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.z.const.wf": {
            "type": "constant",
            "sample": 0.15,
        },
        "q2.z.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q2.z.cz.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q2.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.056010324243378824,
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
        "q3.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.007781272983842789, 0.0297796391458189, 0.06219138112764007, 0.09941221428921737, 0.1350063184362603, 0.16281914959946048] + [0.17804161659906168] * 2 + [0.16281914959946048, 0.13500631843626035, 0.09941221428921745, 0.06219138112764009, 0.02977963914581888, 0.007781272983842789, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.012389634822228572, -0.022636989227395012, -0.028970202544553833, -0.030294204655731537, -0.026380063567723915, -0.017904569833502963, -0.006333213317158819, 0.006333213317158812, 0.017904569833502956, 0.026380063567723905, 0.03029420465573153, 0.028970202544553833, 0.022636989227395005, 0.012389634822228572, 3.451567947366297e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.003859386399616804, 0.014770222628548728, 0.030845925980978097, 0.049306861304089726, 0.06696096516818531, 0.08075568263266411] + [0.08830578172684382] * 2 + [0.08075568263266411, 0.06696096516818534, 0.04930686130408977, 0.030845925980978108, 0.014770222628548718, 0.003859386399616804, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0066889491438483295, -0.012221318213530013, -0.015640510336901926, -0.016355316133447793, -0.014242139187138025, -0.009666366989599462, -0.003419192123371915, 0.003419192123371911, 0.009666366989599459, 0.014242139187138018, 0.01635531613344779, 0.01564051033690193, 0.01222131821353001, 0.0066889491438483295, 1.8634417234839303e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.003859386399616803, -0.014770222628548726, -0.030845925980978094, -0.049306861304089726, -0.06696096516818531, -0.08075568263266411] + [-0.08830578172684382] * 2 + [-0.08075568263266411, -0.06696096516818534, -0.04930686130408977, -0.03084592598097811, -0.01477022262854872, -0.0038593863996168047, -2.2820579420622225e-33],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.00668894914384833, 0.012221318213530015, 0.01564051033690193, 0.0163553161334478, 0.014242139187138033, 0.009666366989599473, 0.0034191921233719257, -0.0034191921233719, -0.009666366989599448, -0.014242139187138009, -0.016355316133447782, -0.015640510336901926, -0.012221318213530008, -0.006688949143848329, -1.8634417234839303e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.012389634822228572, 0.022636989227395016, 0.028970202544553837, 0.030294204655731544, 0.026380063567723922, 0.017904569833502974, 0.00633321331715883, -0.006333213317158801, -0.017904569833502946, -0.026380063567723898, -0.030294204655731523, -0.02897020254455383, -0.022636989227395002, -0.012389634822228572, -3.451567947366297e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.007781272983842788, 0.0297796391458189, 0.06219138112764007, 0.09941221428921737, 0.1350063184362603, 0.16281914959946048] + [0.17804161659906168] * 2 + [0.16281914959946048, 0.13500631843626035, 0.09941221428921745, 0.06219138112764009, 0.02977963914581888, 0.00778127298384279, 2.113475819390868e-33],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0066889491438483295, 0.012221318213530015, 0.01564051033690193, 0.016355316133447796, 0.014242139187138028, 0.009666366989599467, 0.00341919212337192, -0.0034191921233719057, -0.009666366989599454, -0.014242139187138014, -0.016355316133447786, -0.015640510336901926, -0.012221318213530008, -0.0066889491438483295, -1.8634417234839303e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0038593863996168034, 0.014770222628548728, 0.030845925980978097, 0.049306861304089726, 0.06696096516818531, 0.08075568263266411] + [0.08830578172684382] * 2 + [0.08075568263266411, 0.06696096516818534, 0.04930686130408977, 0.030845925980978108, 0.014770222628548718, 0.0038593863996168043, 1.1410289710311113e-33],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0066889491438483295, -0.012221318213530011, -0.015640510336901923, -0.01635531613344779, -0.014242139187138021, -0.009666366989599457, -0.0034191921233719096, 0.003419192123371916, 0.009666366989599464, 0.014242139187138021, 0.016355316133447793, 0.015640510336901933, 0.012221318213530011, 0.0066889491438483295, 1.8634417234839303e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0038593863996168043, -0.014770222628548728, -0.030845925980978097, -0.049306861304089726, -0.06696096516818531, -0.08075568263266411] + [-0.08830578172684382] * 2 + [-0.08075568263266411, -0.06696096516818534, -0.04930686130408977, -0.030845925980978108, -0.014770222628548718, -0.0038593863996168034, 1.1410289710311113e-33],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q3.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q3.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05,
        },
        "q3.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q3.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "q3.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q3.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229255,
        },
        "q3.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q3.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.05600920201614627,
        },
        "q3.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q3.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05600920201614627,
        },
        "q3.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q3.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.1274923951483565,
        },
        "q3.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q3.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0039058085470254585, 0.01494788440716442, 0.031216952350088895, 0.049899942728624934, 0.06776639677668451, 0.08172704227771864] + [0.08936795679611573] * 2 + [0.08172704227771864, 0.06776639677668454, 0.049899942728624976, 0.031216952350088905, 0.01494788440716441, 0.0039058085470254585, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.EF_x90.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0038310543317542738, 0.014661793229020795, 0.030619483542288252, 0.048944895645327356, 0.06646939930438014, 0.08016284863167868] + [0.08765752183745126] * 2 + [0.08016284863167868, 0.06646939930438017, 0.0489448956453274, 0.030619483542288262, 0.014661793229020784, 0.0038310543317542738, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.EF_x90.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0063272797074245945, -0.011560515271903889, -0.014794832721886832, -0.01547098918741978, -0.013472071072923433, -0.009143709479995186, -0.0032343174499829446, 0.0032343174499829407, 0.009143709479995182, 0.013472071072923426, 0.01547098918741978, 0.014794832721886836, 0.011560515271903885, 0.0063272797074245945, 1.7626859988630127e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.z.const.wf": {
            "type": "constant",
            "sample": 0.15,
        },
        "q3.z.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q3.z.cz3_2.wf": {
            "type": "constant",
            "sample": -0.07709397834356477,
        },
        "q3.z.cz3_4.wf": {
            "type": "constant",
            "sample": -0.23305216356732317,
        },
        "q3.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.09613408379227202,
        },
        "q3.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q3.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "q3.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q4.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.009874576618942933, 0.03779090247064414, 0.07892199120348593, 0.12615590391134945, 0.1713254679804194, 0.2066204554305464] + [0.22593804228670875] * 2 + [0.2066204554305464, 0.17132546798041945, 0.12615590391134957, 0.07892199120348596, 0.03779090247064412, 0.009874576618942933, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0059785901638416345, -0.010923427774568516, -0.01397950548684655, -0.01461839970062845, -0.012729639802179603, -0.008639809536786816, -0.0030560777122780342, 0.0030560777122780308, 0.008639809536786815, 0.012729639802179598, 0.014618399700628448, 0.013979505486846552, 0.010923427774568515, 0.0059785901638416345, 1.6655462792924684e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004917459842365154, 0.018819565888995474, 0.039302517707760054, 0.06282462684741899, 0.08531870694607228, 0.10289532720236044] + [0.11251533029538907] * 2 + [0.10289532720236044, 0.08531870694607233, 0.06282462684741905, 0.03930251770776007, 0.01881956588899546, 0.004917459842365154, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0004533402092001452, -0.0008282937777630782, -0.0010600278273380347, -0.0011084734355157558, -0.0009652539165277795, -0.0006551332263156106, -0.00023173404957495662, 0.00023173404957495632, 0.0006551332263156105, 0.0009652539165277791, 0.0011084734355157556, 0.001060027827338035, 0.000828293777763078, 0.0004533402092001452, 1.262938381783635e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.004917459842365154, -0.018819565888995474, -0.039302517707760054, -0.06282462684741899, -0.08531870694607228, -0.10289532720236044] + [-0.11251533029538907] * 2 + [-0.10289532720236044, -0.08531870694607233, -0.06282462684741905, -0.03930251770776007, -0.01881956588899546, -0.004917459842365154, -1.5466534467716666e-34],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0004533402092001458, 0.0008282937777630805, 0.0010600278273380395, 0.0011084734355157634, 0.0009652539165277899, 0.0006551332263156232, 0.0002317340495749704, -0.00023173404957494255, -0.0006551332263155979, -0.0009652539165277687, -0.001108473435515748, -0.0010600278273380302, -0.0008282937777630758, -0.0004533402092001446, -1.262938381783635e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.005978590163841635, 0.010923427774568518, 0.013979505486846556, 0.014618399700628457, 0.012729639802179614, 0.008639809536786829, 0.003056077712278048, -0.003056077712278017, -0.008639809536786803, -0.012729639802179588, -0.014618399700628441, -0.013979505486846547, -0.010923427774568513, -0.005978590163841634, -1.6655462792924684e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.009874576618942933, 0.03779090247064414, 0.07892199120348593, 0.12615590391134945, 0.1713254679804194, 0.2066204554305464] + [0.22593804228670875] * 2 + [0.2066204554305464, 0.17132546798041945, 0.12615590391134957, 0.07892199120348596, 0.03779090247064412, 0.009874576618942933, 1.0198529598836525e-33],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.00045334020920014554, 0.0008282937777630794, 0.0010600278273380371, 0.0011084734355157598, 0.0009652539165277847, 0.0006551332263156169, 0.0002317340495749635, -0.00023173404957494944, -0.0006551332263156042, -0.0009652539165277739, -0.0011084734355157517, -0.0010600278273380326, -0.0008282937777630768, -0.0004533402092001449, -1.262938381783635e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.004917459842365154, 0.018819565888995474, 0.039302517707760054, 0.06282462684741899, 0.08531870694607228, 0.10289532720236044] + [0.11251533029538907] * 2 + [0.10289532720236044, 0.08531870694607233, 0.06282462684741905, 0.03930251770776007, 0.01881956588899546, 0.004917459842365154, 7.733267233858333e-35],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0004533402092001449, -0.000828293777763077, -0.0010600278273380323, -0.001108473435515752, -0.0009652539165277743, -0.0006551332263156043, -0.00023173404957494974, 0.0002317340495749632, 0.0006551332263156168, 0.0009652539165277843, 0.0011084734355157595, 0.0010600278273380373, 0.0008282937777630792, 0.00045334020920014554, 1.262938381783635e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.004917459842365154, -0.018819565888995474, -0.039302517707760054, -0.06282462684741899, -0.08531870694607228, -0.10289532720236044] + [-0.11251533029538907] * 2 + [-0.10289532720236044, -0.08531870694607233, -0.06282462684741905, -0.03930251770776007, -0.01881956588899546, -0.004917459842365154, 7.733267233858333e-35],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q4.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q4.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05,
        },
        "q4.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q4.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "q4.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q4.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229255,
        },
        "q4.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q4.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.05600920201614627,
        },
        "q4.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q4.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05600920201614627,
        },
        "q4.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q4.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.1785114534008873,
        },
        "q4.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q4.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.007119076296882763, 0.02724535222102223, 0.056898811823657976, 0.090952102546271, 0.12351710105848734, 0.14896302327382097] + [0.16289003807229452] * 2 + [0.14896302327382097, 0.1235171010584874, 0.09095210254627106, 0.05689881182365799, 0.02724535222102221, 0.007119076296882763, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.EF_x90.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.01296818135360987, 0.04963040904617126, 0.10364745084375789, 0.165679269490148, 0.22499999999999995, 0.27135254915624213] + [0.29672214011007086] * 2 + [0.27135254915624213, 0.22500000000000006, 0.16567926949014813, 0.10364745084375791, 0.04963040904617123, 0.01296818135360987, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.EF_x90.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.00017685586774239802, -0.0003231317492670167, -0.00041353521578358374, -0.0004324346866416748, -0.0003765622716336233, -0.0002555788188992768, -9.040346651656707e-05, 9.040346651656696e-05, 0.0002555788188992767, 0.00037656227163362314, 0.00043243468664167477, 0.0004135352157835838, 0.0003231317492670166, 0.00017685586774239802, 4.926941376093873e-19],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.z.const.wf": {
            "type": "constant",
            "sample": 0.15,
        },
        "q4.z.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q4.z.cz.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q4.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.056010324243378824,
        },
        "q4.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q4.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "q4.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q5.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.007400940096490705, 0.028324070607849552, 0.059151592187989224, 0.09455314629646207, 0.12840748261489066, 0.15486087883682856] + [0.16933930243394257] * 2 + [0.15486087883682856, 0.1284074826148907, 0.09455314629646214, 0.059151592187989245, 0.028324070607849534, 0.007400940096490705, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.00887630031746843, -0.016217807671389722, -0.02075510874477039, -0.0217036629619367, -0.01889945668139607, -0.012827362644468267, -0.0045373010733806615, 0.004537301073380656, 0.012827362644468264, 0.01889945668139606, 0.021703662961936696, 0.02075510874477039, 0.01621780767138972, 0.00887630031746843, 2.4728052203769877e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.003685309583474932, 0.014104014826930397, 0.029454626942807247, 0.047082885640337316, 0.063940704843015, 0.07711321446221778] + [0.0843227679162924] * 2 + [0.07711321446221778, 0.06394070484301503, 0.04708288564033735, 0.029454626942807257, 0.014104014826930387, 0.003685309583474932, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0022535285797811557, -0.004117401595453714, -0.005269338469853118, -0.005510158852377748, -0.004798222711104596, -0.003256630272596592, -0.0011519368743994056, 0.0011519368743994043, 0.00325663027259659, 0.004798222711104594, 0.005510158852377746, 0.00526933846985312, 0.004117401595453712, 0.0022535285797811557, 6.277995377629245e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0036853095834749315, -0.014104014826930397, -0.029454626942807247, -0.047082885640337316, -0.063940704843015, -0.07711321446221778] + [-0.0843227679162924] * 2 + [-0.07711321446221778, -0.06394070484301503, -0.04708288564033735, -0.029454626942807257, -0.014104014826930387, -0.0036853095834749323, -7.688326944275534e-34],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.002253528579781156, 0.0041174015954537155, 0.005269338469853122, 0.005510158852377754, 0.004798222711104604, 0.0032566302725966015, 0.001151936874399416, -0.001151936874399394, -0.0032566302725965807, -0.004798222711104586, -0.00551015885237774, -0.005269338469853117, -0.00411740159545371, -0.0022535285797811552, -6.277995377629245e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.00887630031746843, 0.016217807671389722, 0.020755108744770392, 0.021703662961936706, 0.018899456681396077, 0.012827362644468276, 0.004537301073380672, -0.004537301073380646, -0.012827362644468255, -0.018899456681396053, -0.02170366296193669, -0.020755108744770386, -0.01621780767138972, -0.00887630031746843, -2.4728052203769877e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.007400940096490704, 0.028324070607849552, 0.059151592187989224, 0.09455314629646207, 0.12840748261489066, 0.15486087883682856] + [0.16933930243394257] * 2 + [0.15486087883682856, 0.1284074826148907, 0.09455314629646214, 0.059151592187989245, 0.028324070607849534, 0.007400940096490706, 1.5141564990247717e-33],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.002253528579781156, 0.004117401595453715, 0.00526933846985312, 0.00551015885237775, 0.0047982227111046, 0.0032566302725965967, 0.0011519368743994108, -0.0011519368743993991, -0.0032566302725965854, -0.00479822271110459, -0.005510158852377743, -0.005269338469853118, -0.004117401595453711, -0.0022535285797811552, -6.277995377629245e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.003685309583474932, 0.014104014826930397, 0.029454626942807247, 0.047082885640337316, 0.063940704843015, 0.07711321446221778] + [0.0843227679162924] * 2 + [0.07711321446221778, 0.06394070484301503, 0.04708288564033735, 0.029454626942807257, 0.014104014826930387, 0.003685309583474932, 3.844163472137767e-34],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0022535285797811552, -0.004117401595453713, -0.005269338469853117, -0.005510158852377745, -0.0047982227111045914, -0.003256630272596587, -0.0011519368743994004, 0.0011519368743994095, 0.003256630272596595, 0.004798222711104598, 0.0055101588523777485, 0.005269338469853122, 0.004117401595453713, 0.002253528579781156, 6.277995377629245e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.003685309583474932, -0.014104014826930397, -0.029454626942807247, -0.047082885640337316, -0.063940704843015, -0.07711321446221778] + [-0.0843227679162924] * 2 + [-0.07711321446221778, -0.06394070484301503, -0.04708288564033735, -0.029454626942807257, -0.014104014826930387, -0.003685309583474932, 3.844163472137767e-34],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q5.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q5.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05,
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
            "sample": -0.11201840403229255,
        },
        "q5.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q5.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.05600920201614627,
        },
        "q5.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q5.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05600920201614627,
        },
        "q5.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q5.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.10287614322831423,
        },
        "q5.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q5.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.01296818135360987, 0.04963040904617126, 0.10364745084375789, 0.165679269490148, 0.22499999999999995, 0.27135254915624213] + [0.29672214011007086] * 2 + [0.27135254915624213, 0.22500000000000006, 0.16567926949014813, 0.10364745084375791, 0.04963040904617123, 0.01296818135360987, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.EF_x90.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004618343160591265, 0.017674819153568368, 0.03691184466538491, 0.05900316322215746, 0.08012898515209145, 0.09663646392143418] + [0.10567130648522964] * 2 + [0.09663646392143418, 0.08012898515209148, 0.059003163222157505, 0.03691184466538492, 0.017674819153568357, 0.004618343160591265, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.EF_x90.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0008065355175237464, -0.001473613716922486, -0.0018858907173049866, -0.001972080079886148, -0.0017172788808699087, -0.0011655445623624013, -0.0004122770003825005, 0.00041227700038250003, 0.001165544562362401, 0.001717278880869908, 0.001972080079886148, 0.0018858907173049866, 0.0014736137169224855, 0.0008065355175237464, 2.246887967757484e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.z.const.wf": {
            "type": "constant",
            "sample": 0.15,
        },
        "q5.z.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q5.z.cz5_4.wf": {
            "type": "constant",
            "sample": -0.1732650364352627,
        },
        "q5.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.056010324243378824,
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
        "coupler_q1_q2.const.wf": {
            "type": "constant",
            "sample": 0.15,
        },
        "coupler_q1_q2.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "coupler_q1_q2.cz.wf": {
            "type": "constant",
            "sample": -0.04662292425886875,
        },
        "q1.z.SWAP_Coupler.flux_pulse_control_q2.wf": {
            "type": "arbitrary",
            "samples": [0.1379194010900913] * 15 + [0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "coupler_q1_q2.SWAP_Coupler.coupler_pulse_control_q2.wf": {
            "type": "arbitrary",
            "samples": [0.04600000000000054] * 15 + [0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.z.Cz_unipolar.flux_pulse_control_q2_q1.wf": {
            "type": "arbitrary",
            "samples": [0.07534407924424193] * 88 + [0.0] * 4,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "coupler_q1_q2.Cz_unipolar.coupler_flux_pulse_q2_q1.wf": {
            "type": "arbitrary",
            "samples": [0.1641] * 88 + [0.0] * 4,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "coupler_q2_q3.const.wf": {
            "type": "constant",
            "sample": 0.15,
        },
        "coupler_q2_q3.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "coupler_q2_q3.cz.wf": {
            "type": "constant",
            "sample": -0.10043328056510244,
        },
        "q3.z.SWAP_Coupler.flux_pulse_control_q2_q3.wf": {
            "type": "arbitrary",
            "samples": [0.1379194010900913] * 15 + [0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "coupler_q2_q3.SWAP_Coupler.coupler_pulse_control_q2_q3.wf": {
            "type": "arbitrary",
            "samples": [0.04600000000000054] * 15 + [0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.z.Cz_unipolar.flux_pulse_control_q2_q3.wf": {
            "type": "arbitrary",
            "samples": [0.07410126789587144] * 86 + [0.0] * 2,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "coupler_q2_q3.Cz_unipolar.coupler_flux_pulse_q2_q3.wf": {
            "type": "arbitrary",
            "samples": [0.1795] * 86 + [0.0] * 2,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "coupler_q3_q4.const.wf": {
            "type": "constant",
            "sample": 0.15,
        },
        "coupler_q3_q4.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "coupler_q3_q4.cz.wf": {
            "type": "constant",
            "sample": -0.11048567109,
        },
        "q3.z.SWAP_Coupler.flux_pulse_control_q3_q4.wf": {
            "type": "arbitrary",
            "samples": [0.1379194010900913] * 15 + [0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "coupler_q3_q4.SWAP_Coupler.coupler_pulse_control_q3_q4.wf": {
            "type": "arbitrary",
            "samples": [0.04600000000000054] * 15 + [0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.z.Cz_unipolar.flux_pulse_control_q4_q3.wf": {
            "type": "arbitrary",
            "samples": [0.13154871783227662] * 84 + [0.0] * 4,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "coupler_q3_q4.Cz_unipolar.coupler_flux_pulse_q4_q3.wf": {
            "type": "arbitrary",
            "samples": [0.158] * 84 + [0.0] * 4,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "coupler_q4_q5.const.wf": {
            "type": "constant",
            "sample": 0.15,
        },
        "coupler_q4_q5.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "coupler_q4_q5.cz.wf": {
            "type": "constant",
            "sample": -0.09938541439500001,
        },
    },
    "digital_waveforms": {
        "ON": {
            "samples": [(1, 0)],
        },
    },
    "integration_weights": {
        "q1.resonator.readout.iw1": {
            "cosine": [(-0.3970892049209327, 1040)],
            "sine": [(0.9177800190324812, 1040)],
        },
        "q1.resonator.readout.iw2": {
            "cosine": [(-0.9177800190324812, 1040)],
            "sine": [(-0.3970892049209327, 1040)],
        },
        "q1.resonator.readout.iw3": {
            "cosine": [(0.9177800190324812, 1040)],
            "sine": [(0.3970892049209327, 1040)],
        },
        "q2.resonator.readout.iw1": {
            "cosine": [(0.9413034281447746, 1040)],
            "sine": [(0.3375616331322253, 1040)],
        },
        "q2.resonator.readout.iw2": {
            "cosine": [(-0.3375616331322253, 1040)],
            "sine": [(0.9413034281447746, 1040)],
        },
        "q2.resonator.readout.iw3": {
            "cosine": [(0.3375616331322253, 1040)],
            "sine": [(-0.9413034281447746, 1040)],
        },
        "q3.resonator.readout.iw1": {
            "cosine": [(0.7913782651308222, 1040)],
            "sine": [(0.6113267877972713, 1040)],
        },
        "q3.resonator.readout.iw2": {
            "cosine": [(-0.6113267877972713, 1040)],
            "sine": [(0.7913782651308222, 1040)],
        },
        "q3.resonator.readout.iw3": {
            "cosine": [(0.6113267877972713, 1040)],
            "sine": [(-0.7913782651308222, 1040)],
        },
        "q4.resonator.readout.iw1": {
            "cosine": [(-0.999408409473025, 1040)],
            "sine": [(-0.034392311271538724, 1040)],
        },
        "q4.resonator.readout.iw2": {
            "cosine": [(0.034392311271538724, 1040)],
            "sine": [(-0.999408409473025, 1040)],
        },
        "q4.resonator.readout.iw3": {
            "cosine": [(-0.034392311271538724, 1040)],
            "sine": [(0.999408409473025, 1040)],
        },
        "q5.resonator.readout.iw1": {
            "cosine": [(0.8065831195035316, 1040)],
            "sine": [(0.591120691003074, 1040)],
        },
        "q5.resonator.readout.iw2": {
            "cosine": [(-0.591120691003074, 1040)],
            "sine": [(0.8065831195035316, 1040)],
        },
        "q5.resonator.readout.iw3": {
            "cosine": [(0.591120691003074, 1040)],
            "sine": [(-0.8065831195035316, 1040)],
        },
    },
    "mixers": {},
}

