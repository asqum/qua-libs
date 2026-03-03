
# Single QUA script generated at 2026-03-03 15:01:09.266622
# QUA library version: 1.2.3


from qm import CompilerOptionArguments
from qm.qua import *

with program() as prog:
    v1 = declare(int, )
    v2 = declare(fixed, )
    v3 = declare(fixed, )
    v4 = declare(fixed, )
    v5 = declare(fixed, )
    v6 = declare(fixed, )
    v7 = declare(int, value=22)
    v8 = declare(int, )
    v9 = declare(int, )
    v10 = declare(int, )
    v11 = declare(fixed, value=-0.0002887303876708028)
    v12 = declare(fixed, )
    v13 = declare(bool, )
    v14 = declare(int, value=1)
    v15 = declare(fixed, value=2.442801735378394e-05)
    v16 = declare(fixed, )
    v17 = declare(bool, )
    v18 = declare(int, value=1)
    v19 = declare(fixed, )
    v20 = declare(fixed, )
    a1 = declare(fixed, size=3)
    v21 = declare(fixed, )
    v22 = declare(fixed, )
    set_dc_offset("q1.z", "single", 0.0437340751973524)
    set_dc_offset("q2.z", "single", 0.061309858577262455)
    set_dc_offset("q3.z", "single", 0.05926167042094979)
    set_dc_offset("q4.z", "single", 0.05586821057233096)
    set_dc_offset("q5.z", "single", 0.06604855156725548)
    set_dc_offset("coupler_q1_q2", "single", -0.12)
    set_dc_offset("coupler_q2_q3", "single", -0.07)
    set_dc_offset("coupler_q3_q4", "single", -0.2)
    set_dc_offset("coupler_q4_q5", "single", -0.2)
    wait(24, "q3.z")
    wait(24, "q4.z")
    align("q3.xy", "q3.z", "q3.resonator", "q4.xy", "q4.z", "q4.resonator", "coupler_q3_q4")
    set_dc_offset("coupler_q3_q4", "single", -0.2)
    wait(1000, )
    with for_(v1,0,(v1<100),(v1+1)):
        r1 = declare_stream()
        save(v1, r1)
        with for_(v2,-0.5,(v2<0.5100000000000009),(v2+0.020000000000000018)):
            with for_(v3,-0.16999999999999998,(v3<-0.06899999999999991),(v3+0.0020000000000000018)):
                assign(v14, 1)
                align("q3.xy", "q3.resonator", "q3.z")
                measure("readout", "q3.resonator", dual_demod.full("iw1", "iw2", v11), dual_demod.full("iw3", "iw1", v12))
                assign(v13, (v11>-0.00013645962910537797))
                wait(250, "q3.resonator")
                align("q3.xy", "q3.resonator", "q3.z")
                play("x180", "q3.xy", condition=v13)
                align("q3.xy", "q3.resonator", "q3.z")
                with while_(broadcast.and_((v11>-0.00026248217060982073), (v14<15))):
                    align("q3.xy", "q3.resonator", "q3.z")
                    wait(4, )
                    reset_if_phase("q3.resonator")
                    wait(4, )
                    measure("readout", "q3.resonator", dual_demod.full("iw1", "iw2", v11), dual_demod.full("iw3", "iw1", v12))
                    assign(v13, (v11>-0.00013645962910537797))
                    wait(1000, "q3.resonator")
                    align("q3.xy", "q3.resonator", "q3.z")
                    play("x180", "q3.xy", condition=v13)
                    align("q3.xy", "q3.resonator", "q3.z")
                    assign(v14, (v14+1))
                wait(500, "q3.xy")
                align("q3.xy", "q3.resonator", "q3.z")
                assign(v18, 1)
                align("q4.xy", "q4.resonator", "q4.z")
                measure("readout", "q4.resonator", dual_demod.full("iw1", "iw2", v15), dual_demod.full("iw3", "iw1", v16))
                assign(v17, (v15>0.00013353594684779364))
                wait(250, "q4.resonator")
                align("q4.xy", "q4.resonator", "q4.z")
                play("x180", "q4.xy", condition=v17)
                align("q4.xy", "q4.resonator", "q4.z")
                with while_(broadcast.and_((v15>2.2207288503439945e-05), (v18<15))):
                    align("q4.xy", "q4.resonator", "q4.z")
                    wait(4, )
                    reset_if_phase("q4.resonator")
                    wait(4, )
                    measure("readout", "q4.resonator", dual_demod.full("iw1", "iw2", v15), dual_demod.full("iw3", "iw1", v16))
                    assign(v17, (v15>0.00013353594684779364))
                    wait(1000, "q4.resonator")
                    align("q4.xy", "q4.resonator", "q4.z")
                    play("x180", "q4.xy", condition=v17)
                    align("q4.xy", "q4.resonator", "q4.z")
                    assign(v18, (v18+1))
                wait(500, "q4.xy")
                align("q4.xy", "q4.resonator", "q4.z")
                align("q3.xy", "q3.z", "q3.resonator", "q4.xy", "q4.z", "q4.resonator", "coupler_q3_q4")
                align()
                assign(v4, (v3+(0.02*v2)))
                play("x180", "q3.xy")
                play("x180", "q4.xy")
                align()
                play("const"*amp((v4/0.45)), "q3.z", duration=v7)
                play("const"*amp((v2/0.15)), "coupler_q3_q4", duration=v7)
                align()
                wait(20, )
                update_frequency("q3.resonator", -89402080.0, "Hz", False)
                wait(4, )
                measure("readout", "q3.resonator", dual_demod.full("iw1", "iw2", v19), dual_demod.full("iw3", "iw1", v20))
                wait(4, )
                update_frequency("q3.resonator", -89152080.0, "Hz", False)
                wait(4, )
                assign(a1[0], (Math.abs((v19--0.00024982942272273103))+Math.abs((v20--3.0523041751936786e-05))))
                assign(a1[1], (Math.abs((v19--5.3694105939261274e-05))+Math.abs((v20-0.00020747989671600074))))
                assign(a1[2], (Math.abs((v19--0.00015618483584958014))+Math.abs((v20--4.974338389050015e-05))))
                wait(4, )
                assign(v8, Math.argmin(a1))
                wait(250, "q3.resonator")
                measure("readout", "q4.resonator", dual_demod.full("iw1", "iw2", v21), dual_demod.full("iw3", "iw1", v22))
                assign(v9, Cast.to_int((v21>0.00013353594684779364)))
                wait(250, "q4.resonator")
                assign(v10, ((v8*2)+v9))
                r2 = declare_stream()
                save(v8, r2)
                r3 = declare_stream()
                save(v9, r3)
                r4 = declare_stream()
                save(v10, r4)
    align()
    with stream_processing():
        r1.save("n")
        r2.buffer(51).buffer(51).average().save("state_control1")
        r3.buffer(51).buffer(51).average().save("state_target1")
        r4.buffer(51).buffer(51).average().save("state1")

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "fems": {
                "2": {
                    "type": "LF",
                    "analog_outputs": {
                        "8": {
                            "delay": 92,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "direct",
                            "offset": 0.0,
                        },
                    },
                },
                "1": {
                    "type": "LF",
                    "analog_outputs": {
                        "1": {
                            "delay": 57,
                            "shareable": False,
                            "filter": {
                                "exponential": [(-0.014222759854540867, 117417.44175630683), (0.0030980452178688738, 979.8296492766553), (0.03626492869698257, 19.764228103817157), (0.17260247326976075, 3750.23093928198), (-0.07156597682800245, 26.14624901197679)],
                            },
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "direct",
                            "offset": 0.0,
                        },
                        "2": {
                            "delay": 56,
                            "shareable": False,
                            "filter": {
                                "exponential": [(-0.01769653680971151, 330.60787392639077), (-0.07237663070119806, 10.417561895314897), (0.01264328247686496, 48611.89022732948), (-0.12045387577060747, 1.23381071262837)],
                            },
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "direct",
                            "offset": 0.0,
                        },
                        "3": {
                            "delay": 55,
                            "shareable": False,
                            "crosstalk": {
                                "7": -0.0275,
                            },
                            "filter": {
                                "exponential": [(-0.010260654320882587, 62.84454837804313), (-0.06442551060715064, 10.924763516892996), (0.004364243812839698, 13978176.988923818), (-0.01780539123873645, 15.873930395927458)],
                            },
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "direct",
                            "offset": 0.0,
                        },
                        "4": {
                            "delay": 87,
                            "shareable": False,
                            "filter": {
                                "exponential": [(0.05121889750694114, 682742.4294348056), (-0.011053862447955358, 194.40473375724764), (0.02946608537070822, 1005051.2239588167), (-0.0051886944506610065, 83.71159271751709), (0.02899826727834642, 1879.1354237436165), (-0.09559799628857335, 9.446550360023931)],
                            },
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "direct",
                            "offset": 0.0,
                        },
                        "5": {
                            "delay": 85,
                            "shareable": False,
                            "filter": {
                                "exponential": [(-0.02584887446705352, 24155.108018680057), (-0.040818588224334644, 13.56262165133981), (-0.034567576179888336, 4.36824367678182), (0.0022211663552704113, 54585060.68856716), (-0.019019463770724016, 31.82265239390185)],
                            },
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "direct",
                            "offset": 0.0,
                        },
                        "6": {
                            "delay": 57,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "direct",
                            "offset": 0.0,
                        },
                        "7": {
                            "delay": 55,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "direct",
                            "offset": 0.0,
                        },
                        "8": {
                            "delay": 85,
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
                            "full_scale_power_dbm": 1,
                            "upconverter_frequency": 5950000000,
                        },
                        "2": {
                            "band": 1,
                            "delay": 20,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 7,
                            "upconverter_frequency": 4900000000.0,
                        },
                        "3": {
                            "band": 1,
                            "delay": 20,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 7,
                            "upconverter_frequency": 4900000000.0,
                        },
                        "4": {
                            "band": 1,
                            "delay": 20,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 7,
                            "upconverter_frequency": 5000000000.0,
                        },
                        "5": {
                            "band": 1,
                            "delay": 20,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 7,
                            "upconverter_frequency": 4800000000.0,
                        },
                        "6": {
                            "band": 1,
                            "delay": 20,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 7,
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
            "intermediate_frequency": 222946400.64122882,
            "core": "a",
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
                "Cz_flattop": "q1.z.Cz_flattop.pulse",
                "Cz_bipolar": "q1.z.Cz_bipolar.pulse",
                "SWAP_Coupler.flux_pulse_control_q1_q2": "q1.z.SWAP_Coupler.flux_pulse_control_q1_q2.pulse",
                "Cz_unipolar.flux_pulse_control_q2_q1": "q1.z.Cz_unipolar.flux_pulse_control_q2_q1.pulse",
                "Cz_flattop.flux_pulse_control_q2_q1": "q1.z.Cz_flattop.flux_pulse_control_q2_q1.pulse",
                "Cz_bipolar.flux_pulse_control_q2_q1": "q1.z.Cz_bipolar.flux_pulse_control_q2_q1.pulse",
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
            "intermediate_frequency": -19074242.0,
            "core": "a",
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
                "EF_EF_x180": "q2.xy.EF_EF_x180.pulse",
            },
            "intermediate_frequency": -62998976.43893273,
            "core": "b",
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
                "port": ('con1', 1, 2),
            },
        },
        "q2.resonator": {
            "operations": {
                "readout": "q2.resonator.readout.pulse",
                "const": "q2.resonator.const.pulse",
            },
            "intermediate_frequency": 75584485.0,
            "core": "b",
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
                "EF_EF_x180": "q3.xy.EF_EF_x180.pulse",
            },
            "intermediate_frequency": 195146679.45820576,
            "core": "c",
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
                "Cz_flattop": "q3.z.Cz_flattop.pulse",
                "Cz_bipolar": "q3.z.Cz_bipolar.pulse",
                "SWAP_Coupler.flux_pulse_control_q2_q3": "q3.z.SWAP_Coupler.flux_pulse_control_q2_q3.pulse",
                "Cz_unipolar.flux_pulse_control_q2_q3": "q3.z.Cz_unipolar.flux_pulse_control_q2_q3.pulse",
                "Cz_flattop.flux_pulse_control_q2_q3": "q3.z.Cz_flattop.flux_pulse_control_q2_q3.pulse",
                "Cz_bipolar.flux_pulse_control_q2_q3": "q3.z.Cz_bipolar.flux_pulse_control_q2_q3.pulse",
                "SWAP_Coupler.flux_pulse_control_q3_q4": "q3.z.SWAP_Coupler.flux_pulse_control_q3_q4.pulse",
                "Cz_unipolar.flux_pulse_control_q4_q3": "q3.z.Cz_unipolar.flux_pulse_control_q4_q3.pulse",
                "Cz_flattop.flux_pulse_control_q4_q3": "q3.z.Cz_flattop.flux_pulse_control_q4_q3.pulse",
                "Cz_bipolar.flux_pulse_control_q4_q3": "q3.z.Cz_bipolar.flux_pulse_control_q4_q3.pulse",
            },
            "singleInput": {
                "port": ('con1', 1, 3),
            },
        },
        "q3.resonator": {
            "operations": {
                "readout": "q3.resonator.readout.pulse",
                "const": "q3.resonator.const.pulse",
            },
            "intermediate_frequency": -89152080.0,
            "core": "c",
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
            "intermediate_frequency": -95054563.7123453,
            "core": "d",
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
                "port": ('con1', 1, 4),
            },
        },
        "q4.resonator": {
            "operations": {
                "readout": "q4.resonator.readout.pulse",
                "const": "q4.resonator.const.pulse",
            },
            "intermediate_frequency": 130789974.0,
            "core": "d",
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
            "intermediate_frequency": -3756251.2312046573,
            "core": "e",
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
                "Cz_flattop": "q5.z.Cz_flattop.pulse",
                "Cz_bipolar": "q5.z.Cz_bipolar.pulse",
                "Cz_unipolar.flux_pulse_control_q4_q5": "q5.z.Cz_unipolar.flux_pulse_control_q4_q5.pulse",
                "Cz_flattop.flux_pulse_control_q4_q5": "q5.z.Cz_flattop.flux_pulse_control_q4_q5.pulse",
                "Cz_bipolar.flux_pulse_control_q4_q5": "q5.z.Cz_bipolar.flux_pulse_control_q4_q5.pulse",
            },
            "singleInput": {
                "port": ('con1', 1, 5),
            },
        },
        "q5.resonator": {
            "operations": {
                "readout": "q5.resonator.readout.pulse",
                "const": "q5.resonator.const.pulse",
            },
            "intermediate_frequency": 22227706.0,
            "core": "e",
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
                "SWAP_Coupler.coupler_pulse_control_q1_q2": "coupler_q1_q2.SWAP_Coupler.coupler_pulse_control_q1_q2.pulse",
                "Cz_unipolar.coupler_flux_pulse_q2_q1": "coupler_q1_q2.Cz_unipolar.coupler_flux_pulse_q2_q1.pulse",
                "Cz_flattop.coupler_flux_pulse_q2_q1": "coupler_q1_q2.Cz_flattop.coupler_flux_pulse_q2_q1.pulse",
                "Cz_bipolar.coupler_flux_pulse_q2_q1": "coupler_q1_q2.Cz_bipolar.coupler_flux_pulse_q2_q1.pulse",
            },
            "singleInput": {
                "port": ('con1', 1, 6),
            },
        },
        "coupler_q2_q3": {
            "operations": {
                "const": "coupler_q2_q3.const.pulse",
                "flux_pulse": "coupler_q2_q3.flux_pulse.pulse",
                "cz": "coupler_q2_q3.cz.pulse",
                "SWAP_Coupler.coupler_pulse_control_q2_q3": "coupler_q2_q3.SWAP_Coupler.coupler_pulse_control_q2_q3.pulse",
                "Cz_unipolar.coupler_flux_pulse_q2_q3": "coupler_q2_q3.Cz_unipolar.coupler_flux_pulse_q2_q3.pulse",
                "Cz_flattop.coupler_flux_pulse_q2_q3": "coupler_q2_q3.Cz_flattop.coupler_flux_pulse_q2_q3.pulse",
                "Cz_bipolar.coupler_flux_pulse_q2_q3": "coupler_q2_q3.Cz_bipolar.coupler_flux_pulse_q2_q3.pulse",
            },
            "singleInput": {
                "port": ('con1', 1, 7),
            },
        },
        "coupler_q3_q4": {
            "operations": {
                "const": "coupler_q3_q4.const.pulse",
                "flux_pulse": "coupler_q3_q4.flux_pulse.pulse",
                "cz": "coupler_q3_q4.cz.pulse",
                "SWAP_Coupler.coupler_pulse_control_q3_q4": "coupler_q3_q4.SWAP_Coupler.coupler_pulse_control_q3_q4.pulse",
                "Cz_unipolar.coupler_flux_pulse_q4_q3": "coupler_q3_q4.Cz_unipolar.coupler_flux_pulse_q4_q3.pulse",
                "Cz_flattop.coupler_flux_pulse_q4_q3": "coupler_q3_q4.Cz_flattop.coupler_flux_pulse_q4_q3.pulse",
                "Cz_bipolar.coupler_flux_pulse_q4_q3": "coupler_q3_q4.Cz_bipolar.coupler_flux_pulse_q4_q3.pulse",
            },
            "singleInput": {
                "port": ('con1', 1, 8),
            },
        },
        "coupler_q4_q5": {
            "operations": {
                "const": "coupler_q4_q5.const.pulse",
                "flux_pulse": "coupler_q4_q5.flux_pulse.pulse",
                "cz": "coupler_q4_q5.cz.pulse",
                "Cz_unipolar.coupler_flux_pulse_q4_q5": "coupler_q4_q5.Cz_unipolar.coupler_flux_pulse_q4_q5.pulse",
                "Cz_flattop.coupler_flux_pulse_q4_q5": "coupler_q4_q5.Cz_flattop.coupler_flux_pulse_q4_q5.pulse",
                "Cz_bipolar.coupler_flux_pulse_q4_q5": "coupler_q4_q5.Cz_bipolar.coupler_flux_pulse_q4_q5.pulse",
            },
            "singleInput": {
                "port": ('con1', 2, 8),
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
        "q1.z.Cz_flattop.pulse": {
            "operation": "control",
            "length": 88,
            "waveforms": {
                "single": "q1.z.Cz_flattop.wf",
            },
        },
        "q1.z.Cz_bipolar.pulse": {
            "operation": "control",
            "length": 88,
            "waveforms": {
                "single": "q1.z.Cz_bipolar.wf",
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
        "q2.xy.EF_EF_x180.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.EF_EF_x180.wf.I",
                "Q": "q2.xy.EF_EF_x180.wf.Q",
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
            "length": 1200,
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
        "q3.xy.EF_EF_x180.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.EF_EF_x180.wf.I",
                "Q": "q3.xy.EF_EF_x180.wf.Q",
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
        "q3.z.Cz_flattop.pulse": {
            "operation": "control",
            "length": 128,
            "waveforms": {
                "single": "q3.z.Cz_flattop.wf",
            },
        },
        "q3.z.Cz_bipolar.pulse": {
            "operation": "control",
            "length": 88,
            "waveforms": {
                "single": "q3.z.Cz_bipolar.wf",
            },
        },
        "q3.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 1200,
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
            "length": 16,
            "waveforms": {
                "single": "q4.z.const.wf",
            },
        },
        "q4.z.flux_pulse.pulse": {
            "operation": "control",
            "length": 16,
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
            "length": 1200,
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
        "q5.z.Cz_flattop.pulse": {
            "operation": "control",
            "length": 128,
            "waveforms": {
                "single": "q5.z.Cz_flattop.wf",
            },
        },
        "q5.z.Cz_bipolar.pulse": {
            "operation": "control",
            "length": 128,
            "waveforms": {
                "single": "q5.z.Cz_bipolar.wf",
            },
        },
        "q5.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 1200,
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
        "q1.z.SWAP_Coupler.flux_pulse_control_q1_q2.pulse": {
            "operation": "control",
            "length": 16,
            "waveforms": {
                "single": "q1.z.SWAP_Coupler.flux_pulse_control_q1_q2.wf",
            },
        },
        "coupler_q1_q2.SWAP_Coupler.coupler_pulse_control_q1_q2.pulse": {
            "operation": "control",
            "length": 16,
            "waveforms": {
                "single": "coupler_q1_q2.SWAP_Coupler.coupler_pulse_control_q1_q2.wf",
            },
        },
        "q1.z.Cz_unipolar.flux_pulse_control_q2_q1.pulse": {
            "operation": "control",
            "length": 88,
            "waveforms": {
                "single": "q1.z.Cz_unipolar.flux_pulse_control_q2_q1.wf",
            },
        },
        "coupler_q1_q2.Cz_unipolar.coupler_flux_pulse_q2_q1.pulse": {
            "operation": "control",
            "length": 88,
            "waveforms": {
                "single": "coupler_q1_q2.Cz_unipolar.coupler_flux_pulse_q2_q1.wf",
            },
        },
        "q1.z.Cz_flattop.flux_pulse_control_q2_q1.pulse": {
            "operation": "control",
            "length": 88,
            "waveforms": {
                "single": "q1.z.Cz_flattop.flux_pulse_control_q2_q1.wf",
            },
        },
        "coupler_q1_q2.Cz_flattop.coupler_flux_pulse_q2_q1.pulse": {
            "operation": "control",
            "length": 88,
            "waveforms": {
                "single": "coupler_q1_q2.Cz_flattop.coupler_flux_pulse_q2_q1.wf",
            },
        },
        "q1.z.Cz_bipolar.flux_pulse_control_q2_q1.pulse": {
            "operation": "control",
            "length": 88,
            "waveforms": {
                "single": "q1.z.Cz_bipolar.flux_pulse_control_q2_q1.wf",
            },
        },
        "coupler_q1_q2.Cz_bipolar.coupler_flux_pulse_q2_q1.pulse": {
            "operation": "control",
            "length": 88,
            "waveforms": {
                "single": "coupler_q1_q2.Cz_bipolar.coupler_flux_pulse_q2_q1.wf",
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
        "q3.z.Cz_flattop.flux_pulse_control_q2_q3.pulse": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "single": "q3.z.Cz_flattop.flux_pulse_control_q2_q3.wf",
            },
        },
        "coupler_q2_q3.Cz_flattop.coupler_flux_pulse_q2_q3.pulse": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "single": "coupler_q2_q3.Cz_flattop.coupler_flux_pulse_q2_q3.wf",
            },
        },
        "q3.z.Cz_bipolar.flux_pulse_control_q2_q3.pulse": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "single": "q3.z.Cz_bipolar.flux_pulse_control_q2_q3.wf",
            },
        },
        "coupler_q2_q3.Cz_bipolar.coupler_flux_pulse_q2_q3.pulse": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "single": "coupler_q2_q3.Cz_bipolar.coupler_flux_pulse_q2_q3.wf",
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
            "length": 80,
            "waveforms": {
                "single": "q3.z.Cz_unipolar.flux_pulse_control_q4_q3.wf",
            },
        },
        "coupler_q3_q4.Cz_unipolar.coupler_flux_pulse_q4_q3.pulse": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "single": "coupler_q3_q4.Cz_unipolar.coupler_flux_pulse_q4_q3.wf",
            },
        },
        "q3.z.Cz_flattop.flux_pulse_control_q4_q3.pulse": {
            "operation": "control",
            "length": 128,
            "waveforms": {
                "single": "q3.z.Cz_flattop.flux_pulse_control_q4_q3.wf",
            },
        },
        "coupler_q3_q4.Cz_flattop.coupler_flux_pulse_q4_q3.pulse": {
            "operation": "control",
            "length": 128,
            "waveforms": {
                "single": "coupler_q3_q4.Cz_flattop.coupler_flux_pulse_q4_q3.wf",
            },
        },
        "q3.z.Cz_bipolar.flux_pulse_control_q4_q3.pulse": {
            "operation": "control",
            "length": 88,
            "waveforms": {
                "single": "q3.z.Cz_bipolar.flux_pulse_control_q4_q3.wf",
            },
        },
        "coupler_q3_q4.Cz_bipolar.coupler_flux_pulse_q4_q3.pulse": {
            "operation": "control",
            "length": 88,
            "waveforms": {
                "single": "coupler_q3_q4.Cz_bipolar.coupler_flux_pulse_q4_q3.wf",
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
        "q5.z.Cz_unipolar.flux_pulse_control_q4_q5.pulse": {
            "operation": "control",
            "length": 76,
            "waveforms": {
                "single": "q5.z.Cz_unipolar.flux_pulse_control_q4_q5.wf",
            },
        },
        "coupler_q4_q5.Cz_unipolar.coupler_flux_pulse_q4_q5.pulse": {
            "operation": "control",
            "length": 76,
            "waveforms": {
                "single": "coupler_q4_q5.Cz_unipolar.coupler_flux_pulse_q4_q5.wf",
            },
        },
        "q5.z.Cz_flattop.flux_pulse_control_q4_q5.pulse": {
            "operation": "control",
            "length": 128,
            "waveforms": {
                "single": "q5.z.Cz_flattop.flux_pulse_control_q4_q5.wf",
            },
        },
        "coupler_q4_q5.Cz_flattop.coupler_flux_pulse_q4_q5.pulse": {
            "operation": "control",
            "length": 128,
            "waveforms": {
                "single": "coupler_q4_q5.Cz_flattop.coupler_flux_pulse_q4_q5.wf",
            },
        },
        "q5.z.Cz_bipolar.flux_pulse_control_q4_q5.pulse": {
            "operation": "control",
            "length": 128,
            "waveforms": {
                "single": "q5.z.Cz_bipolar.flux_pulse_control_q4_q5.wf",
            },
        },
        "coupler_q4_q5.Cz_bipolar.coupler_flux_pulse_q4_q5.pulse": {
            "operation": "control",
            "length": 128,
            "waveforms": {
                "single": "coupler_q4_q5.Cz_bipolar.coupler_flux_pulse_q4_q5.wf",
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
            "samples": [0.0, 0.013349337165626278, 0.051089126991647416, 0.10669381696191726, 0.1705488510278263, 0.231613113694606, 0.27932803919575944] + [0.30544328343564736] * 2 + [0.27932803919575944, 0.23161311369460608, 0.17054885102782644, 0.10669381696191729, 0.05108912699164738, 0.013349337165626262, 0.0],
        },
        "q1.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.008571745181666944, -0.015661357749563374, -0.02004297928359179, -0.02095898761473669, -0.018250996580868675, -0.012387242433069748, -0.004381621534028416, 0.0043816215340284105, 0.012387242433069743, 0.01825099658086867, 0.020958987614736687, 0.020042979283591795, 0.01566135774956337, 0.008571745181666944, 2.3879606902498985e-17],
        },
        "q1.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.005407777688234955, 0.020696056862526, 0.043221355163982164, 0.06908884388037084, 0.09382579921387094, 0.11315497685913609] + [0.12373418640123374] * 2 + [0.11315497685913609, 0.09382579921387098, 0.0690888438803709, 0.04322135516398218, 0.020696056862525986, 0.005407777688234948, 0.0],
        },
        "q1.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.008265624995918104, -0.01510204833919625, -0.01932719032682534, -0.020210485524934353, -0.01759920416928529, -0.01194486052901625, -0.004225141987629084, 0.00422514198762908, 0.011944860529016246, 0.017599204169285287, 0.020210485524934353, 0.01932719032682534, 0.015102048339196247, 0.008265624995918104, 2.3026801604899053e-17],
        },
        "q1.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.005407777688234954, -0.020696056862525996, -0.043221355163982164, -0.06908884388037084, -0.09382579921387094, -0.11315497685913609] + [-0.12373418640123374] * 2 + [-0.11315497685913609, -0.09382579921387098, -0.0690888438803709, -0.04322135516398218, -0.02069605686252599, -0.005407777688234949, -2.819969888004076e-33],
        },
        "q1.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.008265624995918104, 0.015102048339196253, 0.019327190326825346, 0.02021048552493436, 0.0175992041692853, 0.011944860529016265, 0.004225141987629099, -0.004225141987629065, -0.011944860529016232, -0.017599204169285277, -0.020210485524934346, -0.01932719032682533, -0.015102048339196246, -0.008265624995918104, -2.3026801604899053e-17],
        },
        "q1.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.008571745181666944, 0.015661357749563377, 0.0200429792835918, 0.0209589876147367, 0.01825099658086869, 0.012387242433069766, 0.004381621534028435, -0.004381621534028391, -0.012387242433069726, -0.018250996580868657, -0.020958987614736677, -0.020042979283591788, -0.015661357749563367, -0.008571745181666944, -2.3879606902498985e-17],
        },
        "q1.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.013349337165626278, 0.051089126991647416, 0.10669381696191726, 0.1705488510278263, 0.231613113694606, 0.27932803919575944] + [0.30544328343564736] * 2 + [0.27932803919575944, 0.23161311369460608, 0.17054885102782644, 0.10669381696191729, 0.05108912699164738, 0.013349337165626262, 1.4622042079021211e-33],
        },
        "q1.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.008265624995918104, 0.015102048339196253, 0.019327190326825342, 0.020210485524934357, 0.017599204169285298, 0.011944860529016258, 0.004225141987629092, -0.004225141987629072, -0.011944860529016239, -0.01759920416928528, -0.02021048552493435, -0.019327190326825335, -0.015102048339196246, -0.008265624995918104, -2.3026801604899053e-17],
        },
        "q1.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.005407777688234954, 0.020696056862526, 0.043221355163982164, 0.06908884388037084, 0.09382579921387094, 0.11315497685913609] + [0.12373418640123374] * 2 + [0.11315497685913609, 0.09382579921387098, 0.0690888438803709, 0.04322135516398218, 0.020696056862525986, 0.005407777688234949, 1.409984944002038e-33],
        },
        "q1.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.008265624995918104, -0.015102048339196249, -0.019327190326825335, -0.02021048552493435, -0.017599204169285284, -0.011944860529016244, -0.004225141987629077, 0.004225141987629088, 0.011944860529016253, 0.017599204169285294, 0.020210485524934357, 0.019327190326825342, 0.015102048339196249, 0.008265624995918104, 2.3026801604899053e-17],
        },
        "q1.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.005407777688234956, -0.020696056862526, -0.043221355163982164, -0.06908884388037084, -0.09382579921387094, -0.11315497685913609] + [-0.12373418640123374] * 2 + [-0.11315497685913609, -0.09382579921387098, -0.0690888438803709, -0.04322135516398218, -0.020696056862525986, -0.005407777688234947, 1.409984944002038e-33],
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
            "sample": 0.1,
        },
        "q1.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q1.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.006717669725766446, 0.025709132779367188, 0.05369059266680922, 0.08582380076997109, 0.11655263348678496, 0.1405637964778324] + [0.15370554152739355] * 2 + [0.1405637964778324, 0.11655263348678502, 0.08582380076997116, 0.05369059266680924, 0.02570913277936717, 0.006717669725766437, 0.0],
        },
        "q1.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0102278806658164, -0.018687267847134304, -0.023915456649183944, -0.02500844573148675, -0.02177724735221848, -0.014780565065670352, -0.005228188802049635, 0.005228188802049629, 0.014780565065670347, 0.021777247352218473, 0.025008445731486747, 0.023915456649183948, 0.018687267847134304, 0.0102278806658164, 2.849335398673954e-17],
        },
        "q1.xy.EF_x90.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004593670635837815, 0.01758039515822729, 0.036714650916558844, 0.0586879512609605, 0.07970091294071842, 0.09612020398463854] + [0.10510677980665072] * 2 + [0.09612020398463854, 0.07970091294071845, 0.05868795126096055, 0.03671465091655885, 0.017580395158227277, 0.004593670635837809, 0.0],
        },
        "q1.xy.EF_x90.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.002669837807754793, -0.004878036403833742, -0.006242778190120334, -0.006528086913475712, -0.005684630103683355, -0.00385824910572092, -0.0013647417862865906, 0.001364741786286589, 0.0038582491057209184, 0.005684630103683353, 0.006528086913475712, 0.006242778190120334, 0.004878036403833741, 0.002669837807754793, 7.437770954620908e-18],
        },
        "q1.z.const.wf": {
            "type": "constant",
            "sample": 0.45,
        },
        "q1.z.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q1.z.cz1_2.wf": {
            "type": "constant",
            "sample": -0.07009506167631502,
        },
        "q1.z.Cz_flattop.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.0005591865604883849, 0.0022214930902739803, 0.004941576200533949, 0.008645239114747865, 0.013231455563344617, 0.0185751255169805, 0.024530487589228253, 0.03093509502722724, 0.03761424683563438, 0.04438575316436563, 0.05106490497277278, 0.05746951241077176, 0.0634248744830195, 0.06876854443665538, 0.07335476088525215, 0.07705842379946606, 0.07977850690972603, 0.08144081343951164] + [0.08200000000000002] * 51 + [0.08144081343951164, 0.07977850690972603, 0.07705842379946606, 0.07335476088525215, 0.06876854443665541, 0.06342487448301952, 0.057469512410771764, 0.05106490497277278, 0.04438575316436564, 0.037614246835634385, 0.030935095027227244, 0.024530487589228267, 0.01857512551698051, 0.013231455563344627, 0.00864523911474787, 0.004941576200533954, 0.0022214930902739803, 0.0005591865604883849],
        },
        "q1.z.Cz_bipolar.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.0011913854855318678, 0.004696302948218395, 0.010311059324984858, 0.017709345384022617, 0.02646119963125605, 0.03605799610953176, 0.04594200389046825, 0.05553880036874397, 0.0642906546159774, 0.07168894067501516, 0.07730369705178161, 0.08080861451446815] + [0.08200000000000002] * 26 + [0.07920591775570361, 0.07101408311032399, 0.057982756057296914, 0.041000000000000016, 0.021223161698406705, 5.021051876504149e-18, -0.021223161698406715, -0.04099999999999999, -0.05798275605729691, -0.07101408311032399, -0.07920591775570361] + [-0.08200000000000002] * 26 + [-0.08080861451446815, -0.07730369705178161, -0.07168894067501516, -0.0642906546159774, -0.05553880036874397, -0.04594200389046825, -0.03605799610953177, -0.02646119963125604, -0.01770934538402262, -0.010311059324984854, -0.004696302948218399, -0.0011913854855318678],
        },
        "q1.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.04269504351718363,
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
            "samples": [0.0, 0.009567007456445571, 0.036613807323028616, 0.07646376219035395, 0.1222264528363863, 0.1659890943074338, 0.20018472832203574] + [0.21890061910148506] * 2 + [0.20018472832203574, 0.16598909430743386, 0.1222264528363864, 0.07646376219035397, 0.03661380732302859, 0.009567007456445559, 0.0],
        },
        "q2.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.016125390729056558, -0.029462554906483456, -0.037705375681670816, -0.03942859445891286, -0.03433425105667332, -0.023303203729856305, -0.008242820775187357, 0.008242820775187346, 0.023303203729856295, 0.034334251056673315, 0.03942859445891286, 0.037705375681670816, 0.029462554906483453, 0.016125390729056558, 4.4922939681250316e-17],
        },
        "q2.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0047081729608491945, 0.018018605966057376, 0.03762980422753636, 0.06015081340373337, 0.08168754641113861, 0.09851610645769468] + [0.10772668262868351] * 2 + [0.09851610645769468, 0.08168754641113865, 0.06015081340373342, 0.037629804227536374, 0.018018605966057363, 0.0047081729608491885, 0.0],
        },
        "q2.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.00696055063940401, -0.01271755883863767, -0.016275585579275897, -0.017019415714204164, -0.014820434255608732, -0.010058865074800154, -0.003558026740638228, 0.0035580267406382237, 0.01005886507480015, 0.014820434255608729, 0.01701941571420416, 0.0162755855792759, 0.012717558838637668, 0.00696055063940401, 1.93910585967258e-17],
        },
        "q2.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.004708172960849194, -0.018018605966057376, -0.03762980422753636, -0.06015081340373337, -0.08168754641113861, -0.09851610645769468] + [-0.10772668262868351] * 2 + [-0.09851610645769468, -0.08168754641113865, -0.06015081340373342, -0.037629804227536374, -0.018018605966057363, -0.004708172960849189, -2.3747197842559014e-33],
        },
        "q2.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.006960550639404011, 0.012717558838637672, 0.0162755855792759, 0.01701941571420417, 0.014820434255608743, 0.010058865074800166, 0.003558026740638241, -0.0035580267406382106, -0.010058865074800138, -0.014820434255608719, -0.017019415714204154, -0.016275585579275897, -0.012717558838637667, -0.006960550639404009, -1.93910585967258e-17],
        },
        "q2.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.016125390729056558, 0.02946255490648346, 0.03770537568167082, 0.03942859445891287, 0.03433425105667333, 0.02330320372985632, 0.00824282077518737, -0.008242820775187332, -0.02330320372985628, -0.03433425105667331, -0.039428594458912856, -0.03770537568167081, -0.02946255490648345, -0.016125390729056558, -4.4922939681250316e-17],
        },
        "q2.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.00956700745644557, 0.036613807323028616, 0.07646376219035395, 0.1222264528363863, 0.1659890943074338, 0.20018472832203574] + [0.21890061910148506] * 2 + [0.20018472832203574, 0.16598909430743386, 0.1222264528363864, 0.07646376219035397, 0.03661380732302859, 0.00956700745644556, 2.750736714446641e-33],
        },
        "q2.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.00696055063940401, 0.012717558838637672, 0.0162755855792759, 0.017019415714204168, 0.014820434255608738, 0.01005886507480016, 0.0035580267406382345, -0.003558026740638217, -0.010058865074800145, -0.014820434255608724, -0.017019415714204157, -0.016275585579275897, -0.012717558838637667, -0.00696055063940401, -1.93910585967258e-17],
        },
        "q2.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0047081729608491945, 0.018018605966057376, 0.03762980422753636, 0.06015081340373337, 0.08168754641113861, 0.09851610645769468] + [0.10772668262868351] * 2 + [0.09851610645769468, 0.08168754641113865, 0.06015081340373342, 0.037629804227536374, 0.018018605966057363, 0.0047081729608491885, 1.1873598921279507e-33],
        },
        "q2.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.00696055063940401, -0.012717558838637668, -0.016275585579275893, -0.01701941571420416, -0.014820434255608727, -0.010058865074800149, -0.0035580267406382215, 0.00355802674063823, 0.010058865074800156, 0.014820434255608734, 0.017019415714204164, 0.016275585579275904, 0.01271755883863767, 0.00696055063940401, 1.93910585967258e-17],
        },
        "q2.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0047081729608491945, -0.018018605966057376, -0.03762980422753636, -0.06015081340373337, -0.08168754641113861, -0.09851610645769468] + [-0.10772668262868351] * 2 + [-0.09851610645769468, -0.08168754641113865, -0.06015081340373342, -0.037629804227536374, -0.018018605966057363, -0.0047081729608491885, 1.1873598921279507e-33],
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
            "sample": 0.1,
        },
        "q2.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.00522147920474363, 0.01998307562882494, 0.0417323751456268, 0.06670872628251098, 0.09059349102488344, 0.1092567765625123] + [0.11947153127531518] * 2 + [0.1092567765625123, 0.09059349102488348, 0.06670872628251102, 0.04173237514562681, 0.019983075628824926, 0.005221479204743623, 0.0],
        },
        "q2.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0077614939281682414, -0.014180955045197449, -0.018148400204979865, -0.01897782209628136, -0.01652580613903457, -0.011216328168113118, -0.003967445159782416, 0.003967445159782412, 0.011216328168113113, 0.016525806139034562, 0.018977822096281356, 0.01814840020497987, 0.014180955045197445, 0.0077614939281682414, 2.1622367447085848e-17],
        },
        "q2.xy.EF_x90.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.005559686906605402, 0.021277427252099614, 0.044435480939558375, 0.07102961010210704, 0.09646144820746218, 0.11633359940621195] + [0.1272099882232664] * 2 + [0.11633359940621195, 0.09646144820746223, 0.0710296101021071, 0.044435480939558396, 0.0212774272520996, 0.005559686906605395, 0.0],
        },
        "q2.xy.EF_x90.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0036246827501982067, -0.006622624923678124, -0.008475455083195447, -0.008862802061736062, -0.007717690047773467, -0.005238119311537855, -0.0018528301595173225, 0.0018528301595173203, 0.005238119311537853, 0.007717690047773464, 0.008862802061736062, 0.008475455083195449, 0.006622624923678123, 0.0036246827501982067, 1.0097826916988402e-17],
        },
        "q2.xy.EF_EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.005272133834660322, 0.020176935502796242, 0.04213722939011479, 0.06735588118049173, 0.09147235687511178, 0.11031669873507154] + [0.12063054885731256] * 2 + [0.11031669873507154, 0.09147235687511182, 0.06735588118049178, 0.04213722939011481, 0.02017693550279623, 0.005272133834660315, 0.0],
        },
        "q2.xy.EF_EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.007836789756632871, -0.014318527369344046, -0.018324461520158156, -0.01916192980163018, -0.016686126339733112, -0.011325140044997309, -0.004005934150814109, 0.0040059341508141035, 0.011325140044997305, 0.01668612633973311, 0.01916192980163018, 0.01832446152015816, 0.014318527369344042, 0.007836789756632871, 2.1832130423822366e-17],
        },
        "q2.z.const.wf": {
            "type": "constant",
            "sample": 0.45,
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
            "sample": 0.02861308911066331,
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
            "samples": [0.0, 0.01072567564137174, 0.041048135807389544, 0.08572435166444636, 0.13702939962017774, 0.18609217079130932, 0.22442926632106924] + [0.2454118541107828] * 2 + [0.22442926632106924, 0.1860921707913094, 0.13702939962017785, 0.08572435166444639, 0.041048135807389516, 0.010725675641371725, 0.0],
        },
        "q3.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0197264549822217, -0.03604202668879978, -0.046125604549351155, -0.0482336463453575, -0.0420016524993308, -0.028507191363135797, -0.010083577860551377, 0.010083577860551364, 0.028507191363135786, 0.042001652499330785, 0.04823364634535749, 0.04612560454935116, 0.03604202668879977, 0.0197264549822217, 5.4954968979104667e-17],
        },
        "q3.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.00525768413792732, 0.020121635199700744, 0.042021741011983485, 0.0671712743236165, 0.09122165234868099, 0.11001434623581818] + [0.12029992848567776] * 2 + [0.11001434623581818, 0.09122165234868103, 0.06717127432361655, 0.0420217410119835, 0.02012163519970073, 0.005257684137927313, 0.0],
        },
        "q3.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.010122313390106262, -0.018494386836732907, -0.02366861278305865, -0.02475032055659675, -0.021552473056295852, -0.014628007166490492, -0.005174225946325744, 0.005174225946325738, 0.014628007166490485, 0.02155247305629585, 0.02475032055659675, 0.023668612783058655, 0.018494386836732903, 0.010122313390106262, 2.8199259261301655e-17],
        },
        "q3.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.005257684137927319, -0.02012163519970074, -0.042021741011983485, -0.0671712743236165, -0.09122165234868099, -0.11001434623581818] + [-0.12029992848567776] * 2 + [-0.11001434623581818, -0.09122165234868103, -0.06717127432361655, -0.0420217410119835, -0.020121635199700734, -0.005257684137927314, -3.4534132592679426e-33],
        },
        "q3.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.010122313390106262, 0.01849438683673291, 0.023668612783058655, 0.024750320556596758, 0.021552473056295863, 0.014628007166490506, 0.005174225946325758, -0.005174225946325723, -0.014628007166490471, -0.02155247305629584, -0.024750320556596744, -0.02366861278305865, -0.0184943868367329, -0.010122313390106262, -2.8199259261301655e-17],
        },
        "q3.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0197264549822217, 0.03604202668879978, 0.04612560454935116, 0.048233646345357505, 0.04200165249933081, 0.02850719136313581, 0.010083577860551392, -0.010083577860551349, -0.028507191363135773, -0.04200165249933077, -0.048233646345357484, -0.046125604549351155, -0.03604202668879977, -0.0197264549822217, -5.4954968979104667e-17],
        },
        "q3.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.010725675641371737, 0.041048135807389544, 0.08572435166444636, 0.13702939962017774, 0.18609217079130932, 0.22442926632106924] + [0.2454118541107828] * 2 + [0.22442926632106924, 0.1860921707913094, 0.13702939962017785, 0.08572435166444639, 0.041048135807389516, 0.010725675641371727, 3.365021342875131e-33],
        },
        "q3.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.010122313390106262, 0.018494386836732907, 0.023668612783058655, 0.024750320556596754, 0.02155247305629586, 0.014628007166490499, 0.005174225946325751, -0.005174225946325731, -0.014628007166490478, -0.021552473056295842, -0.024750320556596747, -0.02366861278305865, -0.018494386836732903, -0.010122313390106262, -2.8199259261301655e-17],
        },
        "q3.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.005257684137927319, 0.020121635199700744, 0.042021741011983485, 0.0671712743236165, 0.09122165234868099, 0.11001434623581818] + [0.12029992848567776] * 2 + [0.11001434623581818, 0.09122165234868103, 0.06717127432361655, 0.0420217410119835, 0.02012163519970073, 0.005257684137927314, 1.7267066296339713e-33],
        },
        "q3.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.010122313390106262, -0.018494386836732907, -0.023668612783058648, -0.024750320556596747, -0.021552473056295846, -0.014628007166490485, -0.005174225946325737, 0.0051742259463257445, 0.014628007166490492, 0.021552473056295856, 0.024750320556596754, 0.023668612783058658, 0.018494386836732903, 0.010122313390106262, 2.8199259261301655e-17],
        },
        "q3.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.005257684137927321, -0.020121635199700744, -0.042021741011983485, -0.0671712743236165, -0.09122165234868099, -0.11001434623581818] + [-0.12029992848567776] * 2 + [-0.11001434623581818, -0.09122165234868103, -0.06717127432361655, -0.0420217410119835, -0.02012163519970073, -0.005257684137927312, 1.7267066296339713e-33],
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
            "sample": -0.11201840403229253,
        },
        "q3.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q3.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "q3.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q3.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "q3.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q3.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q3.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q3.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0038293854544515934, 0.01465540628385698, 0.03060614513550034, 0.04892357435401924, 0.06644044401891148, 0.08012792822935215] + [0.08761933661846567] * 2 + [0.08012792822935215, 0.06644044401891151, 0.048923574354019284, 0.030606145135500352, 0.014655406283856972, 0.0038293854544515886, 0.0],
        },
        "q3.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
        },
        "q3.xy.EF_x90.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0038310543317542738, 0.014661793229020795, 0.030619483542288252, 0.048944895645327356, 0.06646939930438014, 0.08016284863167868] + [0.08765752183745126] * 2 + [0.08016284863167868, 0.06646939930438017, 0.0489448956453274, 0.030619483542288262, 0.014661793229020784, 0.003831054331754269, 0.0],
        },
        "q3.xy.EF_x90.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0063272797074245945, -0.011560515271903887, -0.014794832721886832, -0.01547098918741978, -0.013472071072923433, -0.009143709479995186, -0.0032343174499829446, 0.0032343174499829407, 0.009143709479995182, 0.013472071072923428, 0.01547098918741978, 0.014794832721886836, 0.011560515271903885, 0.0063272797074245945, 1.7626859988630127e-17],
        },
        "q3.xy.EF_EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.005060618233354027, 0.019367446066595836, 0.040446702993907395, 0.06465359399351918, 0.08780252769889745, 0.10589084317092169] + [0.11579090633729168] * 2 + [0.10589084317092169, 0.08780252769889747, 0.06465359399351923, 0.04044670299390741, 0.019367446066595823, 0.00506061823335402, 0.0],
        },
        "q3.xy.EF_EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
        },
        "q3.z.const.wf": {
            "type": "constant",
            "sample": 0.45,
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
        "q3.z.Cz_flattop.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.00020996429830990007, 0.000838495494195144, 0.001881517321663134, 0.003332265383922333, 0.005181331023091136, 0.007416722338954665, 0.010023941961039295, 0.01298608106962331, 0.016283929055921218, 0.01989609811025345, 0.023799161930199344, 0.027967807649157687, 0.03237500000000002, 0.03699215664915227, 0.04178933356399586, 0.04673541921140972, 0.05179833632800116, 0.056945249953467884, 0.06214278037791542, 0.06735721962208469, 0.07255475004653222, 0.07770166367199893, 0.08276458078859038, 0.08771066643600425, 0.09250784335084786, 0.09712500000000007, 0.10153219235084242, 0.10570083806980077, 0.10960390188974667, 0.1132160709440789, 0.11651391893037678, 0.1194760580389608, 0.12208327766104544, 0.12431866897690898, 0.12616773461607778, 0.12761848267833698, 0.12866150450580496, 0.1292900357016902] + [0.12950000000000012] * 51 + [0.1292900357016902, 0.12866150450580496, 0.12761848267833698, 0.12616773461607778, 0.12431866897690898, 0.12208327766104544, 0.11947605803896082, 0.11651391893037681, 0.1132160709440789, 0.10960390188974667, 0.10570083806980077, 0.10153219235084242, 0.09712500000000009, 0.09250784335084784, 0.08771066643600425, 0.08276458078859039, 0.07770166367199895, 0.07255475004653222, 0.0673572196220847, 0.062142780377915444, 0.05694524995346789, 0.051798336328001175, 0.04673541921140973, 0.041789333563995855, 0.036992156649152265, 0.03237500000000004, 0.027967807649157694, 0.02379916193019935, 0.019896098110253457, 0.01628392905592121, 0.012986081069623324, 0.010023941961039309, 0.007416722338954672, 0.005181331023091136, 0.003332265383922333, 0.001881517321663134, 0.000838495494195144, 0.00020996429830990725],
        },
        "q3.z.Cz_bipolar.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.0018644112605240833, 0.007349292236493861, 0.016135881582317133, 0.027713534663167415, 0.041409400365042356, 0.05642752475579524, 0.07189510903449654, 0.08691323342524944, 0.10060909912712437, 0.11218675220797467, 0.12097334155379792, 0.1264582225297677] + [0.1283226337902918] * 26 + [0.12395014607547712, 0.1111306607429201, 0.09073780453283334, 0.06416131689514591, 0.03321234154264378, 7.857495136471942e-18, -0.0332123415426438, -0.06416131689514587, -0.09073780453283332, -0.1111306607429201, -0.12395014607547711] + [-0.1283226337902918] * 26 + [-0.1264582225297677, -0.12097334155379792, -0.11218675220797467, -0.10060909912712437, -0.08691323342524944, -0.07189510903449654, -0.056427524755795246, -0.04140940036504235, -0.027713534663167422, -0.016135881582317126, -0.007349292236493868, -0.0018644112605240833],
        },
        "q3.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.037424917343165574,
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
            "samples": [0.0, 0.008257508891827761, 0.031602237262800786, 0.06599766949737855, 0.10549652288951035, 0.14326908685187867, 0.17278414192241925] + [0.18893826694357801] * 2 + [0.17278414192241925, 0.14326908685187872, 0.10549652288951043, 0.06599766949737858, 0.031602237262800766, 0.00825750889182775, 0.0],
        },
        "q4.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0060186330029075715, -0.010996589682048117, -0.01407313610428153, -0.014716309443656846, -0.01281489918674971, -0.008697676440749274, -0.003076546422233412, 0.0030765464222334083, 0.00869767644074927, 0.012814899186749707, 0.014716309443656843, 0.014073136104281533, 0.010996589682048115, 0.0060186330029075715, 1.6767016185598995e-17],
        },
        "q4.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004229934478643018, 0.016188343815466334, 0.03380751039739494, 0.054040920257309985, 0.0733900330156742, 0.08850921129539542] + [0.09678421181848716] * 2 + [0.08850921129539542, 0.07339003301567423, 0.054040920257310034, 0.03380751039739495, 0.016188343815466324, 0.004229934478643013, 0.0],
        },
        "q4.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0023426596349290776, -0.004280252136584266, -0.0054777501589541085, -0.005728095415042857, -0.0049880009357575125, -0.003385435780113779, -0.0011974980223698427, 0.0011974980223698414, 0.003385435780113778, 0.00498800093575751, 0.005728095415042857, 0.005477750158954109, 0.004280252136584265, 0.0023426596349290776, 6.52630123771122e-18],
        },
        "q4.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.004229934478643018, -0.016188343815466334, -0.03380751039739494, -0.054040920257309985, -0.0733900330156742, -0.08850921129539542] + [-0.09678421181848716] * 2 + [-0.08850921129539542, -0.07339003301567423, -0.054040920257310034, -0.03380751039739495, -0.016188343815466324, -0.004229934478643013, -7.992413921034455e-34],
        },
        "q4.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.002342659634929078, 0.004280252136584267, 0.005477750158954113, 0.005728095415042864, 0.004988000935757521, 0.00338543578011379, 0.0011974980223698546, -0.0011974980223698295, -0.003385435780113767, -0.004988000935757501, -0.00572809541504285, -0.005477750158954105, -0.004280252136584263, -0.002342659634929077, -6.52630123771122e-18],
        },
        "q4.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.006018633002907572, 0.010996589682048119, 0.014073136104281533, 0.014716309443656853, 0.01281489918674972, 0.008697676440749284, 0.003076546422233424, -0.0030765464222333966, -0.00869767644074926, -0.012814899186749698, -0.014716309443656836, -0.01407313610428153, -0.010996589682048113, -0.006018633002907571, -1.6767016185598995e-17],
        },
        "q4.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.008257508891827761, 0.031602237262800786, 0.06599766949737855, 0.10549652288951035, 0.14326908685187867, 0.17278414192241925] + [0.18893826694357801] * 2 + [0.17278414192241925, 0.14326908685187872, 0.10549652288951043, 0.06599766949737858, 0.031602237262800766, 0.00825750889182775, 1.0266836351472837e-33],
        },
        "q4.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.002342659634929078, 0.0042802521365842665, 0.00547775015895411, 0.00572809541504286, 0.004988000935757517, 0.0033854357801137843, 0.0011974980223698486, -0.0011974980223698355, -0.0033854357801137726, -0.004988000935757506, -0.005728095415042853, -0.005477750158954108, -0.004280252136584264, -0.002342659634929077, -6.52630123771122e-18],
        },
        "q4.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.004229934478643018, 0.016188343815466334, 0.03380751039739494, 0.054040920257309985, 0.0733900330156742, 0.08850921129539542] + [0.09678421181848716] * 2 + [0.08850921129539542, 0.07339003301567423, 0.054040920257310034, 0.03380751039739495, 0.016188343815466324, 0.004229934478643013, 3.996206960517227e-34],
        },
        "q4.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.002342659634929077, -0.004280252136584265, -0.005477750158954107, -0.005728095415042853, -0.004988000935757508, -0.003385435780113774, -0.0011974980223698369, 0.0011974980223698473, 0.003385435780113783, 0.004988000935757514, 0.00572809541504286, 0.005477750158954111, 0.004280252136584266, 0.002342659634929078, 6.52630123771122e-18],
        },
        "q4.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.004229934478643018, -0.016188343815466334, -0.03380751039739494, -0.054040920257309985, -0.0733900330156742, -0.08850921129539542] + [-0.09678421181848716] * 2 + [-0.08850921129539542, -0.07339003301567423, -0.054040920257310034, -0.03380751039739495, -0.016188343815466324, -0.004229934478643013, 3.996206960517227e-34],
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
            "sample": -0.11201840403229253,
        },
        "q4.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q4.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "q4.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q4.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "q4.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q4.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q4.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q4.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0067834767221322725, 0.025960982137321047, 0.05421655133711296, 0.08666453971307633, 0.11769439529428694, 0.14194077415336528] + [0.15521125711413988] * 2 + [0.14194077415336528, 0.11769439529428699, 0.0866645397130764, 0.05421655133711298, 0.02596098213732103, 0.006783476722132264, 0.0],
        },
        "q4.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.005788982394790307, -0.010576997142267343, -0.013536152994843896, -0.014154785022522516, -0.012325926127622034, -0.008365802627732209, -0.0029591558525765525, 0.0029591558525765486, 0.008365802627732205, 0.01232592612762203, 0.014154785022522515, 0.013536152994843897, 0.010576997142267341, 0.005788982394790307, 1.6127243755302174e-17],
        },
        "q4.xy.EF_x90.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.021613635589349783, 0.08271734841028544, 0.17274575140626314, 0.2761321158169133, 0.37499999999999994, 0.45225424859373686] + [0.4945369001834514] * 2 + [0.45225424859373686, 0.3750000000000001, 0.27613211581691355, 0.1727457514062632, 0.08271734841028539, 0.021613635589349756, 0.0],
        },
        "q4.xy.EF_x90.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.01844495986931454, -0.03370061861002473, -0.04312913423255304, -0.045100230730388935, -0.03927308761220707, -0.026655270861074398, -0.009428515622528312, 0.0094285156225283, 0.026655270861074387, 0.03927308761220706, 0.045100230730388935, 0.04312913423255304, 0.03370061861002472, 0.01844495986931454, 5.1384914235859794e-17],
        },
        "q4.z.const.wf": {
            "type": "constant",
            "sample": 0.45,
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
            "sample": 0.034534179061011985,
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
            "samples": [0.0, 0.008639583896092217, 0.033064473040578836, 0.06905138220722629, 0.11037784787031495, 0.14989814867754264, 0.18077886558867814] + [0.1976804421072802] * 2 + [0.18077886558867814, 0.1498981486775427, 0.11037784787031503, 0.0690513822072263, 0.033064473040578815, 0.008639583896092205, 0.0],
        },
        "q5.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.011356376365670863, -0.020749132088276813, -0.026554174372872368, -0.027767758671297417, -0.024180045233287022, -0.016411382305626554, -0.005805042284595555, 0.005805042284595549, 0.016411382305626544, 0.02418004523328702, 0.027767758671297414, 0.02655417437287237, 0.02074913208827681, 0.011356376365670863, 3.1637175126140755e-17],
        },
        "q5.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004425653581720868, 0.016937378616803344, 0.03537178418122809, 0.05654139408158904, 0.07678579044624545, 0.09260453322918102] + [0.1012624180944595] * 2 + [0.09260453322918102, 0.07678579044624549, 0.056541394081589084, 0.035371784181228105, 0.01693737861680333, 0.004425653581720862, 0.0],
        },
        "q5.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.004230311231936023, -0.007729163220699258, -0.009891572671360102, -0.01034363934502616, -0.009007196806923771, -0.006113328113090138, -0.0021624094506608443, 0.0021624094506608417, 0.0061133281130901366, 0.009007196806923768, 0.01034363934502616, 0.009891572671360104, 0.0077291632206992564, 0.004230311231936023, 1.1785017770933489e-17],
        },
        "q5.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0044256535817208675, -0.016937378616803344, -0.03537178418122809, -0.05654139408158904, -0.07678579044624545, -0.09260453322918102] + [-0.1012624180944595] * 2 + [-0.09260453322918102, -0.07678579044624549, -0.056541394081589084, -0.035371784181228105, -0.01693737861680333, -0.004425653581720863, -1.4432484291068373e-33],
        },
        "q5.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.004230311231936024, 0.00772916322069926, 0.009891572671360106, 0.010343639345026167, 0.00900719680692378, 0.0061133281130901496, 0.002162409450660857, -0.002162409450660829, -0.006113328113090125, -0.009007196806923759, -0.010343639345026153, -0.0098915726713601, -0.007729163220699255, -0.004230311231936022, -1.1785017770933489e-17],
        },
        "q5.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.011356376365670863, 0.020749132088276816, 0.02655417437287237, 0.027767758671297424, 0.024180045233287033, 0.016411382305626564, 0.0058050422845955674, -0.005805042284595537, -0.016411382305626533, -0.02418004523328701, -0.027767758671297407, -0.026554174372872368, -0.020749132088276806, -0.011356376365670863, -3.1637175126140755e-17],
        },
        "q5.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.008639583896092217, 0.033064473040578836, 0.06905138220722629, 0.11037784787031495, 0.14989814867754264, 0.18077886558867814] + [0.1976804421072802] * 2 + [0.18077886558867814, 0.1498981486775427, 0.11037784787031503, 0.0690513822072263, 0.033064473040578815, 0.008639583896092205, 1.9372182626146267e-33],
        },
        "q5.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004230311231936023, 0.007729163220699259, 0.009891572671360104, 0.010343639345026164, 0.009007196806923776, 0.006113328113090144, 0.0021624094506608503, -0.0021624094506608356, -0.0061133281130901305, -0.009007196806923762, -0.010343639345026157, -0.009891572671360102, -0.007729163220699256, -0.004230311231936023, -1.1785017770933489e-17],
        },
        "q5.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.004425653581720868, 0.016937378616803344, 0.03537178418122809, 0.05654139408158904, 0.07678579044624545, 0.09260453322918102] + [0.1012624180944595] * 2 + [0.09260453322918102, 0.07678579044624549, 0.056541394081589084, 0.035371784181228105, 0.01693737861680333, 0.004425653581720862, 7.216242145534186e-34],
        },
        "q5.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.004230311231936023, -0.007729163220699257, -0.0098915726713601, -0.010343639345026157, -0.009007196806923766, -0.006113328113090132, -0.002162409450660838, 0.0021624094506608477, 0.006113328113090143, 0.009007196806923773, 0.010343639345026164, 0.009891572671360106, 0.007729163220699257, 0.004230311231936023, 1.1785017770933489e-17],
        },
        "q5.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.004425653581720868, -0.016937378616803344, -0.03537178418122809, -0.05654139408158904, -0.07678579044624545, -0.09260453322918102] + [-0.1012624180944595] * 2 + [-0.09260453322918102, -0.07678579044624549, -0.056541394081589084, -0.035371784181228105, -0.01693737861680333, -0.004425653581720862, 7.216242145534186e-34],
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
            "sample": 0.1,
        },
        "q5.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q5.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004819079185159693, 0.018443054169564665, 0.038516215908484155, 0.06156773237830947, 0.08361178696495504, 0.10083676236644093] + [0.11026430385186127] * 2 + [0.10083676236644093, 0.08361178696495507, 0.061567732378309516, 0.03851621590848416, 0.01844305416956465, 0.004819079185159687, 0.0],
        },
        "q5.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0057586841987536455, -0.010521639583539227, -0.013465307898236053, -0.014080702148845938, -0.01226141507875718, -0.008322017950092292, -0.0029436683146968233, 0.00294366831469682, 0.00832201795009229, 0.012261415078757176, 0.014080702148845936, 0.013465307898236053, 0.010521639583539227, 0.0057586841987536455, 1.6042837488447935e-17],
        },
        "q5.xy.EF_x90.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004618343160591265, 0.017674819153568368, 0.03691184466538491, 0.05900316322215746, 0.08012898515209145, 0.09663646392143418] + [0.10567130648522964] * 2 + [0.09663646392143418, 0.08012898515209148, 0.059003163222157505, 0.03691184466538492, 0.017674819153568357, 0.004618343160591259, 0.0],
        },
        "q5.xy.EF_x90.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0055188094574622724, -0.01008336662291937, -0.012904418094763738, -0.013494178445065425, -0.011750672751452688, -0.007975368987603152, -0.002821051471844364, 0.0028210514718443607, 0.007975368987603149, 0.011750672751452687, 0.013494178445065425, 0.012904418094763738, 0.010083366622919369, 0.0055188094574622724, 1.5374582144118778e-17],
        },
        "q5.z.const.wf": {
            "type": "constant",
            "sample": 0.45,
        },
        "q5.z.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q5.z.cz5_4.wf": {
            "type": "constant",
            "sample": -0.1732650364352627,
        },
        "q5.z.Cz_flattop.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.000162134593289498, 0.0006474868681043578, 0.0014529091286973996, 0.002573177902642726, 0.0040010278170587886, 0.005727198717339505, 0.00774049572281026, 0.010027861829824942, 0.012574462591444947, 0.015363782324520027, 0.018377731220231137, 0.02159676266344221, 0.024999999999999994, 0.02856537192984729, 0.032269755647873224, 0.03608912680417736, 0.03999871531119778, 0.04397316598723385, 0.04798670299452924, 0.05201329700547075, 0.05602683401276615, 0.06000128468880222, 0.06391087319582263, 0.06773024435212678, 0.07143462807015272, 0.075, 0.0784032373365578, 0.08162226877976886, 0.08463621767547998, 0.08742553740855506, 0.08997213817017505, 0.09225950427718974, 0.0942728012826605, 0.09599897218294122, 0.09742682209735727, 0.0985470908713026, 0.09935251313189564, 0.0998378654067105] + [0.1] * 51 + [0.09983786540671051, 0.09935251313189564, 0.0985470908713026, 0.09742682209735727, 0.09599897218294122, 0.0942728012826605, 0.09225950427718975, 0.08997213817017508, 0.08742553740855506, 0.08463621767547998, 0.08162226877976886, 0.0784032373365578, 0.07500000000000001, 0.0714346280701527, 0.06773024435212678, 0.06391087319582264, 0.060001284688802226, 0.05602683401276615, 0.05201329700547076, 0.04798670299452926, 0.04397316598723386, 0.03999871531119779, 0.03608912680417737, 0.03226975564787322, 0.028565371929847285, 0.025000000000000012, 0.021596762663442216, 0.018377731220231144, 0.015363782324520032, 0.012574462591444941, 0.010027861829824953, 0.00774049572281027, 0.00572719871733951, 0.0040010278170587886, 0.002573177902642726, 0.0014529091286973996, 0.0006474868681043578, 0.00016213459328950355],
        },
        "q5.z.Cz_bipolar.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.0003645562950973014, 0.0014529091286973996, 0.0032491878657292584, 0.005727198717339505, 0.008850806705317182, 0.012574462591444947, 0.016843867087960235, 0.02159676266344221, 0.026763841397811572, 0.032269755647873224, 0.03803421678562211, 0.04397316598723385, 0.049999999999999996, 0.05602683401276615, 0.06196578321437788, 0.06773024435212678, 0.07323615860218843, 0.0784032373365578, 0.08315613291203976, 0.08742553740855506, 0.09114919329468282, 0.0942728012826605, 0.09675081213427074, 0.0985470908713026, 0.0996354437049027] + [0.1] * 26 + [0.0992708874098054, 0.0970941817426052, 0.09350162426854149, 0.088545602565321, 0.08229838658936564, 0.0748510748171101, 0.06631226582407954, 0.05680647467311559, 0.04647231720437686, 0.03546048870425356, 0.023931566428755782, 0.012053668025532302, 6.123233995736766e-18, -0.012053668025532288, -0.02393156642875575, -0.03546048870425357, -0.046472317204376855, -0.05680647467311557, -0.0663122658240795, -0.07485107481711012, -0.08229838658936564, -0.08854560256532099, -0.09350162426854147, -0.0970941817426052, -0.0992708874098054] + [-0.1] * 26 + [-0.0996354437049027, -0.0985470908713026, -0.09675081213427075, -0.0942728012826605, -0.09114919329468282, -0.08742553740855506, -0.08315613291203977, -0.0784032373365578, -0.07323615860218843, -0.06773024435212678, -0.061965783214377894, -0.05602683401276615, -0.05, -0.04397316598723386, -0.03803421678562213, -0.03226975564787322, -0.026763841397811572, -0.021596762663442216, -0.016843867087960252, -0.012574462591444941, -0.008850806705317182, -0.00572719871733951, -0.003249187865729264, -0.0014529091286973996, -0.0003645562950973014],
        },
        "q5.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.031125563791230847,
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
            "sample": 0.45,
        },
        "coupler_q1_q2.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "coupler_q1_q2.cz.wf": {
            "type": "constant",
            "sample": -0.035,
        },
        "q1.z.SWAP_Coupler.flux_pulse_control_q1_q2.wf": {
            "type": "arbitrary",
            "samples": [0.1379194010900913] * 15 + [0.0],
        },
        "coupler_q1_q2.SWAP_Coupler.coupler_pulse_control_q1_q2.wf": {
            "type": "arbitrary",
            "samples": [-0.035] * 15 + [0.0],
        },
        "q1.z.Cz_unipolar.flux_pulse_control_q2_q1.wf": {
            "type": "arbitrary",
            "samples": [-0.09913528255422949] * 84 + [0.0] * 4,
        },
        "coupler_q1_q2.Cz_unipolar.coupler_flux_pulse_q2_q1.wf": {
            "type": "arbitrary",
            "samples": [-0.013275000000000004] * 84 + [0.0] * 4,
        },
        "q1.z.Cz_flattop.flux_pulse_control_q2_q1.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.002006682831898706, 0.007830303230627157, 0.016900804656008604, 0.02833030323062716, 0.041, 0.05366969676937286, 0.0650991953439914, 0.07416969676937286, 0.07999331716810132] + [0.08200000000000002] * 69 + [0.07999331716810132, 0.07416969676937286, 0.06509919534399142, 0.05366969676937286, 0.04100000000000001, 0.028330303230627164, 0.01690080465600861, 0.00783030323062716, 0.002006682831898706],
        },
        "coupler_q1_q2.Cz_flattop.coupler_flux_pulse_q2_q1.wf": {
            "type": "constant",
            "sample": -0.035,
        },
        "q1.z.Cz_bipolar.flux_pulse_control_q2_q1.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.004060276416000816, 0.015436918123791925, 0.031876641707791115, 0.050123358292208896, 0.06656308187620809, 0.0779397235839992] + [0.08200000000000002] * 35 + [0.07101408311032399, 0.041000000000000016, 5.021051876504149e-18, -0.04099999999999999, -0.07101408311032399] + [-0.08200000000000002] * 35 + [-0.0779397235839992, -0.06656308187620809, -0.050123358292208896, -0.03187664170779112, -0.01543691812379193, -0.00406027641600082],
        },
        "coupler_q1_q2.Cz_bipolar.coupler_flux_pulse_q2_q1.wf": {
            "type": "constant",
            "sample": -0.035,
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
            "samples": [0.08560805836349752] * 88,
        },
        "coupler_q2_q3.Cz_unipolar.coupler_flux_pulse_q2_q3.wf": {
            "type": "arbitrary",
            "samples": [-0.087] * 88,
        },
        "q3.z.Cz_flattop.flux_pulse_control_q2_q3.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.003344274688013456, 0.012867963246729308, 0.027121170425534843, 0.04393397463710173, 0.06074677884866863, 0.07499998602747417, 0.08452367458619002] + [0.08786794927420348] * 65 + [0.08452367458619002, 0.07499998602747417, 0.06074677884866864, 0.04393397463710174, 0.02712117042553485, 0.012867963246729314, 0.003344274688013456],
        },
        "coupler_q2_q3.Cz_flattop.coupler_flux_pulse_q2_q3.wf": {
            "type": "constant",
            "sample": -0.087,
        },
        "q3.z.Cz_bipolar.flux_pulse_control_q2_q3.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.005973455815072423, 0.022293240598618046, 0.04458648119723609, 0.06687972179585414, 0.08319950657939978] + [0.0891729623944722] * 33 + [0.07214244201588611, 0.027555960818650017, -0.027555960818650006, -0.07214244201588611] + [-0.0891729623944722] * 33 + [-0.08065770220517916, -0.058364461606561106, -0.030808500787911097, -0.008515260189293047],
        },
        "coupler_q2_q3.Cz_bipolar.coupler_flux_pulse_q2_q3.wf": {
            "type": "constant",
            "sample": -0.093,
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
            "samples": [0.129] * 80,
        },
        "coupler_q3_q4.Cz_unipolar.coupler_flux_pulse_q4_q3.wf": {
            "type": "arbitrary",
            "samples": [-0.1177842092514038] * 80,
        },
        "q3.z.Cz_flattop.flux_pulse_control_q4_q3.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.0031690905698888118, 0.012366149614222164, 0.02669090491406239, 0.044741149614222195, 0.06475000000000004, 0.08475885038577792, 0.10280909508593772, 0.11713385038577795, 0.1263309094301113] + [0.12950000000000012] * 109 + [0.1263309094301113, 0.11713385038577795, 0.10280909508593773, 0.08475885038577792, 0.06475000000000006, 0.0447411496142222, 0.026690904914062397, 0.012366149614222171, 0.0031690905698888118],
        },
        "coupler_q3_q4.Cz_flattop.coupler_flux_pulse_q4_q3.wf": {
            "type": "constant",
            "sample": -0.09199999999999992,
        },
        "q3.z.Cz_bipolar.flux_pulse_control_q4_q3.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.03208065844757294, 0.09624197534271883] + [0.1283226337902918] * 41 + [0.06416131689514591, -0.06416131689514587] + [-0.1283226337902918] * 40 + [-0.09624197534271885, -0.03208065844757296],
        },
        "coupler_q3_q4.Cz_bipolar.coupler_flux_pulse_q4_q3.wf": {
            "type": "constant",
            "sample": -0.109,
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
        "q5.z.Cz_unipolar.flux_pulse_control_q4_q5.wf": {
            "type": "arbitrary",
            "samples": [0.16330603996076692] * 60 + [0.0] * 16,
        },
        "coupler_q4_q5.Cz_unipolar.coupler_flux_pulse_q4_q5.wf": {
            "type": "arbitrary",
            "samples": [-0.13191014618685923] * 60 + [0.0] * 16,
        },
        "q5.z.Cz_flattop.flux_pulse_control_q4_q5.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.0024471741852423235, 0.009549150281252628, 0.020610737385376346, 0.034549150281252626, 0.049999999999999996, 0.06545084971874737, 0.07938926261462365, 0.09045084971874738, 0.09755282581475769] + [0.1] * 109 + [0.09755282581475769, 0.09045084971874738, 0.07938926261462366, 0.06545084971874737, 0.05, 0.03454915028125263, 0.02061073738537635, 0.009549150281252633, 0.0024471741852423235],
        },
        "coupler_q4_q5.Cz_flattop.coupler_flux_pulse_q4_q5.wf": {
            "type": "constant",
            "sample": 0.1,
        },
        "q5.z.Cz_bipolar.flux_pulse_control_q4_q5.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.004951556604879043, 0.01882550990706332, 0.03887395330218428, 0.06112604669781572, 0.08117449009293667, 0.09504844339512096] + [0.1] * 55 + [0.08660254037844388, 0.05000000000000002, 6.123233995736766e-18, -0.04999999999999998, -0.08660254037844388] + [-0.1] * 55 + [-0.09504844339512096, -0.08117449009293669, -0.06112604669781572, -0.03887395330218429, -0.018825509907063328, -0.004951556604879049],
        },
        "coupler_q4_q5.Cz_bipolar.coupler_flux_pulse_q4_q5.wf": {
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
            "cosine": [(-0.9533486847849908, 1200)],
            "sine": [(-0.3018713057227006, 1200)],
        },
        "q1.resonator.readout.iw2": {
            "cosine": [(0.3018713057227006, 1200)],
            "sine": [(-0.9533486847849908, 1200)],
        },
        "q1.resonator.readout.iw3": {
            "cosine": [(-0.3018713057227006, 1200)],
            "sine": [(0.9533486847849908, 1200)],
        },
        "q2.resonator.readout.iw1": {
            "cosine": [(-0.7963394329167912, 1200)],
            "sine": [(-0.6048499876678212, 1200)],
        },
        "q2.resonator.readout.iw2": {
            "cosine": [(0.6048499876678212, 1200)],
            "sine": [(-0.7963394329167912, 1200)],
        },
        "q2.resonator.readout.iw3": {
            "cosine": [(-0.6048499876678212, 1200)],
            "sine": [(0.7963394329167912, 1200)],
        },
        "q3.resonator.readout.iw1": {
            "cosine": [(0.9324853686940456, 1200)],
            "sine": [(0.36120774793950594, 1200)],
        },
        "q3.resonator.readout.iw2": {
            "cosine": [(-0.36120774793950594, 1200)],
            "sine": [(0.9324853686940456, 1200)],
        },
        "q3.resonator.readout.iw3": {
            "cosine": [(0.36120774793950594, 1200)],
            "sine": [(-0.9324853686940456, 1200)],
        },
        "q4.resonator.readout.iw1": {
            "cosine": [(0.7963131694988432, 1200)],
            "sine": [(0.6048845642622289, 1200)],
        },
        "q4.resonator.readout.iw2": {
            "cosine": [(-0.6048845642622289, 1200)],
            "sine": [(0.7963131694988432, 1200)],
        },
        "q4.resonator.readout.iw3": {
            "cosine": [(0.6048845642622289, 1200)],
            "sine": [(-0.7963131694988432, 1200)],
        },
        "q5.resonator.readout.iw1": {
            "cosine": [(-0.5971635673658889, 1200)],
            "sine": [(-0.802119488487124, 1200)],
        },
        "q5.resonator.readout.iw2": {
            "cosine": [(0.802119488487124, 1200)],
            "sine": [(-0.5971635673658889, 1200)],
        },
        "q5.resonator.readout.iw3": {
            "cosine": [(-0.802119488487124, 1200)],
            "sine": [(0.5971635673658889, 1200)],
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
                        "8": {
                            "offset": 0.0,
                            "delay": 92,
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
                            "output_mode": "direct",
                        },
                    },
                },
                "1": {
                    "type": "LF",
                    "analog_outputs": {
                        "1": {
                            "offset": 0.0,
                            "delay": 57,
                            "shareable": False,
                            "filter": {
                                "feedforward": [],
                                "exponential": [(-0.014222759854540867, 117417.44175630683), (0.0030980452178688738, 979.8296492766553), (0.03626492869698257, 19.764228103817157), (0.17260247326976075, 3750.23093928198), (-0.07156597682800245, 26.14624901197679)],
                                "high_pass": None,
                                "exponential_dc_gain": None,
                            },
                            "crosstalk": {},
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "direct",
                        },
                        "2": {
                            "offset": 0.0,
                            "delay": 56,
                            "shareable": False,
                            "filter": {
                                "feedforward": [],
                                "exponential": [(-0.01769653680971151, 330.60787392639077), (-0.07237663070119806, 10.417561895314897), (0.01264328247686496, 48611.89022732948), (-0.12045387577060747, 1.23381071262837)],
                                "high_pass": None,
                                "exponential_dc_gain": None,
                            },
                            "crosstalk": {},
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "direct",
                        },
                        "3": {
                            "offset": 0.0,
                            "delay": 55,
                            "shareable": False,
                            "filter": {
                                "feedforward": [],
                                "exponential": [(-0.010260654320882587, 62.84454837804313), (-0.06442551060715064, 10.924763516892996), (0.004364243812839698, 13978176.988923818), (-0.01780539123873645, 15.873930395927458)],
                                "high_pass": None,
                                "exponential_dc_gain": None,
                            },
                            "crosstalk": {
                                "7": -0.0275,
                            },
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "direct",
                        },
                        "4": {
                            "offset": 0.0,
                            "delay": 87,
                            "shareable": False,
                            "filter": {
                                "feedforward": [],
                                "exponential": [(0.05121889750694114, 682742.4294348056), (-0.011053862447955358, 194.40473375724764), (0.02946608537070822, 1005051.2239588167), (-0.0051886944506610065, 83.71159271751709), (0.02899826727834642, 1879.1354237436165), (-0.09559799628857335, 9.446550360023931)],
                                "high_pass": None,
                                "exponential_dc_gain": None,
                            },
                            "crosstalk": {},
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "direct",
                        },
                        "5": {
                            "offset": 0.0,
                            "delay": 85,
                            "shareable": False,
                            "filter": {
                                "feedforward": [],
                                "exponential": [(-0.02584887446705352, 24155.108018680057), (-0.040818588224334644, 13.56262165133981), (-0.034567576179888336, 4.36824367678182), (0.0022211663552704113, 54585060.68856716), (-0.019019463770724016, 31.82265239390185)],
                                "high_pass": None,
                                "exponential_dc_gain": None,
                            },
                            "crosstalk": {},
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "direct",
                        },
                        "6": {
                            "offset": 0.0,
                            "delay": 57,
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
                            "output_mode": "direct",
                        },
                        "7": {
                            "offset": 0.0,
                            "delay": 55,
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
                            "output_mode": "direct",
                        },
                        "8": {
                            "offset": 0.0,
                            "delay": 85,
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
                            "output_mode": "direct",
                        },
                    },
                },
                "6": {
                    "type": "MW",
                    "analog_outputs": {
                        "1": {
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 1,
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
                            "full_scale_power_dbm": 7,
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
                            "full_scale_power_dbm": 7,
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
                            "full_scale_power_dbm": 7,
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
                            "full_scale_power_dbm": 7,
                            "band": 1,
                            "delay": 20,
                            "shareable": False,
                            "upconverters": {
                                "1": {
                                    "frequency": 4800000000.0,
                                },
                            },
                        },
                        "6": {
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 7,
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
            "core": "a",
            "MWInput": {
                "port": ('con1', 6, 2),
                "upconverter": 1,
            },
            "intermediate_frequency": 222946400.64122882,
        },
        "q1.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "q1.z.const.pulse",
                "flux_pulse": "q1.z.flux_pulse.pulse",
                "cz1_2": "q1.z.cz1_2.pulse",
                "Cz_flattop": "q1.z.Cz_flattop.pulse",
                "Cz_bipolar": "q1.z.Cz_bipolar.pulse",
                "SWAP_Coupler.flux_pulse_control_q1_q2": "q1.z.SWAP_Coupler.flux_pulse_control_q1_q2.pulse",
                "Cz_unipolar.flux_pulse_control_q2_q1": "q1.z.Cz_unipolar.flux_pulse_control_q2_q1.pulse",
                "Cz_flattop.flux_pulse_control_q2_q1": "q1.z.Cz_flattop.flux_pulse_control_q2_q1.pulse",
                "Cz_bipolar.flux_pulse_control_q2_q1": "q1.z.Cz_bipolar.flux_pulse_control_q2_q1.pulse",
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
            "core": "a",
            "MWInput": {
                "port": ('con1', 6, 1),
                "upconverter": 1,
            },
            "MWOutput": {
                "port": ('con1', 6, 1),
            },
            "smearing": 0,
            "time_of_flight": 384,
            "intermediate_frequency": -19074242.0,
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
                "EF_EF_x180": "q2.xy.EF_EF_x180.pulse",
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
                "port": ('con1', 6, 3),
                "upconverter": 1,
            },
            "intermediate_frequency": -62998976.43893273,
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
            "core": "b",
            "MWInput": {
                "port": ('con1', 6, 1),
                "upconverter": 1,
            },
            "MWOutput": {
                "port": ('con1', 6, 1),
            },
            "smearing": 0,
            "time_of_flight": 376,
            "intermediate_frequency": 75584485.0,
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
                "EF_EF_x180": "q3.xy.EF_EF_x180.pulse",
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
                "port": ('con1', 6, 4),
                "upconverter": 1,
            },
            "intermediate_frequency": 195146679.45820576,
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
                "Cz_flattop": "q3.z.Cz_flattop.pulse",
                "Cz_bipolar": "q3.z.Cz_bipolar.pulse",
                "SWAP_Coupler.flux_pulse_control_q2_q3": "q3.z.SWAP_Coupler.flux_pulse_control_q2_q3.pulse",
                "Cz_unipolar.flux_pulse_control_q2_q3": "q3.z.Cz_unipolar.flux_pulse_control_q2_q3.pulse",
                "Cz_flattop.flux_pulse_control_q2_q3": "q3.z.Cz_flattop.flux_pulse_control_q2_q3.pulse",
                "Cz_bipolar.flux_pulse_control_q2_q3": "q3.z.Cz_bipolar.flux_pulse_control_q2_q3.pulse",
                "SWAP_Coupler.flux_pulse_control_q3_q4": "q3.z.SWAP_Coupler.flux_pulse_control_q3_q4.pulse",
                "Cz_unipolar.flux_pulse_control_q4_q3": "q3.z.Cz_unipolar.flux_pulse_control_q4_q3.pulse",
                "Cz_flattop.flux_pulse_control_q4_q3": "q3.z.Cz_flattop.flux_pulse_control_q4_q3.pulse",
                "Cz_bipolar.flux_pulse_control_q4_q3": "q3.z.Cz_bipolar.flux_pulse_control_q4_q3.pulse",
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
            "core": "c",
            "MWInput": {
                "port": ('con1', 6, 1),
                "upconverter": 1,
            },
            "MWOutput": {
                "port": ('con1', 6, 1),
            },
            "smearing": 0,
            "time_of_flight": 384,
            "intermediate_frequency": -89152080.0,
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
            "core": "d",
            "MWInput": {
                "port": ('con1', 6, 5),
                "upconverter": 1,
            },
            "intermediate_frequency": -95054563.7123453,
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
            "singleInput": {
                "port": ('con1', 1, 4),
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
            "core": "d",
            "MWInput": {
                "port": ('con1', 6, 1),
                "upconverter": 1,
            },
            "MWOutput": {
                "port": ('con1', 6, 1),
            },
            "smearing": 0,
            "time_of_flight": 384,
            "intermediate_frequency": 130789974.0,
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
            "core": "e",
            "MWInput": {
                "port": ('con1', 6, 6),
                "upconverter": 1,
            },
            "intermediate_frequency": -3756251.2312046573,
        },
        "q5.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "q5.z.const.pulse",
                "flux_pulse": "q5.z.flux_pulse.pulse",
                "cz5_4": "q5.z.cz5_4.pulse",
                "Cz_flattop": "q5.z.Cz_flattop.pulse",
                "Cz_bipolar": "q5.z.Cz_bipolar.pulse",
                "Cz_unipolar.flux_pulse_control_q4_q5": "q5.z.Cz_unipolar.flux_pulse_control_q4_q5.pulse",
                "Cz_flattop.flux_pulse_control_q4_q5": "q5.z.Cz_flattop.flux_pulse_control_q4_q5.pulse",
                "Cz_bipolar.flux_pulse_control_q4_q5": "q5.z.Cz_bipolar.flux_pulse_control_q4_q5.pulse",
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
                "port": ('con1', 1, 5),
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
            "core": "e",
            "MWInput": {
                "port": ('con1', 6, 1),
                "upconverter": 1,
            },
            "MWOutput": {
                "port": ('con1', 6, 1),
            },
            "smearing": 0,
            "time_of_flight": 384,
            "intermediate_frequency": 22227706.0,
        },
        "coupler_q1_q2": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "coupler_q1_q2.const.pulse",
                "flux_pulse": "coupler_q1_q2.flux_pulse.pulse",
                "cz": "coupler_q1_q2.cz.pulse",
                "SWAP_Coupler.coupler_pulse_control_q1_q2": "coupler_q1_q2.SWAP_Coupler.coupler_pulse_control_q1_q2.pulse",
                "Cz_unipolar.coupler_flux_pulse_q2_q1": "coupler_q1_q2.Cz_unipolar.coupler_flux_pulse_q2_q1.pulse",
                "Cz_flattop.coupler_flux_pulse_q2_q1": "coupler_q1_q2.Cz_flattop.coupler_flux_pulse_q2_q1.pulse",
                "Cz_bipolar.coupler_flux_pulse_q2_q1": "coupler_q1_q2.Cz_bipolar.coupler_flux_pulse_q2_q1.pulse",
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
                "port": ('con1', 1, 6),
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
                "Cz_flattop.coupler_flux_pulse_q2_q3": "coupler_q2_q3.Cz_flattop.coupler_flux_pulse_q2_q3.pulse",
                "Cz_bipolar.coupler_flux_pulse_q2_q3": "coupler_q2_q3.Cz_bipolar.coupler_flux_pulse_q2_q3.pulse",
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
                "port": ('con1', 1, 7),
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
                "Cz_flattop.coupler_flux_pulse_q4_q3": "coupler_q3_q4.Cz_flattop.coupler_flux_pulse_q4_q3.pulse",
                "Cz_bipolar.coupler_flux_pulse_q4_q3": "coupler_q3_q4.Cz_bipolar.coupler_flux_pulse_q4_q3.pulse",
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
                "port": ('con1', 1, 8),
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
                "Cz_unipolar.coupler_flux_pulse_q4_q5": "coupler_q4_q5.Cz_unipolar.coupler_flux_pulse_q4_q5.pulse",
                "Cz_flattop.coupler_flux_pulse_q4_q5": "coupler_q4_q5.Cz_flattop.coupler_flux_pulse_q4_q5.pulse",
                "Cz_bipolar.coupler_flux_pulse_q4_q5": "coupler_q4_q5.Cz_bipolar.coupler_flux_pulse_q4_q5.pulse",
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
                "port": ('con1', 2, 8),
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
        "q1.z.Cz_flattop.pulse": {
            "length": 88,
            "waveforms": {
                "single": "q1.z.Cz_flattop.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q1.z.Cz_bipolar.pulse": {
            "length": 88,
            "waveforms": {
                "single": "q1.z.Cz_bipolar.wf",
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
        "q2.xy.EF_EF_x180.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q2.xy.EF_EF_x180.wf.I",
                "Q": "q2.xy.EF_EF_x180.wf.Q",
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
            "length": 1200,
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
        "q3.xy.EF_EF_x180.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q3.xy.EF_EF_x180.wf.I",
                "Q": "q3.xy.EF_EF_x180.wf.Q",
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
        "q3.z.Cz_flattop.pulse": {
            "length": 128,
            "waveforms": {
                "single": "q3.z.Cz_flattop.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q3.z.Cz_bipolar.pulse": {
            "length": 88,
            "waveforms": {
                "single": "q3.z.Cz_bipolar.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q3.resonator.readout.pulse": {
            "length": 1200,
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
            "length": 16,
            "waveforms": {
                "single": "q4.z.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q4.z.flux_pulse.pulse": {
            "length": 16,
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
            "length": 1200,
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
        "q5.z.Cz_flattop.pulse": {
            "length": 128,
            "waveforms": {
                "single": "q5.z.Cz_flattop.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q5.z.Cz_bipolar.pulse": {
            "length": 128,
            "waveforms": {
                "single": "q5.z.Cz_bipolar.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q5.resonator.readout.pulse": {
            "length": 1200,
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
        "q1.z.SWAP_Coupler.flux_pulse_control_q1_q2.pulse": {
            "length": 16,
            "waveforms": {
                "single": "q1.z.SWAP_Coupler.flux_pulse_control_q1_q2.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q1_q2.SWAP_Coupler.coupler_pulse_control_q1_q2.pulse": {
            "length": 16,
            "waveforms": {
                "single": "coupler_q1_q2.SWAP_Coupler.coupler_pulse_control_q1_q2.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q1.z.Cz_unipolar.flux_pulse_control_q2_q1.pulse": {
            "length": 88,
            "waveforms": {
                "single": "q1.z.Cz_unipolar.flux_pulse_control_q2_q1.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q1_q2.Cz_unipolar.coupler_flux_pulse_q2_q1.pulse": {
            "length": 88,
            "waveforms": {
                "single": "coupler_q1_q2.Cz_unipolar.coupler_flux_pulse_q2_q1.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q1.z.Cz_flattop.flux_pulse_control_q2_q1.pulse": {
            "length": 88,
            "waveforms": {
                "single": "q1.z.Cz_flattop.flux_pulse_control_q2_q1.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q1_q2.Cz_flattop.coupler_flux_pulse_q2_q1.pulse": {
            "length": 88,
            "waveforms": {
                "single": "coupler_q1_q2.Cz_flattop.coupler_flux_pulse_q2_q1.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q1.z.Cz_bipolar.flux_pulse_control_q2_q1.pulse": {
            "length": 88,
            "waveforms": {
                "single": "q1.z.Cz_bipolar.flux_pulse_control_q2_q1.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q1_q2.Cz_bipolar.coupler_flux_pulse_q2_q1.pulse": {
            "length": 88,
            "waveforms": {
                "single": "coupler_q1_q2.Cz_bipolar.coupler_flux_pulse_q2_q1.wf",
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
        "q3.z.Cz_flattop.flux_pulse_control_q2_q3.pulse": {
            "length": 80,
            "waveforms": {
                "single": "q3.z.Cz_flattop.flux_pulse_control_q2_q3.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q2_q3.Cz_flattop.coupler_flux_pulse_q2_q3.pulse": {
            "length": 80,
            "waveforms": {
                "single": "coupler_q2_q3.Cz_flattop.coupler_flux_pulse_q2_q3.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q3.z.Cz_bipolar.flux_pulse_control_q2_q3.pulse": {
            "length": 80,
            "waveforms": {
                "single": "q3.z.Cz_bipolar.flux_pulse_control_q2_q3.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q2_q3.Cz_bipolar.coupler_flux_pulse_q2_q3.pulse": {
            "length": 80,
            "waveforms": {
                "single": "coupler_q2_q3.Cz_bipolar.coupler_flux_pulse_q2_q3.wf",
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
            "length": 80,
            "waveforms": {
                "single": "q3.z.Cz_unipolar.flux_pulse_control_q4_q3.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q3_q4.Cz_unipolar.coupler_flux_pulse_q4_q3.pulse": {
            "length": 80,
            "waveforms": {
                "single": "coupler_q3_q4.Cz_unipolar.coupler_flux_pulse_q4_q3.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q3.z.Cz_flattop.flux_pulse_control_q4_q3.pulse": {
            "length": 128,
            "waveforms": {
                "single": "q3.z.Cz_flattop.flux_pulse_control_q4_q3.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q3_q4.Cz_flattop.coupler_flux_pulse_q4_q3.pulse": {
            "length": 128,
            "waveforms": {
                "single": "coupler_q3_q4.Cz_flattop.coupler_flux_pulse_q4_q3.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q3.z.Cz_bipolar.flux_pulse_control_q4_q3.pulse": {
            "length": 88,
            "waveforms": {
                "single": "q3.z.Cz_bipolar.flux_pulse_control_q4_q3.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q3_q4.Cz_bipolar.coupler_flux_pulse_q4_q3.pulse": {
            "length": 88,
            "waveforms": {
                "single": "coupler_q3_q4.Cz_bipolar.coupler_flux_pulse_q4_q3.wf",
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
        "q5.z.Cz_unipolar.flux_pulse_control_q4_q5.pulse": {
            "length": 76,
            "waveforms": {
                "single": "q5.z.Cz_unipolar.flux_pulse_control_q4_q5.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q4_q5.Cz_unipolar.coupler_flux_pulse_q4_q5.pulse": {
            "length": 76,
            "waveforms": {
                "single": "coupler_q4_q5.Cz_unipolar.coupler_flux_pulse_q4_q5.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q5.z.Cz_flattop.flux_pulse_control_q4_q5.pulse": {
            "length": 128,
            "waveforms": {
                "single": "q5.z.Cz_flattop.flux_pulse_control_q4_q5.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q4_q5.Cz_flattop.coupler_flux_pulse_q4_q5.pulse": {
            "length": 128,
            "waveforms": {
                "single": "coupler_q4_q5.Cz_flattop.coupler_flux_pulse_q4_q5.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q5.z.Cz_bipolar.flux_pulse_control_q4_q5.pulse": {
            "length": 128,
            "waveforms": {
                "single": "q5.z.Cz_bipolar.flux_pulse_control_q4_q5.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q4_q5.Cz_bipolar.coupler_flux_pulse_q4_q5.pulse": {
            "length": 128,
            "waveforms": {
                "single": "coupler_q4_q5.Cz_bipolar.coupler_flux_pulse_q4_q5.wf",
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
            "samples": [0.0, 0.013349337165626278, 0.051089126991647416, 0.10669381696191726, 0.1705488510278263, 0.231613113694606, 0.27932803919575944] + [0.30544328343564736] * 2 + [0.27932803919575944, 0.23161311369460608, 0.17054885102782644, 0.10669381696191729, 0.05108912699164738, 0.013349337165626262, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.008571745181666944, -0.015661357749563374, -0.02004297928359179, -0.02095898761473669, -0.018250996580868675, -0.012387242433069748, -0.004381621534028416, 0.0043816215340284105, 0.012387242433069743, 0.01825099658086867, 0.020958987614736687, 0.020042979283591795, 0.01566135774956337, 0.008571745181666944, 2.3879606902498985e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.005407777688234955, 0.020696056862526, 0.043221355163982164, 0.06908884388037084, 0.09382579921387094, 0.11315497685913609] + [0.12373418640123374] * 2 + [0.11315497685913609, 0.09382579921387098, 0.0690888438803709, 0.04322135516398218, 0.020696056862525986, 0.005407777688234948, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.008265624995918104, -0.01510204833919625, -0.01932719032682534, -0.020210485524934353, -0.01759920416928529, -0.01194486052901625, -0.004225141987629084, 0.00422514198762908, 0.011944860529016246, 0.017599204169285287, 0.020210485524934353, 0.01932719032682534, 0.015102048339196247, 0.008265624995918104, 2.3026801604899053e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.005407777688234954, -0.020696056862525996, -0.043221355163982164, -0.06908884388037084, -0.09382579921387094, -0.11315497685913609] + [-0.12373418640123374] * 2 + [-0.11315497685913609, -0.09382579921387098, -0.0690888438803709, -0.04322135516398218, -0.02069605686252599, -0.005407777688234949, -2.819969888004076e-33],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.008265624995918104, 0.015102048339196253, 0.019327190326825346, 0.02021048552493436, 0.0175992041692853, 0.011944860529016265, 0.004225141987629099, -0.004225141987629065, -0.011944860529016232, -0.017599204169285277, -0.020210485524934346, -0.01932719032682533, -0.015102048339196246, -0.008265624995918104, -2.3026801604899053e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.008571745181666944, 0.015661357749563377, 0.0200429792835918, 0.0209589876147367, 0.01825099658086869, 0.012387242433069766, 0.004381621534028435, -0.004381621534028391, -0.012387242433069726, -0.018250996580868657, -0.020958987614736677, -0.020042979283591788, -0.015661357749563367, -0.008571745181666944, -2.3879606902498985e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.013349337165626278, 0.051089126991647416, 0.10669381696191726, 0.1705488510278263, 0.231613113694606, 0.27932803919575944] + [0.30544328343564736] * 2 + [0.27932803919575944, 0.23161311369460608, 0.17054885102782644, 0.10669381696191729, 0.05108912699164738, 0.013349337165626262, 1.4622042079021211e-33],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.008265624995918104, 0.015102048339196253, 0.019327190326825342, 0.020210485524934357, 0.017599204169285298, 0.011944860529016258, 0.004225141987629092, -0.004225141987629072, -0.011944860529016239, -0.01759920416928528, -0.02021048552493435, -0.019327190326825335, -0.015102048339196246, -0.008265624995918104, -2.3026801604899053e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.005407777688234954, 0.020696056862526, 0.043221355163982164, 0.06908884388037084, 0.09382579921387094, 0.11315497685913609] + [0.12373418640123374] * 2 + [0.11315497685913609, 0.09382579921387098, 0.0690888438803709, 0.04322135516398218, 0.020696056862525986, 0.005407777688234949, 1.409984944002038e-33],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.008265624995918104, -0.015102048339196249, -0.019327190326825335, -0.02021048552493435, -0.017599204169285284, -0.011944860529016244, -0.004225141987629077, 0.004225141987629088, 0.011944860529016253, 0.017599204169285294, 0.020210485524934357, 0.019327190326825342, 0.015102048339196249, 0.008265624995918104, 2.3026801604899053e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.005407777688234956, -0.020696056862526, -0.043221355163982164, -0.06908884388037084, -0.09382579921387094, -0.11315497685913609] + [-0.12373418640123374] * 2 + [-0.11315497685913609, -0.09382579921387098, -0.0690888438803709, -0.04322135516398218, -0.020696056862525986, -0.005407777688234947, 1.409984944002038e-33],
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
            "sample": 0.1,
        },
        "q1.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q1.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.006717669725766446, 0.025709132779367188, 0.05369059266680922, 0.08582380076997109, 0.11655263348678496, 0.1405637964778324] + [0.15370554152739355] * 2 + [0.1405637964778324, 0.11655263348678502, 0.08582380076997116, 0.05369059266680924, 0.02570913277936717, 0.006717669725766437, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0102278806658164, -0.018687267847134304, -0.023915456649183944, -0.02500844573148675, -0.02177724735221848, -0.014780565065670352, -0.005228188802049635, 0.005228188802049629, 0.014780565065670347, 0.021777247352218473, 0.025008445731486747, 0.023915456649183948, 0.018687267847134304, 0.0102278806658164, 2.849335398673954e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.EF_x90.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004593670635837815, 0.01758039515822729, 0.036714650916558844, 0.0586879512609605, 0.07970091294071842, 0.09612020398463854] + [0.10510677980665072] * 2 + [0.09612020398463854, 0.07970091294071845, 0.05868795126096055, 0.03671465091655885, 0.017580395158227277, 0.004593670635837809, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.EF_x90.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.002669837807754793, -0.004878036403833742, -0.006242778190120334, -0.006528086913475712, -0.005684630103683355, -0.00385824910572092, -0.0013647417862865906, 0.001364741786286589, 0.0038582491057209184, 0.005684630103683353, 0.006528086913475712, 0.006242778190120334, 0.004878036403833741, 0.002669837807754793, 7.437770954620908e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.z.const.wf": {
            "type": "constant",
            "sample": 0.45,
        },
        "q1.z.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q1.z.cz1_2.wf": {
            "type": "constant",
            "sample": -0.07009506167631502,
        },
        "q1.z.Cz_flattop.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.0005591865604883849, 0.0022214930902739803, 0.004941576200533949, 0.008645239114747865, 0.013231455563344617, 0.0185751255169805, 0.024530487589228253, 0.03093509502722724, 0.03761424683563438, 0.04438575316436563, 0.05106490497277278, 0.05746951241077176, 0.0634248744830195, 0.06876854443665538, 0.07335476088525215, 0.07705842379946606, 0.07977850690972603, 0.08144081343951164] + [0.08200000000000002] * 51 + [0.08144081343951164, 0.07977850690972603, 0.07705842379946606, 0.07335476088525215, 0.06876854443665541, 0.06342487448301952, 0.057469512410771764, 0.05106490497277278, 0.04438575316436564, 0.037614246835634385, 0.030935095027227244, 0.024530487589228267, 0.01857512551698051, 0.013231455563344627, 0.00864523911474787, 0.004941576200533954, 0.0022214930902739803, 0.0005591865604883849],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.z.Cz_bipolar.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.0011913854855318678, 0.004696302948218395, 0.010311059324984858, 0.017709345384022617, 0.02646119963125605, 0.03605799610953176, 0.04594200389046825, 0.05553880036874397, 0.0642906546159774, 0.07168894067501516, 0.07730369705178161, 0.08080861451446815] + [0.08200000000000002] * 26 + [0.07920591775570361, 0.07101408311032399, 0.057982756057296914, 0.041000000000000016, 0.021223161698406705, 5.021051876504149e-18, -0.021223161698406715, -0.04099999999999999, -0.05798275605729691, -0.07101408311032399, -0.07920591775570361] + [-0.08200000000000002] * 26 + [-0.08080861451446815, -0.07730369705178161, -0.07168894067501516, -0.0642906546159774, -0.05553880036874397, -0.04594200389046825, -0.03605799610953177, -0.02646119963125604, -0.01770934538402262, -0.010311059324984854, -0.004696302948218399, -0.0011913854855318678],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.04269504351718363,
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
            "samples": [0.0, 0.009567007456445571, 0.036613807323028616, 0.07646376219035395, 0.1222264528363863, 0.1659890943074338, 0.20018472832203574] + [0.21890061910148506] * 2 + [0.20018472832203574, 0.16598909430743386, 0.1222264528363864, 0.07646376219035397, 0.03661380732302859, 0.009567007456445559, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.016125390729056558, -0.029462554906483456, -0.037705375681670816, -0.03942859445891286, -0.03433425105667332, -0.023303203729856305, -0.008242820775187357, 0.008242820775187346, 0.023303203729856295, 0.034334251056673315, 0.03942859445891286, 0.037705375681670816, 0.029462554906483453, 0.016125390729056558, 4.4922939681250316e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0047081729608491945, 0.018018605966057376, 0.03762980422753636, 0.06015081340373337, 0.08168754641113861, 0.09851610645769468] + [0.10772668262868351] * 2 + [0.09851610645769468, 0.08168754641113865, 0.06015081340373342, 0.037629804227536374, 0.018018605966057363, 0.0047081729608491885, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.00696055063940401, -0.01271755883863767, -0.016275585579275897, -0.017019415714204164, -0.014820434255608732, -0.010058865074800154, -0.003558026740638228, 0.0035580267406382237, 0.01005886507480015, 0.014820434255608729, 0.01701941571420416, 0.0162755855792759, 0.012717558838637668, 0.00696055063940401, 1.93910585967258e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.004708172960849194, -0.018018605966057376, -0.03762980422753636, -0.06015081340373337, -0.08168754641113861, -0.09851610645769468] + [-0.10772668262868351] * 2 + [-0.09851610645769468, -0.08168754641113865, -0.06015081340373342, -0.037629804227536374, -0.018018605966057363, -0.004708172960849189, -2.3747197842559014e-33],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.006960550639404011, 0.012717558838637672, 0.0162755855792759, 0.01701941571420417, 0.014820434255608743, 0.010058865074800166, 0.003558026740638241, -0.0035580267406382106, -0.010058865074800138, -0.014820434255608719, -0.017019415714204154, -0.016275585579275897, -0.012717558838637667, -0.006960550639404009, -1.93910585967258e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.016125390729056558, 0.02946255490648346, 0.03770537568167082, 0.03942859445891287, 0.03433425105667333, 0.02330320372985632, 0.00824282077518737, -0.008242820775187332, -0.02330320372985628, -0.03433425105667331, -0.039428594458912856, -0.03770537568167081, -0.02946255490648345, -0.016125390729056558, -4.4922939681250316e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.00956700745644557, 0.036613807323028616, 0.07646376219035395, 0.1222264528363863, 0.1659890943074338, 0.20018472832203574] + [0.21890061910148506] * 2 + [0.20018472832203574, 0.16598909430743386, 0.1222264528363864, 0.07646376219035397, 0.03661380732302859, 0.00956700745644556, 2.750736714446641e-33],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.00696055063940401, 0.012717558838637672, 0.0162755855792759, 0.017019415714204168, 0.014820434255608738, 0.01005886507480016, 0.0035580267406382345, -0.003558026740638217, -0.010058865074800145, -0.014820434255608724, -0.017019415714204157, -0.016275585579275897, -0.012717558838637667, -0.00696055063940401, -1.93910585967258e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0047081729608491945, 0.018018605966057376, 0.03762980422753636, 0.06015081340373337, 0.08168754641113861, 0.09851610645769468] + [0.10772668262868351] * 2 + [0.09851610645769468, 0.08168754641113865, 0.06015081340373342, 0.037629804227536374, 0.018018605966057363, 0.0047081729608491885, 1.1873598921279507e-33],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.00696055063940401, -0.012717558838637668, -0.016275585579275893, -0.01701941571420416, -0.014820434255608727, -0.010058865074800149, -0.0035580267406382215, 0.00355802674063823, 0.010058865074800156, 0.014820434255608734, 0.017019415714204164, 0.016275585579275904, 0.01271755883863767, 0.00696055063940401, 1.93910585967258e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0047081729608491945, -0.018018605966057376, -0.03762980422753636, -0.06015081340373337, -0.08168754641113861, -0.09851610645769468] + [-0.10772668262868351] * 2 + [-0.09851610645769468, -0.08168754641113865, -0.06015081340373342, -0.037629804227536374, -0.018018605966057363, -0.0047081729608491885, 1.1873598921279507e-33],
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
            "sample": 0.1,
        },
        "q2.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.00522147920474363, 0.01998307562882494, 0.0417323751456268, 0.06670872628251098, 0.09059349102488344, 0.1092567765625123] + [0.11947153127531518] * 2 + [0.1092567765625123, 0.09059349102488348, 0.06670872628251102, 0.04173237514562681, 0.019983075628824926, 0.005221479204743623, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0077614939281682414, -0.014180955045197449, -0.018148400204979865, -0.01897782209628136, -0.01652580613903457, -0.011216328168113118, -0.003967445159782416, 0.003967445159782412, 0.011216328168113113, 0.016525806139034562, 0.018977822096281356, 0.01814840020497987, 0.014180955045197445, 0.0077614939281682414, 2.1622367447085848e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.EF_x90.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.005559686906605402, 0.021277427252099614, 0.044435480939558375, 0.07102961010210704, 0.09646144820746218, 0.11633359940621195] + [0.1272099882232664] * 2 + [0.11633359940621195, 0.09646144820746223, 0.0710296101021071, 0.044435480939558396, 0.0212774272520996, 0.005559686906605395, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.EF_x90.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0036246827501982067, -0.006622624923678124, -0.008475455083195447, -0.008862802061736062, -0.007717690047773467, -0.005238119311537855, -0.0018528301595173225, 0.0018528301595173203, 0.005238119311537853, 0.007717690047773464, 0.008862802061736062, 0.008475455083195449, 0.006622624923678123, 0.0036246827501982067, 1.0097826916988402e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.EF_EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.005272133834660322, 0.020176935502796242, 0.04213722939011479, 0.06735588118049173, 0.09147235687511178, 0.11031669873507154] + [0.12063054885731256] * 2 + [0.11031669873507154, 0.09147235687511182, 0.06735588118049178, 0.04213722939011481, 0.02017693550279623, 0.005272133834660315, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.EF_EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.007836789756632871, -0.014318527369344046, -0.018324461520158156, -0.01916192980163018, -0.016686126339733112, -0.011325140044997309, -0.004005934150814109, 0.0040059341508141035, 0.011325140044997305, 0.01668612633973311, 0.01916192980163018, 0.01832446152015816, 0.014318527369344042, 0.007836789756632871, 2.1832130423822366e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.z.const.wf": {
            "type": "constant",
            "sample": 0.45,
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
            "sample": 0.02861308911066331,
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
            "samples": [0.0, 0.01072567564137174, 0.041048135807389544, 0.08572435166444636, 0.13702939962017774, 0.18609217079130932, 0.22442926632106924] + [0.2454118541107828] * 2 + [0.22442926632106924, 0.1860921707913094, 0.13702939962017785, 0.08572435166444639, 0.041048135807389516, 0.010725675641371725, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0197264549822217, -0.03604202668879978, -0.046125604549351155, -0.0482336463453575, -0.0420016524993308, -0.028507191363135797, -0.010083577860551377, 0.010083577860551364, 0.028507191363135786, 0.042001652499330785, 0.04823364634535749, 0.04612560454935116, 0.03604202668879977, 0.0197264549822217, 5.4954968979104667e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.00525768413792732, 0.020121635199700744, 0.042021741011983485, 0.0671712743236165, 0.09122165234868099, 0.11001434623581818] + [0.12029992848567776] * 2 + [0.11001434623581818, 0.09122165234868103, 0.06717127432361655, 0.0420217410119835, 0.02012163519970073, 0.005257684137927313, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.010122313390106262, -0.018494386836732907, -0.02366861278305865, -0.02475032055659675, -0.021552473056295852, -0.014628007166490492, -0.005174225946325744, 0.005174225946325738, 0.014628007166490485, 0.02155247305629585, 0.02475032055659675, 0.023668612783058655, 0.018494386836732903, 0.010122313390106262, 2.8199259261301655e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.005257684137927319, -0.02012163519970074, -0.042021741011983485, -0.0671712743236165, -0.09122165234868099, -0.11001434623581818] + [-0.12029992848567776] * 2 + [-0.11001434623581818, -0.09122165234868103, -0.06717127432361655, -0.0420217410119835, -0.020121635199700734, -0.005257684137927314, -3.4534132592679426e-33],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.010122313390106262, 0.01849438683673291, 0.023668612783058655, 0.024750320556596758, 0.021552473056295863, 0.014628007166490506, 0.005174225946325758, -0.005174225946325723, -0.014628007166490471, -0.02155247305629584, -0.024750320556596744, -0.02366861278305865, -0.0184943868367329, -0.010122313390106262, -2.8199259261301655e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0197264549822217, 0.03604202668879978, 0.04612560454935116, 0.048233646345357505, 0.04200165249933081, 0.02850719136313581, 0.010083577860551392, -0.010083577860551349, -0.028507191363135773, -0.04200165249933077, -0.048233646345357484, -0.046125604549351155, -0.03604202668879977, -0.0197264549822217, -5.4954968979104667e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.010725675641371737, 0.041048135807389544, 0.08572435166444636, 0.13702939962017774, 0.18609217079130932, 0.22442926632106924] + [0.2454118541107828] * 2 + [0.22442926632106924, 0.1860921707913094, 0.13702939962017785, 0.08572435166444639, 0.041048135807389516, 0.010725675641371727, 3.365021342875131e-33],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.010122313390106262, 0.018494386836732907, 0.023668612783058655, 0.024750320556596754, 0.02155247305629586, 0.014628007166490499, 0.005174225946325751, -0.005174225946325731, -0.014628007166490478, -0.021552473056295842, -0.024750320556596747, -0.02366861278305865, -0.018494386836732903, -0.010122313390106262, -2.8199259261301655e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.005257684137927319, 0.020121635199700744, 0.042021741011983485, 0.0671712743236165, 0.09122165234868099, 0.11001434623581818] + [0.12029992848567776] * 2 + [0.11001434623581818, 0.09122165234868103, 0.06717127432361655, 0.0420217410119835, 0.02012163519970073, 0.005257684137927314, 1.7267066296339713e-33],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.010122313390106262, -0.018494386836732907, -0.023668612783058648, -0.024750320556596747, -0.021552473056295846, -0.014628007166490485, -0.005174225946325737, 0.0051742259463257445, 0.014628007166490492, 0.021552473056295856, 0.024750320556596754, 0.023668612783058658, 0.018494386836732903, 0.010122313390106262, 2.8199259261301655e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.005257684137927321, -0.020121635199700744, -0.042021741011983485, -0.0671712743236165, -0.09122165234868099, -0.11001434623581818] + [-0.12029992848567776] * 2 + [-0.11001434623581818, -0.09122165234868103, -0.06717127432361655, -0.0420217410119835, -0.02012163519970073, -0.005257684137927312, 1.7267066296339713e-33],
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
            "sample": -0.11201840403229253,
        },
        "q3.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q3.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "q3.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q3.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "q3.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q3.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q3.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q3.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0038293854544515934, 0.01465540628385698, 0.03060614513550034, 0.04892357435401924, 0.06644044401891148, 0.08012792822935215] + [0.08761933661846567] * 2 + [0.08012792822935215, 0.06644044401891151, 0.048923574354019284, 0.030606145135500352, 0.014655406283856972, 0.0038293854544515886, 0.0],
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
            "samples": [0.0, 0.0038310543317542738, 0.014661793229020795, 0.030619483542288252, 0.048944895645327356, 0.06646939930438014, 0.08016284863167868] + [0.08765752183745126] * 2 + [0.08016284863167868, 0.06646939930438017, 0.0489448956453274, 0.030619483542288262, 0.014661793229020784, 0.003831054331754269, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.EF_x90.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0063272797074245945, -0.011560515271903887, -0.014794832721886832, -0.01547098918741978, -0.013472071072923433, -0.009143709479995186, -0.0032343174499829446, 0.0032343174499829407, 0.009143709479995182, 0.013472071072923428, 0.01547098918741978, 0.014794832721886836, 0.011560515271903885, 0.0063272797074245945, 1.7626859988630127e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.EF_EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.005060618233354027, 0.019367446066595836, 0.040446702993907395, 0.06465359399351918, 0.08780252769889745, 0.10589084317092169] + [0.11579090633729168] * 2 + [0.10589084317092169, 0.08780252769889747, 0.06465359399351923, 0.04044670299390741, 0.019367446066595823, 0.00506061823335402, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.EF_EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.z.const.wf": {
            "type": "constant",
            "sample": 0.45,
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
        "q3.z.Cz_flattop.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.00020996429830990007, 0.000838495494195144, 0.001881517321663134, 0.003332265383922333, 0.005181331023091136, 0.007416722338954665, 0.010023941961039295, 0.01298608106962331, 0.016283929055921218, 0.01989609811025345, 0.023799161930199344, 0.027967807649157687, 0.03237500000000002, 0.03699215664915227, 0.04178933356399586, 0.04673541921140972, 0.05179833632800116, 0.056945249953467884, 0.06214278037791542, 0.06735721962208469, 0.07255475004653222, 0.07770166367199893, 0.08276458078859038, 0.08771066643600425, 0.09250784335084786, 0.09712500000000007, 0.10153219235084242, 0.10570083806980077, 0.10960390188974667, 0.1132160709440789, 0.11651391893037678, 0.1194760580389608, 0.12208327766104544, 0.12431866897690898, 0.12616773461607778, 0.12761848267833698, 0.12866150450580496, 0.1292900357016902] + [0.12950000000000012] * 51 + [0.1292900357016902, 0.12866150450580496, 0.12761848267833698, 0.12616773461607778, 0.12431866897690898, 0.12208327766104544, 0.11947605803896082, 0.11651391893037681, 0.1132160709440789, 0.10960390188974667, 0.10570083806980077, 0.10153219235084242, 0.09712500000000009, 0.09250784335084784, 0.08771066643600425, 0.08276458078859039, 0.07770166367199895, 0.07255475004653222, 0.0673572196220847, 0.062142780377915444, 0.05694524995346789, 0.051798336328001175, 0.04673541921140973, 0.041789333563995855, 0.036992156649152265, 0.03237500000000004, 0.027967807649157694, 0.02379916193019935, 0.019896098110253457, 0.01628392905592121, 0.012986081069623324, 0.010023941961039309, 0.007416722338954672, 0.005181331023091136, 0.003332265383922333, 0.001881517321663134, 0.000838495494195144, 0.00020996429830990725],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.z.Cz_bipolar.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.0018644112605240833, 0.007349292236493861, 0.016135881582317133, 0.027713534663167415, 0.041409400365042356, 0.05642752475579524, 0.07189510903449654, 0.08691323342524944, 0.10060909912712437, 0.11218675220797467, 0.12097334155379792, 0.1264582225297677] + [0.1283226337902918] * 26 + [0.12395014607547712, 0.1111306607429201, 0.09073780453283334, 0.06416131689514591, 0.03321234154264378, 7.857495136471942e-18, -0.0332123415426438, -0.06416131689514587, -0.09073780453283332, -0.1111306607429201, -0.12395014607547711] + [-0.1283226337902918] * 26 + [-0.1264582225297677, -0.12097334155379792, -0.11218675220797467, -0.10060909912712437, -0.08691323342524944, -0.07189510903449654, -0.056427524755795246, -0.04140940036504235, -0.027713534663167422, -0.016135881582317126, -0.007349292236493868, -0.0018644112605240833],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.037424917343165574,
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
            "samples": [0.0, 0.008257508891827761, 0.031602237262800786, 0.06599766949737855, 0.10549652288951035, 0.14326908685187867, 0.17278414192241925] + [0.18893826694357801] * 2 + [0.17278414192241925, 0.14326908685187872, 0.10549652288951043, 0.06599766949737858, 0.031602237262800766, 0.00825750889182775, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0060186330029075715, -0.010996589682048117, -0.01407313610428153, -0.014716309443656846, -0.01281489918674971, -0.008697676440749274, -0.003076546422233412, 0.0030765464222334083, 0.00869767644074927, 0.012814899186749707, 0.014716309443656843, 0.014073136104281533, 0.010996589682048115, 0.0060186330029075715, 1.6767016185598995e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004229934478643018, 0.016188343815466334, 0.03380751039739494, 0.054040920257309985, 0.0733900330156742, 0.08850921129539542] + [0.09678421181848716] * 2 + [0.08850921129539542, 0.07339003301567423, 0.054040920257310034, 0.03380751039739495, 0.016188343815466324, 0.004229934478643013, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0023426596349290776, -0.004280252136584266, -0.0054777501589541085, -0.005728095415042857, -0.0049880009357575125, -0.003385435780113779, -0.0011974980223698427, 0.0011974980223698414, 0.003385435780113778, 0.00498800093575751, 0.005728095415042857, 0.005477750158954109, 0.004280252136584265, 0.0023426596349290776, 6.52630123771122e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.004229934478643018, -0.016188343815466334, -0.03380751039739494, -0.054040920257309985, -0.0733900330156742, -0.08850921129539542] + [-0.09678421181848716] * 2 + [-0.08850921129539542, -0.07339003301567423, -0.054040920257310034, -0.03380751039739495, -0.016188343815466324, -0.004229934478643013, -7.992413921034455e-34],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.002342659634929078, 0.004280252136584267, 0.005477750158954113, 0.005728095415042864, 0.004988000935757521, 0.00338543578011379, 0.0011974980223698546, -0.0011974980223698295, -0.003385435780113767, -0.004988000935757501, -0.00572809541504285, -0.005477750158954105, -0.004280252136584263, -0.002342659634929077, -6.52630123771122e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.006018633002907572, 0.010996589682048119, 0.014073136104281533, 0.014716309443656853, 0.01281489918674972, 0.008697676440749284, 0.003076546422233424, -0.0030765464222333966, -0.00869767644074926, -0.012814899186749698, -0.014716309443656836, -0.01407313610428153, -0.010996589682048113, -0.006018633002907571, -1.6767016185598995e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.008257508891827761, 0.031602237262800786, 0.06599766949737855, 0.10549652288951035, 0.14326908685187867, 0.17278414192241925] + [0.18893826694357801] * 2 + [0.17278414192241925, 0.14326908685187872, 0.10549652288951043, 0.06599766949737858, 0.031602237262800766, 0.00825750889182775, 1.0266836351472837e-33],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.002342659634929078, 0.0042802521365842665, 0.00547775015895411, 0.00572809541504286, 0.004988000935757517, 0.0033854357801137843, 0.0011974980223698486, -0.0011974980223698355, -0.0033854357801137726, -0.004988000935757506, -0.005728095415042853, -0.005477750158954108, -0.004280252136584264, -0.002342659634929077, -6.52630123771122e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.004229934478643018, 0.016188343815466334, 0.03380751039739494, 0.054040920257309985, 0.0733900330156742, 0.08850921129539542] + [0.09678421181848716] * 2 + [0.08850921129539542, 0.07339003301567423, 0.054040920257310034, 0.03380751039739495, 0.016188343815466324, 0.004229934478643013, 3.996206960517227e-34],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.002342659634929077, -0.004280252136584265, -0.005477750158954107, -0.005728095415042853, -0.004988000935757508, -0.003385435780113774, -0.0011974980223698369, 0.0011974980223698473, 0.003385435780113783, 0.004988000935757514, 0.00572809541504286, 0.005477750158954111, 0.004280252136584266, 0.002342659634929078, 6.52630123771122e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.004229934478643018, -0.016188343815466334, -0.03380751039739494, -0.054040920257309985, -0.0733900330156742, -0.08850921129539542] + [-0.09678421181848716] * 2 + [-0.08850921129539542, -0.07339003301567423, -0.054040920257310034, -0.03380751039739495, -0.016188343815466324, -0.004229934478643013, 3.996206960517227e-34],
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
            "sample": -0.11201840403229253,
        },
        "q4.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q4.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "q4.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q4.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "q4.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q4.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q4.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q4.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0067834767221322725, 0.025960982137321047, 0.05421655133711296, 0.08666453971307633, 0.11769439529428694, 0.14194077415336528] + [0.15521125711413988] * 2 + [0.14194077415336528, 0.11769439529428699, 0.0866645397130764, 0.05421655133711298, 0.02596098213732103, 0.006783476722132264, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.005788982394790307, -0.010576997142267343, -0.013536152994843896, -0.014154785022522516, -0.012325926127622034, -0.008365802627732209, -0.0029591558525765525, 0.0029591558525765486, 0.008365802627732205, 0.01232592612762203, 0.014154785022522515, 0.013536152994843897, 0.010576997142267341, 0.005788982394790307, 1.6127243755302174e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.EF_x90.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.021613635589349783, 0.08271734841028544, 0.17274575140626314, 0.2761321158169133, 0.37499999999999994, 0.45225424859373686] + [0.4945369001834514] * 2 + [0.45225424859373686, 0.3750000000000001, 0.27613211581691355, 0.1727457514062632, 0.08271734841028539, 0.021613635589349756, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.EF_x90.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.01844495986931454, -0.03370061861002473, -0.04312913423255304, -0.045100230730388935, -0.03927308761220707, -0.026655270861074398, -0.009428515622528312, 0.0094285156225283, 0.026655270861074387, 0.03927308761220706, 0.045100230730388935, 0.04312913423255304, 0.03370061861002472, 0.01844495986931454, 5.1384914235859794e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.z.const.wf": {
            "type": "constant",
            "sample": 0.45,
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
            "sample": 0.034534179061011985,
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
            "samples": [0.0, 0.008639583896092217, 0.033064473040578836, 0.06905138220722629, 0.11037784787031495, 0.14989814867754264, 0.18077886558867814] + [0.1976804421072802] * 2 + [0.18077886558867814, 0.1498981486775427, 0.11037784787031503, 0.0690513822072263, 0.033064473040578815, 0.008639583896092205, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.011356376365670863, -0.020749132088276813, -0.026554174372872368, -0.027767758671297417, -0.024180045233287022, -0.016411382305626554, -0.005805042284595555, 0.005805042284595549, 0.016411382305626544, 0.02418004523328702, 0.027767758671297414, 0.02655417437287237, 0.02074913208827681, 0.011356376365670863, 3.1637175126140755e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004425653581720868, 0.016937378616803344, 0.03537178418122809, 0.05654139408158904, 0.07678579044624545, 0.09260453322918102] + [0.1012624180944595] * 2 + [0.09260453322918102, 0.07678579044624549, 0.056541394081589084, 0.035371784181228105, 0.01693737861680333, 0.004425653581720862, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.004230311231936023, -0.007729163220699258, -0.009891572671360102, -0.01034363934502616, -0.009007196806923771, -0.006113328113090138, -0.0021624094506608443, 0.0021624094506608417, 0.0061133281130901366, 0.009007196806923768, 0.01034363934502616, 0.009891572671360104, 0.0077291632206992564, 0.004230311231936023, 1.1785017770933489e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0044256535817208675, -0.016937378616803344, -0.03537178418122809, -0.05654139408158904, -0.07678579044624545, -0.09260453322918102] + [-0.1012624180944595] * 2 + [-0.09260453322918102, -0.07678579044624549, -0.056541394081589084, -0.035371784181228105, -0.01693737861680333, -0.004425653581720863, -1.4432484291068373e-33],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.004230311231936024, 0.00772916322069926, 0.009891572671360106, 0.010343639345026167, 0.00900719680692378, 0.0061133281130901496, 0.002162409450660857, -0.002162409450660829, -0.006113328113090125, -0.009007196806923759, -0.010343639345026153, -0.0098915726713601, -0.007729163220699255, -0.004230311231936022, -1.1785017770933489e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.011356376365670863, 0.020749132088276816, 0.02655417437287237, 0.027767758671297424, 0.024180045233287033, 0.016411382305626564, 0.0058050422845955674, -0.005805042284595537, -0.016411382305626533, -0.02418004523328701, -0.027767758671297407, -0.026554174372872368, -0.020749132088276806, -0.011356376365670863, -3.1637175126140755e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.008639583896092217, 0.033064473040578836, 0.06905138220722629, 0.11037784787031495, 0.14989814867754264, 0.18077886558867814] + [0.1976804421072802] * 2 + [0.18077886558867814, 0.1498981486775427, 0.11037784787031503, 0.0690513822072263, 0.033064473040578815, 0.008639583896092205, 1.9372182626146267e-33],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004230311231936023, 0.007729163220699259, 0.009891572671360104, 0.010343639345026164, 0.009007196806923776, 0.006113328113090144, 0.0021624094506608503, -0.0021624094506608356, -0.0061133281130901305, -0.009007196806923762, -0.010343639345026157, -0.009891572671360102, -0.007729163220699256, -0.004230311231936023, -1.1785017770933489e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.004425653581720868, 0.016937378616803344, 0.03537178418122809, 0.05654139408158904, 0.07678579044624545, 0.09260453322918102] + [0.1012624180944595] * 2 + [0.09260453322918102, 0.07678579044624549, 0.056541394081589084, 0.035371784181228105, 0.01693737861680333, 0.004425653581720862, 7.216242145534186e-34],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.004230311231936023, -0.007729163220699257, -0.0098915726713601, -0.010343639345026157, -0.009007196806923766, -0.006113328113090132, -0.002162409450660838, 0.0021624094506608477, 0.006113328113090143, 0.009007196806923773, 0.010343639345026164, 0.009891572671360106, 0.007729163220699257, 0.004230311231936023, 1.1785017770933489e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.004425653581720868, -0.016937378616803344, -0.03537178418122809, -0.05654139408158904, -0.07678579044624545, -0.09260453322918102] + [-0.1012624180944595] * 2 + [-0.09260453322918102, -0.07678579044624549, -0.056541394081589084, -0.035371784181228105, -0.01693737861680333, -0.004425653581720862, 7.216242145534186e-34],
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
            "sample": 0.1,
        },
        "q5.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q5.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004819079185159693, 0.018443054169564665, 0.038516215908484155, 0.06156773237830947, 0.08361178696495504, 0.10083676236644093] + [0.11026430385186127] * 2 + [0.10083676236644093, 0.08361178696495507, 0.061567732378309516, 0.03851621590848416, 0.01844305416956465, 0.004819079185159687, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0057586841987536455, -0.010521639583539227, -0.013465307898236053, -0.014080702148845938, -0.01226141507875718, -0.008322017950092292, -0.0029436683146968233, 0.00294366831469682, 0.00832201795009229, 0.012261415078757176, 0.014080702148845936, 0.013465307898236053, 0.010521639583539227, 0.0057586841987536455, 1.6042837488447935e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.EF_x90.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004618343160591265, 0.017674819153568368, 0.03691184466538491, 0.05900316322215746, 0.08012898515209145, 0.09663646392143418] + [0.10567130648522964] * 2 + [0.09663646392143418, 0.08012898515209148, 0.059003163222157505, 0.03691184466538492, 0.017674819153568357, 0.004618343160591259, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.EF_x90.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0055188094574622724, -0.01008336662291937, -0.012904418094763738, -0.013494178445065425, -0.011750672751452688, -0.007975368987603152, -0.002821051471844364, 0.0028210514718443607, 0.007975368987603149, 0.011750672751452687, 0.013494178445065425, 0.012904418094763738, 0.010083366622919369, 0.0055188094574622724, 1.5374582144118778e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.z.const.wf": {
            "type": "constant",
            "sample": 0.45,
        },
        "q5.z.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q5.z.cz5_4.wf": {
            "type": "constant",
            "sample": -0.1732650364352627,
        },
        "q5.z.Cz_flattop.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.000162134593289498, 0.0006474868681043578, 0.0014529091286973996, 0.002573177902642726, 0.0040010278170587886, 0.005727198717339505, 0.00774049572281026, 0.010027861829824942, 0.012574462591444947, 0.015363782324520027, 0.018377731220231137, 0.02159676266344221, 0.024999999999999994, 0.02856537192984729, 0.032269755647873224, 0.03608912680417736, 0.03999871531119778, 0.04397316598723385, 0.04798670299452924, 0.05201329700547075, 0.05602683401276615, 0.06000128468880222, 0.06391087319582263, 0.06773024435212678, 0.07143462807015272, 0.075, 0.0784032373365578, 0.08162226877976886, 0.08463621767547998, 0.08742553740855506, 0.08997213817017505, 0.09225950427718974, 0.0942728012826605, 0.09599897218294122, 0.09742682209735727, 0.0985470908713026, 0.09935251313189564, 0.0998378654067105] + [0.1] * 51 + [0.09983786540671051, 0.09935251313189564, 0.0985470908713026, 0.09742682209735727, 0.09599897218294122, 0.0942728012826605, 0.09225950427718975, 0.08997213817017508, 0.08742553740855506, 0.08463621767547998, 0.08162226877976886, 0.0784032373365578, 0.07500000000000001, 0.0714346280701527, 0.06773024435212678, 0.06391087319582264, 0.060001284688802226, 0.05602683401276615, 0.05201329700547076, 0.04798670299452926, 0.04397316598723386, 0.03999871531119779, 0.03608912680417737, 0.03226975564787322, 0.028565371929847285, 0.025000000000000012, 0.021596762663442216, 0.018377731220231144, 0.015363782324520032, 0.012574462591444941, 0.010027861829824953, 0.00774049572281027, 0.00572719871733951, 0.0040010278170587886, 0.002573177902642726, 0.0014529091286973996, 0.0006474868681043578, 0.00016213459328950355],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.z.Cz_bipolar.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.0003645562950973014, 0.0014529091286973996, 0.0032491878657292584, 0.005727198717339505, 0.008850806705317182, 0.012574462591444947, 0.016843867087960235, 0.02159676266344221, 0.026763841397811572, 0.032269755647873224, 0.03803421678562211, 0.04397316598723385, 0.049999999999999996, 0.05602683401276615, 0.06196578321437788, 0.06773024435212678, 0.07323615860218843, 0.0784032373365578, 0.08315613291203976, 0.08742553740855506, 0.09114919329468282, 0.0942728012826605, 0.09675081213427074, 0.0985470908713026, 0.0996354437049027] + [0.1] * 26 + [0.0992708874098054, 0.0970941817426052, 0.09350162426854149, 0.088545602565321, 0.08229838658936564, 0.0748510748171101, 0.06631226582407954, 0.05680647467311559, 0.04647231720437686, 0.03546048870425356, 0.023931566428755782, 0.012053668025532302, 6.123233995736766e-18, -0.012053668025532288, -0.02393156642875575, -0.03546048870425357, -0.046472317204376855, -0.05680647467311557, -0.0663122658240795, -0.07485107481711012, -0.08229838658936564, -0.08854560256532099, -0.09350162426854147, -0.0970941817426052, -0.0992708874098054] + [-0.1] * 26 + [-0.0996354437049027, -0.0985470908713026, -0.09675081213427075, -0.0942728012826605, -0.09114919329468282, -0.08742553740855506, -0.08315613291203977, -0.0784032373365578, -0.07323615860218843, -0.06773024435212678, -0.061965783214377894, -0.05602683401276615, -0.05, -0.04397316598723386, -0.03803421678562213, -0.03226975564787322, -0.026763841397811572, -0.021596762663442216, -0.016843867087960252, -0.012574462591444941, -0.008850806705317182, -0.00572719871733951, -0.003249187865729264, -0.0014529091286973996, -0.0003645562950973014],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.031125563791230847,
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
            "sample": 0.45,
        },
        "coupler_q1_q2.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "coupler_q1_q2.cz.wf": {
            "type": "constant",
            "sample": -0.035,
        },
        "q1.z.SWAP_Coupler.flux_pulse_control_q1_q2.wf": {
            "type": "arbitrary",
            "samples": [0.1379194010900913] * 15 + [0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "coupler_q1_q2.SWAP_Coupler.coupler_pulse_control_q1_q2.wf": {
            "type": "arbitrary",
            "samples": [-0.035] * 15 + [0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.z.Cz_unipolar.flux_pulse_control_q2_q1.wf": {
            "type": "arbitrary",
            "samples": [-0.09913528255422949] * 84 + [0.0] * 4,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "coupler_q1_q2.Cz_unipolar.coupler_flux_pulse_q2_q1.wf": {
            "type": "arbitrary",
            "samples": [-0.013275000000000004] * 84 + [0.0] * 4,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.z.Cz_flattop.flux_pulse_control_q2_q1.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.002006682831898706, 0.007830303230627157, 0.016900804656008604, 0.02833030323062716, 0.041, 0.05366969676937286, 0.0650991953439914, 0.07416969676937286, 0.07999331716810132] + [0.08200000000000002] * 69 + [0.07999331716810132, 0.07416969676937286, 0.06509919534399142, 0.05366969676937286, 0.04100000000000001, 0.028330303230627164, 0.01690080465600861, 0.00783030323062716, 0.002006682831898706],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "coupler_q1_q2.Cz_flattop.coupler_flux_pulse_q2_q1.wf": {
            "type": "constant",
            "sample": -0.035,
        },
        "q1.z.Cz_bipolar.flux_pulse_control_q2_q1.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.004060276416000816, 0.015436918123791925, 0.031876641707791115, 0.050123358292208896, 0.06656308187620809, 0.0779397235839992] + [0.08200000000000002] * 35 + [0.07101408311032399, 0.041000000000000016, 5.021051876504149e-18, -0.04099999999999999, -0.07101408311032399] + [-0.08200000000000002] * 35 + [-0.0779397235839992, -0.06656308187620809, -0.050123358292208896, -0.03187664170779112, -0.01543691812379193, -0.00406027641600082],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "coupler_q1_q2.Cz_bipolar.coupler_flux_pulse_q2_q1.wf": {
            "type": "constant",
            "sample": -0.035,
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
            "samples": [0.08560805836349752] * 88,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "coupler_q2_q3.Cz_unipolar.coupler_flux_pulse_q2_q3.wf": {
            "type": "arbitrary",
            "samples": [-0.087] * 88,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.z.Cz_flattop.flux_pulse_control_q2_q3.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.003344274688013456, 0.012867963246729308, 0.027121170425534843, 0.04393397463710173, 0.06074677884866863, 0.07499998602747417, 0.08452367458619002] + [0.08786794927420348] * 65 + [0.08452367458619002, 0.07499998602747417, 0.06074677884866864, 0.04393397463710174, 0.02712117042553485, 0.012867963246729314, 0.003344274688013456],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "coupler_q2_q3.Cz_flattop.coupler_flux_pulse_q2_q3.wf": {
            "type": "constant",
            "sample": -0.087,
        },
        "q3.z.Cz_bipolar.flux_pulse_control_q2_q3.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.005973455815072423, 0.022293240598618046, 0.04458648119723609, 0.06687972179585414, 0.08319950657939978] + [0.0891729623944722] * 33 + [0.07214244201588611, 0.027555960818650017, -0.027555960818650006, -0.07214244201588611] + [-0.0891729623944722] * 33 + [-0.08065770220517916, -0.058364461606561106, -0.030808500787911097, -0.008515260189293047],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "coupler_q2_q3.Cz_bipolar.coupler_flux_pulse_q2_q3.wf": {
            "type": "constant",
            "sample": -0.093,
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
            "samples": [0.129] * 80,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "coupler_q3_q4.Cz_unipolar.coupler_flux_pulse_q4_q3.wf": {
            "type": "arbitrary",
            "samples": [-0.1177842092514038] * 80,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.z.Cz_flattop.flux_pulse_control_q4_q3.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.0031690905698888118, 0.012366149614222164, 0.02669090491406239, 0.044741149614222195, 0.06475000000000004, 0.08475885038577792, 0.10280909508593772, 0.11713385038577795, 0.1263309094301113] + [0.12950000000000012] * 109 + [0.1263309094301113, 0.11713385038577795, 0.10280909508593773, 0.08475885038577792, 0.06475000000000006, 0.0447411496142222, 0.026690904914062397, 0.012366149614222171, 0.0031690905698888118],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "coupler_q3_q4.Cz_flattop.coupler_flux_pulse_q4_q3.wf": {
            "type": "constant",
            "sample": -0.09199999999999992,
        },
        "q3.z.Cz_bipolar.flux_pulse_control_q4_q3.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.03208065844757294, 0.09624197534271883] + [0.1283226337902918] * 41 + [0.06416131689514591, -0.06416131689514587] + [-0.1283226337902918] * 40 + [-0.09624197534271885, -0.03208065844757296],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "coupler_q3_q4.Cz_bipolar.coupler_flux_pulse_q4_q3.wf": {
            "type": "constant",
            "sample": -0.109,
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
        "q5.z.Cz_unipolar.flux_pulse_control_q4_q5.wf": {
            "type": "arbitrary",
            "samples": [0.16330603996076692] * 60 + [0.0] * 16,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "coupler_q4_q5.Cz_unipolar.coupler_flux_pulse_q4_q5.wf": {
            "type": "arbitrary",
            "samples": [-0.13191014618685923] * 60 + [0.0] * 16,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.z.Cz_flattop.flux_pulse_control_q4_q5.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.0024471741852423235, 0.009549150281252628, 0.020610737385376346, 0.034549150281252626, 0.049999999999999996, 0.06545084971874737, 0.07938926261462365, 0.09045084971874738, 0.09755282581475769] + [0.1] * 109 + [0.09755282581475769, 0.09045084971874738, 0.07938926261462366, 0.06545084971874737, 0.05, 0.03454915028125263, 0.02061073738537635, 0.009549150281252633, 0.0024471741852423235],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "coupler_q4_q5.Cz_flattop.coupler_flux_pulse_q4_q5.wf": {
            "type": "constant",
            "sample": 0.1,
        },
        "q5.z.Cz_bipolar.flux_pulse_control_q4_q5.wf": {
            "type": "arbitrary",
            "samples": [0.0, 0.004951556604879043, 0.01882550990706332, 0.03887395330218428, 0.06112604669781572, 0.08117449009293667, 0.09504844339512096] + [0.1] * 55 + [0.08660254037844388, 0.05000000000000002, 6.123233995736766e-18, -0.04999999999999998, -0.08660254037844388] + [-0.1] * 55 + [-0.09504844339512096, -0.08117449009293669, -0.06112604669781572, -0.03887395330218429, -0.018825509907063328, -0.004951556604879049],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "coupler_q4_q5.Cz_bipolar.coupler_flux_pulse_q4_q5.wf": {
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
            "cosine": [(-0.9533486847849908, 1200)],
            "sine": [(-0.3018713057227006, 1200)],
        },
        "q1.resonator.readout.iw2": {
            "cosine": [(0.3018713057227006, 1200)],
            "sine": [(-0.9533486847849908, 1200)],
        },
        "q1.resonator.readout.iw3": {
            "cosine": [(-0.3018713057227006, 1200)],
            "sine": [(0.9533486847849908, 1200)],
        },
        "q2.resonator.readout.iw1": {
            "cosine": [(-0.7963394329167912, 1200)],
            "sine": [(-0.6048499876678212, 1200)],
        },
        "q2.resonator.readout.iw2": {
            "cosine": [(0.6048499876678212, 1200)],
            "sine": [(-0.7963394329167912, 1200)],
        },
        "q2.resonator.readout.iw3": {
            "cosine": [(-0.6048499876678212, 1200)],
            "sine": [(0.7963394329167912, 1200)],
        },
        "q3.resonator.readout.iw1": {
            "cosine": [(0.9324853686940456, 1200)],
            "sine": [(0.36120774793950594, 1200)],
        },
        "q3.resonator.readout.iw2": {
            "cosine": [(-0.36120774793950594, 1200)],
            "sine": [(0.9324853686940456, 1200)],
        },
        "q3.resonator.readout.iw3": {
            "cosine": [(0.36120774793950594, 1200)],
            "sine": [(-0.9324853686940456, 1200)],
        },
        "q4.resonator.readout.iw1": {
            "cosine": [(0.7963131694988432, 1200)],
            "sine": [(0.6048845642622289, 1200)],
        },
        "q4.resonator.readout.iw2": {
            "cosine": [(-0.6048845642622289, 1200)],
            "sine": [(0.7963131694988432, 1200)],
        },
        "q4.resonator.readout.iw3": {
            "cosine": [(0.6048845642622289, 1200)],
            "sine": [(-0.7963131694988432, 1200)],
        },
        "q5.resonator.readout.iw1": {
            "cosine": [(-0.5971635673658889, 1200)],
            "sine": [(-0.802119488487124, 1200)],
        },
        "q5.resonator.readout.iw2": {
            "cosine": [(0.802119488487124, 1200)],
            "sine": [(-0.5971635673658889, 1200)],
        },
        "q5.resonator.readout.iw3": {
            "cosine": [(-0.802119488487124, 1200)],
            "sine": [(0.5971635673658889, 1200)],
        },
    },
    "mixers": {},
}

