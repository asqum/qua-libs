# COUPLER MEASUREMETS WORKFLOW
##  Basic Requirements 
    1. Qubit's Readout Fidelity > 85%. Otherwise, increase your shot please.
    2. Active Reset for qubits is doing good.
    3. Amplitude for aSWAP operation had been updated in node 02b_resonator_spectroscopy_ns_flux.
    4. Check the RD and T1 is in your coupler.extras (it should be initialized when populate_quam) like :
        "coupler_q7_q8":{
            "extras":{
                "T1": 3.0e-05,
                "RD": {
                    "LO": 3300000000.0,
                    "IF": -100415623.102556,
                    "readout_q": "q8",
                    "driven_q": "q7",
                    "swap_direction": 1,
                    "aswap_supplier": "q"
                }
            }
        }
    5. 
##  _**Standard**_ - Sweet Spot Information included
    1. 03xa - Tuning driving LO to widely search a suspicious signal by 2-tone measurement.
                - Update: driving frequency, and Initialize π-pulse parameters.
    2. 04x  - Check a correct driving frequency with Rabi Oscillation.
                - Warnings: You might need to extend the π-pulse duration for coupler (default 32 ns).
                - Update: a correct π-pulse amplitude.
    3. 03xb - Manually tunes flux bias and LO for driving frequency to observe coupler's sweet spot. 
                - Warnings: You might need to enforce aSWAP applied on coupler itself once coupler's frequency is higher than qubit's.
                - Update: maximum driving frequency, bias to sweet spot, quad_term and neighboring_qubit_detune_flux_amp.
    ---
    * Calibrations, you may skip it as well
        c1. 06xa - Calibrate driving frequency by Ramsey.
                    - Update: a more precise driving frequency.
        c2. 03xx - Calibrate the decouple offset by the current driving frequency.
                    - Update: a corrected decouple offset.
        c3. 04xx - Calibrate the amplitude for π-pulse.
                    - Update: a corrected π-pulse amplitude.
    --- 
    4. 08x  - Check aSWAP as a reset method is doing good as the thermalize doing, then you may use active_reset to measure T1 and T2 statistics.
                - Save: the results dataset and figure only.
    5. 05x  - Statistics for coupler's T1 at idle point.
                - Update: T1 and its deviation T1_dev.
    6. 05xx - Statistics for coupler's T1 at sweet spot.
                - Save: the results dataset and figure only.
    7. 06x  - Statistics for coupler's T2 at idle point.
                - Update: T2 and its deviation T2_dev.


## _**SIMPLIFIED**_ - Idle Information Only
    1. 03xa - Tuning driving LO to widely search a suspicious signal by 2-tone measurement.
                - Update: driving frequency, and Initialize π-pulse parameters.
    2. 04x  - Check a correct driving frequency with Rabi Oscillation.
                - Warnings: You might need to extend the π-pulse duration for coupler (default 32 ns).
                - Update: a correct π-pulse amplitude.
    3. 08x  - Check aSWAP as a reset method is doing good as the thermalize doing, then you may use active_reset to measure T1 and T2 statistics.
                - Save: the results dataset and figure only.
    4. 05x  - Statistics for coupler's T1 at idle point.
                - Update: T1 and its deviation T1_dev.
    5. 06x  - Statistics for coupler's T2 at idle point.
                - Update: T2 and its deviation T2_dev.

## **CZ Focusing** - The operation point frequency, make sure your CZ gate is now available (node 61 updated)
    1. 12x  - Use BOTH SQUARE flux pulse with different duration and coupler flux amplitude to measure CZ coupling strength.
                - Warning: Since your CZ may not be composed with BOTH SQUARE, the measured coupling strength might be slightly different.
                - Update: An additional information 'CZgMHz_bias_conversion' in extras recorded the conversion between specific g_CZ to coupler's flux amplitude.   
    2. 03xc - Applied the flux amplitude obtained from node 12x while coupler's 2tone measurement.
                - Save: The operation frequency figure.


## Another useful information in RD
    1. swap_direction can be either -1 or 1, depends on your coupler's RO SNR.
    2. aswap_supplier is default as "q" which means aSWAP pulse will be applied on the readout_q. You may set it "c" makes it applied on coupler itself.
        - Warnings:  Once you set it "c", please make sure you have aSWAP operation in your coupler.coupler.operation. This amplitude needs to be dynamically adjusted by node 03xb.


* If any **BUG** were found, please reach out to Ratis Wu (AS) **:P**
* Congrats ! You have gone through this doc. **`readme_password = '0xffe8' `** Use it in 03xa's node parameters then start the measurements.