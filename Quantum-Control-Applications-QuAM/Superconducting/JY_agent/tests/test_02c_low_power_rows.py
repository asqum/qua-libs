"""02c ignores low-power rows whose dip is lost in the noise.

Operator rule 2026-10-11 (CRSQ_26009_die(5,-1) q5, run #401): the dressed
plateau is read from the lowest power that still shows the resonance clearly.
"""

from __future__ import annotations

import unittest

import numpy as np
import xarray as xr

from jy_agent.analysis_02c import _usable_power_mask, analyze_02c_transitions


def sweep(noisy_below_dbm: float) -> xr.Dataset:
    rng = np.random.default_rng(7)
    power = np.linspace(-55, -15, 30)
    freq = np.arange(-300e3, 200e3, 10e3)
    rows = []
    centres = []
    for p in power:
        centre = 60e3 if p < -32 else -60e3  # dressed at low power, bare above
        depth = 1.0
        noise = 0.6 if p < noisy_below_dbm else 0.02
        row = 2.0 - depth * np.exp(-(((freq - centre) / 25e3) ** 2))
        row = row + rng.normal(0, noise, freq.size)
        rows.append(row)
        centres.append(freq[int(np.argmin(row))])
    amplitude = np.asarray(rows)[None, :, :]
    return xr.Dataset(
        {
            "IQ_abs": (("qubit", "power_dbm", "freq"), amplitude),
            "rr_min_response": (("qubit", "power_dbm"), np.asarray(centres)[None, :]),
        },
        coords={"qubit": ["q5"], "power_dbm": power, "freq": freq},
    )


class LowPowerRowTests(unittest.TestCase):
    def test_noisy_low_power_rows_are_dropped(self) -> None:
        dataset = sweep(noisy_below_dbm=-44)
        result = analyze_02c_transitions(dataset, {"min_power_row_snr": 6.0})["qubits"]["q5"]
        self.assertGreater(result["ignored_low_power_rows"], 0)
        self.assertGreaterEqual(result["usable_min_power_dbm"], -45)
        self.assertEqual(result["validation_failures"], [])
        self.assertAlmostEqual(result["bare_dressed_separation_hz"], 120e3, delta=20e3)

    def test_clean_sweep_keeps_every_row(self) -> None:
        dataset = sweep(noisy_below_dbm=-100)
        result = analyze_02c_transitions(dataset, {"min_power_row_snr": 6.0})["qubits"]["q5"]
        self.assertNotIn("ignored_low_power_rows", result)

    def test_mask_needs_two_consecutive_clear_rows(self) -> None:
        power = np.array([-50.0, -48, -46, -44, -42])
        snr = np.array([7.0, 2.0, 7.0, 8.0, 9.0])
        self.assertEqual(_usable_power_mask(power, snr, {"min_power_row_snr": 6}).tolist(),
                         [False, False, True, True, True])
        self.assertIsNone(_usable_power_mask(power, snr, {"min_power_row_snr": 0}))


if __name__ == "__main__":
    unittest.main()
