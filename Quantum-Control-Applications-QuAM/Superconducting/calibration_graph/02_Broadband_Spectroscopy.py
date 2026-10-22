# %%
"""
        BROADBAND RESONATOR SPECTROSCOPY (ONE SWEEP PER READOUT LINE)
For every MW input port used by a readout resonator (i.e. every readout line), keep the line's LO fixed and sweep the
readout IF over the full instrument band (+/- max_if_in_mhz around the LO). The qubits do not need to be selected: the
number of broadband traces equals the number of distinct readout MW inputs found in the state (e.g. 2 for the
as-qpu-10qV2 chip, 1 for IQM_4SQ).

Analysis, per line:
    1. Remove the electrical delay (linear phase vs frequency) estimated from the median phase step.
    2. Remove the slowly varying complex baseline (standing waves, amplifier ripple) with a running median.
    3. Resonator candidates are the narrow peaks of |S21/baseline - 1| above a noise threshold. The `res_num`
       strongest ones (split over the lines according to the wiring of q1..q_res_num) are marked as selected.

Design-frequency assignment (res_design_freq is not None):
    res_design_freq[i] is the expected frequency of the i-th qubit (q1..qn, n = res_num). A single global frequency
    shift is fitted so that the shifted design pattern best overlaps the measured candidates (each design frequency
    can only be assigned to a candidate on its own readout line). Assignments whose residual after the shift exceeds
    design_match_tolerance_in_mhz are flagged as mismatched and are not written to the state.

Prerequisites:
    - Ensure calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Readout LO (upconverter frequency) of each line set to the centre of the expected resonator band.

State update:
    - res_design_freq is None: nothing is written to the state.
    - Otherwise: the readout IF of every matched (within tolerance) qubit is set to its measured resonator frequency.
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration
from quam_libs.lib.save_utils import restore_load_data_id
from quam_libs.trackable_object import tracked_updates
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from typing import Optional, List
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


# %% {Node_parameters}
class Parameters(NodeParameters):

    # None: every readout line. Otherwise only the lines of these qubits are swept, and only these qubits receive
    # outcomes and state updates (res_num / res_design_freq still refer to q1..qn of the whole state).
    qubits: Optional[List[str]] = None
    num_averages: int = 100
    # The sweep covers LO +/- max_if_in_mhz on every readout line.
    max_if_in_mhz: float = 400.0
    frequency_step_in_mhz: float = 0.1
    # Common probe settings for the broadband sweep (reverted after acquisition).
    readout_length_in_ns: int = 2000
    # None: same probe power as 02x (0.1 V -> -10 dBm).
    readout_power_dbm: Optional[float] = None
    # Total number of resonators expected over all readout lines. None: number of qubits in the state.
    res_num: Optional[int] = None
    # None, or a list of length res_num with the expected frequencies of q1..qn (Hz; values < 1e6 are read as GHz).
    res_design_freq: Optional[List[float]] = None
    design_match_tolerance_in_mhz: float = 10.0
    # Candidate detection
    baseline_window_in_mhz: float = 15.0
    min_resonator_separation_in_mhz: float = 5.0
    # Ignore |IF| below this value (LO leakage / DC artifacts around the LO)
    lo_guard_in_mhz: float = 10.0
    # Features narrower than this (spurs, single-point glitches) are not resonator candidates
    min_linewidth_in_mhz: float = 0.3
    detection_snr: float = 6.0
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None


node = QualibrationNode(name="02_Broadband_Spectroscopy", parameters=Parameters())
assert not (
    node.parameters.simulate and node.parameters.load_data_id is not None
), "If simulate is True, load_data_id must be None, and vice versa."


# %% {Helper_functions}
def port_key(port) -> str:
    return f"{port.controller_id}-{port.fem_id}-{port.port_id}"


def group_qubits_by_readout_line(machine):
    """Return {line_name: [qubits]} with one entry per readout MW input, in state order."""
    lines = {}
    for q in machine.qubits.values():
        if getattr(q, "resonator", None) is None:
            continue
        lines.setdefault(port_key(q.resonator.opx_input), []).append(q)
    return lines


def expected_qubits(machine, res_num):
    """The resonators q1..q_res_num (state order), or None when res_num exceeds the number of qubits."""
    all_qubits = [q for q in machine.qubits.values() if getattr(q, "resonator", None) is not None]
    return all_qubits[:res_num] if res_num <= len(all_qubits) else None


def expected_count_per_line(lines, machine, res_num):
    qubits = expected_qubits(machine, res_num)
    if qubits is None:
        assert len(lines) == 1, (
            f"res_num={res_num} exceeds the number of qubits in the state and there are {len(lines)} readout lines, "
            "so the resonators cannot be split over the lines."
        )
        return {next(iter(lines)): res_num}
    names = {q.name for q in qubits}
    return {line: sum(q.name in names for q in line_qubits) for line, line_qubits in lines.items()}


def design_frequencies_in_hz(res_design_freq, res_num):
    assert len(res_design_freq) == res_num, (
        f"res_design_freq has {len(res_design_freq)} entries but res_num is {res_num}."
    )
    return [float(f) * 1e9 if abs(float(f)) < 1e6 else float(f) for f in res_design_freq]


def remove_electrical_delay(freq, s21):
    """Remove the linear phase roll 2*pi*f*tau. The median phase step is insensitive to resonator phase features."""
    dphi = np.angle(s21[1:] * np.conj(s21[:-1]))
    df = np.diff(freq)
    tau = -np.median(dphi / (2 * np.pi * df))
    s21_corr = s21 * np.exp(1j * 2 * np.pi * (freq - freq[0]) * tau)
    s21_corr *= np.exp(-1j * np.angle(np.mean(s21_corr)))
    return tau, s21_corr


def detect_candidates(freq, s21_corr, window_hz, min_sep_hz, min_width_hz, snr, excluded=None):
    step = abs(freq[1] - freq[0])
    window = max(int(round(window_hz / step)) | 1, 5)
    baseline = median_filter(s21_corr.real, size=window, mode="nearest") + 1j * median_filter(
        s21_corr.imag, size=window, mode="nearest"
    )
    baseline = uniform_filter1d(baseline.real, max(window // 4, 1), mode="nearest") + 1j * uniform_filter1d(
        baseline.imag, max(window // 4, 1), mode="nearest"
    )
    s21_norm = s21_corr / baseline
    metric = np.abs(s21_norm - 1)
    if excluded is not None:
        metric[excluded] = 0.0
    noise = 1.4826 * np.median(np.abs(metric - np.median(metric)))
    threshold = np.median(metric) + snr * noise
    # The prominence condition rejects noise bumps riding on the shoulder of a strong resonance,
    # the width condition rejects spurs that are much narrower than any resonator
    min_sep = max(int(round(min_sep_hz / step)), 1)
    peaks, props = find_peaks(
        metric,
        height=threshold,
        prominence=snr * noise,
        distance=min_sep,
        width=max(min_width_hz / step, 1),
        rel_height=0.5,
    )
    candidates = []
    for p, width in zip(peaks, props["widths"]):
        dip = np.min(np.abs(s21_norm[max(p - min_sep // 2, 0) : p + min_sep // 2 + 1]))
        # Parabolic interpolation of the peak position
        f0 = freq[p]
        if 0 < p < len(freq) - 1:
            y0, y1, y2 = metric[p - 1], metric[p], metric[p + 1]
            denom = y0 - 2 * y1 + y2
            if denom != 0:
                f0 = freq[p] + 0.5 * (y0 - y2) / denom * (freq[1] - freq[0])
        candidates.append(
            {
                "freq": float(f0),
                "height": float(metric[p]),
                "snr": float((metric[p] - np.median(metric)) / noise) if noise > 0 else float("inf"),
                "width_mhz": float(width * step / 1e6),
                "depth_db": float(20 * np.log10(max(dip, 1e-12))),
            }
        )
    return baseline, s21_norm, metric, float(threshold), candidates


def match_design_to_candidates(design, candidates, tol_hz):
    """
    design: {qubit_name: (line, f_design)}; candidates: {line: [f_measured]}.
    Find the global shift s (and per-line assignment) maximising the number of |f_meas - (f_design + s)| <= tol,
    then minimising the squared residuals. Returns (shift, {qubit_name: assignment}).
    """
    design_lines = {}
    for name, (line, f) in design.items():
        design_lines.setdefault(line, []).append((name, f))
    cap = (3 * tol_hz) ** 2

    def assign(shift):
        result = {}
        for line, entries in design_lines.items():
            cands = np.asarray(candidates.get(line, []), dtype=float)
            if len(cands) == 0:
                continue
            d = np.array([f for _, f in entries]) + shift
            cost = np.minimum((cands[None, :] - d[:, None]) ** 2, cap)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                result[entries[r][0]] = (float(cands[c]), float(cands[c] - d[r]))
        return result

    def score(result):
        res = np.array([r for _, r in result.values()])
        ok = np.abs(res) <= tol_hz
        return (int(ok.sum()), -float(np.sum(res[ok] ** 2)))

    shifts = {0.0}
    for line, entries in design_lines.items():
        for _, f in entries:
            for c in candidates.get(line, []):
                shifts.add(float(c - f))
    best_shift, best_result, best_score = 0.0, {}, (-1, 0.0)
    for s in shifts:
        result = assign(s)
        res = np.array([r for _, r in result.values()])
        if len(res) and np.any(np.abs(res) <= tol_hz):
            # Re-centre the shift on the matched pairs and re-assign
            s = s + float(np.median(res[np.abs(res) <= tol_hz]))
            result = assign(s)
        sc = score(result) if result else (0, 0.0)
        if sc > best_score:
            best_shift, best_result, best_score = s, result, sc

    assignments = {}
    for name, (line, f) in design.items():
        entry = {"line": line, "design_freq": float(f), "shifted_design_freq": float(f + best_shift)}
        if name in best_result:
            f_meas, residual = best_result[name]
            entry.update(
                measured_freq=f_meas,
                residual=residual,
                status="matched" if abs(residual) <= tol_hz else "mismatch",
            )
        else:
            entry.update(measured_freq=None, residual=None, status="not_found")
        assignments[name] = entry
    return float(best_shift), assignments


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
node.machine = machine

lines = group_qubits_by_readout_line(machine)
selected_names = set(node.parameters.qubits) if node.parameters.qubits else {q.name for l in lines.values() for q in l}
line_names = [line for line in lines if any(q.name in selected_names for q in lines[line])]
# One resonator element per line drives the broadband sweep (all resonators of a line share its LO and ports).
scan_resonators = [lines[line][0].resonator for line in line_names]
for line in line_names:
    outputs = {port_key(q.resonator.opx_output) for q in lines[line]}
    assert len(outputs) == 1, f"Readout line {line} is driven by several MW outputs {outputs}; cannot share one LO."
num_lines = len(line_names)
print(f"Found {num_lines} readout line(s): " + ", ".join(f"{l} -> {[q.name for q in lines[l]]}" for l in line_names))

readout_power_dbm = node.parameters.readout_power_dbm
if readout_power_dbm is None:
    readout_power_dbm = u.volts2dBm(0.1)
tracked_resonators = []
if node.parameters.load_data_id is None:
    for rr in scan_resonators:
        with tracked_updates(rr, auto_revert=False, dont_assign_to_none=True) as resonator:
            resonator.operations["readout"].length = node.parameters.readout_length_in_ns
            resonator.set_output_power(readout_power_dbm, operation="readout")
            tracked_resonators.append(resonator)

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()


# %% {QUA_program}
n_avg = node.parameters.num_averages
max_if = node.parameters.max_if_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
assert max_if <= 500 * u.MHz, "The MW-FEM IF must stay within +/- 500 MHz."
ifs = np.arange(-max_if, max_if + step / 2, step).astype(int)

with program() as broadband_spec:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_lines)
    df = declare(int)  # QUA variable for the readout IF

    # Bring the qubits to the minimum frequency point
    machine.apply_all_flux_to_min()

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(df, ifs)):
            # The lines are on different ports and are swept in parallel
            for i, rr in enumerate(scan_resonators):
                update_frequency(rr.name, df)
                rr.measure("readout", qua_vars=(I[i], Q[i]))
                rr.wait(machine.depletion_time * u.ns)
                save(I[i], I_st[i])
                save(Q[i], Q_st[i])

    with stream_processing():
        n_st.save("n")
        for i in range(num_lines):
            I_st[i].buffer(len(ifs)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(ifs)).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, broadband_spec, simulation_config)
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()), 1, i + 1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    node.results = {"figure": plt.gcf()}
    for tracked_resonator in tracked_resonators:
        tracked_resonator.revert_changes()
    node.save()

elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(broadband_spec)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            progress_counter(n, n_avg, start_time=results.start_time)
        I_data = np.array([job.result_handles.get(f"I{i + 1}").fetch_all() for i in range(num_lines)])
        Q_data = np.array([job.result_handles.get(f"Q{i + 1}").fetch_all() for i in range(num_lines)])
    for tracked_resonator in tracked_resonators:
        tracked_resonator.revert_changes()

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is not None:
        load_data_id = node.parameters.load_data_id
        # Re-analyse the stored data with the current analysis parameters (e.g. a new res_design_freq)
        analysis_parameters = node.parameters
        node = node.load_from_id(load_data_id)
        ds = node.results["ds"]
        node.parameters = analysis_parameters
        restore_load_data_id(node, load_data_id)
        machine = node.machine
        lines = group_qubits_by_readout_line(machine)
        selected_names = set(node.parameters.qubits) if node.parameters.qubits else {q.name for l in lines.values() for q in l}
        line_names = [str(l) for l in ds.line.values]
    else:
        to_volts = 2**12 / node.parameters.readout_length_in_ns
        los = np.array([rr.upconverter_frequency for rr in scan_resonators], dtype=float)
        ds = xr.Dataset(
            {"I": (["line", "freq"], I_data * to_volts), "Q": (["line", "freq"], Q_data * to_volts)},
            coords={
                "line": line_names,
                "freq": ifs,
                "LO": ("line", los),
                "freq_full": (["line", "freq"], los[:, None] + ifs[None, :]),
            },
        )
        ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
    node.results = {"ds": ds}

    # %% {Data_analysis}
    p = node.parameters
    res_num = p.res_num if p.res_num is not None else len(machine.qubits)
    expected_per_line = expected_count_per_line(lines, machine, res_num)

    line_results = {}
    analysis = {}
    for line in line_names:
        dsl = ds.sel(line=line)
        freq = dsl.freq_full.values.astype(float)
        s21 = dsl.I.values + 1j * dsl.Q.values
        tau, s21_corr = remove_electrical_delay(freq, s21)
        baseline, s21_norm, metric, threshold, candidates = detect_candidates(
            freq,
            s21_corr,
            p.baseline_window_in_mhz * u.MHz,
            p.min_resonator_separation_in_mhz * u.MHz,
            p.min_linewidth_in_mhz * u.MHz,
            p.detection_snr,
            excluded=np.abs(dsl.freq.values) < p.lo_guard_in_mhz * u.MHz,
        )
        n_expected = expected_per_line.get(line, 0)
        strongest = sorted(candidates, key=lambda c: c["height"], reverse=True)[:n_expected]
        strongest_ids = {id(c) for c in strongest}
        for c in candidates:
            c["selected"] = id(c) in strongest_ids
        analysis[line] = {
            "freq": freq,
            "s21_corr": s21_corr,
            "s21_norm": s21_norm,
            "metric": metric,
            "threshold": threshold,
        }
        line_results[line] = {
            "LO": float(dsl.LO.values),
            "electrical_delay_ns": float(tau * 1e9),
            "num_candidates": len(candidates),
            "num_expected": n_expected,
            "enough_candidates": len(candidates) >= n_expected,
            "candidates": candidates,
            "selected_freqs": sorted(c["freq"] for c in strongest),
        }
        print(
            f"Line {line} (LO {dsl.LO.values / 1e9:.3f} GHz): electrical delay {tau * 1e9:.1f} ns, "
            f"{len(candidates)} candidate(s), {n_expected} expected"
        )
        for c in candidates:
            print(
                f"    {c['freq'] / 1e9:.5f} GHz  SNR {c['snr']:6.1f}  depth {c['depth_db']:6.2f} dB  "
                f"width {c['width_mhz']:.2f} MHz"
                + ("  [selected]" if c["selected"] else "")
            )

    design_shift, assignments = None, {}
    if p.res_design_freq is not None:
        design_hz = design_frequencies_in_hz(p.res_design_freq, res_num)
        design_qubits = expected_qubits(machine, res_num)
        assert design_qubits is not None, (
            f"res_design_freq maps to q1..q{res_num} but the state only has {len(machine.qubits)} qubits."
        )
        qubit_line = {q.name: line for line in lines for q in lines[line]}
        design = {q.name: (qubit_line[q.name], f) for q, f in zip(design_qubits, design_hz)}
        assert all(qubit_line[q.name] in line_names for q in design_qubits), (
            "res_design_freq covers qubits on readout lines that were not swept; include them in `qubits`."
        )
        candidate_freqs = {line: [c["freq"] for c in line_results[line]["candidates"]] for line in line_names}
        design_shift, assignments = match_design_to_candidates(
            design, candidate_freqs, p.design_match_tolerance_in_mhz * u.MHz
        )
        print(f"\nGlobal design shift: {design_shift / 1e6:+.2f} MHz")
        for name, a in assignments.items():
            if a["measured_freq"] is None:
                print(f"    {name}: design {a['design_freq'] / 1e9:.5f} GHz -> NOT FOUND")
            else:
                flag = "" if a["status"] == "matched" else "  <-- MISMATCH, not updated"
                print(
                    f"    {name}: design {a['design_freq'] / 1e9:.5f} GHz -> measured {a['measured_freq'] / 1e9:.5f} GHz "
                    f"(residual after shift {a['residual'] / 1e6:+.2f} MHz){flag}"
                )

    node.results["fit_results"] = {
        "lines": line_results,
        "res_num": res_num,
        "design_shift": design_shift,
        "assignments": assignments,
    }

    # %% {Plotting}
    status_colors = {"matched": "tab:green", "mismatch": "tab:red", "not_found": "tab:gray"}
    fig, axes = plt.subplots(num_lines, 1, figsize=(16, 4.5 * num_lines), squeeze=False)
    for ax, line in zip(axes[:, 0], line_names):
        a = analysis[line]
        f_ghz = a["freq"] / 1e9
        ax.plot(f_ghz, 20 * np.log10(np.abs(a["s21_norm"])), lw=0.6, color="tab:blue")
        ax.set_ylabel("|S21 / baseline| [dB]")
        ax.set_xlabel("Frequency [GHz]")
        ax.set_title(
            f"Line {line}  (LO {line_results[line]['LO'] / 1e9:.3f} GHz, delay "
            f"{line_results[line]['electrical_delay_ns']:.1f} ns): {line_results[line]['num_candidates']} candidates, "
            f"{line_results[line]['num_expected']} expected"
        )
        lo = line_results[line]["LO"]
        guard = p.lo_guard_in_mhz * u.MHz
        ax.axvspan((lo - guard) / 1e9, (lo + guard) / 1e9, color="0.9", zorder=0)
        for c in line_results[line]["candidates"]:
            ax.axvline(c["freq"] / 1e9, color="tab:orange" if c["selected"] else "0.75", lw=0.8, ls="-", zorder=0)
        y_top = ax.get_ylim()[1]
        for name, asg in assignments.items():
            if asg["line"] != line:
                continue
            color = status_colors[asg["status"]]
            ax.axvline(asg["shifted_design_freq"] / 1e9, color=color, ls="--", lw=0.8)
            x = (asg["measured_freq"] if asg["measured_freq"] is not None else asg["shifted_design_freq"]) / 1e9
            ax.annotate(name, (x, y_top), color=color, ha="center", va="bottom", fontsize=9, annotation_clip=False)
    fig.suptitle(
        "Broadband spectroscopy (delay & baseline removed; orange: selected candidates"
        + (", dashed: shifted design" if assignments else "")
        + ")"
    )
    fig.tight_layout()
    node.results["broadband_amplitude"] = fig

    fig, axes = plt.subplots(num_lines, 2, figsize=(16, 4 * num_lines), squeeze=False)
    for (ax_phase, ax_metric), line in zip(axes, line_names):
        a = analysis[line]
        f_ghz = a["freq"] / 1e9
        ax_phase.plot(f_ghz, np.unwrap(np.angle(a["s21_corr"])), lw=0.6)
        ax_phase.set_ylabel("Phase, delay removed [rad]")
        ax_phase.set_xlabel("Frequency [GHz]")
        ax_phase.set_title(f"Line {line}")
        ax_metric.plot(f_ghz, a["metric"], lw=0.6)
        ax_metric.axhline(a["threshold"], color="tab:red", ls="--", lw=0.8, label="threshold")
        for c in line_results[line]["candidates"]:
            ax_metric.plot(c["freq"] / 1e9, c["height"], "o", color="tab:orange" if c["selected"] else "0.6")
        ax_metric.set_ylabel("|S21 / baseline - 1|")
        ax_metric.set_xlabel("Frequency [GHz]")
        ax_metric.legend(loc="upper right")
    fig.suptitle("Broadband spectroscopy diagnostics")
    fig.tight_layout()
    node.results["broadband_diagnostics"] = fig
    plt.show()

    # %% {Update_state}
    if node.parameters.load_data_id is None:
        if assignments:
            with node.record_state_updates():
                for name, asg in assignments.items():
                    if asg["status"] != "matched" or name not in selected_names:
                        continue
                    rr = machine.qubits[name].resonator
                    rr.intermediate_frequency = int(round(asg["measured_freq"] - rr.upconverter_frequency))
            node.outcomes = {
                name: "successful" if asg["status"] == "matched" else "failed"
                for name, asg in assignments.items()
                if name in selected_names
            }
        else:
            qubits = expected_qubits(machine, res_num) or []
            node.outcomes = {
                q.name: "successful" if line_results[port_key(q.resonator.opx_input)]["enough_candidates"] else "failed"
                for q in qubits
                if q.name in selected_names
            }

        # %% {Save_results}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.save()
        print("Results saved")


# %%
