"""Fit the main spectroscopy ridge in runs #1608-#1618 from 2026-11-04.

Each frequency trace is fitted with a Lorentzian and a linear background.
A reduced SQUID envelope fits the ridge, with an empirical frequency offset
for #1615, whose acquisition conditions differ. This does not extract EJ/EC.
The sweet spot is also estimated independently from #1615 with a parabola.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patheffects as pe
from matplotlib.ticker import MultipleLocator
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from merge_coupler_flux_spectroscopy import DEFAULT_DATA_DIR, DEFAULT_RUN_IDS, cell_edges, label_scan, load_scans


MODEL_SOURCE = "https://scqubits.readthedocs.io/en/v4.1/guide/qubits/tunable_transmon.html"
PARAMETER_NAMES = ["reference_max_ghz", "sweet_flux_v", "effective_period_v", "run_1615_offset_ghz"]
INITIAL = np.array([3.91, -0.236, 0.71, 0.053])
BOUNDS = ([3.5, -0.26, 0.60, -0.20], [4.5, -0.20, 1.20, 0.20])


def envelope(parameters, flux, group):
    amplitude, sweet, period, offset = parameters
    cosine = np.cos(np.pi * (flux - sweet) / period)
    return amplitude * np.sqrt(np.maximum(cosine, 1e-12)) + offset * group


def robust_fit(function, initial, bounds, x, y, group):
    result = least_squares(
        lambda p: (function(p, x, group) - y) * 1000,
        initial, bounds=bounds, loss="soft_l1", f_scale=3, max_nfev=2000,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def extract_peaks(scans):
    records = []
    for run in scans.run.values:
        scan = scans.sel(run=run)
        signal = scan.state.values
        frequency = scan.frequency_hz.values / 1e9
        step = float(np.median(np.diff(frequency)) * 1000)
        smooth = gaussian_filter1d(signal, 1, axis=0)
        differences = np.diff(signal, axis=0)
        noise = float(np.median(np.abs(differences - np.median(differences))) / 0.67449 / np.sqrt(2))
        for column, flux in enumerate(scan.flux_v.values):
            row = dict(run=int(run), flux_index=column, flux_v=float(flux),
                       frequency_ghz=np.nan, center_sigma_mhz=np.nan,
                       fwhm_mhz=np.nan, snr=np.nan, prominence=np.nan,
                       status="no_resolved_peak", used_for_global_fit=False)
            records.append(row)
            peaks, properties = find_peaks(
                smooth[:, column], prominence=max(3 * noise, 0.035), width=1.5, distance=4,
            )
            if not len(peaks):
                continue
            strongest = int(np.argmax(properties["prominences"]))
            peak = peaks[strongest]
            width = properties["widths"][strongest] * step
            window = max(25, 2 * width)
            mask = np.abs((frequency - frequency[peak]) * 1000) <= window
            x = (frequency[mask] - frequency[peak]) * 1000
            y = signal[mask, column]
            if len(x) < 9:
                row["status"] = "too_few_points"
                continue
            background = float(np.percentile(y, 20))
            amplitude = max(float(y.max() - background), 0.01)

            def line_shape(p):
                baseline, slope, height, center, half_width = p
                return baseline + slope * x + height / (1 + ((x - center) / half_width) ** 2)

            fit = least_squares(
                lambda p: (line_shape(p) - y) / noise,
                [background, 0, amplitude, 0, max(step, width / 2)],
                bounds=([0, -0.005, 0, -max(8, 2 * step), step / 3],
                        [1, 0.005, 1, max(8, 2 * step), 40]),
                loss="soft_l1",
            )
            if not fit.success:
                row["status"] = "line_fit_failed"
                continue
            _, _, height, center, half_width = fit.x
            center_ghz = float(frequency[peak] + center / 1000)
            covariance = np.linalg.pinv(fit.jac.T @ fit.jac) * np.sum(fit.fun**2) / max(1, len(x) - 5)
            row.update(
                frequency_ghz=center_ghz,
                center_sigma_mhz=float(max(np.sqrt(covariance[3, 3]), step / np.sqrt(12))),
                fwhm_mhz=float(2 * half_width), snr=float(height / noise),
                prominence=float(properties["prominences"][strongest]),
            )
            edge_distance = min(center_ghz - frequency[0], frequency[-1] - center_ghz) * 1000
            if edge_distance < max(2 * step, 1.5 * half_width):
                row["status"] = "edge_clipped"
            elif height < 3 * noise or half_width > 39 or half_width < step / 2:
                row["status"] = "unreliable_line_shape"
            else:
                row["status"] = "candidate"
    return records


def fit_ridge(records, bootstrap_count):
    candidates = [row for row in records if row["status"] == "candidate"]
    x = np.array([r["flux_v"] for r in candidates])
    y = np.array([r["frequency_ghz"] for r in candidates])
    runs = np.array([r["run"] for r in candidates])
    group = (runs == 1615).astype(float)
    parameters = robust_fit(envelope, INITIAL, BOUNDS, x, y, group)
    residual = (y - envelope(parameters, x, group)) * 1000
    scatter = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    threshold_mhz = max(10, 4 * scatter)
    good = np.abs(residual) < threshold_mhz
    for row, accepted in zip(candidates, good):
        row["used_for_global_fit"] = bool(accepted)
        row["status"] = "accepted" if accepted else "off_main_ridge"
    x, y, runs, group = x[good], y[good], runs[good], group[good]
    parameters = robust_fit(envelope, parameters, BOUNDS, x, y, group)
    residual = (y - envelope(parameters, x, group)) * 1000

    # Model comparisons use exactly the same accepted points.
    shared = lambda p, v, g: envelope([*p, 0], v, g)
    parabola = lambda p, v, g: p[0] + p[2] * (v - p[1])**2 + p[3] * g
    shared_p = robust_fit(shared, [3.94, -0.245, 0.73],
                          (BOUNDS[0][:3], BOUNDS[1][:3]), x, y, group)
    quadratic_p = robust_fit(parabola, [3.93, -0.235, -21, 0.04],
                             ([3.5, -0.3, -100, -0.2], [4.5, -0.15, -1, 0.2]), x, y, group)
    comparisons = []
    for name, model, p in [("sqrt_cos_with_run_offset", envelope, parameters),
                            ("sqrt_cos_without_offset", shared, shared_p),
                            ("quadratic_with_run_offset", parabola, quadratic_p)]:
        errors = (y - model(p, x, group)) * 1000
        comparisons.append(dict(model=name, points=len(x), parameters=len(p),
                                rmse_mhz=float(np.sqrt(np.mean(errors**2))),
                                median_absolute_error_mhz=float(np.median(np.abs(errors)))))

    local = runs == 1615
    local_coeff = np.polyfit(x[local], y[local], 2)

    def vertex(coefficients):
        sweet = -coefficients[1] / (2 * coefficients[0])
        return [float(sweet), float(np.polyval(coefficients, sweet)), float(coefficients[0])]

    local_values = vertex(local_coeff)
    local_errors = (y[local] - np.polyval(local_coeff, x[local])) * 1000
    rng = np.random.default_rng(20261104)
    indices_by_run = [np.flatnonzero(runs == run) for run in np.unique(runs)]
    boot_global, boot_local = [], []
    for _ in range(bootstrap_count):
        indices = np.concatenate([rng.choice(indices, len(indices), replace=True) for indices in indices_by_run])
        boot_global.append(robust_fit(envelope, parameters, BOUNDS, x[indices], y[indices], group[indices]))
        local_indices = rng.choice(np.flatnonzero(local), local.sum(), replace=True)
        boot_local.append(vertex(np.polyfit(x[local_indices], y[local_indices], 2)))
    global_intervals = np.percentile(boot_global, [2.5, 97.5], axis=0).T
    local_intervals = np.percentile(boot_local, [2.5, 97.5], axis=0).T
    per_run = []
    for run in np.unique(runs):
        selected = runs == run
        per_run.append(dict(run=int(run), points=int(selected.sum()),
                            mean_residual_mhz=float(residual[selected].mean()),
                            rmse_mhz=float(np.sqrt(np.mean(residual[selected]**2)))))
    result = dict(
        model="f(V, run) = A * sqrt(cos(pi * (V - V_sweet) / V_period)) + delta * (run == 1615)",
        interpretation="Effective envelope on the measured lobe, with an empirical acquisition-condition offset; not a determination of EJ, EC, junction asymmetry, or the cause of the offset.",
        model_motivation_source=MODEL_SOURCE,
        parameters={name: dict(value=float(value), conditional_95_percent_ci=interval.tolist())
                    for name, value, interval in zip(PARAMETER_NAMES, parameters, global_intervals)},
        global_rmse_mhz=float(np.sqrt(np.mean(residual**2))),
        reference_validated_flux_range_v=[float(x[~local].min()), float(x[~local].max())],
        run_1615_validated_flux_range_v=[float(x[local].min()), float(x[local].max())],
        peak_extraction=dict(total_traces=len(records), candidates=len(candidates),
                             accepted=int(good.sum()), rejected_off_ridge=int((~good).sum()),
                             off_ridge_threshold_mhz=float(threshold_mhz),
                             method="Positive Lorentzian plus linear baseline, initialized by strongest smoothed peak; edge-clipped and low-SNR lines excluded."),
        local_run_1615_quadratic=dict(
            model="f(V) = f_max + curvature * (V - V_sweet)^2",
            parameters={name: dict(value=value, conditional_95_percent_ci=interval.tolist())
                        for name, value, interval in zip(
                            ["sweet_flux_v", "maximum_frequency_ghz", "curvature_ghz_per_v2"],
                            local_values, local_intervals)},
            polynomial_coefficients_descending=local_coeff.tolist(),
            points=int(local.sum()), rmse_mhz=float(np.sqrt(np.mean(local_errors**2))),
        ),
        uncertainty="95% intervals from stratified resampling of accepted flux traces with fixed extraction and model selection; acquisition systematics and model uncertainty are excluded.",
        bootstrap_replicates=bootstrap_count,
        model_comparison_same_points=comparisons, residual_by_run=per_run,
    )
    return parameters, local_coeff, result


def save_csv(path, records):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot_results(scans, records, parameters, local_coeff, result, output):
    good = [r for r in records if r["used_for_global_fit"]]
    x = np.array([r["flux_v"] for r in good])
    y = np.array([r["frequency_ghz"] for r in good])
    runs = np.array([r["run"] for r in good])
    group = (runs == 1615).astype(float)
    all_runs = scans.run.values.astype(int)
    run_range = f"{all_runs.min()}-{all_runs.max()}"
    coupler = scans.attrs["coupler"]
    date = Path(scans.attrs["source_data_directory"]).name
    figure, ax = plt.subplots(figsize=(13, 8))
    figure.subplots_adjust(left=0.085, right=0.9, bottom=0.16, top=0.88)
    norm = Normalize(*np.percentile(scans.state.values, [2, 98]))
    for run in scans.run.values:
        scan = scans.sel(run=run)
        xe, ye = cell_edges(scan.flux_v.values), cell_edges(scan.frequency_hz.values / 1e9)
        mesh = ax.pcolormesh(xe, ye, scan.state.values, cmap="viridis", norm=norm,
                             shading="flat", rasterized=True, antialiased=False)
        label_scan(ax, int(run), xe, ye)
    for index, regime in enumerate([0, 1]):
        selected = group == regime
        grid = np.linspace(x[selected].min(), x[selected].max(), 700)
        ax.plot(grid, envelope(parameters, grid, regime), "--", color="#ff7755", lw=1.9,
                label="Global fit with acquisition offset" if index == 0 else None,
                path_effects=[pe.withStroke(linewidth=3.2, foreground="#172f44")], zorder=5)
    ax.scatter(x, y, s=9, facecolors="none", edgecolors="white", linewidths=0.5,
               alpha=0.75, label="Accepted line centers", zorder=4)
    local_sweet = -local_coeff[1] / (2 * local_coeff[0])
    local_max = np.polyval(local_coeff, local_sweet)
    ax.scatter([local_sweet], [local_max], marker="*", s=160, color="#ffb54c",
               edgecolors="#172f44", linewidths=0.8, label="#1615 quadratic maximum", zorder=6)
    ax.text(0.035, 0.52,
            "#1615 local sweet spot\n"
            f"Flux pulse = {local_sweet:.5f} V\nFrequency = {local_max:.5f} GHz\n\n"
            f"Global fit RMSE = {result['global_rmse_mhz']:.2f} MHz\n"
            f"#1615 empirical offset = {parameters[3] * 1000:+.1f} MHz\n"
            f"Accepted line centers: {len(good)}",
            transform=ax.transAxes, fontsize=11, va="top", color="#273a50",
            bbox=dict(facecolor="white", edgecolor="#d4dde6", boxstyle="round,pad=0.7"))
    frequency_span_ghz = float(scans.frequency_hz.max() - scans.frequency_hz.min()) / 1e9
    ax.set(xlim=(float(scans.flux_v.min()) - 0.005, float(scans.flux_v.max()) + 0.006),
           ylim=(float(scans.frequency_hz.min() / 1e9) - 0.02,
                 float(scans.frequency_hz.max() / 1e9) + max(0.075, 0.07 * frequency_span_ghz)),
           xlabel="Coupler flux pulse amplitude (V)",
           ylabel="Drive frequency (GHz)")
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.2 if frequency_span_ghz > 1.7 else 0.1))
    ax.legend(loc="lower left", bbox_to_anchor=(0.025, 0.07), framealpha=0.95, fontsize=10)
    ax.set_title(f"{date} | #{all_runs.min()}-#{all_runs.max()} | raw data and fitted main ridge", fontsize=11, pad=12)
    figure.suptitle(f"Coupler flux spectroscopy fit | {coupler}", fontsize=17, y=0.965)
    figure.colorbar(mesh, ax=ax, pad=0.025, fraction=0.035, extend="both").set_label("Mean measured state (raw)")
    figure.text(0.085, 0.075,
                 "#1615: drive amplitude factor 0.18, qubit detuning flux 0.30 V; fitted offset is empirical.", fontsize=10)
    figure.text(0.085, 0.04,
                 "White = unmeasured; later runs shown on top in overlaps. Reference-condition maximum is extrapolated.", fontsize=9)
    for extension in ["png", "pdf"]:
        figure.savefig(output / f"{coupler}_flux_spectroscopy_{run_range}_fit.{extension}", dpi=240)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [1.3, 1]})
    colors = plt.get_cmap("tab20" if len(np.unique(runs)) > 10 else "tab10")
    residual = (y - envelope(parameters, x, group)) * 1000
    for index, run in enumerate(np.unique(runs)):
        selected = runs == run
        color = colors(index)
        axes[0].scatter(x[selected], y[selected], s=10, color=color, label=f"#{run}")
        axes[1].scatter(x[selected], residual[selected], s=13, color=color)
    for regime in [0, 1]:
        selected = group == regime
        grid = np.linspace(x[selected].min(), x[selected].max(), 600)
        axes[0].plot(grid, envelope(parameters, grid, regime), color="#15293e", lw=1)
    rejected = [r for r in records if r["status"] == "off_main_ridge"]
    axes[0].scatter([r["flux_v"] for r in rejected], [r["frequency_ghz"] for r in rejected],
                    s=30, marker="x", color="red", label="Off main ridge (excluded)")
    axes[0].legend(ncol=3, fontsize=8, loc="lower left")
    axes[0].set_ylabel("Line center (GHz)")
    axes[1].axhline(0, color="#15293e", lw=1)
    axes[1].set(xlabel="Coupler flux pulse amplitude (V)", ylabel="Data - global fit (MHz)")
    for ax in axes:
        ax.grid(alpha=0.2)
    figure.suptitle(f"Extracted main ridge and residuals | same {len(good)} accepted points")
    figure.tight_layout()
    figure.savefig(output / "fit_diagnostics.png", dpi=220)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--runs", nargs="+", type=int, default=list(DEFAULT_RUN_IDS))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=300)
    args = parser.parse_args()
    if args.bootstrap < 20:
        parser.error("Use at least 20 bootstrap replicates")
    if len(set(args.runs)) != len(args.runs):
        parser.error("Run IDs must be unique")
    if 1615 not in args.runs:
        parser.error("Include #1615 for the acquisition-offset and local sweet-spot estimates")
    output = args.output_dir or Path(__file__).parent / "analysis" / f"{args.data_dir.name}_{min(args.runs)}-{max(args.runs)}" / "fit"
    scans, _ = load_scans(args.data_dir, args.runs)
    records = extract_peaks(scans)
    parameters, local_coeff, result = fit_ridge(records, args.bootstrap)
    result["source_runs"] = [int(run) for run in scans.run.values]
    result["source_data_directory"] = str(args.data_dir.resolve())
    output.mkdir(parents=True, exist_ok=True)
    (output / "fit_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    save_csv(output / "peak_fits.csv", records)
    save_csv(output / "model_comparison.csv", result["model_comparison_same_points"])
    plot_results(scans, records, parameters, local_coeff, result, output)
    print(json.dumps(result, indent=2))
    print(f"Outputs: {output.resolve()}")


if __name__ == "__main__":
    main()
