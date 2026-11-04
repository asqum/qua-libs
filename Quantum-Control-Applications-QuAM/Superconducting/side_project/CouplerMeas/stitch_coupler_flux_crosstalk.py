"""Stitch #1608-#1618 by correcting only the flux coordinates of #1615.

A shared local even-quartic ridge registers #1614 and #1615 horizontally.
An empirical smoothing spline connects all registered main-ridge samples.
The estimated crosstalk is the combined voltage-equivalent response to q4
and q5 each receiving +0.30 V; their individual coefficients are unresolved.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties
import matplotlib.patheffects as pe
from matplotlib.ticker import MultipleLocator
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import least_squares

from merge_coupler_flux_spectroscopy import (
    DEFAULT_DATA_DIR, DEFAULT_RUN_IDS, cell_edges, label_scan, load_scans,
)
from fit_coupler_flux_spectroscopy import extract_peaks, save_csv


REGISTRATION_SCALE_V = 0.15
SHIFTED_RUN = 1615


def local_ridge(parameters, corrected_flux):
    maximum, sweet, _, *coefficients = parameters
    z = (corrected_flux - sweet) / REGISTRATION_SCALE_V
    return maximum + sum(value * z ** (2 * (i + 1)) for i, value in enumerate(coefficients))


def register(x, frequency, shifted, degree=4, initial=None):
    terms = degree // 2
    if initial is None:
        initial = [3.962, -0.253, -0.016, -0.42] + [-0.02] * (terms - 1)
    fit = least_squares(
        lambda p: (local_ridge(p, x + p[2] * shifted) - frequency) * 1000,
        initial,
        bounds=([3.8, -0.30, -0.05, -2] + [-2] * (terms - 1),
                [4.1, -0.20, 0.05, -0.05] + [2] * (terms - 1)),
        loss="soft_l1", f_scale=2, max_nfev=3000,
    )
    if not fit.success:
        raise RuntimeError(fit.message)
    return fit.x


def fit_registration(records, detuning_v, bootstrap_count):
    candidates = [r for r in records if r["status"] == "candidate"]
    region = [r for r in candidates if r["run"] in [1614, SHIFTED_RUN]]
    x = np.array([r["flux_v"] for r in region])
    y = np.array([r["frequency_ghz"] for r in region])
    runs = np.array([r["run"] for r in region])
    shifted = (runs == SHIFTED_RUN).astype(float)
    parameters = register(x, y, shifted)
    errors = (y - local_ridge(parameters, x + parameters[2] * shifted)) * 1000

    rng = np.random.default_rng(20261104)
    groups = [np.flatnonzero(runs == run) for run in np.unique(runs)]
    samples = []
    for _ in range(bootstrap_count):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        samples.append(register(x[indices], y[indices], shifted[indices], initial=parameters))
    intervals = np.percentile(samples, [2.5, 97.5], axis=0).T

    comparisons = []
    for first_run, degree in [(1614, 4), (1614, 6), (1613, 4), (1613, 6)]:
        subset = [r for r in candidates if first_run <= r["run"] <= SHIFTED_RUN]
        sx = np.array([r["flux_v"] for r in subset])
        sy = np.array([r["frequency_ghz"] for r in subset])
        group = np.array([r["run"] == SHIFTED_RUN for r in subset], dtype=float)
        p = register(sx, sy, group, degree=degree)
        residual = (sy - local_ridge(p, sx + p[2] * group)) * 1000
        comparisons.append(dict(
            registration_runs=f"{first_run}-{SHIFTED_RUN}", degree=degree, points=len(subset),
            shift_v=float(p[2]), combined_crosstalk_percent=float(100 * p[2] / detuning_v),
            rmse_mhz=float(np.sqrt(np.mean(residual**2))),
        ))

    names = ["maximum_frequency_ghz", "sweet_flux_corrected_v", "run_1615_flux_shift_v",
             "quadratic_scaled_ghz", "quartic_scaled_ghz"]
    result = dict(
        coordinate_rule="V_corrected = V_raw + delta_V * (run == 1615)",
        frequency_correction_hz=0.0,
        interpretation="Assume the #1615 change is a constant flux displacement caused by simultaneous qubit detuning. Register the neighboring scans using a shared local ridge; retain the measured frequencies.",
        registration=dict(
            runs=[1614, SHIFTED_RUN], points=len(region), degree=4,
            model="f(V_corrected) = f_max + c2*z^2 + c4*z^4; z=(V_corrected-V_sweet)/0.15 V",
            parameters={name: dict(value=float(value), conditional_95_percent_ci=interval.tolist())
                        for name, value, interval in zip(names, parameters, intervals)},
            rmse_mhz=float(np.sqrt(np.mean(errors**2))),
        ),
        crosstalk=dict(
            source_qubits=["q4", "q5"], voltage_per_qubit_v=float(detuning_v),
            definition="V_effective = V_command + k_q4*V_q4 + k_q5*V_q5; k_q4+k_q5 = delta_V / 0.30 V",
            combined_voltage_equivalent_coefficient=float(parameters[2] / detuning_v),
            combined_percent=float(100 * parameters[2] / detuning_v),
            conditional_95_percent_ci=(100 * intervals[2] / detuning_v).tolist(),
            model_sensitivity_percent_range=[
                min(r["combined_crosstalk_percent"] for r in comparisons),
                max(r["combined_crosstalk_percent"] for r in comparisons),
            ],
            interpretation="Combined voltage-equivalent coefficient for both qubit Z lines each driven by +0.30 V. Individual line coefficients and a mutual-inductance flux ratio cannot be separated from these scans.",
        ),
        sensitivity_models=comparisons,
        bootstrap_replicates=bootstrap_count,
        uncertainty="Bootstrap intervals condition on the shared local quartic model and peak extraction. The sensitivity-model range is a separate model-dependence check, not a confidence interval. Registration spans an unmeasured gap rather than matching overlapping resonance peaks.",
    )
    return parameters, result


def fit_continuous_curve(records, shift, target_mhz=1.5):
    candidates = [r for r in records if r["status"] == "candidate"]
    x = np.array([r["flux_v"] + shift * (r["run"] == SHIFTED_RUN) for r in candidates])
    y = np.array([r["frequency_ghz"] for r in candidates])
    center, scale = (x.max() + x.min()) / 2, (x.max() - x.min()) / 2
    matrix = np.polynomial.polynomial.polyvander((x - center) / scale, 6)
    initial = np.linalg.lstsq(matrix, y, rcond=None)[0]
    coarse = least_squares(lambda p: (matrix @ p - y) * 1000, initial,
                            loss="soft_l1", f_scale=3)
    # Reject obvious secondary branches, independently of the former y-offset fit.
    good = np.abs(matrix @ coarse.x - y) < 0.020
    for row in records:
        row["flux_raw_v"] = row["flux_v"]
        row["flux_axis_correction_v"] = float(shift if row["run"] == SHIFTED_RUN else 0)
        row["flux_corrected_v"] = row["flux_v"] + row["flux_axis_correction_v"]
    for row, accepted in zip(candidates, good):
        row["used_for_global_fit"] = bool(accepted)
        row["status"] = "accepted" if accepted else "off_main_ridge"
    x, y = x[good], y[good]
    # Collapse duplicate coordinates only for spline fitting, retaining every raw sample.
    unique_x, inverse = np.unique(np.round(x, 12), return_inverse=True)
    counts = np.bincount(inverse)
    mean_y = np.bincount(inverse, weights=y) / counts
    spline = UnivariateSpline(
        unique_x, mean_y, w=np.sqrt(counts), s=len(x) * (target_mhz / 1000)**2,
        k=3, ext=2, bbox=[min(unique_x.min(), x.min()) - 1e-12,
                         max(unique_x.max(), x.max()) + 1e-12],
    )
    errors = (y - spline(x)) * 1000
    return spline, dict(
        method="Cubic smoothing spline on registered main-ridge centers; interpolation in unsampled intervals.",
        accepted_peak_centers=len(x), excluded_secondary_branch_points=int((~good).sum()),
        smoothing_target_mhz=target_mhz, rmse_mhz=float(np.sqrt(np.mean(errors**2))),
        knots=spline.get_knots().tolist(), validated_flux_domain_v=[float(x.min()), float(x.max())],
    )


def plot_stitched(scans, records, parameters, result, spline, output):
    selected = [r for r in records if r["used_for_global_fit"]]
    x = np.array([r["flux_corrected_v"] for r in selected])
    y = np.array([r["frequency_ghz"] for r in selected])
    shift = parameters[2]
    figure, ax = plt.subplots(figsize=(13, 8.5))
    figure.subplots_adjust(left=0.085, right=0.90, bottom=0.22, top=0.88)
    norm = Normalize(*np.percentile(scans.state.values, [2, 98]))
    for run in scans.run.values:
        scan = scans.sel(run=run)
        xe = cell_edges(scan.flux_corrected_v.values)
        ye = cell_edges(scan.frequency_hz.values / 1e9)
        mesh = ax.pcolormesh(xe, ye, scan.state.values, cmap="viridis", norm=norm,
                             rasterized=True, shading="flat", antialiased=False)
        label_scan(ax, int(run), xe, ye)
    grid = np.linspace(x.min(), x.max(), 1800)
    ax.plot(grid, spline(grid), "--", color="#ff7852", lw=1.8,
            path_effects=[pe.withStroke(linewidth=3.1, foreground="#20394b")],
            label="Continuous fit after flux correction", zorder=5)
    ax.scatter(x, y, s=8, facecolors="none", edgecolors="white", linewidths=0.45, alpha=0.7)
    local = [r for r in selected if r["run"] == SHIFTED_RUN]
    polynomial = np.polyfit([r["flux_corrected_v"] for r in local],
                            [r["frequency_ghz"] for r in local], 2)
    sweet = -polynomial[1] / (2 * polynomial[0])
    maximum = float(np.polyval(polynomial, sweet))
    ax.scatter([sweet], [maximum], marker="*", s=150, color="#ffb94c",
               edgecolors="#20394b", zorder=6, label="Sweet spot on corrected flux axis")
    ax.text(0.035, 0.57,
            f"#1615 flux correction: {shift * 1000:+.2f} mV\n"
            f"Combined voltage crosstalk: {result['crosstalk']['combined_percent']:.2f}%\n"
            f"q4, q5: +0.30 V each\n\n"
            f"Corrected sweet spot: {sweet:.5f} V\nMaximum frequency: {maximum:.5f} GHz\n"
            f"Local registration RMSE: {result['registration']['rmse_mhz']:.2f} MHz",
            transform=ax.transAxes, va="top", fontsize=10.5, color="#263c50",
            bbox=dict(boxstyle="round,pad=0.65", facecolor="white", edgecolor="#d1dae3"))
    ax.set(xlim=(float(scans.flux_corrected_v.min()) - 0.005,
                 float(scans.flux_corrected_v.max()) + 0.006), ylim=(1.99, 4.15),
           xlabel="Corrected coupler-equivalent flux pulse amplitude (V)", ylabel="Drive frequency (GHz)")
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.legend(loc="lower left", bbox_to_anchor=(0.025, 0.08), fontsize=9.5, framealpha=0.95)
    figure.colorbar(mesh, ax=ax, fraction=0.035, pad=0.025, extend="both").set_label("Mean measured state (raw)")
    figure.suptitle("Coupler flux spectroscopy | flux-crosstalk corrected", fontsize=17, y=0.965)
    ax.set_title("coupler_q4_q5 | 2026-11-04 | #1608-#1618", fontsize=11, pad=12)

    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    note = (f"註：#1615 因 q4、q5 各加 +0.3 V 所致 flux crosstalk，Flux 軸左移 {-shift * 1000:.1f} mV；頻率保留原值。")
    if font_path.exists():
        figure.text(0.085, 0.145, note, fontsize=10.5, fontproperties=FontProperties(fname=str(font_path)))
        figure.text(0.085, 0.102,
                    f"合計等效 crosstalk 約 {abs(result['crosstalk']['combined_percent']):.1f}%（帶符號為 {result['crosstalk']['combined_percent']:.1f}%）；此資料無法分離 q4、q5 各自的係數。",
                    fontsize=10.2, fontproperties=FontProperties(fname=str(font_path)))
    else:
        figure.text(0.085, 0.145,
                    f"#1615: +0.30 V on each qubit causes a flux-crosstalk correction of {shift * 1000:+.1f} mV; frequencies retained.", fontsize=10)
        figure.text(0.085, 0.102,
                    f"Combined voltage-equivalent crosstalk = {result['crosstalk']['combined_percent']:.1f}%; individual line coefficients unresolved.", fontsize=10)
    figure.text(0.085, 0.06,
                 "White = unmeasured. The continuous fit bridges the unsampled seam; image pixels retain native values.", fontsize=9)
    for extension in ["png", "pdf"]:
        figure.savefig(output / f"coupler_q4_q5_flux_spectroscopy_1608-1618_flux_corrected.{extension}", dpi=240)
    plt.close(figure)
    return sweet, maximum


def plot_diagnostics(scans, records, parameters, spline, output):
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={"height_ratios": [1.2, 1]})
    colors = {1614: "#2b66a0", 1615: "#d46b2c"}
    for column, corrected in enumerate([False, True]):
        ax = axes[0, column]
        for run in [1614, SHIFTED_RUN]:
            scan = scans.sel(run=run)
            key = "flux_corrected_v" if corrected else "flux_v"
            z = scan.state.values
            contrast = (z - np.percentile(z, 10)) / (np.percentile(z, 98) - np.percentile(z, 10))
            ax.pcolormesh(cell_edges(scan[key].values), cell_edges(scan.frequency_hz.values / 1e9),
                           contrast, cmap="viridis", vmin=0, vmax=1, shading="flat", rasterized=True)
            points = [r for r in records if r["run"] == run and r["used_for_global_fit"]]
            ax.scatter([r["flux_corrected_v"] if corrected else r["flux_v"] for r in points],
                        [r["frequency_ghz"] for r in points], facecolors="none", edgecolors=colors[run],
                        s=18, label=f"#{run}")
        if corrected:
            grid = np.linspace(-0.185, -0.105, 400)
            ax.plot(grid, local_ridge(parameters, grid), color="#ff7852", lw=1.4, ls="--",
                    path_effects=[pe.withStroke(linewidth=2.4, foreground="#20394b")])
        ax.set(xlim=(-0.185, -0.105), ylim=(3.65, 3.88),
               title="After: horizontal flux correction" if corrected else "Before: original flux coordinates",
               xlabel="Corrected flux (V)" if corrected else "Original flux (V)", ylabel="Frequency (GHz)")
        ax.legend(loc="lower left", fontsize=9)

    for run, color in colors.items():
        points = [r for r in records if r["run"] == run and r["used_for_global_fit"]]
        x = np.array([r["flux_corrected_v"] for r in points])
        y = np.array([r["frequency_ghz"] for r in points])
        axes[1, 0].scatter(x, (y - local_ridge(parameters, x)) * 1000, s=15, color=color, label=f"#{run}")
    axes[1, 0].set(xlabel="Corrected flux (V)", ylabel="Data - local registration (MHz)", title="Local registration residuals")
    selected = [r for r in records if r["used_for_global_fit"]]
    x = np.array([r["flux_corrected_v"] for r in selected])
    y = np.array([r["frequency_ghz"] for r in selected])
    axes[1, 1].scatter(x, (y - spline(x)) * 1000, s=10, color="#446b87")
    axes[1, 1].set(xlabel="Corrected flux (V)", ylabel="Data - smooth curve (MHz)", title="Full stitched curve residuals")
    for ax in axes[1]:
        ax.axhline(0, color="#293c4f", lw=0.7)
        ax.grid(alpha=0.2)
    figure.suptitle("Flux-only stitching | original frequencies retained | join panels use per-run display contrast", fontsize=11)
    figure.tight_layout()
    figure.savefig(output / "flux_stitch_diagnostics.png", dpi=220)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    if args.bootstrap < 20:
        parser.error("Use at least 20 bootstrap replicates")
    output = args.output_dir or Path(__file__).parent / "analysis" / f"{args.data_dir.name}_1608-1618" / "flux_crosstalk_corrected"
    scans, metadata = load_scans(args.data_dir, list(DEFAULT_RUN_IDS))
    detuning_v = float(scans.qubits_detune_flux_amp_v.sel(run=SHIFTED_RUN))
    if not np.isclose(detuning_v, 0.30):
        raise ValueError("This registration assumes +0.30 V on each qubit for #1615")
    if any(m["qubits_detune_flux_amp_v"] != 0 for m in metadata if m["run"] != SHIFTED_RUN):
        raise ValueError("Reference scans must use zero additional qubit detuning")
    records = extract_peaks(scans)
    parameters, result = fit_registration(records, detuning_v, args.bootstrap)
    shift = float(parameters[2])
    spline, smooth_result = fit_continuous_curve(records, shift)
    result["continuous_curve"] = smooth_result
    result["source_runs"] = list(DEFAULT_RUN_IDS)
    result["source_data_directory"] = str(args.data_dir.resolve())
    correction = np.where(scans.run.values == SHIFTED_RUN, shift, 0.0)
    corrected = scans.copy(deep=True)
    corrected["flux_axis_correction_v"] = ("run", correction, {"units": "V"})
    corrected = corrected.assign_coords(
        flux_corrected_v=(scans.flux_v + corrected.flux_axis_correction_v)
    )
    corrected.flux_corrected_v.attrs.update(units="V", long_name="Crosstalk-corrected coupler-equivalent flux amplitude")
    corrected.attrs.update(
        description="Native raw state, frequency and flux coordinates retained. flux_corrected_v adds the fitted horizontal correction only to #1615.",
        correction_rule="flux_corrected_v = flux_v + flux_axis_correction_v; frequency correction = 0 Hz",
    )
    seam_left = float(corrected.flux_corrected_v.sel(run=SHIFTED_RUN).max())
    seam_right = float(corrected.flux_corrected_v.sel(run=1614).min())
    result["unsampled_seam"] = dict(flux_interval_v=[seam_left, seam_right],
                                    width_mv=float((seam_right - seam_left) * 1000),
                                    rendering="Unmeasured background remains white; smooth curve provides an interpolated connection.")
    output.mkdir(parents=True, exist_ok=True)
    corrected.to_netcdf(output / "combined_flux_corrected.nc", engine="scipy")
    sweet, maximum = plot_stitched(corrected, records, parameters, result, spline, output)
    result["local_run_1615_peak_after_axis_correction"] = dict(sweet_flux_v=float(sweet), maximum_frequency_ghz=maximum)
    plot_diagnostics(corrected, records, parameters, spline, output)
    save_csv(output / "registered_peak_fits.csv", records)
    save_csv(output / "registration_model_sensitivity.csv", result["sensitivity_models"])
    grid = np.linspace(*smooth_result["validated_flux_domain_v"], 2000)
    save_csv(output / "continuous_curve.csv", [dict(flux_corrected_v=float(v), frequency_ghz=float(f))
                                                for v, f in zip(grid, spline(grid))])
    (output / "flux_stitch_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Outputs: {output.resolve()}")


if __name__ == "__main__":
    main()
