"""Add every fitting function to the stitched spectrum and constrain EC.

The provided f01 and f02/2 are treated as transitions at the same reference
operating point. Their voltage need not be assumed: the local EJ_eff and EC
are inferred first, at each ng, then the full flux ridge is fitted with that
EC constraint. Frequencies and the established flux correction are retained.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties
import matplotlib.patheffects as pe
import numpy as np
from scipy.constants import elementary_charge, h
from scipy.interpolate import BSpline, CubicSpline
from scipy.optimize import brentq, least_squares
import xarray as xr

from fit_coupler_flux_spectroscopy import save_csv
from fit_coupler_transmon_model import ChargeBasisModel, dimensionless_levels, plot_analysis, summarize
from merge_coupler_flux_spectroscopy import cell_edges, label_scan


BASE = Path(__file__).parent / "analysis/2026-11-04_1608-1618"
LOWER = np.array([3.5, 0.0, 0.62, -0.30, 0.0])
UPPER = np.array([6.0, 0.9025, 1.5, -0.20, 0.5])


class ReferenceConstraint:
    def __init__(self, f01, half_f02):
        self.f01, self.half_f02 = f01, half_f02
        self.ng = np.linspace(0, 0.5, 61)
        self.solutions = [self.solve(ng) for ng in self.ng]
        self.ec = CubicSpline(self.ng, [r[0] for r in self.solutions])

    def solve(self, ng):
        target = 2 * self.half_f02 / self.f01

        def ratio_error(ratio):
            e = dimensionless_levels(ratio, ng=ng, ncut=55)
            return (e[2] - e[0]) / (e[1] - e[0]) - target

        ratio = brentq(ratio_error, 20, 200, xtol=1e-10)
        e = dimensionless_levels(ratio, ng=ng, ncut=80)
        ec = self.f01 / (e[1] - e[0])
        return float(ec), float(ratio * ec)

    def unpack(self, p):
        plasma, d_squared, period, sweet, ng = p
        return np.array([plasma, float(self.ec(ng)), d_squared, period, sweet, ng])


def fit(model, constraint, x, y, starts, fixed=None):
    fixed = fixed or {}
    free = np.array([i for i in range(5) if i not in fixed])
    best = None
    for start in starts:
        template = np.asarray(start).copy()
        for index, value in fixed.items():
            template[index] = value

        def unpack(values):
            p = template.copy()
            p[free] = values
            return p

        result = least_squares(
            lambda values: (model(constraint.unpack(unpack(values)), x) - y) * 1000,
            template[free], bounds=(LOWER[free], UPPER[free]),
            loss="soft_l1", f_scale=3, max_nfev=2000,
        )
        if result.success and (best is None or result.cost < best[0]):
            best = (float(result.cost), unpack(result.x))
    if best is None:
        raise RuntimeError("Constrained fit failed")
    return best[1], best[0]


def function_panel(ax, functions, stitch, best, reference):
    ax.set_axis_off()
    local = stitch["registration"]["parameters"]
    maximum = local["maximum_frequency_ghz"]["value"]
    sweet = local["sweet_flux_corrected_v"]["value"]
    c2, c4 = [local[k]["value"] for k in ["quadratic_scaled_ghz", "quartic_scaled_ghz"]]
    shift = local["run_1615_flux_shift_v"]["value"]
    coefficients = functions["full_curve"]["coefficients_ghz"]
    knots = functions["full_curve"]["knots_with_multiplicity"]

    def line(y, text, size=10.5, bold=False):
        ax.text(0, y, text, transform=ax.transAxes, va="top", fontsize=size,
                fontweight="bold" if bold else "normal", color="#24384c")

    line(0.998, "Fitting functions and constraints", 14, True)
    line(0.958, "1. Resonance extraction at each flux", bold=True)
    line(0.931, r"$P(f)=b_0+b_1(f-f_s)+\frac{A}{1+[(f-f_r)/\gamma]^2}$", 12)
    line(0.901, r"$f_r(V)$ is the fitted line center; FWHM $=2\gamma$.", 9.5)

    line(0.864, "2. Flux-crosstalk registration", bold=True)
    line(0.837, rf"$V_c=V_{{raw}}{shift:+.8f}\,I_{{1615}}\quad[\mathrm{{V}}]$", 11)
    line(0.809, r"$\kappa_{q4}+\kappa_{q5}=\Delta V/0.30=-5.40\%$", 11)
    line(0.784, "q4 and q5 each receive +0.30 V; frequencies retained.", 9.3)

    line(0.749, "3. Local join: #1614 and #1615", bold=True)
    line(0.723, rf"$f_{{join}}/\mathrm{{GHz}}={maximum:.7f}{c2:+.7f}z^2{c4:+.7f}z^4$", 10.7)
    line(0.695, rf"$z=(V_c{-sweet:+.8f})/0.15$", 11)

    line(0.659, "4. Full empirical curve: cubic B-spline", bold=True)
    line(0.634, r"$f_{spline}(V_c)=\sum_j c_j B_{j,3}(V_c;t)$", 12)
    line(0.605, r"$t=[a,a,a,a,-0.1402,-0.096,-0.0322,b,b,b,b]$", 9.8)
    line(0.582, rf"$a={knots[0]:.8f},\ b={knots[-1]:.8f}$", 10)
    line(0.559, "c [GHz] = [" + ", ".join(f"{v:.6f}" for v in coefficients[:4]) + ",", 9.2)
    line(0.538, "                 " + ", ".join(f"{v:.6f}" for v in coefficients[4:]) + "]", 9.2)

    line(0.504, "5. Physical SQUID / transmon model", bold=True)
    line(0.478, r"$H/h=4e_C(\hat n-n_g)^2-e_{J,eff}(V_c)\cos\hat\varphi$", 11.2)
    line(0.451, r"$e_{J,eff}=e_{J\Sigma}\sqrt{\cos^2\theta+d^2\sin^2\theta}$", 11.2)
    line(0.424, r"$\theta=\pi(V_c-V_s)/V_{\Phi_0},\quad d=|E_{J1}-E_{J2}|/E_{J\Sigma}$", 10.4)
    line(0.398, r"$f_{01}=\lambda_1-\lambda_0,\quad f_{02}/2=(\lambda_2-\lambda_0)/2$", 10.7)
    line(0.372, r"$\lambda_i$: eigenvalues of $H/h$; $e_C=E_C/h$, $e_{J\Sigma}=E_{J\Sigma}/h$.", 9.1)
    line(0.340, "Large EJ/EC approximation (not the numerical fit):", 9.4)
    line(0.315, r"$f_{01}\simeq\sqrt{8e_Ce_{J\Sigma}}(\cos^2\theta+d^2\sin^2\theta)^{1/4}-e_C$", 10.5)

    line(0.277, "6. Provided two-transition constraint", bold=True)
    line(0.251, rf"$f_{{01}}={reference['f01_ghz']:.6f},\quad f_{{02}}/2={reference['half_f02_ghz']:.6f}\ \mathrm{{GHz}}$", 10.7)
    line(0.226, rf"$\alpha=f_{{02}}-2f_{{01}}={reference['alpha_mhz']:.0f}\ \mathrm{{MHz}}$", 10.7)
    line(0.199, rf"$e_C\approx-\alpha=0.176\ \mathrm{{GHz}}$ (leading approximation)", 9.8)
    line(0.172, rf"$e_C={best['ec_over_h_ghz']:.6f},\quad e_{{J\Sigma}}={best['ej_sum_over_h_ghz']:.4f}\ \mathrm{{GHz}}$", 11)
    line(0.145, rf"$V_{{\Phi_0}}={best['flux_period_v']:.6f}\ \mathrm{{V}},\quad V_s={best['sweet_flux_v']:.6f}\ \mathrm{{V}}$", 10.5)
    line(0.119, rf"$n_g={best['ng']:.3f},\quad d\to0$ (boundary; not resolved)", 10.3)
    line(0.091, rf"Full-spectrum RMSE = {best['rmse_mhz']:.2f} MHz; systematic residuals remain.", 9.5)
    line(0.065, "Energy values are conditional on the isolated-transmon model.", 9.2)
    line(0.040, "Numerical EC uses both supplied transitions, not EC = -alpha.", 9.1)
    line(0.015, "Displayed coefficients rounded; full precision saved in JSON.", 8.9)


def plot_all_functions(scans, rows, model, p, functions, stitch, best, reference, output):
    fig = plt.figure(figsize=(21, 13))
    ax = fig.add_axes([0.055, 0.155, 0.545, 0.755])
    color_ax = fig.add_axes([0.612, 0.155, 0.014, 0.755])
    panel = fig.add_axes([0.67, 0.04, 0.322, 0.89])
    norm = Normalize(*np.percentile(scans.state.values, [2, 98]))
    for run in scans.run.values:
        scan = scans.sel(run=run)
        xedges = cell_edges(scan.flux_corrected_v.values)
        yedges = cell_edges(scan.frequency_hz.values / 1e9)
        mesh = ax.pcolormesh(xedges, yedges, scan.state.values, shading="flat", cmap="viridis",
                             norm=norm, rasterized=True, antialiased=False)
        label_scan(ax, int(run), xedges, yedges)
    representation = functions["full_curve"]
    spline = BSpline(representation["knots_with_multiplicity"], representation["coefficients_ghz"],
                     representation["degree"], extrapolate=False)
    domain = stitch["continuous_curve"]["validated_flux_domain_v"]
    grid = np.linspace(*domain, 1800)
    ax.plot(grid, spline(grid), color="#ff7852", lw=1.7, ls="--", zorder=6,
            path_effects=[pe.withStroke(linewidth=3, foreground="#20394b")], label="Empirical spline (continuous stitch)")
    ax.plot(grid, model(p, grid), color="#193451", lw=1.1, ls=":", zorder=7,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")], label="Two-transition-constrained physical model")
    x = np.array([float(r["flux_corrected_v"]) for r in rows])
    y = np.array([float(r["frequency_ghz"]) for r in rows])
    ax.scatter(x, y, s=7, facecolors="none", edgecolors="white", linewidths=0.4, alpha=0.7)
    peak = stitch["local_run_1615_peak_after_axis_correction"]
    ax.scatter([peak["sweet_flux_v"]], [peak["maximum_frequency_ghz"]], marker="*", s=180,
               color="#ffbf4d", edgecolors="#20394b", zorder=8, label="Empirical sweet spot")
    ax.set(xlim=(float(scans.flux_corrected_v.min()) - 0.005, 0.076), ylim=(1.99, 4.15),
           xlabel="Corrected coupler-equivalent flux pulse amplitude (V)", ylabel="Drive frequency (GHz)")
    ax.set_title("coupler_q4_q5 | 2026-11-04 | #1608-#1618", fontsize=13, pad=14)
    ax.tick_params(labelsize=11)
    ax.legend(loc="lower left", bbox_to_anchor=(0.025, 0.10), fontsize=10, framealpha=0.95)
    fig.colorbar(mesh, cax=color_ax, extend="both").set_label("Mean measured state (raw)", fontsize=11)
    function_panel(panel, functions, stitch, best, reference)
    fig.suptitle("Coupler flux spectroscopy | all fitting functions and transition constraint", fontsize=20, y=0.975)
    font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    fig.text(0.055, 0.103, "註：#1615 因 q4、q5 各加 +0.3 V 的 flux crosstalk，Flux 軸左移 16.21 mV；合計等效係數約 -5.4%。", fontproperties=font, fontsize=11)
    fig.text(0.055, 0.070, "橘色虛線為經驗拼接曲線；深色點線為加入 f01、f02 約束的物理模型。原始頻率與訊號保留。", fontproperties=font, fontsize=10.5)
    fig.text(0.055, 0.040, "白色區域未量測，跨越接縫的曲線為插值；物理模型仍有系統性殘差，d 尚未可靠辨識。", fontproperties=font, fontsize=10.5)
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"coupler_flux_spectroscopy_all_fitting_functions.{extension}", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f01-ghz", type=float, default=2.760584)
    parser.add_argument("--half-f02-ghz", type=float, default=2.672584)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not 0 < args.f01_ghz / 2 < args.half_f02_ghz < args.f01_ghz:
        parser.error("Expected a negative-anharmonicity 0-2 transition: f01/2 < f02/2 < f01")
    source = BASE / "flux_crosstalk_corrected"
    output = args.output_dir or BASE / "transmon_model_anharmonicity_constrained"
    output.mkdir(parents=True, exist_ok=True)
    with (source / "registered_peak_fits.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["used_for_global_fit"] == "True"]
    x = np.array([float(r["flux_corrected_v"]) for r in rows])
    y = np.array([float(r["frequency_ghz"]) for r in rows])
    shifted = np.array([r["run"] == "1615" for r in rows], dtype=float)
    stitch = json.loads((source / "flux_stitch_results.json").read_text())
    functions = json.loads((BASE / "transmon_model/empirical_fitting_functions.json").read_text())
    shift = stitch["registration"]["parameters"]["run_1615_flux_shift_v"]["value"]
    model = ChargeBasisModel(BASE / "transmon_model/charge_basis_lookup.npz")
    constraint = ReferenceConstraint(args.f01_ghz, args.half_f02_ghz)
    starts = [[4.12, 0.005, 0.75, -0.253, ng] for ng in [0.02, 0.25, 0.48]]
    p5, cost = fit(model, constraint, x, y, starts)
    p = constraint.unpack(p5)
    best = summarize(model, p, x, y, cost)
    ec_reference, ej_reference = constraint.solve(p[5])
    levels = ec_reference * dimensionless_levels(ej_reference / ec_reference, ng=p[5], ncut=85)
    reference = dict(
        source="User-provided transitions at the same reference operating point; its flux voltage was not assumed",
        f01_ghz=args.f01_ghz, half_f02_ghz=args.half_f02_ghz,
        f02_ghz=2 * args.half_f02_ghz, f12_ghz=2 * args.half_f02_ghz - args.f01_ghz,
        alpha_mhz=(2 * args.half_f02_ghz - 2 * args.f01_ghz) * 1000,
        leading_ec_over_h_ghz=2 * (args.f01_ghz - args.half_f02_ghz),
        exact_ec_over_h_ghz=ec_reference, exact_ej_effective_over_h_ghz=ej_reference,
        predicted_f01_ghz=float(levels[1] - levels[0]),
        predicted_half_f02_ghz=float((levels[2] - levels[0]) / 2),
        ec_range_over_ng_0_to_half_ghz=[min(r[0] for r in constraint.solutions), max(r[0] for r in constraint.solutions)],
        implied_capacitance_ff=float(elementary_charge**2 / (2 * h * ec_reference * 1e9) * 1e15),
    )
    np.testing.assert_allclose([reference["predicted_f01_ghz"], reference["predicted_half_f02_ghz"]],
                               [args.f01_ghz, args.half_f02_ghz], rtol=0, atol=1e-8)
    ng_profiles, d_profiles, shift_profiles = [], [], []
    for ng in [0, 0.25, 0.30, 0.35, 0.40, 0.425, 0.45, 0.475, 0.50]:
        q5, qcost = fit(model, constraint, x, y, [p5], {4: ng})
        ng_profiles.append(summarize(model, constraint.unpack(q5), x, y, qcost))
    for d in [0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15]:
        q5, qcost = fit(model, constraint, x, y, [p5], {1: d*d})
        d_profiles.append(dict(fixed_d=d, **summarize(model, constraint.unpack(q5), x, y, qcost)))
    for trial_shift in sorted({r["shift_v"] for r in stitch["sensitivity_models"]}):
        xx = x + (trial_shift - shift) * shifted
        q5, qcost = fit(model, constraint, xx, y, [p5])
        shift_profiles.append(dict(shift_v=trial_shift, **summarize(model, constraint.unpack(q5), xx, y, qcost)))
    ratios = model.ratios(p, x)
    direct = []
    for ratio in ratios:
        e = p[1] * dimensionless_levels(ratio, ng=p[5], ncut=85)
        direct.append(e[1] - e[0])
    validation_hz = float(np.max(np.abs(model(p, x) - direct)) * 1e9)
    if validation_hz > 1000:
        raise ValueError("Numerical lookup differs from direct diagonalization by more than 1 kHz")
    result = dict(
        reference_transition_constraint=reference,
        physical_fit=best, numerical_parameters=p.tolist(),
        energy_units="EJ/h and EC/h in GHz; f01 and f02/2 are ordinary frequencies, not angular frequencies",
        model="H/h=4 ec (n-ng)^2 - ej_eff(V) cos(phi); f01=lambda1-lambda0; f02/2=(lambda2-lambda0)/2",
        flux_rule="#1615: V_corrected=V_raw-0.01621188060019709 V; other runs unchanged",
        interpretation="EC is constrained by the provided 0-1 and 0-2 transitions. EJ_sum and d are conditional on the isolated-transmon flux model; systematic ridge residuals remain and d reaches its lower bound.",
        frequency_correction_hz=0.0, source_peak_count=len(rows),
        numerical_validation_max_hz=validation_hz,
        ng_profiles=ng_profiles, d_profiles=d_profiles, flux_registration_sensitivity=shift_profiles,
        uncertainty="Profiles and registration sensitivity diagnose model dependence; no statistical confidence interval is claimed without transition uncertainties.",
        function_data=functions,
    )
    (output / "constrained_transmon_fit_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    save_csv(output / "reference_constraint_vs_ng.csv", [dict(ng=float(ng), ec_over_h_ghz=ec, ej_effective_over_h_ghz=ej)
                                                           for ng, (ec, ej) in zip(constraint.ng, constraint.solutions)])
    save_csv(output / "asymmetry_profile.csv", d_profiles)
    save_csv(output / "flux_registration_sensitivity.csv", shift_profiles)
    save_csv(output / "physical_model_predictions.csv", [dict(
        run=int(row["run"]), flux_corrected_v=float(v), measured_frequency_ghz=float(f),
        model_frequency_ghz=float(prediction), residual_mhz=float((f - prediction) * 1000),
    ) for row, v, f, prediction in zip(rows, x, y, model(p, x))])
    curve = np.genfromtxt(source / "continuous_curve.csv", delimiter=",", names=True)
    plot_analysis(model, p, x, y, curve, ng_profiles, d_profiles, output)
    with xr.open_dataset(source / "combined_flux_corrected.nc", engine="scipy") as scans:
        plot_all_functions(scans, rows, model, p, functions, stitch, best, reference, output)
    print(json.dumps(dict(reference=reference, physical_fit=best, numerical_validation_max_hz=validation_hz), indent=2))
    print(f"Outputs: {output.resolve()}")


if __name__ == "__main__":
    main()
