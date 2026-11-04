"""Explore EJ, EC and junction asymmetry from the flux-corrected spectrum.

The numerical fit uses the charge-basis Hamiltonian H/h = 4 ec (n-ng)^2
- ej_eff cos(phi), with ec=EC/h and ej=EJ/h expressed in GHz. A lookup
of dimensionless eigenvalue gaps accelerates fitting and is checked against
direct diagonalization. Parameter profiles expose d/ng and flux-shift
dependence; these fits are conditional isolated-circuit model estimates.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline, RectBivariateSpline
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import least_squares

from fit_coupler_flux_spectroscopy import save_csv
from stitch_coupler_flux_crosstalk import fit_continuous_curve


DEFAULT_SOURCE = Path(__file__).parent / "analysis/2026-11-04_1608-1618/flux_crosstalk_corrected"
SOURCES = ["https://scqubits.readthedocs.io/en/v4.1/guide/qubits/tunable_transmon.html",
           "https://arxiv.org/html/cond-mat/0703002v2"]
LOWER = np.array([3.5, 0.005, 0.0, 0.62, -0.30, 0.0])
UPPER = np.array([6.0, 0.8, 0.9025, 1.5, -0.20, 0.5])


def f01_approx(flux_v, ej_max_ghz, ec_ghz, d, period_v, sweet_v):
    """Leading transmon approximation; EJmax/h and EC/h are in GHz."""
    theta = np.pi * (np.asarray(flux_v) - sweet_v) / period_v
    shape = (np.cos(theta)**2 + d**2 * np.sin(theta)**2)**0.25
    return np.sqrt(8 * ec_ghz * ej_max_ghz) * shape - ec_ghz


def dimensionless_levels(ratio, ng=0.0, ncut=55):
    """Lowest three energies in units of EC, for ratio=EJ_eff/EC."""
    charges = np.arange(-ncut, ncut + 1)
    return eigh_tridiagonal(
        4 * (charges - ng)**2, -ratio / 2 * np.ones(2 * ncut),
        select="i", select_range=(0, 2), eigvals_only=True, check_finite=False,
    )


class ChargeBasisModel:
    def __init__(self, cache_path):
        if cache_path.exists():
            with np.load(cache_path) as saved:
                log_ratio, ng, gap = saved["log_ratio"], saved["ng"], saved["gap"]
        else:
            log_ratio = np.linspace(np.log(0.1), np.log(2e5), 2001)
            ng = np.linspace(0, 0.5, 41)
            gap = np.empty((len(log_ratio), len(ng)))
            for column, charge in enumerate(ng):
                for row, ratio in enumerate(np.exp(log_ratio)):
                    levels = dimensionless_levels(ratio, ng=charge)
                    gap[row, column] = levels[1] - levels[0]
            np.savez_compressed(cache_path, log_ratio=log_ratio, ng=ng, gap=gap)
        self.interpolator = RectBivariateSpline(log_ratio, ng, gap, s=0)
        self.ratio_limits = (float(np.exp(log_ratio[0])), float(np.exp(log_ratio[-1])))

    @staticmethod
    def ratios(parameters, voltage):
        plasma, ec, d_squared, period, sweet, _ = parameters
        theta = np.pi * (np.asarray(voltage) - sweet) / period
        return plasma**2 / (8 * ec**2) * np.sqrt(
            np.cos(theta)**2 + d_squared * np.sin(theta)**2
        )

    def __call__(self, parameters, voltage):
        ratio = self.ratios(parameters, voltage)
        log_ratio = np.log(np.clip(ratio, *self.ratio_limits))
        ng = np.full_like(log_ratio, parameters[5])
        return parameters[1] * self.interpolator.ev(log_ratio, ng)


def fit_model(model, voltage, frequency, starts, fixed=None):
    fixed = fixed or {}
    free = np.array([i for i in range(6) if i not in fixed])
    best = None
    for start in starts:
        template = np.asarray(start, dtype=float).copy()
        for index, value in fixed.items():
            template[index] = value

        def unpack(values):
            p = template.copy()
            p[free] = values
            return p

        fit = least_squares(
            lambda values: (model(unpack(values), voltage) - frequency) * 1000,
            template[free], bounds=(LOWER[free], UPPER[free]),
            loss="soft_l1", f_scale=3, max_nfev=2000,
        )
        if fit.success and (best is None or fit.cost < best[0]):
            best = (float(fit.cost), unpack(fit.x))
    if best is None:
        raise RuntimeError("Physical-model optimization did not converge")
    return best[1], best[0]


def summarize(model, parameters, x, y, cost):
    plasma, ec, d_squared, period, sweet, ng = parameters
    ej = plasma**2 / (8 * ec)
    d = np.sqrt(d_squared)
    errors = (y - model(parameters, x)) * 1000
    levels = ec * dimensionless_levels(ej / ec, ng=ng, ncut=85)
    return dict(
        ej_sum_over_h_ghz=float(ej), ec_over_h_ghz=float(ec), d=float(d),
        ej1_over_h_ghz=float(ej * (1 + d) / 2), ej2_over_h_ghz=float(ej * (1 - d) / 2),
        flux_period_v=float(period), sweet_flux_v=float(sweet), ng=float(ng),
        maximum_f01_ghz=float(levels[1] - levels[0]),
        predicted_f12_at_sweet_ghz=float(levels[2] - levels[1]),
        predicted_anharmonicity_at_sweet_mhz=float((levels[2] - 2 * levels[1] + levels[0]) * 1000),
        rmse_mhz=float(np.sqrt(np.mean(errors**2))), robust_cost=cost,
    )


def export_empirical_functions(source, stitch, output):
    with (source / "registered_peak_fits.csv").open() as handle:
        records = list(csv.DictReader(handle))
    for row in records:
        row["run"] = int(row["run"])
        row["flux_v"] = float(row["flux_v"])
        row["frequency_ghz"] = float(row["frequency_ghz"])
        if row["status"] in ["accepted", "off_main_ridge"]:
            row["status"] = "candidate"
    shift = stitch["registration"]["parameters"]["run_1615_flux_shift_v"]["value"]
    spline, _ = fit_continuous_curve(records, shift)
    degree = 3
    knots = spline.get_knots()
    t = np.r_[np.repeat(knots[0], degree), knots, np.repeat(knots[-1], degree)]
    c = spline.get_coeffs()
    reconstructed = BSpline(t, c, degree, extrapolate=False)
    grid = np.linspace(knots[0], knots[-1], 2000)
    np.testing.assert_allclose(reconstructed(grid), spline(grid), rtol=0, atol=1e-12)
    representation = dict(
        full_curve=dict(expression="f(V_corrected) = sum_j c_j B_{j,3}(V_corrected; t)",
                        degree=degree, knots_with_multiplicity=t.tolist(), coefficients_ghz=c.tolist(),
                        python="scipy.interpolate.BSpline(knots_with_multiplicity, coefficients_ghz, degree, extrapolate=False)"),
        local_join=dict(expression="f/GHz=f_max+c2*z^2+c4*z^4; z=(V_corrected-V_sweet)/0.15 V",
                        fitted_parameters=stitch["registration"]["parameters"]),
        coordinate_rule=stitch["coordinate_rule"],
        energy_parameter_interpretation="These empirical functions describe the data and do not themselves determine EJ, EC, or d.",
    )
    (output / "empirical_fitting_functions.json").write_text(json.dumps(representation, indent=2) + "\n")


def plot_analysis(model, p, x, y, spline_curve, ng_profiles, d_profiles, output):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
    grid = np.linspace(x.min(), x.max(), 1200)
    axes[0, 0].scatter(x, y, s=8, color="#437ba0", label="Extracted main ridge")
    axes[0, 0].plot(grid, model(p, grid), color="#d46b39", lw=1.3, label="Charge-basis model (conditional fit)")
    axes[0, 0].plot(spline_curve["flux_corrected_v"], spline_curve["frequency_ghz"],
                     color="#465160", lw=1, ls="--", label="Previous empirical spline")
    axes[0, 0].set(xlabel="Corrected flux amplitude (V)", ylabel="Frequency (GHz)")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    axes[0, 1].scatter(x, (y - model(p, x)) * 1000, s=11, color="#437ba0")
    axes[0, 1].axhline(0, color="#283e50", lw=0.7)
    axes[0, 1].set(xlabel="Corrected flux amplitude (V)", ylabel="Data - physical model (MHz)")
    axes[1, 0].plot([r["ng"] for r in ng_profiles], [r["rmse_mhz"] for r in ng_profiles], "o-", color="#437ba0")
    axes[1, 0].set(xlabel="Assumed offset charge ng", ylabel="Refitted RMSE (MHz)", title="Model depends on the offset-charge assumption")
    axes[1, 1].plot([r["fixed_d"] for r in d_profiles], [r["rmse_mhz"] for r in d_profiles], "o-", color="#d46b39")
    axes[1, 1].set(xlabel="Fixed junction asymmetry d", ylabel="Refitted RMSE (MHz)", title="Other parameters refitted at every d")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle("Tunable-transmon model exploration | #1608-#1618 | flux correction fixed", fontsize=13, y=0.985)
    fig.text(0.05, 0.94,
             r"$H/h=4e_C(\hat n-n_g)^2-e_{J,\mathrm{eff}}(V)\cos\hat\varphi,\quad f_{01}=\lambda_1-\lambda_0$",
             fontsize=12)
    fig.text(0.05, 0.90,
             r"$e_{J,\mathrm{eff}}=e_{J\Sigma}\sqrt{\cos^2\theta+d^2\sin^2\theta},\quad \theta=\pi(V_{\mathrm{corr}}-V_s)/V_{\Phi_0}$",
             fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.87))
    fig.text(0.06, 0.025, "Conditional isolated-transmon estimates. The weak d/ng distinction and flux-registration uncertainty limit parameter interpretation.", fontsize=9)
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"transmon_model_diagnostics.{extension}", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ec-ghz", type=float, help="Independently known EC/h in GHz")
    args = parser.parse_args()
    output = args.output_dir or args.source_dir.parent / "transmon_model"
    output.mkdir(parents=True, exist_ok=True)
    with (args.source_dir / "registered_peak_fits.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["used_for_global_fit"] == "True"]
    x = np.array([float(r["flux_corrected_v"]) for r in rows])
    y = np.array([float(r["frequency_ghz"]) for r in rows])
    shifted = np.array([r["run"] == "1615" for r in rows], dtype=float)
    stitch = json.loads((args.source_dir / "flux_stitch_results.json").read_text())
    shift = stitch["registration"]["parameters"]["run_1615_flux_shift_v"]["value"]
    model = ChargeBasisModel(output / "charge_basis_lookup.npz")
    starts = [[3.96 + ec, ec, 0.005, 0.80, -0.253, ng]
              for ec, ng in [(0.05, 0.1), (0.15, 0.25), (0.30, 0.40), (0.40, 0.49)]]
    fixed = {} if args.ec_ghz is None else {1: args.ec_ghz}
    if args.ec_ghz is not None and not LOWER[1] <= args.ec_ghz <= UPPER[1]:
        parser.error(f"EC/h must lie between {LOWER[1]} and {UPPER[1]} GHz")
    p, cost = fit_model(model, x, y, starts, fixed)
    best = summarize(model, p, x, y, cost)

    ng_profiles = []
    for ng in [0, 0.25, 0.30, 0.35, 0.40, 0.425, 0.45, 0.475, 0.50]:
        q, qcost = fit_model(model, x, y, [p, starts[0]], {**fixed, 5: ng})
        ng_profiles.append(summarize(model, q, x, y, qcost))
    d_profiles = []
    for d in [0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15]:
        start = p.copy()
        start[5] = 0.45
        q, qcost = fit_model(model, x, y, [start, starts[0]], {**fixed, 2: d*d})
        d_profiles.append(dict(fixed_d=d, **summarize(model, q, x, y, qcost)))
    shift_profiles = []
    shifts = sorted({r["shift_v"] for r in stitch["sensitivity_models"]})
    for trial_shift in shifts:
        xx = x + (trial_shift - shift) * shifted
        q, qcost = fit_model(model, xx, y, [p], fixed)
        shift_profiles.append(dict(shift_v=trial_shift, **summarize(model, q, xx, y, qcost)))

    ratio = model.ratios(p, x)
    if ratio.min() < model.ratio_limits[0] or ratio.max() > model.ratio_limits[1]:
        raise ValueError("Best fit falls outside the numerical lookup domain")
    direct = []
    for r in ratio:
        levels = dimensionless_levels(r, ng=p[5], ncut=85)
        direct.append(p[1] * (levels[1] - levels[0]))
    validation_hz = float(np.max(np.abs(model(p, x) - direct)) * 1e9)
    if validation_hz > 1000:
        raise ValueError(f"Numerical-model check exceeded 1 kHz: {validation_hz} Hz")

    result = dict(
        status="Conditional physical-model exploration; d is weakly identified and model-systematic uncertainty is unresolved",
        sources=SOURCES,
        energy_units="EJ/h and EC/h in GHz; not angular-frequency units",
        model="H(V)/h = 4 ec (n-ng)^2 - ej_eff(V) cos(phi); f01=lambda1-lambda0",
        josephson_model="ej_eff=ej_sum*sqrt(cos(theta)^2+d^2*sin(theta)^2); theta=pi*(V_corrected-V_sweet)/V_period",
        asymmetry_definition="d=abs(EJ1-EJ2)/(EJ1+EJ2), 0<=d<1; the junction labels and asymmetry sign are unidentifiable from this spectrum",
        previous_plot_function="Full curve: cubic smoothing spline. Local join: f/GHz=3.961871679826394-0.419400078801085*z^2+0.021771851859479833*z^4, z=(V_corrected+0.253018787249107)/0.15 V",
        source_peak_count=len(rows), fixed_flux_shift_v=shift, frequency_correction_hz=0.0,
        independent_ec_constraint_ghz=args.ec_ghz,
        conditional_best_fit=best,
        ng_profiles=ng_profiles, d_profiles=d_profiles, flux_registration_sensitivity=shift_profiles,
        parameter_interpretation="Only f01 was used. Alternative d and ng combinations fit nearly equally well; the lower-bound EC solution at ng=0 and the finite-EC charge-sensitive branch are distinct possibilities under different assumptions. Conditional fits do not establish bare circuit constants or transition assignment.",
        useful_followup="Independently identify f12/anharmonicity at a stated flux and extend coverage toward a flux minimum/full period; compare against coupling-induced spectral shifts if present.",
        numerical_validation_max_hz=validation_hz,
        uncertainty="Profile and registration-sensitivity comparisons are model diagnostics, not statistical confidence intervals.",
    )
    (output / "transmon_fit_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    save_csv(output / "offset_charge_profile.csv", ng_profiles)
    save_csv(output / "asymmetry_profile.csv", d_profiles)
    save_csv(output / "flux_registration_sensitivity.csv", shift_profiles)
    save_csv(output / "physical_model_predictions.csv", [
        dict(run=int(r["run"]), flux_corrected_v=float(v), measured_frequency_ghz=float(f),
             model_frequency_ghz=float(prediction), residual_mhz=float((f-prediction)*1000))
        for r, v, f, prediction in zip(rows, x, y, model(p, x))
    ])
    curve = np.genfromtxt(args.source_dir / "continuous_curve.csv", delimiter=",", names=True)
    export_empirical_functions(args.source_dir, stitch, output)
    plot_analysis(model, p, x, y, curve, ng_profiles, d_profiles, output)
    print(json.dumps(dict(conditional_best_fit=best, numerical_validation_max_hz=validation_hz), indent=2))
    print(f"Outputs: {output.resolve()}")


if __name__ == "__main__":
    main()
