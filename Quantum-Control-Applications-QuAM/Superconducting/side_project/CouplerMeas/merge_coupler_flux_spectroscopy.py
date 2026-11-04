"""Combine saved coupler flux spectroscopy scans on their physical coordinates.

Run with Python, NumPy, SciPy, xarray and Matplotlib; no QPU connection is used.
Defaults reproduce the 2026-11-04 scans #1608 through #1618.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
import numpy as np
import xarray as xr


DEFAULT_DATA_DIR = Path("/home/ratiswu/Qualibrate_data/as-qpu-10qV2/2026-11-04")
DEFAULT_RUN_IDS = tuple(range(1608, 1619))


def load_scans(data_dir, run_ids):
    scans, metadata = [], []
    for run_id in run_ids:
        matches = sorted(data_dir.glob(f"#{run_id}_*"))
        if len(matches) != 1:
            raise ValueError(f"Expected one directory for #{run_id}, found {len(matches)}")
        folder = matches[0]
        node = json.loads((folder / "node.json").read_text())
        params = node["data"]["parameters"]["model"]
        coupler, = params["coupler"]
        # These .h5 files are actually NetCDF3 files, readable by SciPy.
        with xr.open_dataset(folder / "ds.h5", engine="scipy") as ds:
            signal = ds.state.sel(qubit=coupler).transpose("freq", "flux").values.copy()
            frequency = ds.freq_full.sel(qubit=coupler).values.copy()
            detuning = ds.freq.values.copy()
            flux = ds.flux.values.copy()
        if not all(np.isfinite(a).all() for a in (signal, frequency, flux)):
            raise ValueError(f"Nonfinite data in #{run_id}")
        if not (np.all(np.diff(frequency) > 0) and np.all(np.diff(flux) > 0)):
            raise ValueError(f"Scan axes are not strictly increasing in #{run_id}")
        if signal.shape != (len(frequency), len(flux)):
            raise ValueError(f"Unexpected signal dimensions in #{run_id}")

        # Validate saved absolute frequencies against the saved LO and IF.
        machine = json.loads((folder / "quam_state" / "state.json").read_text())
        rd = machine["qubit_pairs"][coupler]["extras"]["RD"]
        lo = rd["LO"] if params["Driving_LO_GHz"] is None else params["Driving_LO_GHz"] * 1e9
        expected = detuning + rd["IF"] + lo + params["frequency_shift_in_mhz"] * 1e6
        np.testing.assert_allclose(frequency, expected, rtol=0, atol=1e-3)

        scan = xr.Dataset(
            {"state": (("frequency_point", "flux_point"), signal)},
            coords={
                # freq_full is numerically in Hz, despite its saved GHz attribute.
                "frequency_hz": ("frequency_point", frequency, {"units": "Hz"}),
                "detuning_hz": ("frequency_point", detuning, {"units": "Hz"}),
                "flux_v": ("flux_point", flux, {"units": "V"}),
            },
        ).expand_dims(run=[run_id])
        scans.append(scan)
        metadata.append({
            "run": run_id,
            "coupler": coupler,
            "source_directory": str(folder),
            "created_at": node["created_at"],
            "flux_min_v": float(flux.min()),
            "flux_max_v": float(flux.max()),
            "flux_step_v": float(np.median(np.diff(flux))),
            "frequency_min_ghz": float(frequency.min() / 1e9),
            "frequency_max_ghz": float(frequency.max() / 1e9),
            "frequency_step_mhz": float(np.median(np.diff(frequency)) / 1e6),
            "num_averages": params["num_averages"],
            "operation_amplitude_factor": params["operation_amplitude_factor"],
            "qubits_detune_flux_amp_v": params["qubits_detune_flux_amp"],
        })

    if len({m["coupler"] for m in metadata}) != 1:
        raise ValueError("All scans must measure the same coupler")
    if len({s.state.shape[1:] for s in scans}) != 1:
        raise ValueError("Native data export requires equal scan array dimensions")
    combined = xr.concat(scans, dim="run", coords="all", join="exact")
    for key in ("num_averages", "operation_amplitude_factor", "qubits_detune_flux_amp_v"):
        combined[key] = ("run", [m[key] for m in metadata])
    combined.state.attrs["long_name"] = "Mean measured readout state (raw)"
    combined.attrs.update(
        coupler=metadata[0]["coupler"],
        source_data_directory=str(data_dir),
        description="Native samples grouped by run; axes vary by run. No interpolation, normalization, or averaging across runs.",
    )
    return combined, metadata


def cell_edges(centers):
    midpoints = (centers[:-1] + centers[1:]) / 2
    return np.r_[2 * centers[0] - midpoints[0], midpoints, 2 * centers[-1] - midpoints[-1]]


def label_scan(ax, run_id, x_edges, y_edges):
    # #1616 overlaps #1608 in frequency; separate their source labels.
    offset = (8, 26) if run_id == 1616 else (0, 8)
    ax.annotate(
        f"#{run_id}", xy=((x_edges[0] + x_edges[-1]) / 2, y_edges[-1]),
        xytext=offset, textcoords="offset points", ha="center", va="bottom",
        fontsize=9, color="#263244",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.3, "alpha": 0.9},
        arrowprops={"arrowstyle": "-", "color": "#526071", "linewidth": 0.5} if run_id == 1616 else None,
    )


def draw_combined(scans, metadata, data_dir, output_dir):
    vmin, vmax = np.percentile(scans.state.values, [2, 98])
    norm = Normalize(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=(13, 7.8))
    fig.subplots_adjust(left=0.085, right=0.90, bottom=0.20, top=0.88)
    ax.set_axisbelow(True)
    ax.grid(color="#d6dce3", alpha=0.5, linewidth=0.5)

    for item in metadata:
        scan = scans.sel(run=item["run"])
        x = cell_edges(scan.flux_v.values)
        y = cell_edges(scan.frequency_hz.values / 1e9)
        # Each tile keeps its own sampling resolution and absolute coordinates.
        mesh = ax.pcolormesh(
            x, y, scan.state.values, cmap="viridis", norm=norm,
            shading="flat", rasterized=True, linewidth=0, antialiased=False,
        )
        ax.add_patch(Rectangle(
            (x[0], y[0]), x[-1] - x[0], y[-1] - y[0],
            fill=False, edgecolor="#485465", linewidth=0.65, alpha=0.65,
        ))
        label_scan(ax, item["run"], x, y)

    frequency_span_ghz = float(scans.frequency_hz.max() - scans.frequency_hz.min()) / 1e9
    ax.set_xlim(float(scans.flux_v.min()) - 0.005, float(scans.flux_v.max()) + 0.006)
    ax.set_ylim(float(scans.frequency_hz.min() / 1e9) - 0.02,
                float(scans.frequency_hz.max() / 1e9) + max(0.075, 0.07 * frequency_span_ghz))
    ax.set_xlabel("Coupler flux pulse amplitude (V)", fontsize=12)
    ax.set_ylabel("Drive frequency (GHz)", fontsize=12)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_locator(MultipleLocator(0.2 if frequency_span_ghz > 1.7 else 0.1))
    ax.tick_params(labelsize=10)
    coupler = metadata[0]["coupler"]
    run_text = ", ".join(f"#{m['run']}" for m in metadata)
    fig.suptitle(f"Coupler flux spectroscopy | {coupler}", fontsize=17, y=0.965)
    ax.set_title(f"{data_dir.name}  |  {run_text}", fontsize=10, color="#526071", pad=12)
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.025, fraction=0.035, extend="both")
    colorbar.set_label("Mean measured state (raw)", fontsize=11)
    fig.text(
        0.085, 0.115,
        "Native sampling; shared 2-98% color scale; white = unmeasured; later runs displayed on top in overlaps.",
        fontsize=10, color="#425166",
    )
    amplitude_groups = []
    for item in metadata:
        if (amplitude_groups and amplitude_groups[-1][-1]["run"] + 1 == item["run"]
                and amplitude_groups[-1][-1]["operation_amplitude_factor"] == item["operation_amplitude_factor"]):
            amplitude_groups[-1].append(item)
        else:
            amplitude_groups.append([item])
    settings = ";  ".join(
        (f"#{group[0]['run']}" if len(group) == 1 else f"#{group[0]['run']}-#{group[-1]['run']}")
        + f": {group[0]['operation_amplitude_factor']:g}"
        for group in amplitude_groups
    )
    fig.text(0.085, 0.078, f"Drive amplitude factors: {settings}", fontsize=8.5, color="#425166")
    detuned = [m for m in metadata if m["qubits_detune_flux_amp_v"] != 0]
    if detuned:
        text = ";  ".join(
            f"#{m['run']}: qubit detuning flux {m['qubits_detune_flux_amp_v']:g} V, {m['num_averages']} averages"
            for m in detuned
        )
        fig.text(0.085, 0.043, text, fontsize=9, color="#425166")
    run_ids = [m["run"] for m in metadata]
    stem = output_dir / f"{coupler}_flux_spectroscopy_{min(run_ids)}-{max(run_ids)}"
    fig.savefig(stem.with_suffix(".png"), dpi=240, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return stem.with_suffix(".png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--runs", nargs="+", type=int, default=list(DEFAULT_RUN_IDS))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if len(set(args.runs)) != len(args.runs):
        parser.error("Run IDs must be unique")
    output_dir = args.output_dir or Path(__file__).parent / "analysis" / f"{args.data_dir.name}_{min(args.runs)}-{max(args.runs)}"
    scans, metadata = load_scans(args.data_dir, args.runs)
    output_dir.mkdir(parents=True, exist_ok=True)
    scans.to_netcdf(output_dir / "combined_native_data.nc", engine="scipy")
    with (output_dir / "scan_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)
    image = draw_combined(scans, metadata, args.data_dir, output_dir)
    print(f"Combined {len(metadata)} scans, {scans.state.size:,} native samples.")
    print(f"Figure: {image.resolve()}")
    print(f"Data and metadata: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
