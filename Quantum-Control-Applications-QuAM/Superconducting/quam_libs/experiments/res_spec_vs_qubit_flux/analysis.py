import math

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from scipy.signal import detrend
from scipy.ndimage import uniform_filter1d
from sklearn.cluster import KMeans


def fit_resonator_spectroscopy_vs_flux(ds: xr.Dataset, smoothing_filter_size_in_mv: int) -> dict:
    """
    Analyze resonator phase response versus flux for each qubit in a dataset.

    This function:
    - Computes the phase of the complex signal I + iQ
    - Detrends and unwraps the phase
    - Smooths it using a uniform filter
    - Applies KMeans clustering to categorize regions (low, medium, high)
    - Identifies representative flux values for each region based on local derivative minima

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing 'I', 'Q', 'flux', and 'qubit' dimensions.
    smoothing_filter_size_in_mv : float
        Window size for smoothing the average phase signal in units of mV.

    Returns
    -------
    dict
        Dictionary mapping each qubit to a set of flux values:
        {
            "0": {"minimum": float, "crossing": float, "insensitive": float},
            ...
        }
    """

    S = ds.I + 1j * ds.Q  # Complex signal
    fit = {}

    smoothing_filter_size_in_points = math.ceil(ds.flux.diff('flux').median() * smoothing_filter_size_in_mv / 1e3)

    for qubit in ds.qubit:
        flux = ds.sel(qubit=qubit).flux.values
        phase = detrend(np.unwrap(np.angle(S.sel(qubit=qubit))))
        for i in range(len(phase)):
            phase[i] -= phase[i].mean()
        avg_phase = uniform_filter1d(np.mean(phase, axis=0), size=smoothing_filter_size_in_points)

        # Apply KMeans to segment phase response
        avg_reshaped = avg_phase.reshape(-1, 1)
        kmeans = KMeans(n_clusters=2)
        thresholds = np.sort(kmeans.fit(avg_reshaped).cluster_centers_.flatten())
        region_ids = np.digitize(avg_reshaped, thresholds, right=True)

        derivative = np.gradient(avg_phase)
        flux_by_region = {0: [], 1: [], 2: []}

        for region in np.unique(region_ids):
            indices = np.where(region_ids == region)[0]
            contiguous_groups = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)

            for group in contiguous_groups:
                if region == 0:
                    min_idx = np.argmin(avg_phase[group])
                    flux_by_region[region].append(flux[group[min_idx]])
                elif region == 1:
                    if len(group) == 0:
                        continue
                    min_idx = np.argmin(np.abs(derivative[group]))
                    flux_by_region[region].append(flux[group[min_idx]])
                elif region == 2:
                    max_idx = np.argmax(avg_phase[group])
                    flux_by_region[region].append(flux[group[max_idx]])


        fit[str(qubit.item())] = {
            "minimum": min(flux_by_region[0], key=abs),
            "crossing": min(flux_by_region[1], key=abs),
            "insensitive": min(flux_by_region[2], key=abs),
        }

    S -= S.mean(axis=2)
    S = abs(S)
    for qubit in ds.qubit:
        minimum_flux_point = fit[str(qubit.item())]["insensitive"]
        frequency_at_insensitive_flux_point = int(S.sel(flux=minimum_flux_point).sel(qubit=qubit).idxmax("freq"))
        fit[str(qubit.item())]["frequency_at_insensitive"] = frequency_at_insensitive_flux_point
        minimum_flux_point = fit[str(qubit.item())]["minimum"]
        frequency_at_minimum_flux_point = int(S.sel(flux=minimum_flux_point).sel(qubit=qubit).idxmax("freq"))
        fit[str(qubit.item())]["frequency_at_minimum"] = frequency_at_minimum_flux_point

    return fit


def find_symmetric_peak_center(x, y):
    """
    Find the center of a symmetric, flat-topped peak in a 1D signal using symmetry matching.

    Args:
        x (np.ndarray): 1D array of x values.
        y (np.ndarray): 1D array of y values (same length as x).
        window_size (int): Number of samples on each side to consider for symmetry.

    Returns:
        float: Estimated x-coordinate of the center of the symmetric peak.
    """
    assert len(x) == len(y), "x and y must be the same length"
    n = len(y)
    best_score = float('inf')
    best_index = None

    window_size = min(len(x) // 2 - 1, 10)
    for i in range(window_size, n - window_size):
        left = y[i - window_size:i]
        right = y[i + 1:i + window_size + 1][::-1]  # Mirror the right side
        score = np.sum((left - right) ** 2)

        if score < best_score:
            best_score = score
            best_index = i

    return x[best_index]
