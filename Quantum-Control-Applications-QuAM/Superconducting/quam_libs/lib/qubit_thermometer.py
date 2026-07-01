""" Copied from repo LCH-QCAT """
import numpy as np
import xarray as xr
import os
from scipy import special
from scipy.integrate import quad
from lmfit import Model,Parameter
import matplotlib.pyplot as plt
from lmfit import Model, Parameters
from sklearn.mixture import GaussianMixture
from abc import ABC, abstractmethod
from matplotlib.axes import Axes
class FunctionFitting(ABC):
    """
    Abstract base class for all function fitting routines.
    Child classes must implement model_function, guess, and fit methods.
    """
    def __init__(self):
        pass

    @abstractmethod
    def model_function(self, *args, **kwargs):
        pass

    @abstractmethod
    def guess(self):
        pass

    @abstractmethod
    def fit(self, data=None):
        pass

    def fitting_curve(self, x):
        """Return the model evaluated at x using current parameters."""
        return self.model(x)


def prepare_dataset_for_qcat(ds):
    """
    將原始 ds 轉換為符合 StateDiscrimination 要求的格式：
    1. 合併 I_g/I_e 與 Q_g/Q_e 為單一 DataArray
    2. 將 N 更名為 shot_idx
    3. 建立 prepared_state 座標 (0 為 ground, 1 為 excited)
    """
    # 合併 I 分量
    # concat 會沿著新維度 'prepared_state' 堆疊資料
    I = xr.concat([ds.I_g, ds.I_e], dim="prepared_state").assign_coords(prepared_state=[0, 1])
    # 合併 Q 分量
    Q = xr.concat([ds.Q_g, ds.Q_e], dim="prepared_state").assign_coords(prepared_state=[0, 1])
    
    # 建立新的 Dataset 並重命名維度 N 為 shot_idx
    new_ds = xr.Dataset({"I": I, "Q": Q})
    new_ds = new_ds.rename({"N": "shot_idx"})
    
    return new_ds

H_BAR = 1.0545718e-34
K_B = 1.38e-23 

def Relax_cal(inte_i,tf,T1):
    tau= tf - inte_i
    Relax= 1+(T1/tau)*(np.exp(-(inte_i+tau)/T1)-np.exp(-inte_i/T1))
    return Relax

def PetoT(Pe, Wa):
    Pe = np.clip(Pe,1e-10, 1-1e-10)  
    Pg = 1 - Pe
    try:
        T = (-H_BAR*Wa)/(K_B*np.log(Pe/Pg)) * 1000  
    except Exception as e:
        T = np.nan
    return T

def gaussian2D_function( x, y, amp, x0, y0, sigma_x, sigma_y):
    return amp * np.exp(
        -( ((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2) )
    )


def plot_2d_fit_residue(fit_residues, norm_res):
    """
    Plot 2D fit residue (difference between density and best fit) for each prepared_state.
    Args:
        fit_residues (xr.DataArray): Residue arrays with dims (prepared_state, y, x).
        norm_res (list or array): Normalized residue values for each prepared_state.
    Returns:
        fig: matplotlib Figure
        axes: matplotlib Axes
    """

    n_states = fit_residues.sizes['prepared_state']
    fig, axes = plt.subplots(1, n_states, figsize=(6 * n_states, 5), dpi=150)
    if n_states == 1:
        axes = [axes]
    x = fit_residues['x'].values
    y = fit_residues['y'].values
    xedges = np.concatenate([x - (x[1] - x[0])/2, [x[-1] + (x[1] - x[0])/2]]) if len(x) > 1 else np.array([x[0]-0.5, x[0]+0.5])
    yedges = np.concatenate([y - (y[1] - y[0])/2, [y[-1] + (y[1] - y[0])/2]]) if len(y) > 1 else np.array([y[0]-0.5, y[0]+0.5])
    vmin = float(np.nanmin(fit_residues.values))
    vmax = float(np.nanmax(fit_residues.values))
    absmax = max(abs(vmin), abs(vmax))
    for i, state in enumerate(fit_residues['prepared_state'].values):
        residue = fit_residues.sel(prepared_state=state).values
        pcm = axes[i].pcolormesh(xedges, yedges, residue, shading='auto', cmap='bwr', vmin=-absmax, vmax=absmax)
        axes[i].set_title(f"Fit Residue prepared_state={state}")
        axes[i].set_xlabel('I')
        axes[i].set_ylabel('Q')
        fig.colorbar(pcm, ax=axes[i], label='Residue (density - fit)')
        # Show normalized sum of absolute residues over density as a text box
        axes[i].text(
            0.02, 0.98,
            f"res/density: {norm_res[i]:.3e}",
            transform=axes[i].transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
    plt.tight_layout()
    plt.close(fig)
    return fig, axes

def plot_gmm_mean_on_axes(axes, mean):
    """
    Plot GMM mean as black dots on each axis in axes.
    Args:
        axes: list of matplotlib Axes
        mean: array-like, shape (2, 2), GMM mean for two components
    """
    axes.scatter(mean[0][0], mean[0][1], c='k', s=40, marker='o')
    axes.scatter(mean[1][0], mean[1][1], c='k', s=40, marker='o')

def plot_gmm_circles_on_axis(axes, analysis_result, n_std=[1,2,3], **circle_kwargs):
    """
    Plot GMM mean as centers and covariance as radii (dashed circles) on a given axis.
    Args:
        axes: matplotlib Axes
        analysis_result: dict, expects 'mean' (N,2) and 'covariance' (N,) from GMM
        n_std: list or float, number(s) of standard deviations for the radius
        circle_kwargs: additional kwargs for Ellipse
    """
    from matplotlib.patches import Ellipse
    mean = analysis_result['mean']
    std = analysis_result['std']
    # Accept n_std as a list or a single float
    if isinstance(n_std, (int, float)):
        n_std_list = [n_std]
    else:
        n_std_list = list(n_std)
    for i in range(mean.shape[0]):
        center = mean[i]
        for n in n_std_list:
            radius = std * n
            circle = Ellipse(xy=center, width=2*radius, height=2*radius, angle=0,
                             edgecolor='k', facecolor='none', linestyle='--', linewidth=1.5, **circle_kwargs)
            axes.add_patch(circle)

def plot_outliers(data, outlier_mask, analysis_result=None):
    """
    Plot scatter plots of I vs Q for each prepared_state, showing only the outlier points as defined by outlier_mask.
    Args:
        data (xr.Dataset): Dataset with variables 'I', 'Q', coords 'shot_idx', 'prepared_state'.
        outlier_mask (dict): Dictionary mapping prepared_state index to boolean mask array (same length as shot_idx for that state).
    Returns:
        fig: matplotlib Figure
    """

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=150)

    for i in range(2):
        mask = outlier_mask[i]

        # Extract I and Q for both prepared states
        I_vals = data['I'].sel(prepared_state=i).values[mask]
        Q_vals = data['Q'].sel(prepared_state=i).values[mask]
        axes[i].scatter(I_vals, Q_vals, s=10, alpha=0.8, color='orange', marker='o', edgecolor='none', label='Outlier')

        # Optionally plot mean as black dots
        if analysis_result is not None:

            trained_paras = analysis_result.get('trained_paras', None)
            if trained_paras is not None and 'mean' in trained_paras:
                mean = trained_paras['mean']
                plot_gmm_mean_on_axes(axes[i], mean)
            if trained_paras is not None and 'std' in trained_paras:
                plot_gmm_circles_on_axis(axes[i], trained_paras)

            y_offset = 0.98
            if 'outlier_probability' in analysis_result:
                outlier_prob = analysis_result['outlier_probability']
                text_msg = f"Outlier prob.: {outlier_prob[i]:.3e}"
                axes[i].text(
                    0.02, y_offset, text_msg,
                    transform=axes[i].transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )
                        
    fig.tight_layout()
    plt.close(fig)
    return fig, axes

def plot_prepared_state_scatter(data, analysis_result=None):
    """
    Plot two scatter plots of I vs Q for prepared_state=0 and prepared_state=1, sharing the same axis limits.
    Optionally plot GMM mean as black dots if analysis_result is provided.
    Args:
        data (xr.Dataset): Dataset with variables 'I', 'Q', coords 'shot_idx', 'prepared_state'.
        analysis_result (dict, optional): Dictionary with GMM parameters (expects 'mean').
    Returns:
        fig: matplotlib Figure
    """
    import numpy as np
    import matplotlib.pyplot as plt
    I_list = []
    Q_list = []
    for i in range(2):
        # Extract I and Q for both prepared states
        I_list.append(data['I'].sel(prepared_state=i).values)
        Q_list.append(data['Q'].sel(prepared_state=i).values)



    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
    for i in range(2):
        # Default color is blue, but if state_label is provided, use it for coloring
        if analysis_result is not None and 'state_label' in analysis_result:
            labels = np.array(analysis_result['state_label'][i])
            # Use a colormap for 2 classes
            cmap = plt.get_cmap('coolwarm')
            colors = cmap(labels / (labels.max() if labels.max() > 0 else 1))
            axes[i].scatter(I_list[i], Q_list[i], s=6, alpha=0.7, c=colors, marker='o', edgecolor='none')
        else:
            axes[i].scatter(I_list[i], Q_list[i], s=1, alpha=0.5, color='blue')


        # Optionally plot GMM mean as black dots
        if analysis_result is not None:

            trained_paras = analysis_result.get('trained_paras', None)
            if trained_paras is not None and 'mean' in trained_paras:
                mean = trained_paras['mean']
                plot_gmm_mean_on_axes(axes[i], mean)
            if trained_paras is not None and 'covariance' in trained_paras:
                plot_gmm_circles_on_axis(axes[i], trained_paras)

            y_offset = 0.98
            if 'direct_counts' in analysis_result:
                direct_counts = analysis_result['direct_counts']
                text_msg = f"Direct counts:\n{direct_counts[i]}"
                axes[i].text(
                    0.02, y_offset, text_msg,
                    transform=axes[i].transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )
                y_offset -= 0.18  # Move next box down
            if 'gaussian_norms' in analysis_result:
                gaussian_norms = analysis_result['gaussian_norms']
                text_msg = f"Gaussian norms:\n{gaussian_norms[i]}"
                axes[i].text(
                    0.02, y_offset, text_msg,
                    transform=axes[i].transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )

    fig.tight_layout()
    plt.close(fig)
    return fig, axes

def compute_shared_axis_limits(data, n_std=5):
    """
    Compute shared axis limits for I and Q from a dataset with 'prepared_state' axis.
    Args:
        data: xarray Dataset with variables 'I', 'Q', and 'prepared_state' coordinate
        n_std: number of standard deviations for the axis limits (default 5)
    Returns:
        lim_I: tuple (min, max) for I axis
        lim_Q: tuple (min, max) for Q axis
    """
    I_list = []
    Q_list = []
    for i in range(2):
        I_list.append(data['I'].sel(prepared_state=i).values)
        Q_list.append(data['Q'].sel(prepared_state=i).values)
    all_I = np.concatenate(I_list)
    all_Q = np.concatenate(Q_list)
    I_mean, Q_mean = np.mean(all_I), np.mean(all_Q)
    I_std, Q_std = np.std(all_I), np.std(all_Q)
    lim_I = (I_mean - n_std*I_std, I_mean + n_std*I_std)
    lim_Q = (Q_mean - n_std*Q_std, Q_mean + n_std*Q_std)
    return lim_I, lim_Q

def axis_formatter(axes, lim_I, lim_Q, i):

    from matplotlib.ticker import ScalarFormatter
    axes.set_xlim(lim_I)
    axes.set_ylim(lim_Q)
    axes.set_aspect('equal')
    # Use ScalarFormatter for scientific notation
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    axes.xaxis.set_major_formatter(formatter)
    axes.yaxis.set_major_formatter(formatter)
    # Force scientific notation if needed
    axes.ticklabel_format(style='sci', axis='both', scilimits=(-2,2))
    # Get offset text (exponential part)
    axes.xaxis.offsetText.set_visible(True)
    axes.yaxis.offsetText.set_visible(True)
    # Set axis labels with exponent if present
    xlabel = r"$I$"
    ylabel = r"$Q$"
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(f'prepared_state={i}')
    

def load_xarray_h5(file_path: str, engine_order: list[str] | None = None, load_into_memory: bool = True) -> "xr.Dataset":
    """
    Load an xarray.Dataset stored in an HDF5 (.h5) file.

    Parameters
    ----------
    file_path : str
        Path to the .h5 file.
    group : str | None
        HDF5 group name where the dataset is stored (if any).
    engine_order : list[str] | None
        List of xarray engines to try (default: ["h5netcdf", "netcdf4"]).
    load_into_memory : bool
        If True, call .load() on the returned dataset to read it into memory.

    Returns
    -------
    xr.Dataset
        The loaded xarray Dataset.

    Raises
    ------
    FileNotFoundError
        If file_path does not exist.
    RuntimeError
        If the file cannot be opened as an xarray Dataset with the tried engines.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: {file_path}")

    engines = engine_order or ["h5netcdf", "netcdf4"]
    last_exc = None
    for eng in engines:
        try:
            ds = xr.open_dataset(file_path, engine=eng)
            if load_into_memory:
                ds = ds.load()
            return ds
        except Exception as exc:
            last_exc = exc

    # Final fallback: let xarray choose the engine
    try:
        ds = xr.open_dataset(file_path)
        if load_into_memory:
            ds = ds.load()
        return ds
    except Exception as exc:
        raise RuntimeError(f"Failed to open '{file_path}' as an xarray Dataset. Tried engines {engines}. Last error: {last_exc}") from exc

def repetition_data( ds: xr.Dataset, repetition_dim: str = "qubit"):
    n_qubits = ds.sizes[repetition_dim]
    output_data = []
    for qubit_idx in range(n_qubits):
        data = ds.isel(**{repetition_dim: qubit_idx})
        output_data.append(data)
    return output_data

def plot_2d_histogram(hist_dataset, analysis_result=None):
    """
    Plot 2D histogram (density) for each prepared_state from hist_dataset.
    If analysis_result is provided, overlay GMM mean and covariance.
    Args:
        hist_dataset (xr.Dataset): Dataset with variable 'density' and coords 'prepared_state', 'x', 'y'.
        analysis_result (dict, optional): GMM fit results to overlay.
        axes: matplotlib Axes or None.
        cmap (str): Colormap for the plot (default 'viridis').
    Returns:
        fig: matplotlib Figure
    """
    
    from matplotlib.colors import LogNorm
    n_states = hist_dataset.sizes['prepared_state']
    fig, axes = plt.subplots(1, n_states, figsize=(6 * n_states, 5), dpi=150)
    if n_states == 1:
        axes = [axes]
    
    x = hist_dataset['x'].values
    y = hist_dataset['y'].values
    xedges = np.concatenate([x - (x[1] - x[0])/2, [x[-1] + (x[1] - x[0])/2]]) if len(x) > 1 else np.array([x[0]-0.5, x[0]+0.5])
    yedges = np.concatenate([y - (y[1] - y[0])/2, [y[-1] + (y[1] - y[0])/2]]) if len(y) > 1 else np.array([y[0]-0.5, y[0]+0.5])
    for i, state in enumerate(hist_dataset.coords['prepared_state'].values):
        density = hist_dataset['density'].sel(prepared_state=state).values  # shape (len(y), len(x))
        # Mask zero values so they are not shown in the log plot
        density_masked = np.ma.masked_where(density <= 0, density)
        pcm = axes[i].pcolormesh(xedges, yedges, density_masked, shading='auto', cmap='viridis', norm=LogNorm())
        axes[i].set_title(f"prepared_state={state}")
        axes[i].set_xlabel('I')
        axes[i].set_ylabel('Q')

        # Optionally plot GMM mean as black dots
        if analysis_result is not None:
            trained_paras = analysis_result.get('trained_paras', None)
            if trained_paras is not None and 'mean' in trained_paras:
                mean = trained_paras['mean']
                plot_gmm_mean_on_axes(axes[i], mean)
            if trained_paras is not None and 'covariance' in trained_paras:
                plot_gmm_circles_on_axis(axes[i], trained_paras)

            y_offset = 0.98
            if 'direct_counts' in analysis_result:
                direct_counts = analysis_result['direct_counts']
                text_msg = f"Direct counts:\n{direct_counts[i]}"
                axes[i].text(
                    0.02, y_offset, text_msg,
                    transform=axes[i].transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )
                y_offset -= 0.18  # Move next box down
            if 'gaussian_norms' in analysis_result:
                gaussian_norms = analysis_result['gaussian_norms']
                text_msg = f"Gaussian norms:\n{gaussian_norms[i]}"
                axes[i].text(
                    0.02, y_offset, text_msg,
                    transform=axes[i].transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )

    plt.tight_layout()
    plt.close(fig)
    return fig, axes



class FitMultiGaussian2D(FunctionFitting):

    def __init__(self, data, x, y, n_gauss=2):
        super().__init__()
        self.data = data
        self.x = x
        self.y = y
        self.n_gauss = n_gauss

        m_offset = Model(lambda x, y, offset: offset * np.ones_like(x), independent_vars=['x', 'y'])
        self.model = m_offset
        for i in range(n_gauss):
            self.model += Model(gaussian2D_function, independent_vars=['x', 'y'], prefix=f'g{i}_')

        self.params = self.guess()
    

    def guess(self):
        from sklearn.cluster import KMeans
        params = Parameters()
        # Flatten coordinate grids and data
        X, Y = np.meshgrid(self.x, self.y)
        coords = np.column_stack([X.ravel(), Y.ravel()])
        data_flat = self.data.ravel()
        # Use data as weights for clustering
        kmeans = KMeans(n_clusters=self.n_gauss, n_init=10)
        # To avoid NaN, mask out zero/negative density points
        mask = data_flat > 0
        if np.sum(mask) >= self.n_gauss:
            kmeans.fit(coords[mask], sample_weight=data_flat[mask])
            centers = kmeans.cluster_centers_
        else:
            # fallback: uniform grid
            centers = np.column_stack([
                np.linspace(self.x.min(), self.x.max(), self.n_gauss),
                np.linspace(self.y.min(), self.y.max(), self.n_gauss)
            ])
        for i in range(self.n_gauss):
            # print(f"Gaussian {i} initial center: {centers[i]}")
            x0_guess, y0_guess = centers[i]
            params.add(f'g{i}_x0', value=x0_guess)
            params.add(f'g{i}_y0', value=y0_guess)
            params.add(f'g{i}_sigma_x', value=(self.x.max()-self.x.min())/4, min=1e-6)
            params.add(f'g{i}_sigma_y', value=(self.y.max()-self.y.min())/4, min=1e-6)
            # Amplitude guess: max in region near center
            ix = np.abs(self.x - x0_guess).argmin()
            iy = np.abs(self.y - y0_guess).argmin()
            amp_guess = self.data[iy, ix]
            params.add(f'g{i}_amp', value=amp_guess, min=0, vary=True)
        params.add('offset', value=np.min(self.data))
        return params
    
    def model_function(self, *args, **kwargs):
        pass

    def fit(self):

        result = self.model.fit(
            self.data.ravel(),
            x=np.tile(self.x, len(self.y)),
            y=np.repeat(self.y, len(self.x)),
            params=self.params
        )
        return result


class StateDiscrimination():

    
    """
    Class for analyzing exponential decay data with the flux coordinate.
    This is adapted from the repetition analysis code but replaces "repetition" with "flux".
    """

    def __init__(self, data: xr.Dataset, user_mean=None, user_std=None):
        # super().__init__()
        self.user_mean = user_mean
        self.user_std = user_std
        self._import_data(data)

    def _import_data(self, data):
        # Ensure input data is an xarray Dataset and has 'shot_idx' and 'prepared_state' coordinates.
        if not isinstance(data, xr.Dataset):
            raise ValueError("Input data must be an xarray.Dataset.")

        for coords_name in ["shot_idx", "prepared_state"]:
            if coords_name not in data.coords:
                raise ValueError(f"No {coords_name} coordinate in the input xarray.Dataset.")

        # Store original data in self.data
        self.data = data


    def _preprocess_data(self, bins=20):
        # Compute std for initialization (in original units)
        prepared_state_num = self.data.coords['prepared_state'].size
        std_init = []
        std_I = self.data['I'].std(dim='shot_idx').values
        std_Q = self.data['Q'].std(dim='shot_idx').values
        for i in range(prepared_state_num):
            std_init.append(np.array([std_I[i], std_Q[i]]))
        self.std_init = np.min(np.array(std_init))

        # Compute 2D histograms for each prepared_state and build a dataset
        prepared_states = self.data.coords['prepared_state'].values
        # Find global I/Q min/max for binning
        I_all = self.data['I'].values.ravel()
        Q_all = self.data['Q'].values.ravel()
        I_min, I_max = I_all.min(), I_all.max()
        Q_min, Q_max = Q_all.min(), Q_all.max()

        # Use std/5 as step for bins, prefer user_std if set
        std_val = self.user_std if self.user_std is not None else self.std_init
        step = std_val / 3
        # Ensure step is positive and not too small
        if step <= 0:
            step = 1e-3
        xedges = np.arange(I_min, I_max + step, step)
        yedges = np.arange(Q_min, Q_max + step, step)
        # If only one bin, fallback to linspace
        if len(xedges) < 2:
            xedges = np.linspace(I_min, I_max, 2)
        if len(yedges) < 2:
            yedges = np.linspace(Q_min, Q_max, 2)
        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])
        bins_x = len(xcenters)
        bins_y = len(ycenters)
        density_arr = np.zeros((len(prepared_states),bins_y,  bins_x))
        mean_init = []
        for i, state in enumerate(prepared_states):
            I = self.data['I'].sel(prepared_state=state).values
            Q = self.data['Q'].sel(prepared_state=state).values
            H, _, _ = np.histogram2d(I, Q, bins=[xedges, yedges], density=True)
                    # Plot density_all for visual inspection

            density_arr[i, :, :] = H.T
            
            # Find the coordinates of maximum value in H for mean_init
            max_idx = np.unravel_index(np.argmax(H), H.shape)
            max_I = xcenters[max_idx[0]]
            max_Q = ycenters[max_idx[1]]
            mean_init.append(np.array([max_I, max_Q]))
        
        self.mean_init = np.array(mean_init)
        self.hist_dataset = xr.Dataset(
            {'density': (['prepared_state', 'y', 'x'], density_arr)},
            coords={
                'prepared_state': prepared_states,
                'x': xcenters,
                'y': ycenters
            }
        )

    def _start_analysis(self):
        self._preprocess_data()
        self._analysis_by_multi_2Dgaussian( outlier_sigma = 3 )
        print(self.analysis_result['trained_paras'])

    def _plot_results(self, fig_group_name=None, save_path=None):

        figs = {}
        

        fig_raw, axes_raw = plot_prepared_state_scatter( self.data, self.analysis_result )
        fig_2Dhist, axes_2Dhist = plot_2d_histogram( self.hist_dataset, analysis_result=self.analysis_result)
        fig_outliers, axes_outliers = plot_outliers(self.data, self.analysis_result["outlier_mask"], analysis_result=self.analysis_result)
        fig_residue, axes_residue = plot_2d_fit_residue(self.analysis_result['fit_residues'],self.analysis_result['norm_res'])
        
        
        lim_I, lim_Q = compute_shared_axis_limits(self.data)

        for i in range(2):
            axis_formatter(axes_raw[i], lim_I, lim_Q, i)
            axis_formatter(axes_2Dhist[i], lim_I, lim_Q, i)
            axis_formatter(axes_outliers[i], lim_I, lim_Q, i)
            axis_formatter(axes_residue[i], lim_I, lim_Q, i)

        figs["2DHist"] = fig_2Dhist
        figs["raw"] = fig_raw
        figs["outliers"] = fig_outliers
        figs["fit_residue"] = fig_residue
        if save_path is not None:
            for plot_name, fig in figs.items():
                fig.savefig(f"{save_path}\\{fig_group_name}_{plot_name}.png", bbox_inches='tight')
        return figs


    def _analysis_by_multi_2Dgaussian(self, outlier_sigma=3):
        """
        1. Fit the 2D multi-Gaussian model on all data (concatenated over prepared_state).
        2. Use the trained model's parameters as initial guess to fit prepared_state=0 and prepared_state=1 separately.
        3. Store or return the fit results for each case.
        All fitting uses the density dataset (var 'density', coords: prepared_state, x, y).
        """
        # 1. Fit on all data (concatenated)
        # Concatenate all prepared_state densities for global fit
        trained_multi_2Dgaussian_params = self._train_by_multi_2Dgaussian()

        # 2. Fit on each prepared_state using the density dataset
        fit_results = []
        x = self.hist_dataset['x'].values
        y = self.hist_dataset['y'].values
        fit_residues_list = []
        norm_res = []
        for i, state in enumerate(self.hist_dataset['prepared_state'].values):
            density = self.hist_dataset['density'].sel(prepared_state=state).values
            fit_result, fitter = self._fit_histogram_by_multi_2Dgaussian(
                density, x, y, mean=trained_multi_2Dgaussian_params['mean'], std=trained_multi_2Dgaussian_params['std']
            )
            fit_results.append(self._extract_multi_2Dgaussian_params(fit_result, n_gauss=len(self.mean_init)))

            best_fit = fit_result.best_fit.reshape(density.shape)
            residue = density - best_fit
            fit_residues_list.append(residue)
            norm_res.append(np.nansum(residue) / np.nansum(density) if np.nansum(density) != 0 else np.nan)

        # Convert fit_residues_list to a DataArray with dims (prepared_state, y, x)
        fit_residues = xr.DataArray(
            np.stack(fit_residues_list, axis=0),
            dims=["prepared_state", "y", "x"],
            coords={
                "prepared_state": self.hist_dataset["prepared_state"].values,
                "y": self.hist_dataset["y"].values,
                "x": self.hist_dataset["x"].values,
            },
        )
        
        # 3. Use the trained model to assign state labels and count populations
        distance_dataset = self.calc_distances_to_mean(mean_trained=trained_multi_2Dgaussian_params['mean'])
        state_label = distance_dataset['distance'].argmin(dim='center')

        # Get population counts for each state label
        def bincount_1d(arr, minlength=None):
            return np.bincount(arr, minlength=minlength)

        max_label = int(state_label.max().item())
        minlength = max_label + 1

        counts = xr.apply_ufunc(
            bincount_1d,
            state_label,
            input_core_dims=[['idx_shot']],
            output_core_dims=[['count']],
            vectorize=True,
            kwargs={'minlength': minlength},
            output_dtypes=[int]
        )
        gaussian_amp = []
        for i, state in enumerate(self.hist_dataset['prepared_state'].values):
            gaussian_amp.append(fit_results[i]['amp'])
        gaussian_norms = np.array(gaussian_amp) / np.sum(gaussian_amp, axis=1, keepdims=True)

        # Outlier probability
        outlier_mask = distance_dataset['distance'].min(dim='center') > (outlier_sigma * np.mean(trained_multi_2Dgaussian_params['std']))
        n_outlier = np.count_nonzero(outlier_mask, axis=1)
        # print(f"Number of outliers detected: {n_outlier}")
        self.p_outlier = n_outlier / self.data['shot_idx'].size


        self.analysis_result = {
            'trained_paras': trained_multi_2Dgaussian_params,
            'fitted_paras': fit_results,
            'gaussian_norms': gaussian_norms,
            'direct_counts': counts.values/ self.data['shot_idx'].size,
            'state_label': state_label.values,
            'outlier_mask': outlier_mask.values,
            'outlier_probability': self.p_outlier,
            'norm_res': norm_res,
            'fit_residues': fit_residues,
        }

    def _train_by_multi_2Dgaussian(self, mean=None, std=None):
        """
        Fit the 2D multi-Gaussian model on all data (concatenated over prepared_state).
        If both mean and std are provided, skip training and use them directly.
        If only one is provided, fix that value during fitting.
        Returns trained_multi_2Dgaussian_params dict.
        """
        density_all = self.hist_dataset['density'].values
        density_all = np.sum(density_all, axis=0)  # sum over prepared_state

        x = self.hist_dataset['x'].values
        y = self.hist_dataset['y'].values

        # Use class properties if not provided as arguments
        use_mean = mean if mean is not None else self.user_mean
        use_std = std if std is not None else self.user_std

        # If both mean and std are provided (via argument or class property), skip fitting and use them directly
        if use_mean is not None and use_std is not None:
            trained_multi_2Dgaussian_params = {
                'mean': np.array(use_mean),
                'std': use_std,
                'covariance': use_std**2,
                'amp': np.ones(len(use_mean)),  # dummy amplitude
            }
            return trained_multi_2Dgaussian_params

        # If only one is provided, fix that value during fitting
        fit_all_result, fit_all_fitter = self._fit_histogram_by_multi_2Dgaussian(
            density_all, x, y, mean=use_mean, std=use_std
        )
        trained_multi_2Dgaussian_params = self._extract_multi_2Dgaussian_params(fit_all_result, n_gauss=len(self.mean_init))

        return trained_multi_2Dgaussian_params

    def _extract_multi_2Dgaussian_params(self, fit_result, n_gauss=None):
        """
        Extract mean, std, and amp from a fit result (lmfit.ModelResult) and return a dict in the format of self.trained_multi_2Dgaussian_params.
        Args:
            fit_result: lmfit ModelResult object with .params attribute
            n_gauss: number of Gaussians (if None, use self.mean_init)
        Returns:
            dict with keys 'mean', 'std', 'amp'
        """
        if n_gauss is None:
            n_gauss = len(self.mean_init)
        mean = []
        std = []
        amp = []
        for i in range(n_gauss):
            x0 = fit_result.params[f'g{i}_x0'].value
            y0 = fit_result.params[f'g{i}_y0'].value
            g_amp = fit_result.params[f'g{i}_amp'].value
            mean.append(np.array([x0, y0]))
            amp.append(g_amp)
        std = fit_result.params[f'g0_sigma_x'].value
        return {
            'mean': np.array(mean),
            'std': std,
            'covariance': std**2,
            'amp': np.array(amp),
        }
    
    def _fit_histogram_by_multi_2Dgaussian(self, density, x, y, mean=None, std=None):
        """
        Fit a 2D histogram (density) using FitMultiGaussian2D.
        Args:
            density: 2D numpy array (shape: [len(x), len(y)])
            x: 1D array of x bin centers
            y: 1D array of y bin centers
            mean: list/array of initial mean (optional)
            std: list/array of initial std (optional)
        Returns:
            fit_result: lmfit ModelResult from FitMultiGaussian2D.fit()
            fitter: the FitMultiGaussian2D instance
        """
        vary_mean = False
        vary_std = False
        if mean is None:
            mean = self.mean_init
            vary_mean = True
            print("Using default mean_init for fitting.", mean)
        if std is None:
            std = self.std_init
            vary_std = True
            print("Using default std_init for fitting.", std)

        n_gauss = len(mean)
        fitter = FitMultiGaussian2D(density, x, y, n_gauss=n_gauss)
        fitter.params['offset'].set(value=0, vary=False)
        for i in range(n_gauss):
            if vary_mean:
                # Allow parameters to vary with reasonable bounds
                fitter.params[f'g{i}_x0'].set(value=mean[i][0], vary=True, max=mean[i][0]+std*0.5, min=mean[i][0]-std*0.5)
                fitter.params[f'g{i}_y0'].set(value=mean[i][1], vary=True, max=mean[i][1]+std*0.5, min=mean[i][1]-std*0.5)
            else:
                # Fix the parameters if vary_mean is False
                fitter.params[f'g{i}_x0'].set(value=mean[i][0], vary=False)
                fitter.params[f'g{i}_y0'].set(value=mean[i][1], vary=False)
            # fitter.params[f'g{i}_amp'].set(value=np.max(density), vary=True)
            if i == 0:
                fitter.params[f'g{i}_sigma_x'].set(value=std, vary=vary_std)
            else:
                fitter.params[f'g{i}_sigma_x'].set(expr='g0_sigma_x')
            fitter.params[f'g{i}_sigma_y'].set(expr='g0_sigma_x')

        fit_result = fitter.fit()
        return fit_result, fitter
      
    def _export_result(self, save_path=None):
        # Implement result export functionality if needed.
        pass


    
    def rotate_data_to_x_axis(self):
        """
        Return a rotated copy of self.data so that the vector between the two GMM mean (in scaled space) aligns with the x-axis.
        self.data is never modified in-place.
        Returns:
            rotated_data: xarray.Dataset with rotated 'I' and 'Q' variables
            angle: rotation angle in radians (counterclockwise)
        """
        import copy
        # Get mean in scaled space
        mean = self.gmm_model.mean_  # shape (2, 2), columns: I, Q
        v = mean[1] - mean[0]  # vector from mean 0 to mean 1
        angle = np.arctan2(v[1], v[0])  # angle to x-axis
        # Build rotation matrix (counterclockwise)
        R = np.array([[np.cos(-angle), -np.sin(-angle)],
                      [np.sin(-angle),  np.cos(-angle)]])
        # Deep copy to avoid modifying self.data
        rotated_data = copy.deepcopy(self.data)
        for i in self.data.coords['prepared_state']:
            I = self.data['I'].sel(prepared_state=i).values
            Q = self.data['Q'].sel(prepared_state=i).values
            IQ = np.stack([I, Q], axis=-1)
            IQ_rot = IQ @ R.T  # shape (..., 2)
            rotated_data['I'].loc[dict(prepared_state=i)] = IQ_rot[..., 0]
            rotated_data['Q'].loc[dict(prepared_state=i)] = IQ_rot[..., 1]
        return rotated_data, angle
    
    def calc_distances_to_mean(self, mean_trained=None):
        """
        Calculate the Euclidean distances from each (I, Q) point (for all shot_idx and prepared_state)
        to each of the two mean_trained points.
        Args:
            mean_trained: list or array of two mean points [[x0, y0], [x1, y1]]. If None, uses self._trained_multi_2Dgaussian_params['mean'].
        Returns:
            distances: dict with keys 'prepared_state', each value is a (n_shots, 2) array of distances to each mean.
        """
        if mean_trained is None:
            mean_trained = self._trained_multi_2Dgaussian_params['mean']
        # Get coordinate values
        prepared_states = self.data.coords['prepared_state'].values
        n_center = len(mean_trained)
        n_state = len(prepared_states)
        n_shot = self.data.sizes['shot_idx']

        # Allocate array: (center, prepared_state, idx_shot)
        dist_arr = np.zeros((n_center, n_state, n_shot))
        for i_center, mean in enumerate(mean_trained):
            for i_state, state in enumerate(prepared_states):
                I = self.data['I'].sel(prepared_state=state).values.ravel()
                Q = self.data['Q'].sel(prepared_state=state).values.ravel()
                dist_arr[i_center, i_state, :] = np.sqrt((I - mean[0])**2 + (Q - mean[1])**2)

        # Build xarray Dataset
        dis = xr.Dataset({
            'distance': (['center', 'prepared_state', 'idx_shot'], dist_arr)
        }, coords={
            'center': np.arange(n_center),
            'prepared_state': prepared_states,
            'idx_shot': np.arange(n_shot)
        })
        return dis
    
    def show_thermal_analysis(self, qubit_name, freq_ghz, ax:Axes):
        """
        計算有效溫度並繪製 1D 投影高斯分佈圖。
        freq_ghz: Qubit 的頻率，單位為 GHz。
        """
        from scipy.stats import norm

        # 1. 取得數據與參數
        (p00, p01), (p10, p11) = self.analysis_result['gaussian_norms']
        trained_paras = self.analysis_result['trained_paras']
        
        wa = 2 * np.pi * freq_ghz * 1e9 
        temp_mk = PetoT(p01, wa)
        freq_mhz = round(freq_ghz * 1e3)
        
        # 2. 投影計算
        mu_g = trained_paras['mean'][0]
        mu_e = trained_paras['mean'][1]
        unit_vec = (mu_e - mu_g) / np.linalg.norm(mu_e - mu_g)
        
        points = np.stack([self.data.sel(prepared_state=0)['I'].values, 
                           self.data.sel(prepared_state=0)['Q'].values], axis=1)
        projections = np.dot(points - mu_g, unit_vec)
        
        # 3. 繪圖核心
        # 直方圖優化：去除邊框 (lw=0)，提高透明度
        ax.hist(projections, bins=100, density=True, alpha=0.2, color='#7f8c8d', lw=0)
        
        x_axis = np.linspace(np.min(projections), np.max(projections), 500)
        std_1d = trained_paras['std']
        dist_centers = np.linalg.norm(mu_e - mu_g)
        
        y_g = p00 * norm.pdf(x_axis, 0, std_1d)
        y_e = p01 * norm.pdf(x_axis, dist_centers, std_1d)
        
        # 曲線優化：使用更深一點的色調
        ax.plot(x_axis, y_g, color='#0984e3', lw=2, label=f'G-state ({p00:.3f})')
        ax.plot(x_axis, y_e, color='#d63031', lw=2, label=f'E-state ({p01:.3f})')
        # 填滿曲線下方，增加美感
        ax.fill_between(x_axis, y_g, color='#0984e3', alpha=0.1)
        ax.fill_between(x_axis, y_e, color='#d63031', alpha=0.1)
        
        # 4. 裝飾細節
        ax.set_title(f"{qubit_name} @ {freq_mhz} MHz", fontsize=11, fontweight='bold', pad=10)
        
        # 資訊框優化：背景透明度增加，邊框顏色變淡，放置在更適當的位置
        info_text = f"$P_{{01}}$: {p01:.2%}\n$T_{{eff}}$: {temp_mk:.1f} mK"
        ax.text(0.95, 0.75, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ecf0f1', alpha=0.8))
        
        # 座標軸與網格
        ax.set_xlabel('Projected distance', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.grid(True, linestyle='-', color='#f1f2f6', alpha=0.7, zorder=0) # 極淡的背景網格
        ax.tick_params(labelsize=8)
        
        # 圖例優化：無邊框，位置調整
        ax.legend(loc='upper right', fontsize=8, frameon=False)
        
        # 去除上方與右方的邊框，讓圖表更開闊 (Spine removal)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        return temp_mk
    
    def show_scatter_analysis(self, qubit_name, ax:Axes):
        """
        在指定的 ax 上畫出 I-Q 散佈圖，包含 GMM 分類顏色、中心點與標準差圓圈。
        """
        from matplotlib.patches import Ellipse

        # 1. 取得資料與分析結果
        i_data = self.data['I'].values.flatten()
        q_data = self.data['Q'].values.flatten()
        labels = self.analysis_result['state_label'].flatten()
        trained_paras = self.analysis_result['trained_paras']
        means = trained_paras['mean'] 
        std = trained_paras['std']

        # 2. 繪製散佈圖 
        # 增加 s (點大小) 到 0.8，alpha (不透明度) 到 0.3，讓顏色更紮實
        # 使用 'RdYlBu_r' 或 'coolwarm' 會有較明顯的藍紅對比
        scatter = ax.scatter(i_data, q_data, c=labels, s=0.8, alpha=0.3, 
                             cmap='coolwarm', edgecolors='none', zorder=2)

        # 3. 處理中心點與標籤
        # 依照 I 值排序：確保左邊永遠是 g，右邊永遠是 e
        indexed_means = sorted(enumerate(means), key=lambda x: x[1][0])
        state_labels = [r'$|g\rangle$', r'$|e\rangle$'] # 使用 Raw string 修正亂碼

        for i, (idx, center) in enumerate(indexed_means):
            # 畫中心黑點，稍微加大一點
            ax.scatter(center[0], center[1], c='black', s=20, zorder=10)
            
            # 畫 2-sigma 圓圈，線條加粗
            circle = Ellipse(xy=center, width=4*std, height=4*std, 
                             edgecolor='black', facecolor='none', 
                             linestyle='--', linewidth=1.0, alpha=0.7, zorder=5)
            ax.add_patch(circle)

            # 標籤位置：放在圓圈正上方，增加字體大小與飽和度
            ax.text(center[0], center[1] + 2.5*std, state_labels[i], 
                    fontsize=14, fontweight='bold', ha='center', va='bottom', 
                    color='black', zorder=11)

        # 4. 座標軸範圍自動校正 (確保點群在中央)
        all_i = [m[0] for m in means]
        all_q = [m[1] for m in means]
        ax.set_xlim(min(all_i) - 6*std, max(all_i) + 6*std)
        ax.set_ylim(np.mean(all_q) - 6*std, np.mean(all_q) + 6*std)

        # 5. 裝飾與網格
        ax.set_title(f"{qubit_name}", fontsize=12, fontweight='bold')
        ax.set_xlabel('I (V)', fontsize=10)
        ax.set_ylabel('Q (V)', fontsize=10)
        ax.set_aspect('equal')
        
        # 開啟網格，設定為灰色虛線
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5, zorder=0)
        
        # 讓座標軸刻度更明顯
        ax.tick_params(labelsize=9)

    def show_outliers_analysis(self, qubit_name, ax:Axes):
        """
        在指定的 ax 上標示出 IQ 資料中的 Outliers。
        """
    
        from matplotlib.patches import Ellipse

        # 1. 取得資料
        analysis_result = self.analysis_result
        outlier_mask = analysis_result.get('outliers', analysis_result.get('outlier_mask', None))
        
        if outlier_mask is None:
            ax.text(0.5, 0.5, "Data Missing", ha='center', va='center', color='gray')
            return

        # 2. 繪製離群點：使用更鮮艷的顏色 (#e67e22)
        vibrant_orange = '#e67e22' 

        for i in range(2):
            if i in outlier_mask:
                mask = outlier_mask[i]
                i_vals = self.data['I'].sel(prepared_state=i).values[mask]
                q_vals = self.data['Q'].sel(prepared_state=i).values[mask]
                
                
                if len(i_vals) > 0:
                    ax.scatter(i_vals, q_vals, s=10, color=vibrant_orange, 
                               marker='.', zorder=3)

        # 3. 繪製中心點與高斯圓圈
        trained_paras = analysis_result.get('trained_paras', {})
        means = trained_paras.get('mean', [])
        std = trained_paras.get('std', 0)
        
        for mean in means:
            ax.scatter(mean[0], mean[1], c='#2d3436', s=25, zorder=10) # 放大中心點
            for s in [1, 2, 3]:
                circle = Ellipse(xy=mean, width=2*s*std, height=2*s*std,
                                 edgecolor='#b2bec3', facecolor='none', 
                                 linestyle='--', linewidth=0.8, alpha=0.5, zorder=5)
                ax.add_patch(circle)

        # 4. 文字框：確保 LaTeX 換行正確 (使用雙反斜線 \\)
        if 'outlier_probability' in analysis_result:
            prob = analysis_result['outlier_probability']
            info_str = f"$|g\\rangle$={prob[0]:.1%}, $|e\\rangle$={prob[1]:.1%}"
        else:
            info_str = ''


        # 6. 現代感視覺設定
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(f"{qubit_name.upper()} \n{info_str}", fontweight='bold')
        ax.set_xlabel('I (V)', fontsize=10)
        ax.set_ylabel('Q (V)', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.9, zorder=0)
        ax.set_aspect('equal')
        
        # 範圍自動對焦
        if len(means) > 0:
            all_i = [m[0] for m in means]
            ax.set_xlim(min(all_i) - 10*std, max(all_i) + 10*std)