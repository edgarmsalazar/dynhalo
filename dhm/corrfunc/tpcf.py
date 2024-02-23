import math
from typing import Tuple
from warnings import filterwarnings

import numpy as np
from Corrfunc.theory import DD as countDD
from tqdm import tqdm

from dhm.corrfunc.bins import partition_box

filterwarnings("ignore")


def process_DD_pairs(
    data_1: np.ndarray,
    data_2: np.ndarray,
    data_1_id: list,
    radial_edges: np.ndarray,
    boxsize: float,
    gridsize: float,
    weights_1=None,
    weights_2=None,
    nthreads: int = 4,
) -> np.ndarray:
    """Counts data-data pais by using pre-partitiones box into a 3D grid.
    Corrfunc does the heavy lifting. Doc is taken from Corrfunc.theory.DD

    Parameters
    ----------
    data_1 : np.ndarray
        The array of X/Y/Z positions for the first set of points. Calculations
        are done in the precision of the supplied arrays.
    data_2 : np.ndarray
        The array of X/Y/Z positions for the second set of points. Calculations
        are done in the precision of the supplied arrays.
    data_1_id : list
        Box partitioning 3D grid.
    radial_edges : np.ndarray
        The bins need to be contiguous and sorted in increasing order (smallest
        bins come first).
    boxsize : float
        Size of simulation box
    gridsize : float
        Size of sub-volume or cell of the box
    weights_1 : array_like, real (float/double), optional
        A scalar, or an array of weights of shape `(n_weights, n_positions)` or
        `(n_positions,)`. If `None` will be set to uniform weights.
        By default `None`
    weights_2 : array_like, real (float/double), optional
        Same as `weights_1` but for `data_2`, by default None
    nthreads : int, optional
        The number of OpenMP threads to use. Has no effect if OpenMP was not
        enabled during library compilation, by default 4

    Returns
    -------
    np.ndarray
        Data-data pair counts
    """
    # Number of grid cells per side
    cells_per_side = int(math.ceil(boxsize / gridsize))
    n_cells = cells_per_side**3
    # Create pairs list for each radial bin for each cell
    dd_pairs = np.zeros((n_cells, radial_edges.size - 1))
    # Fill value if no pairs are found in cell
    zeros = np.zeros(radial_edges.size - 1)

    # Pair counting for all elements in each cell
    for cell in tqdm(range(n_cells), desc="Pair counting", ncols=100, colour="green"):
        # Get data 1 in cell
        xd1 = data_1[data_1_id[cell], 0]
        yd1 = data_1[data_1_id[cell], 1]
        zd1 = data_1[data_1_id[cell], 2]
        if weights_1 is not None:
            if isinstance(weights_1, np.ndarray):
                wd1 = weights_1[data_1_id[cell]]
            else:
                wd1 = weights_1
        else:
            wd1 = 1.0

        # Get data 2 in cell
        xd2 = data_2[:, 0]
        yd2 = data_2[:, 1]
        zd2 = data_2[:, 2]
        if weights_2 is not None:
            if isinstance(weights_2, np.ndarray):
                wd2 = weights_2[:]
            else:
                wd2 = weights_2
        else:
            wd2 = 1.0

        # Data pair counting if there are elements in cell
        if np.size(xd1) != 0:  # and np.size(xd2) != 0:
            autocorr = 0
            # Data pair counting
            DD_counts = countDD(
                autocorr=autocorr,
                nthreads=nthreads,
                binfile=radial_edges,
                X1=xd1,
                Y1=yd1,
                Z1=zd1,
                X2=xd2,
                Y2=yd2,
                Z2=zd2,
                weights1=wd1,
                weights2=wd2,
                weight_type="pair_product",
                periodic=True,
                boxsize=boxsize,
                verbose=False,
            )["npairs"]
            dd_pairs[cell] = DD_counts
        else:
            dd_pairs[cell] = zeros

    return dd_pairs


def tpcf_jk(
    n_obj_d1: int,
    n_obj_d2: int,
    data_1_id: list,
    radial_edges: np.ndarray,
    dd_pairs: np.ndarray,
    boxsize: float,
    gridsize: float,
) -> Tuple[np.ndarray]:
    """Ultra-fast correlation function estimation with jackknife samples

    Parameters
    ----------
    n_obj_d1 : int
        The number of objects/particles/points in the first set of points
    n_obj_d2 : int
        The number of objects/particles/points in the second set of points
    data_1_id : list
        Box partitioning 3D grid.
    radial_edges : np.ndarray
        The bins need to be contiguous and sorted in increasing order (smallest
        bins come first).
    dd_pairs : np.ndarray
        _description_
    boxsize : float
        Size of simulation box
    gridsize : float
        Size of sub-volume or cell of the box

    Returns
    -------
    Tuple[np.ndarray]
        Returns a tuple with `(xi, xi_samples, xi_mean, cov)`, where `xi` is the
        total correlation function measured directly on the full simulation box.
        `xi_samples` is an array of shape `(Njk, Nbins)` with the Njk samples of
        the correlation function. `xi_mean` and `xi_cov` are the mean and
        covariance of `xi_samples`.
    """
    # Number of radial bins
    n_bins = radial_edges.shape[0] - 1

    # Number of cells per dimension
    cells_per_side = int(math.ceil(boxsize / gridsize))

    # Number of jackknife samples. One sample per cell.
    n_jk_samples = cells_per_side**3

    # Number of objects in d1 and d2
    data_1_row_idx = np.arange(n_obj_d1)

    # Volume of the box and spherical shells.
    volume_box = boxsize**3
    volume_shell = 4.0 / 3.0 * np.pi * np.diff(np.power(radial_edges, 3))

    # Number densities
    num_dens_d1 = float(n_obj_d1) / volume_box
    num_dens_d2 = float(n_obj_d2) / volume_box

    # Sum all data pairs over the same axis
    dd_pairs_total = np.sum(dd_pairs, axis=0)

    # Compute the correlation function for each jk sample. That is, remove the
    # cell from data and compute the correlation function
    xi_samples = np.zeros((n_jk_samples, n_bins))
    dd_pairs_removed_samples = dd_pairs_total[None, :] - dd_pairs
    for sample in range(n_jk_samples):
        # Number of objects in d1 after removing all objects in sample.
        d1_total_sample = n_obj_d1 - np.size(data_1_row_idx[data_1_id[sample]], 0)

        xi_samples[sample] = (
            dd_pairs_removed_samples[sample]
            / (d1_total_sample * num_dens_d2 * volume_shell)
            - 1
        )

    # Compute mean correlation function from all jk samples
    xi_mean = np.mean(xi_samples, axis=0)

    # Compute covariance matrix of the radial bins using all jk samples
    xi_cov = (float(n_jk_samples) - 1.0) * np.cov(xi_samples.T, bias=True)

    # Compute the total correlation function from all pairs
    xi = dd_pairs_total / (num_dens_d1 * num_dens_d2 * volume_box * volume_shell) - 1

    return xi, xi_samples, xi_mean, xi_cov


def cross_tpcf_jk(
    data_1: np.ndarray,
    data_2: np.ndarray,
    radial_edges: np.ndarray,
    boxsize: float,
    gridsize: float,
    weights_1=None,
    weights_2=None,
    nthreads: int = 4,
    jk_estimates: bool = True,
) -> Tuple[np.ndarray]:
    """Compute the cross-correlation function between data 1 and data 2. It is
    assumed that data 1...

    Parameters
    ----------
    data_1 : np.ndarray
        The array of X/Y/Z positions for the first set of points. Calculations
        are done in the precision of the supplied arrays.
    data_2 : np.ndarray
        The array of X/Y/Z positions for the second set of points. Calculations
        are done in the precision of the supplied arrays.
    radial_edges : np.ndarray
        The bins need to be contiguous and sorted in increasing order (smallest
        bins come first).
    boxsize : float
        Size of simulation box
    gridsize : float
        Size of sub-volume or cell of the box
    weights_1 : array_like, real (float/double), optional
        A scalar, or an array of weights of shape `(n_weights, n_positions)` or
        `(n_positions,)`. If `None` will be set to uniform weights.
        By default `None`
    weights_2 : array_like, real (float/double), optional
        Same as `weights_1` but for `data_2`, by default None
    nthreads : int, optional
        The number of OpenMP threads to use. Has no effect if OpenMP was not
        enabled during library compilation, by default 4
    jk_estimates : bool, optional
        If True returns all the jackknife samples and their mean, by default True

    Returns
    -------
    Tuple[np.ndarray]
        Total correlation function and covariance matrix. If `jk_estimates` is
        True, it also returns the jackknife samples and their mean.
    """

    # Partition box
    data_1_id = partition_box(
        data=data_1,
        boxsize=boxsize,
        gridsize=gridsize,
    )

    # Pair counting. NOTE: data_1 and data_2 must have the same dtype.
    dd_pairs = process_DD_pairs(
        data_1=data_1,
        data_2=data_2,
        data_1_id=data_1_id,
        radial_edges=radial_edges,
        boxsize=boxsize,
        gridsize=gridsize,
        weights_1=weights_1,
        weights_2=weights_2,
        nthreads=nthreads,
    )

    # Estimate jackknife samples of the tpcf
    xi, xi_samples, xi_mean, xi_cov = tpcf_jk(
        n_obj_d1=np.size(data_1, 0),
        n_obj_d2=np.size(data_2, 0),
        data_1_id=data_1_id,
        radial_edges=radial_edges,
        dd_pairs=dd_pairs,
        boxsize=boxsize,
        gridsize=gridsize,
    )

    # Total correlation function (measured once)
    # Correlation function per subsample (subvolume)
    # Mean correlation function (sample mean)
    # Covariance
    if jk_estimates:
        return xi, xi_samples, xi_mean, xi_cov
    # Total correlation function (measured once)
    # Covariance
    else:
        return xi, xi_cov


def density(
    n_obj: int,
    radii: np.ndarray,
    radial_edges: np.ndarray,
    mass: float,
) -> np.ndarray:
    """Compute density profile as a function of radial separation r.

    Parameters
    ----------
    n_obj : int
        Number of haloes.
    radii : np.ndarray
        r coordinate for all particles
    radial_edges : np.ndarray
        The bins need to be contiguous and sorted in increasing order (smallest
        bins come first).
    mass : float
        Particle mass.

    Returns
    -------
    np.ndarray
        Density profile.
    """
    n_bins = radial_edges.shape[0] - 1
    volume_shell = 4.0 / 3.0 * np.pi * np.diff((np.power(radial_edges, 3)))

    # Compute mass density per spherical shell
    rho = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (radial_edges[i] < radii) & (radii <= radial_edges[i + 1])
        rho[i] = mass * mask.sum() / (n_obj * volume_shell[i])
    return rho


def density_jk(
    n_obj_d1: int,
    data_1_id: list,
    data_1_hid: np.ndarray,
    radial_data: np.ndarray,
    radial_edges: np.ndarray,
    radial_data_1_id: np.ndarray,
    boxsize: float,
    gridsize: float,
    mass: float,
) -> Tuple[np.ndarray]:
    """Density profile jackknife samples

    Parameters
    ----------
    n_obj_d1 : int
        Number of haloes.
    data_1_id : list
        Box partitioning 3D grid.
    data_1_hid : np.ndarray
        Halo ID.
    radial_data : np.ndarray
        r coordinate for all particles
    radial_edges : np.ndarray
        The bins need to be contiguous and sorted in increasing order (smallest
        bins come first).
    radial_data_1_id : np.ndarray
        Parent halo ID for each particle.
    boxsize : float
        Size of simulation box
    gridsize : float
        Size of sub-volume or cell of the box
    mass : float
        Particle mass.

    Returns
    -------
    Tuple[np.ndarray]
        Returns a tuple with `(rho, rho_samples, rho_mean, cov)`, where `rho` is
        the total correlation function measured directly on the full simulation
        box. `rho_samples` is an array of shape `(Njk, Nbins)` with the Njk
        samples of the density profile. `rho_mean` and `rho_cov` are the mean
        and covariance of `rho_samples`.
    """
    n_bins = radial_edges.shape[0] - 1
    # Number of cells per dimension
    cells_per_side = int(math.ceil(boxsize / gridsize))

    # Number of jackknife samples. One sample per cell
    n_jk_samples = cells_per_side**3

    # Data 1 index array
    data_1_xx = np.arange(n_obj_d1)

    rho_samples = np.zeros((n_jk_samples, n_bins))
    for sample in tqdm(
        range(n_jk_samples), desc="Pair counting", ncols=100, colour="green"
    ):
        d1_total_sample = np.size(data_1_xx[data_1_id[sample]], 0)
        mask = np.isin(radial_data_1_id, data_1_hid[data_1_id[sample]])

        rho_samples[sample] = density(
            n_obj=d1_total_sample,
            radii=radial_data[mask],
            radial_edges=radial_edges,
            mass=mass,
        )

    # Compute mean correlation function from all jk samples
    rho_mean = np.mean(rho_samples, axis=0)

    # Compute covariance matrix of the radial bins using all jk samples
    rho_cov = (
        (float(n_jk_samples) - 1.0)
        * np.cov(rho_samples.T, bias=True)
        / np.sqrt(n_obj_d1)
    )

    rho = density(
        n_obj=n_obj_d1, radii=radial_data, radial_edges=radial_edges, mass=mass
    )

    return rho, rho_samples, rho_mean, rho_cov


def cross_tpcf_jk_radial(
    data_1: np.ndarray,
    data_1_hid: np.ndarray,
    radial_data: np.ndarray,
    radial_edges: np.ndarray,
    radial_data_1_id: np.ndarray,
    boxsize: float,
    gridsize: float,
    mass: float,
    jk_estimates: bool = True,
) -> Tuple[np.ndarray]:
    """Compute the cross-correlation function between data 1 and data 2. It is
    assumed that data 1...

    Parameters
    ----------
    data_1 : np.ndarray
        The array of X/Y/Z positions for the first set of points. Calculations
        are done in the precision of the supplied arrays.
    data_1_hid : np.ndarray
        Halo ID.
    radial_data : np.ndarray
        r coordinate for all particles
    radial_edges : np.ndarray
        The bins need to be contiguous and sorted in increasing order (smallest
        bins come first).
    radial_data_1_id : np.ndarray
        Parent halo ID for each particle.
    boxsize : float
        Size of simulation box
    gridsize : float
        Size of sub-volume or cell of the box
    mass : float
        Particle mass.
    jk_estimates : bool, optional
        If True returns all the jackknife samples and their mean, by default True

    Returns
    -------
    Tuple[np.ndarray]
        Total density profile and covariance matrix. If `jk_estimates` is
        True, it also returns the jackknife samples and their mean.
    """
    # Partition box
    data_1_id = partition_box(
        data=data_1,
        box_size=boxsize,
        grid_size=gridsize,
    )

    rho, rho_samples, rho_mean, rho_cov = density_jk(
        n_obj_d1=np.size(data_1, 0),
        data_1_id=data_1_id,
        data_1_hid=data_1_hid,
        radial_data=radial_data,
        radial_edges=radial_edges,
        radial_data_1_id=radial_data_1_id,
        boxsize=boxsize,
        gridsize=gridsize,
        mass=mass,
    )
    if jk_estimates:
        return rho, rho_samples, rho_mean, rho_cov
    else:
        return rho, rho_cov


if __name__ == "__main__":
    pass
