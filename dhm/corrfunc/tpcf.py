import math

from typing import List, Tuple
from warnings import filterwarnings

from Corrfunc.theory import DD as countDD
import numpy as np

from tqdm import tqdm

filterwarnings('ignore')


def get_radial_bins(
    n_rbins: int,
    rmin: float,
    rmax: float,
    rsoft: float = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates radial bins.

    Args:
        n_rbins (int, optional): number of radial bins below `rmid`.
            Defaults to 20.
        n_ext (int, optional): number of radial bins above `rmid`. 
            Defaults to 7.
        rmin (int | float, optional): lower radial bound. Defaults to `RSOFT`.
        rmid (int | float, optional): generated `n_rbins` up to this radial
            scale. Defaults to 5 Mpc/h.
        rmax (int | float, optional): upper radial bound. Defaults to 50 Mpc/h. 
            If `full` is `False`, then the upper radial bound is `rmin`.
        full (bool, optional): If `True` extends the radial bins from `rmin` to
            `rmax`. Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: radial bins and bin edges.
    """
    if rmin < rsoft:
        rmin = rsoft
    # Equally log-spaced bins
    r_edges = np.logspace(np.log10(rmin), np.log10(rmax),
                          num=n_rbins + 1, base=10)
    # Bin middle point
    rbins = 0.5 * (r_edges[1:] + r_edges[:-1])
    return rbins, r_edges


def gen_binstr(bin_edges: list) -> list:
    return [f'{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}' for i in range(len(bin_edges)-1)]


def get_mbins(mmin, mmax, mnum, bins=None) -> tuple:
    if bins is not None and (isinstance(bins, list) or isinstance(bins, tuple)):
        if isinstance(bins[-1], int) and len(bins) == 3:
            # Equally log-spaced bins
            bin_edges = np.linspace(bins[0], bins[1], num=bins[-1] + 1)
            # Bin middle point
            bin_mid = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        else:
            bin_edges = bins
            bin_mid = 0.5 * (bins[1:] + bins[:-1])
        # Generate labels for each bin
    else:
        # Grab data from
        bin_edges = np.linspace(mmin, mmax,num=mnum + 1)
        # Bin middle point
        bin_mid = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        # Generate labels for each bin
    bin_str = gen_binstr(bins)

    return bin_mid, bin_edges, bin_str


def partition_box(data: np.ndarray, boxsize: float , gridsize: float) -> List[float]:
    # Number of grid cells per side.
    n_cpd = int(math.ceil(boxsize / gridsize))
    # Grid ID for each data point.
    grid_id = data[:] / gridsize
    grid_id = grid_id.astype(int)
    # Correction for points on the edges.
    grid_id[np.where(grid_id == n_cpd)] = n_cpd - 1

    # This list stores all of the particles original IDs in a convenient 3D
    # list. It is kind of a pointer of size n_cpd**3
    data_id = [[] for _ in range(n_cpd**3)]
    cells = n_cpd**2 * grid_id[:, 0] + n_cpd * grid_id[:, 1] + grid_id[:, 2]
    for cell in tqdm(range(np.size(data, 0)), desc="Partitioning box", ncols=100, colour='blue'):
        data_id[cells[cell]].append(cell)

    return data_id


def process_DD_pairs(
    data_1: np.ndarray,
    data_2: np.ndarray,
    data_1_id: list,
    radial_edges: np.ndarray,
    boxsize: float,
    gridsize: float,
    weights_1=None,
    weights_2=None,
    nthreads: int = 16,
) -> np.ndarray:
    # Number of grid cells per side
    n_cpd = int(math.ceil(boxsize / gridsize))

    # Create pairs list
    ddpairs_i = np.zeros((n_cpd**3, radial_edges.size - 1))
    zeros = np.zeros(radial_edges.size - 1)

    # Pair counting
    # Loop for each minibox
    for s1 in tqdm(range(n_cpd**3), desc="Pair counting", ncols=100, colour='green'):
        # Get data1 box
        xd1 = data_1[data_1_id[s1], 0]
        yd1 = data_1[data_1_id[s1], 1]
        zd1 = data_1[data_1_id[s1], 2]
        if weights_1 is not None:
            if isinstance(weights_1, np.ndarray):
                wd1 = weights_1[data_1_id[s1]]
            else:
                wd1 = weights_1
        else:
            wd1 = 1.0

        # Get data2 box
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
        # Data pair counting
        if np.size(xd1) != 0 and np.size(xd2) != 0:
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
                weight_type='pair_product',
                periodic=True,
                boxsize=boxsize,
                verbose=False,
            )["npairs"]
            ddpairs_i[s1] = DD_counts
        else:
            ddpairs_i[s1] = zeros

    return ddpairs_i


def tpck_with_jk_from_DD(
    data_1: np.ndarray,
    data_2: np.ndarray,
    data_1_id: list,
    radial_edges: np.ndarray,
    dd_pairs: np.ndarray,
    boxsize: float,
    gridsize: float,
) -> Tuple[np.ndarray]:
    # Set up bins
    nbins = radial_edges.size - 1

    n_cpd = int(math.ceil(boxsize / gridsize))  # Number of cells per dimension
    n_jk = n_cpd**3  # Number of jackknife samples
    d1tot = np.size(data_1, 0)  # Number of objects in d1
    d2tot = np.size(data_2, 0)  # Number of objects in d2
    Vbox = boxsize**3  # Volume of box
    Vshell = np.zeros(nbins)  # Volume of spherical shell
    # Vjk = (N - 1) / N * Vbox
    for m in range(nbins):
        Vshell[m] = (
            4.0 / 3.0 * np.pi *
            (radial_edges[m + 1] ** 3 - radial_edges[m] ** 3)
        )
    n1 = float(d1tot) / Vbox  # Number density of d2
    n2 = float(d2tot) / Vbox  # Number density of d2

    # Some arrays
    ddpairs_removei = np.zeros((n_jk, nbins))
    xi = np.zeros(nbins)
    xi_i = np.zeros((n_jk, nbins))
    meanxi_i = np.zeros(nbins)
    cov = np.zeros((nbins, nbins))

    ddpairs = np.sum(dd_pairs, axis=0)

    ddpairs_removei = ddpairs[None, :] - dd_pairs
    for s1 in range(n_jk):
        d1tot_s1 = np.size(data_1, 0) - np.size(data_1[data_1_id[s1]], 0)
        # xi_i[s1] = dd_pairs_i[s1] / (n1 * n2 * Vjk * Vshell) - 1
        xi_i[s1] = ddpairs_removei[s1] / (d1tot_s1 * n2 * Vshell) - 1

    # Compute mean xi from all jk samples
    for i in range(nbins):
        meanxi_i[i] = np.mean(xi_i[:, i])

    # Compute covariance matrix
    cov = (float(n_jk) - 1.0) * np.cov(xi_i.T, bias=True)

    # Compute the total xi
    xi = ddpairs / (n1 * n2 * Vbox * Vshell) - 1

    return xi, xi_i, meanxi_i, cov


def cross_tpcf_jk(
    data_1: np.ndarray,
    data_2: np.ndarray,
    radial_edges: np.ndarray,
    weights_1=None,
    weights_2=None,
    nthreads: int = 16,
    jk_estimates: bool = True,
) -> Tuple[np.ndarray]:
    # Partition boxes
    data_1_id = partition_box(data_1)

    # Pair counting. NOTE: data_1 and data_2 must have the same dtype.
    dd_pairs = process_DD_pairs(
        data_1=data_1,
        data_2=data_2,
        data_1_id=data_1_id,
        radial_edges=radial_edges,
        weights_1=weights_1,
        weights_2=weights_2,
        nthreads=nthreads,
    )
    # Estimate jackknife samples of the tpcf
    xi, xi_i, mean_xi_i, cov = tpck_with_jk_from_DD(
        data_1=data_1,
        data_2=data_2,
        data_1_id=data_1_id,
        radial_edges=radial_edges,
        dd_pairs=dd_pairs,
    )

    # Total correlation function (measured once)
    # Correlation function per subsample (subvolume)
    # Mean correlation function (sample mean)
    # Covariance
    if jk_estimates:
        return xi, xi_i, mean_xi_i, cov
    # Mean correlation function
    # Covariance
    else:
        return mean_xi_i, cov


if __name__ == "__main__":
    pass