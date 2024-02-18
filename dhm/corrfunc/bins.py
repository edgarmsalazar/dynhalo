import math
from typing import List, Tuple, Union
from warnings import filterwarnings

import numpy as np
from tqdm import tqdm

filterwarnings("ignore")


def generate_bins(
    bmin: float,
    bmax: float,
    nbins: int,
    logspaced: bool = True,
    soft: float = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates equally spaced bins in linear or logarithmic (base 10) space.

    Parameters
    ----------
    bmin : float
        Minimum or lower bound (inclusive)
    bmax : float
        Maximum or upper bound (inclusive)
    nbins : int
        Number of bins to generate
    logspaced : bool, optional
        Generate log-spaced bins, by default True
    soft : float, optional
        If there is softening value, then bmin >= soft, by default 0

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Bins and edges in units of bmin and bmax
    """
    if bmin < soft:
        bmin = soft
    # Equally log-spaced bins
    if bmin == 0:
        logspaced = False

    if logspaced:
        r_edges = np.logspace(
            start=np.log10(bmin),
            stop=np.log10(bmax),
            num=nbins + 1,
            base=10,
        )
    else:
        r_edges = np.linspace(
            start=bmin,
            stop=bmax,
            num=nbins + 1,
        )
    # Bin middle point
    rbins = 0.5 * (r_edges[1:] + r_edges[:-1])
    return rbins, r_edges


def generate_bin_str(bin_edges: Union[List[float], Tuple[float]]) -> str:
    """Generates a formatted string 'min-max' from bin edges. Intended to be
    compatible with HDF dataset naming and LaTeX format.

    Parameters
    ----------
    bin_edges : Union[List[float], Tuple[float]]
        A 2-tuple or list with minimum or lower bound and maximum or upper bound
        of the bin

    Returns
    -------
    str
        _description_

    Raises
    ------
    ValueError
        If bin_edges is not a list or tuple of length 2.
    """
    if type(bin_edges) in [list, tuple] and len(bin_edges) == 2:
        return f"{bin_edges[0]:.2f}-{bin_edges[-1]:.2f}"
    else:
        raise ValueError("bin_edges must be a list of floats and len=2: [min, max]")


def partition_box(data: np.ndarray, boxsize: float, gridsize: float) -> List[float]:
    """Sorts all data points into a 3D grid with `cells per side = boxsize / gridsize`

    Parameters
    ----------
    data : np.ndarray
        `(N, d)` array with all data points' coordinates, where `N` is the
        number of data points and `d` the dimensions
    boxsize : float
        Simulation box size (per side)
    gridsize : float
        Grid size (per side)

    Returns
    -------
    List[float]

    """
    # Number of grid cells per side.
    cells_per_side = int(math.ceil(boxsize / gridsize))
    # Grid ID for each data point.
    grid_id = (data / gridsize).astype(int, copy=False)
    # Correction for points on the edges.
    grid_id[np.where(grid_id == cells_per_side)] = cells_per_side - 1

    # This list stores all of the particles original IDs in a convenient 3D
    # list. It is kind of a pointer of size n_cpd**3
    data_cell_id = [[] for _ in range(cells_per_side**3)]
    cells = (
        cells_per_side**2 * grid_id[:, 0]
        + cells_per_side * grid_id[:, 1]
        + grid_id[:, 2]
    )
    for cell in tqdm(
        range(np.size(data, 0)), desc="Partitioning box", ncols=100, colour="blue"
    ):
        data_cell_id[cells[cell]].append(cell)

    return data_cell_id
