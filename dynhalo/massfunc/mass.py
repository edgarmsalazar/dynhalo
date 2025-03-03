import math
from typing import List, Tuple, Union

import numpy as np
from colossus.cosmology import cosmology as ccosmo
from colossus.lss import peaks as cpeaks
from scipy.integrate import simps as simpson
from scipy.interpolate import interp1d

from dynhalo.corrfunc.bins import partition_box


def set_colossus_cosmology(params) -> ccosmo.Cosmology:
    ccosmo.addCosmology(cosmo_name='mycosmo', params=params)
    cosmo = ccosmo.setCosmology(cosmo_name='mycosmo')
    return cosmo


def h_parameter(z: float, cosmo: ccosmo.Cosmology) -> float:
    return cosmo.Hz(z=z) / 100.


def peak_height(mass: float, z: float) -> float:
    return cpeaks.peakHeight(M=mass, z=z)


def measure_mass_function_jk(
    positions: np.ndarray,
    masses: np.ndarray,
    boxsize: float,
    gridsize: float,
    nu_edges: Union[List, Tuple, np.ndarray],
    z: float,
) -> np.ndarray:
    # Number of grid cells per side
    cells_per_side = int(math.ceil(boxsize / gridsize))
    # Number of jackknife samples. One sample per cell
    n_jk_samples = cells_per_side**3
    # Number of radial bins
    n_bins = nu_edges.shape[0] - 1

    # Partition box
    data_cell_id = partition_box(
        data=positions, boxsize=boxsize, gridsize=gridsize)

    f_nu_samples = np.zeros((n_jk_samples, n_bins))
    for sample in range(n_jk_samples):
        nu = peak_height(mass=masses[data_cell_id[sample]], z=z)
        f_nu_samples[sample], _ = np.histogram(a=nu, bins=nu_edges)

    # Compute mean from all jk samples
    f_nu_mean = np.mean(f_nu_samples, axis=0)

    # Compute covariance matrix of the radial bins using all jk samples
    f_nu_cov = (float(n_jk_samples) - 1.0) * np.cov(f_nu_samples.T, bias=True)

    # Compute the total using all items
    f_nu, _ = np.histogram(a=peak_height(mass=masses, z=z), bins=nu_edges)

    return f_nu, f_nu_samples, f_nu_mean, f_nu_cov


def moving_barrier_model(
    nu_edges: float,
    delta_sc0: float,
    b: float,
    rhom: float,
    mass_from_nu: interp1d,
    boxsize: float,
) -> float:
    delta_sc = 1.686
    n_points = len(nu_edges) - 1

    nu_orb = np.stack([np.linspace(nu_edges[i], nu_edges[i + 1], 100)
                      for i in range(n_points)])
    nu0 = nu_orb * delta_sc0 / delta_sc
    nu_prime = nu0 * (1. + b*np.exp(-nu0) / delta_sc0)
    integrand = np.sqrt(2. / np.pi) * boxsize**3 * rhom * \
        np.exp(-0.5*nu_prime**2) / mass_from_nu(nu_orb)
    eval_at_nu = simpson(integrand, nu_prime)
    return eval_at_nu


def moving_barrier_lnlike(pars, *data):
    # Check parameter priors
    delta_sc0, b = pars
    out_of_prior_neg = any([p < 0 for p in (delta_sc0, b)])
    out_of_prior_lrg = any([b > 100])
    if out_of_prior_neg or out_of_prior_lrg:
        return -np.inf

    # Unpack data and evaluate parameters
    x_edges, y, covy, rhom, mass_from_nu, boxsize = data

    mask = np.diag(covy) != 0
    y = y[mask]
    covy = covy[mask, :][:, mask]

    y_model = moving_barrier_model(x_edges, delta_sc0, b, rhom, mass_from_nu,
                                   boxsize)
    y_model = y_model[mask]

    d = y - y_model
    log_det_cov = np.log(np.linalg.det(covy))
    chi2 = np.dot(d, np.linalg.solve(covy, d))

    return -chi2 - log_det_cov


if __name__ == '__main__':
    pass
