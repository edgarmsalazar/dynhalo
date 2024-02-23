from itertools import repeat
from multiprocessing.pool import Pool
from typing import Callable, Tuple
from warnings import filterwarnings

import numpy as np
from nbodykit.lab import cosmology
from scipy.interpolate import interp1d
from velocileptors.LPT.cleft_fftw import CLEFT
from velocileptors.Utils.qfuncfft import loginterp
from ZeNBu.zenbu import SphericalBesselTransform

from dhm.corrfunc.model import error_func_pos_incr

filterwarnings("ignore")


def zeldovich_approx_corr_func_prediction(
    h: float, Om: float, Omb: float, ns: float, sigma8: float, z: float = 0,
    power_spectra: bool = True
) -> Tuple[Callable]:
    """Returns the linear and Zel'dovich approximation power spectra and
    correlation functions.

    Parameters
    ----------
    h : float
        H0 / 100, where H0 is the Hubble parameter.
    Om : float
        Matter density
    Omb : float
        Baryonic matter density
    ns : float
        Spectral index
    sigma8 : float
        _description_
    z : float, optional
        Cosmological Redshift, by default 0
    power_spectra : bool
        If True, returns the linear and ZA power spectra

    Returns
    -------
    Tuple[Callable]
        Power spectra and correlation functions as callables to evaluate on
        arbitrary k and r grids.
    """
    c = cosmology.Cosmology(
        h=h,
        Omega0_b=Omb,
        Omega0_cdm=Om - Omb,
        n_s=ns,
        k_max_tau0_over_L_max=15.0,
    ).match(sigma8=sigma8)

    # Linear and ZA power spectra
    pk_lin = cosmology.LinearPower(c, redshift=z, transfer="CLASS")
    pk_zel = cosmology.ZeldovichPower(c, redshift=z)

    # Correlation functions
    cf_lin = cosmology.CorrelationFunction(pk_lin)
    cf_zel = cosmology.CorrelationFunction(pk_zel)

    if power_spectra:
        return cf_lin, cf_zel, pk_lin, pk_zel
    else:
        return cf_lin, cf_zel

def eft_tranform(k: np.ndarray) -> SphericalBesselTransform:
    """Creates a spherical Bessel transform object to compute fast Fourier 
    transformations.

    Parameters
    ----------
    k : np.ndarray
        Fourier modes.

    Returns
    -------
    SphericalBesselTransform

    """
    return SphericalBesselTransform(k, L=5, low_ring=True, fourier=True)


def eft_counter_term_power_spec_prediction(
    klin: np.ndarray,
    plin: np.ndarray,
    cs: float = 0
) -> Tuple[np.ndarray]:
    """Computes the 1-Loop LPT power spectrum prediction from linear power
      spectrum up to the counter term.

    Parameters
    ----------
    klin : _type_
        Fourier modes.
    plin : _type_
        Linear power spectrum evaluated at each k
    cs : int, optional
        Speed sound for the counter term, by default 0

    Returns
    -------
    Tuple[np.ndarray]
        Correlation function prediction and r
    """
    cleft = CLEFT(klin, plin)
    cleft.make_ptable(nk=400)

    # 1-loop matter power spectrum
    lptpk = cleft.pktable[:, 1]
    # Counter term contribution
    cterm = cleft.pktable[:, -1]
    # Model evaluation k modes
    kcleft = cleft.pktable[:, 0]

    # Add counter term
    if cs != 0:
        k_factor = kcleft**2 / (1 + kcleft**2)
        lptpk += cs * k_factor * cterm

    eft_pred = loginterp(kcleft, lptpk)(klin)
    return eft_pred


def eft_counter_term_corr_func_prediction(
    klin: np.ndarray,
    plin: np.ndarray,
    cs: float = 0
) -> Tuple[np.ndarray]:
    """Computes the 1-Loop LPT correlation function prediction from linear power
      spectrum up to the counter term.

    Parameters
    ----------
    klin : _type_
        Fourier modes.
    plin : _type_
        Linear power spectrum evaluated at each k
    cs : int, optional
        Speed sound for the counter term, by default 0

    Returns
    -------
    Tuple[np.ndarray]
        Correlation function prediction and r
    """
    # Get power spectrum
    eftpred = eft_counter_term_power_spec_prediction(klin, plin, cs=cs)
    # Hankel transform object
    sph = eft_tranform(klin)
    r_eft, xi_eft = sph.sph(0, eftpred)

    return r_eft, xi_eft[0]


def power_spec_box_effect(
    k: np.ndarray,
    pk: np.ndarray,
    boxsize: float,
    lamb: float
) -> np.ndarray:
    """Truncates the power spectrum at small k accounting for the effects of a 
    finite simulation box.

    Parameters
    ----------
    k : np.ndarray
        Fourier modes
    pk : np.ndarray
        Power spectrum evaluated at each k
    boxsize : float
        Simulation box size.
    lamb : float
        Attenuation factor at small k

    Returns
    -------
    np.ndarray
        Attenuated power spectrum such that $P(k \rightarrow 0) = 0$
    """
    rbox = boxsize * np.cbrt(3. / 4. / np.pi)
    phat = (1 - np.exp(-lamb * (rbox * k) ** 2)) * pk
    return phat


def loglike_cs(cs: float, data: Tuple[float]) -> float:
    """Log-likelihood for the sound speed parameter of the counter term.

    Parameters
    ----------
    cs : float
        Sound speed
    data : Tuple[float]
        (k, pk, r, xi, cov)

    Returns
    -------
    float

    """
    # Check prior
    if cs < 0:
        return -np.inf

    # Unpack data
    k, pk, r, xi, cov = data
    xi_pred = interp1d(*eft_counter_term_corr_func_prediction(k, pk, cs=cs))
    # Compute chi2
    d = xi - xi_pred(r)
    return -np.dot(d, np.linalg.solve(cov, d))


def loglike_lamb(lamb: float, data: Tuple[float]) -> float:
    """Log-likelihood for the attenuation parameter in the power spectrum low k 
    limit

    Parameters
    ----------
    lamb : float
        Attenuation factor
    data : Tuple[float]
        (k, pk, r, xi, cov, cs, boxsize)

    Returns
    -------
    float

    """
    # Check priors
    if lamb < 0:
        return -np.inf
    # Unpack data

    k, pk, r, xi, cov, cs, boxsize = data
    # Account for the simulation box size in the linear power spectrum
    phat = power_spec_box_effect(k, pk, boxsize, lamb)
    # Compute chi2
    xi_pred = interp1d(*eft_counter_term_corr_func_prediction(k, phat, cs=cs))
    d = xi - xi_pred(r)
    return -np.dot(d, np.linalg.solve(cov, d))


def loglike_B(B: float, data: Tuple[np.ndarray]) -> float:
    """Log-likelihood for the ratio parameter between ZA and data.

    Parameters
    ----------
    B : float
        Ratio parameter
    data : Tuple[float]
        (xi, xi_pred)

    Returns
    -------
    float

    """
    if B < 0:
        return -np.inf

    xi, xi_pred = data
    d = xi - B * xi_pred
    return -np.dot(d, d)


def find_cs(k_lin, p_lin, r, xi, xi_cov) -> float:
    r_mask = (40 < r) & (r < 80)
    args = (k_lin, p_lin, r[r_mask], xi[r_mask], xi_cov[r_mask, :][:, r_mask])
    # Define grids to estimate cs
    ngrid = 16
    grid = np.logspace(0, 1.2, ngrid)
    with Pool(16) as p:
        loglike_grid = p.starmap(loglike_cs, zip(grid, repeat(args, ngrid)))
    cs_max = grid[np.argmax(loglike_grid)]
    # Refine grid around cs_max with 10% deviation
    ngrid = 80
    grid = np.linspace(0.9*cs_max, 1.1*cs_max, ngrid)
    with Pool(16) as p:
        loglike_grid = p.starmap(loglike_cs, zip(grid, repeat(args, ngrid)))
    cs_max = grid[np.argmax(loglike_grid)]
    return cs_max


def find_lamb(k_lin, p_lin, r, xi, xi_cov, cs_max, boxsize) -> float:
    r_mask = (40 < r) & (r < 150)
    args = (k_lin, p_lin, r[r_mask], xi[r_mask], xi_cov[r_mask, :][:, r_mask],
            cs_max, boxsize)
    # Define grids to estimate cs
    ngrid = 16
    grid = np.logspace(-1.5, 1, ngrid)
    with Pool(16) as p:
        loglike_grid = p.starmap(loglike_lamb, zip(grid, repeat(args, ngrid)))
    lamb_max = grid[np.argmax(loglike_grid)]
    # Refine grid around cs_max with 10% deviation below and 50% above
    ngrid = 80
    grid = np.linspace(0.9*lamb_max, 1.5*lamb_max, ngrid)
    with Pool(16) as p:
        loglike_grid = p.starmap(loglike_lamb, zip(grid, repeat(args, ngrid)))
    lamb_max = grid[np.argmax(loglike_grid)]
    return lamb_max


def find_B(xi_data, xi_zel) -> float:
    B_grid = np.linspace(0.8, 1.2, 10_000)
    loglike_grid = [loglike_B(b, (xi_data, xi_zel))
                    for b in B_grid]
    B_max = B_grid[np.argmax(loglike_grid)]
    return B_max


def xi_large_construct(
    r: np.ndarray,
    xi_zel: np.ndarray,
    xi_eft: np.ndarray,
    b: float,
    fmu: float = 40.,
    fsig: float = 3.,
) -> np.ndarray:
    erf_transition = error_func_pos_incr(r, 1.0, fmu, fsig)
    xi_large = (1.0 - erf_transition) * b * xi_zel + \
        erf_transition * xi_eft
    return xi_large


def xi_large_estimation_from_data(
    r: np.ndarray,
    xi: np.ndarray,
    xi_cov: np.ndarray,
    h: float,
    Om: float,
    Omb: float,
    ns: float,
    sigma8: float,
    boxsize: float,
    z: float = 0,
    large_only: bool = True,
    power_spectra: bool = False,
) -> Tuple[np.ndarray]:
    """Computes the large scale limit correlation function using 1-Loop LPT.

    Parameters
    ----------
    r : np.ndarray
        Radial points of the measured correlation function
    xi : np.ndarray
        Measured correlation function from a simulation box
    xi_cov : np.ndarray
        Covariance matrix of the measured correlation function
    h : float
        H0 / 100, where H0 is the Hubble parameter.
    Om : float
        Matter density
    Omb : float
        Baryonic matter density
    ns : float
        Spectral index
    sigma8 : float
        _description_
    boxsize : float
        Length of one side of the simulation box.
    z : float, optional
        Cosmological Redshift, by default 0
    large_only : bool, optional
        if True, only returns `xi_large`, by default True
    power_spectra : bool, optional
        If True, also returns the power spectra, by default False

    Returns
    -------
    Tuple[np.ndarray]

    """
    # Compute ZA
    xi_lin_call, xi_zel_call, pk_lin_call, pk_zel_call = (
        zeldovich_approx_corr_func_prediction(
            h=h, Om=Om, Omb=Omb, ns=ns, sigma8=sigma8, z=z
        )
    )
    k_lin = np.logspace(-4, 3, 1000, base=10)
    p_lin = pk_lin_call(k_lin)

    # Find the best value of cs that minimizes the eft prediction error at
    # intermediate scales
    cs_max = find_cs(k_lin, p_lin, r, xi, xi_cov)

    # Account for the box size effects on large scale Fourier modes
    lamb_max = find_lamb(k_lin, p_lin, r, xi, xi_cov, cs_max, boxsize)

    # Compute the 1-loop EFT approx.
    p_hat = power_spec_box_effect(k_lin, p_lin, boxsize, lamb_max)
    p_eft = eft_counter_term_power_spec_prediction(k_lin, p_hat, cs=cs_max)
    r_eft, xi_eft = eft_counter_term_corr_func_prediction(
        k_lin, p_hat, cs=cs_max)

    # Evaluate ZA in the same grid as EFT
    p_zel = pk_zel_call(k_lin)
    xi_lin = xi_lin_call(r_eft)
    xi_zel = xi_zel_call(r_eft)

    # Find the ratio between EFT and ZA
    r_mask = (30 < r) & (r < 50)
    B_max = find_B(xi[r_mask], interp1d(r_eft, xi_zel)(r[r_mask]))

    # Construct xi large
    xi_large = xi_large_construct(r_eft, xi_zel, xi_eft, B_max)

    power_spectra = (k_lin, p_lin, p_hat, p_eft, p_zel)
    corr_func = (r_eft, xi_lin, xi_eft, xi_zel,
                 xi_large, B_max, cs_max, lamb_max)
    # Return all quantities
    if large_only:
        return r_eft, xi_large
    else:
        if power_spectra:
            return power_spectra, corr_func
        else:
            return corr_func


if __name__ == "__main__":
    pass
