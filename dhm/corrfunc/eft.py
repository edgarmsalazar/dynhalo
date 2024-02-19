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
    h: float, Om: float, Omb: float, ns: float, sigma8: float, z: float = 0
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
        rCosmological Redshift, by default 0

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

    return pk_lin, pk_zel, cf_lin, cf_zel


def eft_tranform(k) -> SphericalBesselTransform:
    return SphericalBesselTransform(k, L=5, low_ring=True, fourier=True)


def eft_counter_term_corr_func_prediction(klin, plin, cs=4.5) -> Tuple[np.ndarray]:
    cleft = CLEFT(klin, plin)
    cleft.make_ptable(nk=400)

    # 1-loop matter power spectrum
    lptpk = cleft.pktable[:, 1]
    # Counter term contribution
    cterm = cleft.pktable[:, -1]
    # Model evaluation k modes
    kcleft = cleft.pktable[:, 0]
    # Hankel transform object
    sph = eft_tranform(klin)

    # No counter term
    # eftpred_cs0 = loginterp(kcleft, lptpk)(klin)
    # r_eft, xi_eft_cs0 = sph.sph(0, eftpred_cs0)

    # With counter term
    k_factor = kcleft**2 / (1 + kcleft**2)
    eftpred = loginterp(kcleft, lptpk + cs * k_factor * cterm)(klin)
    r_eft, xi_eft = sph.sph(0, eftpred)

    return r_eft, xi_eft[0]


def loglike_cs(cs, data) -> float:
    # Check prior
    if not cs > 0:
        return -np.inf

    k, pk, r, xi, cov = data
    xi_pred = interp1d(*eft_counter_term_corr_func_prediction(k, pk, cs=cs))
    d = xi - xi_pred(r)
    return -np.dot(d, np.linalg.solve(cov, d))


def loglike_B(B, data) -> float:
    if not B > 0:
        return -np.inf

    xi, xi_pred = data
    d = xi - B * xi_pred
    return -np.dot(d, d)


def xi_large_estimation(
    r: np.ndarray,
    xi: np.ndarray,
    xi_cov: np.ndarray,
    h: float,
    Om: float,
    Omb: float,
    ns: float,
    sigma8: float,
    z: float = 0,
    large_only: bool = True,
    power_spectra: bool = False,
    linear: bool = False,
):
    # Compute ZA
    pk_lin_call, xi_lin_call, pk_zel_call, xi_zel_call = (
        zeldovich_approx_corr_func_prediction(
            h=h, Om=Om, Omb=Omb, ns=ns, sigma8=sigma8, z=z
        )
    )
    k_lin = np.logspace(-4, 3, 1000, base=10)
    p_lin = pk_lin_call(k_lin)

    # Find the best value of cs that minimizes the eft prediction error at
    # intermediate scales
    r_mask = (40 < r) & (r < 80)
    args = (k_lin, p_lin, r[r_mask], xi[r_mask], xi_cov[r_mask, :][:, r_mask])
    # Define grids to estimate cs
    ngrid = 20
    grid = np.linspace(0.5 * 4.5, 5 * 4.5, ngrid)
    # cost_ = loglike_cs(4.5, args)
    with Pool(16) as p:
        loglike_grid = p.starmap(loglike_cs, zip(grid, repeat(args, ngrid)))
    cs_max = grid[np.argmax(loglike_grid)]
    ngrid = 100
    grid = np.linspace(cs_max - 0.25, cs_max + 0.25, 100)
    with Pool(16) as p:
        loglike_grid = p.starmap(loglike_cs, zip(grid, repeat(args, ngrid)))
    cs_max = grid[np.argmax(loglike_grid)]

    # Compute the 1-loop EFT approx.
    r_eft, xi_eft = eft_counter_term_corr_func_prediction(k_lin, p_lin, cs=cs_max)

    # Evaluate ZA in the same grid as EFT
    p_zel = pk_zel_call(k_lin)
    xi_lin = xi_lin_call(r_eft)
    xi_zel = xi_zel_call(r_eft)

    # Find the ratio between
    r_mask = (30 < r_eft) & (r_eft < 50)
    B_grid = np.linspace(0.8, 1.2, 100)
    loglike_grid = [loglike_B(b, (xi_eft[r_mask], xi_zel[r_mask])) for b in B_grid]
    B_max = B_grid[np.argmax(loglike_grid)]

    erf_transition = error_func_pos_incr(r_eft, 1.0, 40.0, 3.0)

    xi_large = (1.0 - erf_transition) * B_max * xi_zel + erf_transition * xi_eft

    # Return all quantities
    if large_only:
        return r_eft, xi_large
    else:
        if power_spectra and linear:
            return (k_lin, p_lin, p_zel), (r_eft, xi_lin, xi_eft, xi_zel, xi_large, B_max, cs_max)
        if power_spectra and not linear:
            return (k_lin, p_zel), (r_eft, xi_eft, xi_zel, xi_large, B_max, cs_max)
        elif not power_spectra and linear:
            return r_eft, xi_lin, xi_eft, xi_zel, xi_large, B_max, cs_max
        else:
            return r_eft, xi_eft, xi_zel, xi_large, B_max, cs_max


if __name__ == "__main__":
    pass
