import numpy as np
from typing import Tuple, Callable
from nbodykit.lab import cosmology
from velocileptors.LPT.cleft_fftw import CLEFT
from velocileptors.Utils.qfuncfft import loginterp
from ZeNBu.zenbu import SphericalBesselTransform

__all__ = [
    "zeldovich_approx_corr_func_prediction",
    "eft_counter_term_corr_func_prediction",
]


def zeldovich_approx_corr_func_prediction(
    h: float, Om: float, Omb: float, ns: float, sigma8: float, z: float = 0, **kwargs
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
        _description_, by default 0

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
        **kwargs
    ).match(sigma8=sigma8)

    # Linear and ZA power spectra
    pk_lin = cosmology.LinearPower(c, redshift=z, transfer="CLASS")
    pk_zel = cosmology.ZeldovichPower(c, redshift=z)

    # Correlation functions
    cf_lin = cosmology.CorrelationFunction(pk_lin)
    cf_zel = cosmology.CorrelationFunction(pk_zel)

    return pk_lin, pk_zel, cf_lin, cf_zel


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
    sph = SphericalBesselTransform(klin, L=5, low_ring=True, fourier=True)
    
    # With counter term
    k_factor =  kcleft**2 / (1 + kcleft ** 2)
    eftpred = loginterp(kcleft, lptpk + cs * k_factor * cterm)(klin)
    r_eft, xi_eft = sph.sph(0, eftpred)
    
    return


if __name__ == "__main__":
    pass
