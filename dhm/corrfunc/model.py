import h5py
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import erf as scierf


def error_func_pos_incr(x: float, c0: float, cm: float, cv: float) -> float:
    """Evaluates 

                c0/2 (1 + erf( (x-cm) / \sqrt{2}cv) )

    Parameters
    ----------
    x : float
        Point where to evaluate the error function
    c0 : float
        Normalization
    cm : float
        Offset
    cv : float
        Width

    Returns
    -------
    float

    """
    x = (x - cm) / (np.sqrt(2) * cv)
    return 0.5 * c0 * (1 + scierf(x))


def error_func_pos_decr(x: float, c0: float, cm: float, cv: float) -> float:
    """Evaluates 

                c0/2 (1 - erf( (x-cm) / \sqrt{2}cv) )

    Parameters
    ----------
    x : float
        Point where to evaluate the error function
    c0 : float
        Normalization
    cm : float
        Offset
    cv : float
        Width

    Returns
    -------
    float

    """
    x = (x - cm) / (np.sqrt(2) * cv)
    return 0.5 * c0 * (1 - scierf(x))


def get_xi_large_interp(path: str) -> interp1d:
    """Gets large scale limit correlation funciton prediction from file. 

    Parameters
    ----------
    path : str
        Full path of the xi_large.hdf5 file. It must contain 'r' and 'xi_large'
        datasets.

    Returns
    -------
    interp1d
        Interpolation object which takes 'r' as input.

    Raises
    ------
    FileNotFoundError
        If file is not found at specified path or either 'r' or 'xi_large' are 
        not found inside file.
    """
    try:
        with h5py.File(path, 'r') as hdf:
            r = hdf['r'][()]
            xi_large = hdf['xi_large'][()]
        return interp1d(r, xi_large)
    except:
        raise FileNotFoundError("Either path or dataset name does not exist. "
                                "Make sure 'r' and 'xi_large' are dataset names "
                                "and are not within a group.")


def xi_inf_model(
    r: float,
    r_h: float,
    bias: float,
    xi_large: interp1d,
    eta: float,
    gamma: float,
    r_inf: float,
    mu: float,
) -> float:
    """Infall correlation function model. See equation (12) in Salazar et.al. (2024).
    The equations and figures referenced below are from the same article.

    Parameters
    ----------
    r : float
        Radial points in which the model is evaluated
    r_h : float
        Halo radius from orbiting density profile
    bias : float
        Large scale halo bias
    xi_large : interp1d
        Large scale limit correlation function interpolation object (Eq. 14)
    eta : float
        Amplitude of the dip in denominator (Fig. 6)
    gamma : float
        Power law slope index
    r_inf : float
        Power law scale
    mu : float
        Power law core

    Returns
    -------
    float

    """
    num = 1. + np.power(r_inf / (mu * r_h + r), gamma)
    den = 1. + eta * (r / r_h) * np.exp(-(r / r_h))
    return bias * num / den * xi_large(r)


def power_law(x: float, p: float, s: float) -> float:
    """Power law model with pivot and slope. Evaluates

                p x^s

    Typically x is some scaled value by a pivot reference. E.g. for mass-dependent 
    power laws, x = M/M_pivot, such that p is the value at M=M_pivot.

    Parameters
    ----------
    x : float
        Point where to evaluate the power law
    p : float
        Value at pivot (x=0)
    s : float
        Slope

    Returns
    -------
    float
        _description_
    """
    return p * np.power(x, s)


def rho_orb_dist(x: float, alpha: float, a: float) -> float:
    """Orbiting profile as a distribution.

    Parameters
    ----------
    x : float
        Radial points scaled by rh
    alpha : float
        Slope parameter
    a : float
        Small scale parameter

    Returns
    -------
    float

    """
    alpha *= x / (a + x)
    return np.power(x / a, -alpha) * np.exp(-0.5 * x ** 2)


def rho_orb_dens_dist(r: float, r_h: float, alpha: float, a: float) -> float:
    """Orbiting profile density distribution.

    Parameters
    ----------
    r : float
        Radial points
    r_h : float
        Halo radius
    alpha : float
        Slope parameter
    a : float
        Small scale parameter

    Returns
    -------
    float
        Normalized density distribution
    """
    distr = rho_orb_dist(r/r_h, alpha, a)
    distr /= 4. * np.pi * r_h ** 3 * \
        quad(lambda x, alpha, a: x**2 * rho_orb_dist(x, alpha, a),
             a=0, b=np.inf, args=(alpha, a))[0]
    return distr


def rho_orb_model_with_norm(r: float, log10A: float, r_h: float, alpha: float, a: float) -> float:
    """Orbiting density profile with free normalization constant.

    Parameters
    ----------
    r : float
        Radial points
    log10A : float
        Log 10 value of the normalization constant
    r_h : float
        Halo radius
    alpha : float
        Slope parameter
    a : float
        Small scale parameter

    Returns
    -------
    float
        Orbiting density profile
    """
    return np.power(10., log10A) * rho_orb_dist(x=r/r_h, alpha=alpha, a=a)


def rho_orb_model(r: float, morb: float, r_h: float, alpha: float, a: float) -> float:
    """Orbiting density profile imposing the constraint

                M_{\rm orb} = \int \rho_{\rm rob} dV

    Parameters
    ----------
    r : float
        Radial points in which to 
    morb : float
        Orbiting mass
    r_h : float
        Halo radius
    alpha : float
        Slope parameter
    a : float
        Small scale parameter

    Returns
    -------
    float
        Orbiting density profile
    """
    return morb * rho_orb_dens_dist(r=r, r_h=r_h, alpha=alpha, a=a)


def xihm_model(
    r: float,
    morb: float,
    r_h: float,
    alpha: float,
    a: float,
    bias: float,
    xi_large: interp1d,
    eta: float,
    gamma: float,
    r_inf: float,
    mu: float,
    rhom: float,
) -> float:
    orb = rho_orb_model(r, morb, r_h, alpha, a) / rhom
    inf = xi_inf_model(r, r_h, bias, xi_large, eta, gamma, r_inf, mu)
    return orb + inf


if __name__ == "__main__":
    pass
