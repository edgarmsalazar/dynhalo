from typing import Tuple

import numpy as np


def relative_coordinates(
    x0: np.ndarray,
    x: np.ndarray,
    boxsize: float,
    periodic: bool = True
) -> float:
    """Returns the coordinates x relative to x0 accounting for periodic boundary
    conditions

    Parameters
    ----------
    x0 : np.ndarray
        Reference position in cartesian coordinates
    x : np.ndarray
        Position array (N, 3)
    boxsize : float
        Size of simulation box
    periodic : bool, optional
        Set to True if the simulation box is periodic, by default True

    Returns
    -------
    float
        Relative positions
    """
    if periodic:
        return (x - x0 + 0.5*boxsize) % boxsize - 0.5*boxsize
    return x - x0


def get_vr_vt_from_coordinates(
    pos: np.ndarray,
    vel: np.ndarray,
) -> Tuple[np.ndarray]:
    """Computes the radial and tangential velocites from cartesian/rectangular 
    coordinates.

    Parameters
    ----------
    pos : np.ndarray
        Cartesian coordinates
    vel : np.ndarray
        Cartesian velocities

    Returns
    -------
    Tuple[np.ndarray]
        Radial velocity, tangential velocity and magnitude squared of the 
        velocity
    """
    # Transform coordinates from cartesian to spherical
    #   rs = sqrt( x**2 + y**2 + z**2 )
    rps = np.sqrt(np.sum(np.square(pos), axis=1))
    #   theta = arccos( z / rs )
    thetas = np.arccos(pos[:, 2] / rps)
    #   phi = arctan( y / x)
    phis = np.arctan2(pos[:, 1], pos[:, 0])

    # Get radial vector in cartesian coordinates
    rp_hat = np.zeros_like(pos)
    rp_hat[:, 0] = np.sin(thetas) * np.cos(phis)
    rp_hat[:, 1] = np.sin(thetas) * np.sin(phis)
    rp_hat[:, 2] = np.cos(thetas)

    # Get tangential vector in cartesian coordinates
    rt_hat = np.zeros_like(pos)
    rt_hat[:, 0] = np.cos(thetas) * np.cos(phis)
    rt_hat[:, 1] = np.cos(thetas) * np.sin(phis)
    rt_hat[:, 2] = -np.sin(thetas)

    # Compute radial velocity as v dot r_hat
    vr = np.sum(vel * rp_hat, axis=1)
    vt = np.sum(vel * rt_hat, axis=1)
    # Velocity squared
    v2 = np.sum(np.square(vel), axis=1)

    return vr, vt, v2


# Deprecated
def distance_metric(
    rel_pos: np.ndarray,
    rel_vel: np.ndarray,
    sigma_x_sq: float,
    sigma_v_sq: float,
    lamb: float = 1.0,
) -> float:
    """Computes the distance between particles and halo centre in
    six-dimensional phase space.

    Parameters
    ----------
    rel_pos : np.ndarray
        Relative position of paticles to halo
    rel_vel : np.ndarray
        Relative velocity of paticles to halo
    sigma_x_sq : float
        Position dispersion parameter 
    sigma_v_sq : float
        Velocity dispersion parameter 
    lamb : float, optional
        Scaling parameter in distance metric, by default 1.0

    Returns
    -------
    float
        Six-dimensional distance
    """
    return np.sum(np.square(rel_pos), axis=1) / sigma_x_sq + \
            lamb * np.sum(np.square(rel_vel), axis=1) / sigma_v_sq


if __name__ == "__main__":
    pass
