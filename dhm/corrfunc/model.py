import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.special import erf as scierf


def error_func_pos_incr(x: float, c0: float, cm: float, cv: float) -> float:
    x = (x - cm) / (np.sqrt(2) * cv)
    return 0.5 * c0 * (1 + scierf(x))


def error_func_pos_decr(x: float, c0: float, cm: float, cv: float) -> float:
    x = (x - cm) / (np.sqrt(2) * cv)
    return 0.5 * c0 * (1 - scierf(x))


def xi_large_construct(path: str, sim_name: str):
    sim_name = sim_name.lower()
    if sim_name not in ['quijote', 'banerjee']:
        raise AttributeError(f'Simulation {sim_name} not supported')
    
    if sim_name == 'quijote':
        with h5py.File(path, 'r') as hdf:
            r = hdf['r'][()]
            xi_large = hdf['xi_large'][()]
    elif sim_name == 'banerjee':
        with h5py.File(path, 'r') as hdf:
            r = hdf['r'][()]

    return r, xi_large


def xi_inf_model(
    r: float,
    r_inf: float,
    mu: float,
    gamma: float,
    eta: float,
    bias: float,
    r_h: float,
    xi_large: interp1d,
) -> np.ndarray:
    
    num = 1. + np.power(r_inf / (mu * r_h + r), gamma)
    den = 1. + eta * (r / r_h) * np.exp(-(r / r_h))
    return bias * num / den * xi_large(r)


if __name__ == "__main__":
    pass
