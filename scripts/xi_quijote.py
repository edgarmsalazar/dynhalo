import os

import h5py
import numpy as np

from dhm.corrfunc.eft import xi_large_estimation_from_data
from dhm.utils import timer


def load_quijote_cf(
    path: str = '/spiff/edgarmsc/simulations/Quijote/CF/matter/fiducial/all/',
):
    # Find best cs by fitting xi_EFT to xi_mm from Quijote
    file_name = os.getcwd() + '/scripts/data/xi_mm_quijote.hdf5'
    if not os.path.exists(file_name):
        n_obs = 15_000

        # Load data
        r_obs, _ = np.loadtxt(path + 'CF_m_z=0_0.txt', unpack=True)
        xi_obs = np.zeros((n_obs, len(r_obs)))
        for i in range(n_obs):
            _, xi_obs[i, :] = np.loadtxt(
                path + f'CF_m_z=0_{i}.txt', unpack=True)

        xi_obs_cov = np.cov(xi_obs, rowvar=False, bias=False) / n_obs
        xi_obs_mean = np.mean(xi_obs, axis=0)
        with h5py.File(file_name, 'w') as hsave:
            r_obs = hsave.create_dataset('r', data=r_obs)
            xi_obs_mean = hsave.create_dataset('xi', data=xi_obs_mean)
            xi_obs_cov = hsave.create_dataset('cov', data=xi_obs_cov)
    else:
        with h5py.File(file_name, 'r') as hload:
            r_obs = hload['r'][()]
            xi_obs_mean = hload['xi'][()]
            xi_obs_cov = hload['cov'][()]

    return r_obs, xi_obs_mean, xi_obs_cov


@timer
def xi_large_quijote() -> None:
    r_obs, xi_obs_mean, xi_obs_cov = load_quijote_cf()
    # Takes ~10 min
    power_spectra, corr_func = xi_large_estimation_from_data(
            r=r_obs,
            xi=xi_obs_mean,
            xi_cov=xi_obs_cov,
            h=0.6711,
            Om=0.3175,
            Omb=0.049,
            ns=0.9624,
            sigma8=0.834,
            boxsize=1000,
            z=0,
            large_only=False,
            power_spectra=True,
        )
    k_lin, p_lin, p_hat, p_eft, p_zel = power_spectra
    r_eft, xi_lin, xi_eft, xi_zel, xi_large, B_max, cs_max, lamb_max = corr_func
    
    with h5py.File(os.getcwd() + '/scripts/data/xi_large_quijote.hdf5', 'w') as hdf_save:
        hdf_save.create_dataset('k', data=k_lin)
        hdf_save.create_dataset('p_lin', data=p_lin)
        hdf_save.create_dataset('p_lin_hat', data=p_hat)
        hdf_save.create_dataset('p_eft', data=p_eft)
        hdf_save.create_dataset('p_zel', data=p_zel)

        hdf_save.create_dataset('r', data=r_eft)
        hdf_save.create_dataset('xi_lin', data=xi_lin)
        hdf_save.create_dataset('xi_eft', data=xi_eft)
        hdf_save.create_dataset('xi_zel', data=xi_zel)
        hdf_save.create_dataset('xi_large', data=xi_large)
        hdf_save.create_dataset('B', data=B_max)
        hdf_save.create_dataset('cs', data=cs_max)
        hdf_save.create_dataset('lambda', data=lamb_max)

    return None


if __name__ == '__main__':
    xi_large_quijote()
