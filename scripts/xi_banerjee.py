import os

import h5py
import numpy as np

from dhm.corrfunc.bins import generate_bins
from dhm.corrfunc.eft import xi_large_estimation_from_data
from dhm.corrfunc.tpcf import cross_tpcf_jk
from dhm.utils import timer


def matter_auto_corr(pos: np.ndarray, path: str):
    r_bins, r_edges = generate_bins(0.015, 50., 30, soft=0.015)

    xi, xi_samples, xi_mean, xi_cov = cross_tpcf_jk(
        data_1=pos,
        data_2=pos,
        radial_edges=r_edges,
        boxsize=1000.,
        gridsize=250.,
        nthreads=16
    )
    with h5py.File(path, 'w') as hdf:
        hdf.create_dataset('xi', data=xi)
        hdf.create_dataset('xi_samples', data=xi_samples)
        hdf.create_dataset('xi_mean', data=xi_mean)
        hdf.create_dataset('xi_cov', data=xi_cov)
        hdf.create_dataset('r_edges', data=r_edges)
        hdf.create_dataset('r_bins', data=r_bins)

    return


@timer
def compute_matter_auto_corr():
    ds = 100
    with h5py.File('/home/edgarmsc/spiff_edgar/simulations/Banerjee/particle_catalogue.h5', 'r') as hdf:
        p_coord = np.vstack(
            [
                hdf[f"snap99/{ds}/x"][()],
                hdf[f"snap99/{ds}/y"][()],
                hdf[f"snap99/{ds}/z"][()],
            ]
        ).T

    matter_auto_corr(p_coord, os.getcwd() +
                     '/scripts/data/xi_mm_banerjee.hdf5')
    return


@timer
def xi_large_banerjee() -> None:
    with h5py.File(os.getcwd() + '/scripts/data/xi_mm_banerjee.hdf5', "r") as hdf:
        xi_obs_mean = hdf['xi'][()]
        xi_obs_cov = hdf['xi_cov'][()]
        r_obs = hdf['r_bins'][()]

    # Takes ~10 min
    power_spectra, corr_func = xi_large_estimation_from_data(
        r=r_obs,
        xi=xi_obs_mean,
        xi_cov=xi_obs_cov,
        h=0.7,
        Om=0.3,
        Omb=0.0469,
        ns=1.0,
        sigma8=0.8355,
        boxsize=1000,
        z=0,
        large_only=False,
        power_spectra=True,
    )
    k_lin, p_lin, p_hat, p_eft, p_zel = power_spectra
    r_eft, xi_lin, xi_eft, xi_zel, xi_large, B_max, cs_max, lamb_max = corr_func

    with h5py.File(os.getcwd() + '/scripts/data/xi_large_banerjee.hdf5', 'w') as hdf_save:
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


if __name__ == "__main__":
    compute_matter_auto_corr()
    xi_large_banerjee()
