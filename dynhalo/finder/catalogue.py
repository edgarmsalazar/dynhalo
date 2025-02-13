import os
from functools import partial
from multiprocessing import Pool
from typing import Any, List, Tuple, Union
from warnings import filterwarnings

import h5py as h5
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from tqdm import tqdm

from dynhalo.finder.coordinates import (characteristic_density,
                                        relative_coordinates)
from dynhalo.finder.minibox import load_particles, load_seeds
from dynhalo.utils import G_gravity, timer

filterwarnings('ignore')


def rho_nfw_roots(
    x: float,
    delta1: float,
    delta2: float,
    rs1: float,
    rs2: float,
    r12: float,
) -> float:
    """Returns the value of 

    \begin{equation*}
        \frac{\rho_1(R-r)}{\rho_{c}} &= \frac{\rho_2(r)}{\rho_{c}}
    \end{equation*}
    
    where $\rho(r)$ is the NFW profile.

    \begin{equation*}
        \frac{\rho(r)}{\rho_{c}} = 
            \frac{\delta_c}{\frac{r}{R_s}\left(1+\frac{r}{R_s}\right)^2}
    \end{equation*}

    Parameters
    ----------
    x : float
        Radial coordinate
    delta1 : float
        Characteristic density of the central object
    delta2 : float
        Characteristic density of the substructure
    rs1 : float
        Scale radius of the central object
    rs2 : float
        Scale radius of the substructure
    r12 : float
        Radial separation between central and substructure R=|x2-x1|.

    Returns
    -------
    float
        
    """
    x1 = (r12 - x) / rs1
    x2 = x / rs2
    frac1 = delta1 / (x1 * (1 + x1)**2)
    frac2 = delta2 / (x2 * (1 + x2)**2)
    return frac1 - frac2


def classify(
    rel_pos: np.ndarray,
    rel_vel: np.ndarray,
    r200: float,
    m200: float,
    class_pars: Union[List, Tuple, np.ndarray],
    max_radius: float = 2.0
) -> np.ndarray:
    """Classifies particles as orbiting.

    Parameters
    ----------
    rel_pos : np.ndarray
        Relative position of particles around seed
    rel_vel : np.ndarray
        Relative velocity of particles around seed
    r200 : float
        Seed R200
    m200 : float
        Seed M200
    class_pars : Union[List, Tuple, np.ndarray]
        Classification parameters [m_pos, b_pos, m_neg, b_neg]
    max_radius : float
        Maximum radius where orbiting particles can be found. All particles 
        above this value are set to be infalling. By default 2.0.

    Returns
    -------
    np.ndarray
        A boolean array where True == orbiting
    """
    m_pos, b_pos, m_neg, b_neg = class_pars
    # Compute V200
    v200 = G_gravity * m200 / r200

    # Compute the radius to seed_i in r200 units, and ln(v^2) in v200 units
    part_ln_vel = np.log(np.sum(np.square(rel_vel), axis=1) / v200)
    part_radius = np.sqrt(np.sum(np.square(rel_pos), axis=1)) / r200

    # Create a mask for particles with positive radial velocity
    mask_vr_positive = np.sum(rel_vel * rel_pos, axis=1) > 0

    # Orbiting classification for vr > 0
    mask_cut_pos = part_ln_vel < (m_pos * part_radius + b_pos)

    # Orbiting classification for vr < 0
    mask_cut_neg = part_ln_vel < (m_neg * part_radius + b_neg)

    # Particle is infalling if it is below both lines and 2*R00
    mask_orb = (part_radius <= max_radius) & (
        (mask_cut_pos & mask_vr_positive) ^ \
        (mask_cut_neg & ~mask_vr_positive)
    )

    return mask_orb


def classify_single_mini_box(
    mini_box_id: int,
    min_num_part: int,
    boxsize: float,
    minisize: float,
    load_path: str,
    dir_name: str,
    padding: float = 5.0,
    disable_tqdm: bool = True,
) -> None:
    """Runs the classifier for each seed in a mini box...

    Parameters
    ----------
    mini_box_id : int
        Sub-box ID
    min_num_part : int
        Minimum number of particles needed to be considered a halo
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
    load_path : str
        Location from where to load the file
    dir_name : str
        Label for the current run. The directory created will be `run_dir_name`.
    padding : float
        Only particles up to this distance from the mini box edge are considered 
        for classification. Defaults to 5

    Returns
    -------
    None
    """
    save_path = load_path + f'run_{dir_name}/mini_box_catalogues/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load seeds in mini box
    pos_seed_mb, vel_seed_mb, hid_seed_mb, r200b_mb, m200b_mb, rs_mb = \
        load_seeds(mini_box_id, boxsize, minisize, load_path, padding)
    n_seeds = len(hid_seed_mb)

    # Exit if there are no seeds in the mini box.
    if any([p is None for p in (pos_seed_mb, vel_seed_mb, hid_seed_mb)]):
        return None

    # Load adjacent seeds
    pos_seed, vel_seed, hid_seed, r200b_seed, m200b_seed, rs_seed = \
        load_seeds(mini_box_id, boxsize, minisize, load_path, padding, 
                   adjacent=True)

    # Concatenate all seeds for ease of selection.
    hid_seed = np.hstack([hid_seed_mb, hid_seed])
    pos_seed = np.vstack([pos_seed_mb, pos_seed])
    vel_seed = np.vstack([vel_seed_mb, vel_seed])
    r200b_seed = np.hstack([r200b_mb, r200b_seed])
    m200b_seed = np.hstack([m200b_mb, m200b_seed])
    rs_seed = np.hstack([rs_mb, rs_seed])

    # Concentration parameter and characteristic density
    deltac_mb = characteristic_density(r200b_mb, rs_mb)
    deltac_seed = characteristic_density(r200b_seed, rs_seed)

    # Load particles
    pos_part, vel_part, pid_part = \
        load_particles(mini_box_id, boxsize, minisize, load_path, padding)
    
    # Load calibration parameters
    with h5.File(load_path + 'calibration_pars.hdf5', 'r') as hdf:
        pars = (*hdf['pos'][()], *hdf['neg'][()])

    col_names = ('Orig_halo_ID', 'Norb', 'LIDX', 'UIDX', 'SLIDX', 'SUIDX')
    haloes = pd.DataFrame(columns=col_names)
    
    pid_out, subhaloes = [[] for _ in range(2)]
    n_tot = 0
    n_tot_sub = 0

    for i in tqdm(range(n_seeds), ncols=100, desc='Finding haloes', 
                  colour='green', disable=disable_tqdm):
        # ======================================================================
        #                           Classify particles
        # ======================================================================
        rel_pos = relative_coordinates(pos_seed_mb[i], pos_part, boxsize)
        rel_vel = vel_part - vel_seed_mb[i]

        # Only work with particles within a 2*R200b cube box to speedup
        # computations. Probably a better way to create this mask?
        r_max = 2.0 * r200b_mb[i]
        box3d = (np.abs(rel_pos[:, 0]) <= r_max) & \
                (np.abs(rel_pos[:, 1]) <= r_max) & \
                (np.abs(rel_pos[:, 2]) <= r_max)

        rel_pos_box3d = rel_pos[box3d]
        rel_vel_box3d = rel_vel[box3d]

        # Classify
        mask_orb = classify(rel_pos_box3d, rel_vel_box3d, r200b_mb[i], 
                            m200b_mb[i], pars)

        # Ignore seed if it does not have the minimum mass to be considered a
        # halo. Early exit to avoid further computation for a non-halo seed
        is_halo = mask_orb.sum() >= min_num_part
        if not is_halo:
            continue
        
        # ======================================================================
        #                           Classify seeds
        # ======================================================================
        # Seeds inherit the classification from the bulk of particles within a 
        # 6D ball around it. If the fraction of orbiting particles within the 
        # ball is greater than 50%, the seed is orbtiing. The seed is infalling
        # otherwise, and all orbiting particles within the 6D ball are also 
        # infalling.
        # ======================================================================
        # Ignore current seed.
        mask_self = hid_seed != hid_seed_mb[i]

        rel_pos = relative_coordinates(pos_seed_mb[i], pos_seed, boxsize)
        rel_vel = vel_seed[mask_self] - vel_seed_mb[i]
        
        # Only work with seeds within a 2*R200b cube sphere. 
        box3d_seed = mask_self & \
            (np.sum(np.square(rel_pos), axis=1) <= r_max**2)

        # If there are seeds in the vicinity
        orb_hid_seed = []
        slidx = -1
        suidx = -1
        if box3d_seed.sum() > 0:
            pos_seed_box3d = pos_seed[box3d_seed]
            vel_seed_box3d = vel_seed[box3d_seed]
            deltac_seed_box3d = deltac_seed[box3d_seed]
            rs_seed_box3d = rs_seed[box3d_seed]

            j = 0 
            while is_halo and (j < box3d_seed.sum()):
                # Select particles around jth seed.
                rel_pos_part = relative_coordinates(pos_seed_box3d[j],
                                                    pos_part[box3d], boxsize)
                rel_vel_part = vel_part[box3d] - vel_seed_box3d[j]
                rps = np.sum(np.square(rel_pos_part), axis=1)
                vps = np.sum(np.square(rel_vel_part), axis=1)

                # Distance from the current seed to the substructure.
                r_dist = np.linalg.norm(rel_pos[box3d_seed][j])
                # Distance from the substructure where the NFW density of both
                # objects is equal. Defines the search radius of the 6D ball.
                r_ball = fsolve(
                    func=rho_nfw_roots, 
                    x0=r_dist/2, 
                    args=(
                        deltac_mb[i],
                        deltac_seed_box3d[j],
                        rs_mb[i],
                        rs_seed_box3d[j],
                        r_dist
                    )
                )

                # V200 of the jth seed.
                v200bsq_seed = G_gravity * m200b_seed[box3d_seed][j] / \
                    r200b_seed[box3d_seed][j]
                v_fac = 2
                ball6d = (rps <= r_ball**2) & (vps <= (v_fac**2)*v200bsq_seed)
                
                # Check the fraction of orbiting particles in the 6D ball
                frac_inside = (ball6d * mask_orb).sum() / ball6d.sum()

                # If more than half the particles in the vicinity of the seed 
                # are orbiting, the seed is tagged as orbiting.
                if frac_inside > 0.5:
                    orb_hid_seed.append(hid_seed[box3d_seed][j])
                    mask_orb[ball6d] = True
                # The seed is infalling otherwise and all the particles within 
                # the box are tagged as infalling too.
                else:
                    mask_orb[ball6d] = False

                # Check wether seed is still a halo.
                is_halo = mask_orb.sum() >= min_num_part
                
                # Next item.
                j += 1
        
        if is_halo:
            n_orb_subs = len(orb_hid_seed)
            if n_orb_subs > 0:
                # If there are orbiting seeds, append them to the members list.
                subhaloes.append(orb_hid_seed)
                
                # Save subhalo indices
                slidx = n_tot_sub
                suidx = n_tot_sub + n_orb_subs
                n_tot_sub += n_orb_subs

            # Append halo to catalogue =========================================
            haloes.loc[len(haloes.index)] = [
                hid_seed_mb[i],
                mask_orb.sum(),
                n_tot,
                n_tot + mask_orb.sum(),
                slidx,
                suidx
            ]

            n_tot += mask_orb.sum()
            
            # Save particles
            pid_out.append(pid_part[box3d][mask_orb])
            
    # ==========================================================================
    #                           Save catalogue
    # ==========================================================================
    # Exit if no haloes were found in this mini box.
    if len(haloes.index) < 1:
        return None
    
    # Save into file
    with h5.File(save_path + f'{mini_box_id}.hdf5', 'w') as hdf:
        # Halo catalogue
        for i, key in enumerate(haloes.columns):
            data = haloes[key].values
            hdf.create_dataset(f'halo/{key}', data=data)

        # Particles
        hdf.create_dataset('part/PID', data=np.concatenate(pid_out), 
                           dtype=np.dtype(pid_part[0]))
            
        # Halo members: seeds
        if len(subhaloes) > 0:
            hdf.create_dataset('subs/Orig_halo_ID', 
                               data=np.concatenate(subhaloes))

    return None


@timer
def classify_all_mini_boxes(
    load_path: str,
    dir_name: str,
    min_num_part: int,
    boxsize: float,
    minisize: float,
    padding: float,
    n_threads: int = None,
) -> None:
    """Generates a halo catalogue using the kinetic mass criterion to classify
    particles into orbiting or infalling.

    Parameters
    ----------
    load_path : str
        Location from where to load the file
    min_num_part : int
        Minimum number of particles needed to be considered a halo
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
    dir_name : str
        Label for the current run. The directory created will be `run_dir_name`.
    padding : float, optional
        Only particles up to this distance from the mini box edge are considered 
        for classification. Defaults to 5
    n_threads : int
        Number of threads

    Returns
    -------
    None
    """
    # Create directory if it does not exist
    save_path = load_path + f'run_{dir_name}/mini_box_catalogues/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Number of miniboxes
    n_mini_boxes = np.int_(np.ceil(boxsize / minisize))**3

    # Parallel processing of miniboxes.
    func = partial(classify_single_mini_box, min_num_part=min_num_part,
                   boxsize=boxsize, load_path=load_path, minisize=minisize, 
                   padding=padding, dir_name=dir_name, disable_tqdm=True)
    with Pool(n_threads) as pool:
        list(tqdm(pool.imap(func, range(n_mini_boxes)),
                  total=n_mini_boxes, colour="green", ncols=100,
                  desc='Generating halo catalogue'))

    # Consolidate catalogue
    ohid, norb, lidx, uidx, slidx, suidx, subs = ([] for _ in range(7))
    n_part = 0
    n_subs = 0
    first_file = True
    
    # Saved datasets
    with h5.File(save_path + '0.hdf5', 'r') as hdf:
        keys_part = list(hdf['part'].keys())

    hdf_part = h5.File(load_path + f'run_{dir_name}/temp_members.hdf5', 'w')
    for i in tqdm(range(n_mini_boxes), ncols=100, desc='Merging catalogues', 
                  colour='green'):
        with h5.File(save_path + f'{i}.hdf5', 'r') as hdf_load:
            if 'halo' not in hdf_load.keys():
                continue
            # Number of particles in current file
            n_part_this = hdf_load['part/PID'].shape[0]
            n_subs_this = hdf_load['subs/Orig_halo_ID'].shape[0]

            # This reshaping of the dataset after every new file...
            if first_file:  # Create the dataset at first pass.
                for key in keys_part:
                    hdf_part.create_dataset(name=f'part/{key}',
                                            chunks=True, maxshape=(None,),
                                            data=hdf_load[f'part/{key}'][()])
                first_file = False
            else:
                # Number of particles so far plus this file's total.
                new_shape = n_part + n_part_this
                for key in keys_part:
                    name = f'part/{key}'
                    # Resize axes
                    hdf_part[name].resize((new_shape), axis=0)
                    # Save incoming data
                    hdf_part[name][n_part:] = hdf_load[name][()]
            
            # Subs. Only shift indices for haloes with subhaloes
            temp_l = hdf_load['halo/SLIDX'][()]
            temp_l[temp_l != -1] += n_subs
            temp_u = hdf_load['halo/SUIDX'][()]
            temp_u[temp_u != -1] += n_subs

            # Halo data
            norb.append(hdf_load['halo/Norb'][()])
            ohid.append(hdf_load['halo/Orig_halo_ID'][()])
            lidx.append(hdf_load['halo/LIDX'][()] + n_part)
            uidx.append(hdf_load['halo/UIDX'][()] + n_part)
            subs.append(hdf_load['subs/Orig_halo_ID'][()])
            slidx.append(temp_l)
            suidx.append(temp_u)

            # Add the total number of particles in this file to the next.
            n_part += n_part_this
            n_subs += n_subs_this

    hdf_part.close()

    with h5.File(load_path + f'run_{dir_name}/temp_members.hdf5', 'a') as hdf:
        hdf.create_dataset('subs/Orig_halo_ID', data=np.concatenate(subs))

    with h5.File(load_path + f'run_{dir_name}/temp_catalogue.hdf5', 'w') as hdf:
        hdf.create_dataset('Orig_halo_ID', data=np.concatenate(ohid))
        hdf.create_dataset('Norb', data=np.concatenate(norb))
        hdf.create_dataset('LIDX', data=np.concatenate(lidx))
        hdf.create_dataset('UIDX', data=np.concatenate(uidx))
        hdf.create_dataset('SLIDX', data=np.concatenate(slidx))
        hdf.create_dataset('SUIDX', data=np.concatenate(suidx))

    return None


@timer
def percolate_haloes(
    file_seeds: str,
    save_path: str,
    dir_name: str,
) -> None:
    """_summary_

    Parameters
    ----------
    file_seeds : str
        _description_
    save_path : str
        _description_
    dir_name : str
        _description_
    """
    # ==========================================================================
    # Load M200b of new haloes from the seed catalogue.
    # ==========================================================================
    with h5.File(file_seeds, 'r') as hdf:
        ohids_m200b = hdf['Orig_halo_ID'][()]
        m200b = hdf['M200b'][()]
    # Sort by HID
    ohids_order = np.argsort(ohids_m200b)
    ohids_m200b = ohids_m200b[ohids_order]
    m200b = m200b[ohids_order]

    # Load parent candidates
    with h5.File(save_path + f'run_{dir_name}/temp_catalogue.hdf5', 'r') as hdf:
        ohids_morb = hdf['Orig_halo_ID'][()]
        slidx = hdf['SLIDX'][()]
        suidx = hdf['SUIDX'][()]
    
    # Sort by HID
    ohids_order = np.argsort(ohids_morb)
    ohids_morb = ohids_morb[ohids_order]
    slidx = slidx[ohids_order]
    suidx = suidx[ohids_order]

    # Cross-match catalogues making them element-wise compatible.
    mask_match_catalogues = np.isin(ohids_m200b, ohids_morb, assume_unique=True)
    m200b = m200b[mask_match_catalogues]
    
    # ==========================================================================
    # Prepare subhaloes for percolation.
    # ==========================================================================
    # Load subhalo IDs. Convert to signed integers.
    with h5.File(save_path + f'run_{dir_name}/temp_members.hdf5', 'r') as hdf:
        ohids_subs = hdf['subs/Orig_halo_ID'][()].astype(np.int64)

    # Rank order haloes by mass (Rockstar)
    mass_order = np.argsort(m200b)[::-1]
    ohids = ohids_morb[mass_order]
    m200b = m200b[mass_order]
    slidx = slidx[mass_order]
    suidx = suidx[mass_order]

    # Select all haloes with members. SIDX=-1 if there are no subhaloes.
    mask_with_subs = (slidx != -1) & (suidx != -1)
    ohids_with_subs = ohids[mask_with_subs]
    m200b_with_subs = m200b[mask_with_subs]
    slidx_with_subs = slidx[mask_with_subs]
    suidx_with_subs = suidx[mask_with_subs]

    # Populate arrays with parent halo ids and parent mass for each subhalo.
    parent_ohid = np.zeros_like(ohids_subs)
    parent_mass = np.zeros(ohids_subs.shape, dtype=np.dtype(m200b[0]))

    for i in tqdm(range(len(ohids_with_subs)), ncols=100, colour='blue',
                    desc='Populating parent mass'):
        parent_ohid[slidx_with_subs[i]:suidx_with_subs[i]] = ohids_with_subs[i]
        parent_mass[slidx_with_subs[i]:suidx_with_subs[i]] = m200b_with_subs[i]

    # ==========================================================================
    #                                   Step 1
    # 
    #   Objects may only orbit more massive structures.
    # 
    # ==========================================================================
    # It can happen that two or more haloes are mutually orbiting. However, a
    # more massive halo cannot orbit a smaller one (by definition). Therefore,
    # we rank order all haloes, and remove all orbiting haloes that are more
    # massive than the halo itself.
    # ==========================================================================
    # Find unique subhaloes. Since not all subhaloes have an orbiting mass, 
    # select only those found in the generated catalogue of candidate haloes.
    ohids_subs_unique, counts_unique = np.unique(ohids_subs, return_counts=True)
    mask_has_mass = np.isin(ohids_subs_unique, ohids)
    ohids_subs_unique = ohids_subs_unique[mask_has_mass]
    counts_unique = counts_unique[mask_has_mass]
    
    # Set the mass of all subhaloes. Notice how subhaloes without assigned Morb 
    # mass will have set M200b=0. Even though M200b is used to rank haloes, the
    # fact that the seeds have no orbiting mass assigned means they did not pass
    # the minimum mass threshold in the classifier and must be treated as such.
    m200b_subs = np.zeros(ohids_subs.shape, dtype=np.dtype(m200b[0]))
    for item in tqdm(ohids_subs_unique, ncols=100, colour='blue', 
                    desc='Populating subhalo mass'):
        m200b_subs[(ohids_subs == item)] = m200b[(ohids==item)]
    
    # Compare parent to subhalo mass. Using argmax will return 
    #                               = 0 if M_parent > M_sub
    #   argmax(Mparent, Msub) ->    > 0 if M_sub > M_parent
    # 
    # Select all cases where argmax(Mparent, Msub) > 0, i.e. where a subhalo is
    # tagged as orbiting a less massive structure
    sub_is_massive = np.argmax([parent_mass, m200b_subs], axis=0) > 0
    # For all 'subhaloes' with larger mass than their parents, set the subhalo 
    # ID to -1. This effectively removes the ID of a more massive member.
    ohids_subs[sub_is_massive] = -1

    # ==========================================================================
    #                                   Step 2
    # 
    #   Objects may orbit at most one structure.
    # 
    # ==========================================================================
    # By definition, any one object may only orbit a single structure at a time.
    # Because the orbiting classification of an object occurs irrespective of 
    # any previous associations, it can happen that a single object is orbiting 
    # more than one halo at a time (multiple membership). We find all parents 
    # and keep membership to the closest parent, thus removing membership from 
    # all others.
    # ==========================================================================
    # Get unique sub-haloes again to remove all unassigned by previous step
    ohids_subs_unique, counts_unique = np.unique(ohids_subs, return_counts=True)
    # Ensure selected subhaloes have an assigned mass.
    mask_has_mass = np.isin(ohids_subs_unique, ohids) & (ohids_subs_unique != -1)
    ohids_subs_unique = ohids_subs_unique[mask_has_mass]
    counts_unique = counts_unique[mask_has_mass]

    for item in tqdm(ohids_subs_unique[counts_unique > 1], ncols=100,
                    colour='blue', desc='Cleaning repetitions'):
        mask_this = np.isin(ohids_subs, item)
        # This diference is only zero for the largest parent mass, and its less
        # than zero for all others.
        mask_max = (parent_mass - np.max(parent_mass[mask_this])) < 0
        ohids_subs[mask_this & mask_max] = -1

    # ==========================================================================
    #                             Create new catalogue
    # ==========================================================================
    # Load data again to ensure PIDs has the same ordering as the catalogue.
    with h5.File(save_path + f'run_{dir_name}/temp_catalogue.hdf5', 'r') as hdf:
        ohids = hdf['Orig_halo_ID'][()]
        slidx = hdf['SLIDX'][()]
        suidx = hdf['SUIDX'][()]
    
    # Select all haloes with members. SIDX=-1 if there are no subhaloes.
    mask_with_subs = (slidx != -1) & (suidx != -1)
    ohids_with_subs = ohids[mask_with_subs]
    slidx_with_subs = slidx[mask_with_subs]
    suidx_with_subs = suidx[mask_with_subs]

    # Generate parent halo ID list.
    pids = np.full(ohids.shape[0], fill_value=-1, dtype=np.int32)
    for i, hid in enumerate(tqdm(ohids_with_subs, ncols=100, colour='green',
                            desc='Establishing halo hierarchy')):
        subs_ids = ohids_subs[slidx_with_subs[i]:suidx_with_subs[i]]
        mask_subs = np.isin(ohids, subs_ids[subs_ids != -1], assume_unique=True)
        pids[mask_subs] = hid

    # Save results
    with h5.File(save_path + f'run_{dir_name}/temp_pids.hdf5', 'w') as hdf:
        hdf.create_dataset('PID', data=pids)

    return None


@timer
def percolate_particles(
    file_seeds: str,
    min_num_part: int,
    save_path: str,
    dir_name: str,
) -> None:
    # ==========================================================================
    # Load M200b of new haloes from the seed catalogue.
    # ==========================================================================
    with h5.File(file_seeds, 'r') as hdf:
        ohids_m200b = hdf['Orig_halo_ID'][()]
        m200b = hdf['M200b'][()]
    # Sort by HID
    ohids_order = np.argsort(ohids_m200b)
    ohids_m200b = ohids_m200b[ohids_order]
    m200b = m200b[ohids_order]

    # Load parent candidates
    with h5.File(save_path + f'run_{dir_name}/temp_catalogue.hdf5', 'r') as hdf:
        ohids_morb = hdf['Orig_halo_ID'][()]
        # norb = hdf['Norb'][()]
        lidx = hdf['LIDX'][()]
        uidx = hdf['UIDX'][()]
    
    # Select parent haloes only
    with h5.File(save_path + f'run_{dir_name}/temp_pids.hdf5', 'r') as hdf:
        mask_parents = hdf['PID'][()] == -1
    
    # Sort by HID
    ohids_order = np.argsort(ohids_morb[mask_parents])
    ohids_morb = ohids_morb[mask_parents][ohids_order]
    # norb = norb[mask_parents][ohids_order]
    lidx = lidx[mask_parents][ohids_order]
    uidx = uidx[mask_parents][ohids_order]

    # Cross-match catalogues making them element-wise compatible.
    mask_match_catalogues = np.isin(ohids_m200b, ohids_morb, assume_unique=True)
    m200b = m200b[mask_match_catalogues]

    # ==========================================================================
    # Prepare particles for percolation.
    # ==========================================================================
    # Load particle IDs. Convert to signed integers.
    with h5.File(save_path + f'run_{dir_name}/temp_members.hdf5', 'r') as hdf:
        pids = hdf['part/PID'][()].astype(np.int64)

    # Rank order haloes by mass (Rockstar)
    mass_order = np.argsort(m200b)[::-1]
    ohids = ohids_morb[mass_order]
    m200b = m200b[mass_order]
    # norb = norb[mass_order]
    lidx = lidx[mass_order]
    uidx = uidx[mass_order]
    n_haloes = len(ohids)

    # ==========================================================================
    #                                   Step 1
    # 
    #   Objects may orbit at most one structure.
    # 
    # ==========================================================================
    # By definition, any one object may only orbit a single structure at a time.
    # Because the orbiting classification of an object occurs irrespective of 
    # any previous associations, it can happen that a single object is orbiting 
    # more than one halo at a time (multiple membership). We find all parents 
    # and keep membership to the closest parent, thus removing membership from 
    # all others.
    # ==========================================================================
    # Create two arrays containing the halo ID and mass.
    parent_ohid = np.full(pids.shape, -1, dtype=np.int64)
    parent_mass = np.zeros(pids.shape, dtype=np.dtype(m200b[0]))
    for i in tqdm(range(n_haloes), ncols=100, colour='blue',
                  desc='Populating halo mass'):
        parent_ohid[lidx[i] : uidx[i]] = ohids[i]
        parent_mass[lidx[i] : uidx[i]] = m200b[i]

    # Get ordering of PIDs and create new sorted arrays.
    argsort = np.argsort(pids)      # Takes ~1s per 10 million particles
    pids_sorted = pids[argsort]
    parent_ohid_sorted = parent_ohid[argsort]
    parent_mass_sorted = parent_mass[argsort]

    # Get unique PIDs and select those with more than one ocurrence. Only select
    # particles orbiting a parent halo. This means that their parent_ohid wasn't
    # changed in the step above. Should help skip duplicated particles which 
    # have already been percolated via their hosts not being parents.
    pids_unique, counts_unique = np.unique(pids[parent_ohid !=- 1], 
                                           return_counts=True)
    pids_unique = pids_unique[counts_unique > 1]
    n_unique = len(pids_unique)

    # Since the PIDs are sorted, all repetitions are contiguous now. Find the 
    # first and last index for each PID.
    lidx_sorted = np.searchsorted(pids_sorted, pids_unique, side='left')
    ridx_sorted = np.searchsorted(pids_sorted, pids_unique, side='right')

    # Probably a vectorized way to do this loop...
    for i in tqdm(range(n_unique), ncols=100, colour='blue', 
                     desc='Cleaning repetitions'):
        parent_mass_this = parent_mass_sorted[lidx_sorted[i]:ridx_sorted[i]]
        # Keep most massive.
        mask_remove = (parent_mass_this - np.max(parent_mass_this)) < 0
        # Unassign HID to all toher particles
        parent_ohid_sorted[lidx_sorted[i]:ridx_sorted[i]][mask_remove] = -1

    # ==========================================================================
    #                               New halo masses
    # ==========================================================================
    # Select only PIDs with assigned HID
    mask_ohids = parent_ohid_sorted != -1
    pids_sorted = pids_sorted[mask_ohids]
    parent_ohid_sorted = parent_ohid_sorted[mask_ohids]

    # Sort arrays now by HID
    argsort = np.argsort(parent_ohid_sorted)
    pids_sorted = pids_sorted[argsort]
    parent_ohid_sorted = parent_ohid_sorted[argsort]

    # Load data again to ensure PIDs has the same ordering as the catalogue.
    with h5.File(save_path + f'run_{dir_name}/temp_catalogue.hdf5', 'r') as hdf:
        ohids = hdf['Orig_halo_ID'][()]
        norb_old = hdf['Orig_halo_ID'][()]

    # New Left and Right indices for member particles.
    lidx_temp = np.searchsorted(parent_ohid_sorted, ohids, side='left')
    ridx_temp = np.searchsorted(parent_ohid_sorted, ohids, side='right')

    # Get new orbiting mass
    norb_new = ridx_temp - lidx_temp

    # Select haloes above minimum mass thresold.
    mask_new_haloes = norb_new >= min_num_part

    ohids = ohids[mask_new_haloes]
    norb_new = norb_new[mask_new_haloes]
    lidx_temp = lidx_temp[mask_new_haloes]
    ridx_temp = ridx_temp[mask_new_haloes]

    # Select all member particles only for the final haloes
    pids_new = []
    lidx_new = np.zeros_like(lidx_temp)
    ridx_new = np.zeros_like(ridx_temp)
    norb_sanity = np.zeros_like(norb_new)
    n_tot = 0
    for i in tqdm(range(len(ohids)), ncols=100, colour='green', 
                     desc='Saving particles'):
        pids_temp = pids_sorted[lidx_temp[i]:ridx_temp[i]]
        n_parts = len(pids_temp)
        lidx_new[i] = n_tot
        ridx_new[i] = n_tot + n_parts
        norb_sanity[i] = n_parts
        n_tot += n_parts
        pids_new.append(pids_temp)
    pids_new = np.concatenate(pids_new)

    # Save results
    with h5.File(save_path + f'run_{dir_name}/orbiting_particles.hdf5', 'w') as hdf:
        hdf.create_dataset('PID', data=pids_new)
        
    with h5.File(save_path + f'run_{dir_name}/catalogue.hdf5', 'w') as hdf:
        hdf.create_dataset('Orig_halo_ID', data=ohids)
        hdf.create_dataset('Norb', data=norb_new)
        hdf.create_dataset('LIDX', data=lidx_temp)
        hdf.create_dataset('UIDX', data=ridx_temp)

    return None


@timer
def generate_catalogue(
    load_path: str,
    file_seeds: str,
    dir_name: str,
    min_num_part: int,
    boxsize: float,
    minisize: float,
    padding: float = 5.0,
    n_threads: int = None,
):

    classify_all_mini_boxes(
        load_path=load_path,
        dir_name=dir_name,
        min_num_part=min_num_part,
        boxsize=boxsize,
        minisize=minisize,
        padding=padding,
        n_threads=n_threads,
    )

    percolate_haloes(
        file_seeds=file_seeds,
        save_path=load_path,
        dir_name=dir_name,
    )
    
    percolate_particles(
        save_path=load_path,
        min_num_part=min_num_part,
        dir_name=dir_name,
    )

    return None


if __name__ == "__main__":
    pass
