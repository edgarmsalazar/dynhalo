import os
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import List, Tuple, Union, Any
from warnings import filterwarnings

import h5py as h5
import numpy as np
import pandas as pd
from tqdm import tqdm

from dynhalo.finder.coordinates import relative_coordinates
from dynhalo.finder.minibox import load_particles, load_seeds
from dynhalo.utils import G_gravity, timer

filterwarnings('ignore')


def compute_halo_properties(rel_pos, rel_vel, part_mass, rhom, delta: Any = 'rockstar'):
    dists = np.sqrt(np.sum(np.square(rel_pos), axis=1))
    argsort = np.argsort(dists)

    dists = dists[argsort]
    velsq = np.sum(np.square(rel_vel), axis=1)[argsort]
    mass_prof = part_mass * np.arange(1, len(dists)+1)

    # Get R200 and M200
    loc = np.argmax(mass_prof / (4 / 3 * np.pi * dists ** 3) <= 200 * rhom)
    r200 = dists[loc]
    m200 = mass_prof[loc]

    if delta == 'rockstar':
        # Get v_max
        vel_prof_sq = G_gravity * mass_prof / dists
        vmax = np.max(vel_prof_sq)
        # Eq 4 in P. Behroozi (2012) ROCKSTAR paper.
        sigma_x = vmax**2 / (G_gravity * 200 * rhom * 4 * np.pi / 3)
        sigma_v = np.var(rel_vel)
    else:
        # Get R_Delta
        loc2 = np.argmax(mass_prof / (4 / 3 * np.pi * dists ** 3) <= delta * rhom)
        sigma_x = dists[loc2]**2
        sigma_v = np.median(velsq[:loc2])

    return r200, m200, sigma_x, sigma_v


def classify(
    rel_pos: np.ndarray,
    rel_vel: np.ndarray,
    r200: float,
    m200: float,
    class_pars: Union[List, Tuple, np.ndarray],
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
    mask_orb = \
        (mask_cut_pos & mask_vr_positive) ^ \
        (mask_cut_neg & ~mask_vr_positive)

    return mask_orb


def classify_seeds_in_mini_box(
    mini_box_id: int,
    min_num_part: int,
    part_mass: float,
    rhom: float,
    boxsize: float,
    minisize: float,
    path: str,
    dir_name: str,
    padding: float = 5.0,
) -> None:
    """Runs the classifier for each seed in a mini box.
    Additionally, percolates all found haloes by:
        1. Resolves parent-sub halo relationship by only allowing less massive
           structures to orbit more massive ones.
        2. Assigns shared sub-haloes between parents to the closest parent.
        3. Assigns shared particles between parents to the closest parent.
    The distance metric is the phase-space distance:
    \begin{equation*}
    d^{2} = \frac{|\vec{x}_p - \vec{x}_h|^2}{r_{v_\max}^2} + 
                \frac{|\vec{v}_p - \vec{v}_h|^2}{\sigma_v^2}
    \end{equation*}
    where
    \begin{equation*}
    r_{v_\max}^2 = \frac{v_{\max}^2}{\frac{4\pi}{3}G\rho_{200}}
    \end{equation*}
    
    See P. Behroozi (2013) for a discussion on this metric.

    Parameters
    ----------
    mini_box_id : int
        Sub-box ID
    min_num_part : int
        Minimum number of particles needed to be considered a halo
    part_mass : float
        Mass per particle
    rhom : float
        Matter density of the universe
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
    path : str
        Location from where to load the file
    padding : float
        Only particles up to this distance from the mini box edge are considered 
        for classification. Defaults to 5

    Returns
    -------
    None
    """
    save_path = path + f'run_{dir_name}/mini_box_catalogues/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load seeds in mini box
    pos_seed_mb, vel_seed_mb, hid_seed_mb, _ = \
        load_seeds(mini_box_id, boxsize, minisize, path, padding)
    n_seeds = len(hid_seed_mb)

    # Exit if there are no seeds in the mini box.
    if any([p is None for p in (pos_seed_mb, vel_seed_mb, hid_seed_mb)]):
        return None

    # Load adjacent seeds
    pos_seed_adj, vel_seed_adj, hid_seed_adj, _ = \
        load_seeds(mini_box_id, boxsize, minisize,
                   path, padding, adjacent=True)
    
    # Concatenate all seeds for ease of selection.
    hid_seed = np.hstack([hid_seed_mb, hid_seed_adj])
    pos_seed = np.vstack([pos_seed_mb, pos_seed_adj])
    vel_seed = np.vstack([vel_seed_mb, vel_seed_adj])

    # Load particles
    pos_part, vel_part, pid_part, row_part = \
        load_particles(mini_box_id, boxsize, minisize, path, padding)
    
    # Load calibration parameters
    with h5.File(path + 'calibration_pars.hdf5', 'r') as hdf:
        pars = (*hdf['pos'][()], *hdf['neg'][()])

    # Create empty catalog of found halos and a dictionary with the PIDs.
    halo_members = {}
    halo_non_members = {}
    halo_subs = {}

    col_names = ('OHID', 'pos', 'vel', 'R200m', 'M200m', 'Morb')
    haloes = pd.DataFrame(columns=col_names)

    for i in range(n_seeds):
        # ======================================================================
        #                           Classify particles
        # ======================================================================
        rel_pos = relative_coordinates(pos_seed_mb[i], pos_part, boxsize)
        rel_vel = vel_part - vel_seed_mb[i]
        r200, m200, sigma_x, sigma_v = \
            compute_halo_properties(rel_pos, rel_vel, part_mass, rhom)
        
        # Classify
        mask_orb = classify(rel_pos, rel_vel, r200, m200, pars)
        # Ignore seed if it does not have the minimum mass to be considered a
        # halo. Early exit to avoid further computation for a non-halo seed
        if mask_orb.sum() < min_num_part:
            continue
        
        # Compute phase space distance from particle to halo
        dphsq = np.sum(np.square(rel_pos), axis=1) / sigma_x + \
            np.sum(np.square(rel_vel), axis=1) / sigma_v

        # Select orbiting particles' PID
        row_idx_order = np.argsort(row_part[mask_orb])
        orb_pid = pid_part[mask_orb][row_idx_order]
        orb_arg = row_part[mask_orb][row_idx_order]
        orb_dph = dphsq[mask_orb][row_idx_order]

        # Append halo to catalogue if there are at least min_dm_part orbiting
        # particles.
        haloes.loc[len(haloes.index)] = [
            hid_seed_mb[i],
            pos_seed_mb[i],
            vel_seed_mb[i],
            r200,
            m200,
            part_mass * mask_orb.sum(),
        ]
        halo_members[hid_seed_mb[i]] = {
            'PID': orb_pid,
            'row_idx': orb_arg,
            'dph': orb_dph,
        }
        # Save non-members up to a padding distance. This is helpful for
        # computing density profiles of non-orbiting particles.
        non_members_mask = np.sum(np.square(rel_pos), axis=1) < 2 * padding
        halo_non_members[hid_seed_mb[i]] = row_part[~mask_orb*non_members_mask]

        # ======================================================================
        #                           Classify seeds
        # ======================================================================
        # Ignore current seed.
        mask_self = hid_seed != hid_seed_mb[i]
        rel_pos = relative_coordinates(
            pos_seed_mb[i], pos_seed[mask_self], boxsize)
        rel_vel = vel_seed[mask_self] - vel_seed_mb[i]

        # Classify
        mask_orb_seed = classify(rel_pos, rel_vel, r200, m200, pars)

         # If there are orbiting seeds, append them to the members list.
        if mask_orb_seed.sum() > 0:
            # Compute phase space distance from particle to halo
            dphsq = np.sum(np.square(rel_pos), axis=1) / sigma_x + \
                np.sum(np.square(rel_vel), axis=1) / sigma_v

            # Select orbiting particles' PID
            orb_pid_seed = hid_seed[mask_self][mask_orb_seed]
            orb_dph_seed = dphsq[mask_orb_seed]

            halo_subs[hid_seed_mb[i]] = {
                'OHID': orb_pid_seed,
                'dph': orb_dph_seed,
            }

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
            if key in ["pos", "vel"]:
                data = np.stack(haloes[key].values)
            else:
                data = haloes[key].values
            hdf.create_dataset(f'halo/{key}', data=data)

        # Halo members: particles
        # Dataset names must be strings (thus are not properly sorted)
        for item, _ in halo_members.items():
            hdf.create_dataset(f'members/part/{str(item)}/PID',
                               data=halo_members[item]['PID'])
            hdf.create_dataset(f'members/part/{str(item)}/dph',
                               data=halo_members[item]['dph'])
            hdf.create_dataset(f'members/part/{str(item)}/row_idx',
                               data=halo_members[item]['row_idx'])
            hdf.create_dataset(f'non_members/part/{str(item)}/row_idx',
                               data=halo_non_members[item])
        # Halo members: seeds
        if len(halo_subs.keys()) > 0:
            for item, _ in halo_subs.items():
                hdf.create_dataset(f'members/halo/{str(item)}/OHID',
                                   data=halo_subs[item]['OHID'])
                hdf.create_dataset(f'members/halo/{str(item)}/dph',
                                   data=halo_subs[item]['dph'])

    return None


@timer
def generate_full_box_catalogue(
    path: str,
    min_num_part: int,
    part_mass: float,
    rhom: float,
    boxsize: float,
    minisize: float,
    dir_name: str,
    padding: float = 5.0,
    n_threads: int = None,
) -> None:
    """Generates a halo catalogue using the kinetic mass criterion to classify
    particles into orbiting or infalling.

    Parameters
    ----------
    path : str
        Location from where to load the file
    min_num_part : int
        Minimum number of particles needed to be considered a halo
    part_mass : float
        Mass per particle
    rhom : float
        Matter density of the universe
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
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
    save_path = path + f'run_{dir_name}/mini_box_catalogues/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    n_mini_boxes = np.int_(np.ceil(boxsize / minisize))**3

    func = partial(classify_seeds_in_mini_box, min_num_part=min_num_part,
                   part_mass=part_mass, rhom=rhom, boxsize=boxsize, path=path,
                   minisize=minisize, padding=padding, dir_name=dir_name)

    with Pool(n_threads) as pool:
        list(tqdm(pool.imap(func, range(n_mini_boxes)),
                  total=n_mini_boxes, colour="green", ncols=100,
                  desc='Generating halo catalogue'))

    # # Consolidate catalogue
    m200m, morb, ohid, r200m, pos, vel = ([] for _ in range(6))
    files = os.listdir(save_path)
    for f in tqdm(files, ncols=100, desc='Merging catalogues', colour='green'):
        with h5.File(save_path+f, 'r') as hdf:
            if 'halo' in hdf.keys():
                m200m.append(hdf['halo/M200m'][()])
                r200m.append(hdf['halo/R200m'][()])
                morb.append(hdf['halo/Morb'][()])
                ohid.append(hdf['halo/OHID'][()])
                pos.append(hdf['halo/pos'][()])
                vel.append(hdf['halo/vel'][()])

    with h5.File(path + f'run_{dir_name}/catalogue_raw.hdf5', 'w') as hdf:
        hdf.create_dataset('M200m', data=np.concatenate(m200m))
        hdf.create_dataset('R200m', data=np.concatenate(r200m))
        hdf.create_dataset('OHID', data=np.concatenate(ohid))
        hdf.create_dataset('Morb', data=np.concatenate(morb))
        hdf.create_dataset('pos', data=np.concatenate(pos))
        hdf.create_dataset('vel', data=np.concatenate(vel))

    # Consolidate members catalogue
    # Process haloes
    with h5.File(path + f'run_{dir_name}/members_halo_temp.hdf5', 'w') as hdf:
        for f in tqdm(files, ncols=100, desc='Merging members', colour='green'):
            with h5.File(save_path + f, 'r') as hdf_load:
                if 'members' in hdf_load.keys():
                    if 'halo' in hdf_load['members'].keys():
                        for hid in hdf_load['members/halo'].keys():
                            hdf.create_dataset(
                                f'{hid}/OHID',
                                data=hdf_load[f'members/halo/{hid}/OHID'][()])
                            hdf.create_dataset(
                                f'{hid}/dph',
                                data=hdf_load[f'members/halo/{hid}/dph'][()])
                        
    # Process particles
    with h5.File(path + f'run_{dir_name}/members_part_temp.hdf5', 'w') as hdf:
        for f in tqdm(files, ncols=100, desc='Merging members', colour='green'):
            with h5.File(save_path + f, 'r') as hdf_load:
                if 'members' in hdf_load.keys():
                    for hid in hdf_load['members/part'].keys():
                        hdf.create_dataset(
                            f'{hid}/PID',
                            data=hdf_load[f'members/part/{hid}/PID'][()])
                        hdf.create_dataset(
                            f'{hid}/dph',
                            data=hdf_load[f'members/part/{hid}/dph'][()])
                        hdf.create_dataset(
                            f'{hid}/row_idx',
                            data=hdf_load[f'members/part/{hid}/row_idx'][()])
                        hdf.create_dataset(
                            f'{hid}/row_idx_inf',
                            data=hdf_load[f'non_members/part/{hid}/row_idx'][()])
    return


@timer
def percolate_haloes(
    path: str,
    dir_name: str,
) -> None:
    """Percolate parent halo candidates.

    Parameters
    ----------
    path : str
        Location from where to load the file
    dir_name : str
        _description_
    """
    # Load parent candidates
    with h5.File(path + f'run_{dir_name}/catalogue_raw.hdf5', 'r') as hdf:
        ohids = hdf['OHID'][()]
        # Rank order haloes by mass
        morb_order = np.argsort(hdf['Morb'][()])[::-1]
    ohids = ohids[morb_order]

    # Load members
    halo_memb = {}
    with h5.File(path + f'run_{dir_name}/members_halo_temp.hdf5', 'r') as hdf:
        for key in tqdm(hdf.keys(), ncols=100, colour='blue',
                        desc='Loading members'):
            halo_memb[int(key)] = {
                'OHID': hdf[key]['OHID'][()],
                'dph': hdf[key]['dph'][()],
            }
    
    # Select hales with members only
    members_keys = np.array([item for item in halo_memb.keys()])
    mask_with_subs = np.isin(ohids, members_keys, assume_unique=True)
    ohids_with_subs = ohids[mask_with_subs]
    
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
    halo_memb_temp = {}
    # Skip over first item
    for i in tqdm(range(1, len(ohids_with_subs)), ncols=100,
                  colour='blue', desc='Establishing hierarchy'): 
        ohid_current = ohids_with_subs[i]
        members_current = halo_memb[ohid_current]['OHID']
        
        # More massive HIDs
        ohid_massive = ohids_with_subs[:i]
        # Does the current halo has any more massive member?
        mask_is_more_massive = np.isin(members_current, ohid_massive)
        # If so, select only those members with smaller mass.
        if mask_is_more_massive.sum() != 0:
            mask_is_less_massive = np.isin(members_current, ohid_massive, 
                                           invert=True)
        # If all members are less massive, select all members and keep them.
        else:
            mask_is_less_massive = np.full(members_current.shape[0], True)

        # If there are any members left, save them into a new dictionary
        if mask_is_less_massive.sum() != 0:
            halo_memb_temp[ohid_current] = {
                'OHID': members_current[mask_is_less_massive],
                'dph': halo_memb[ohid_current]['dph'][mask_is_less_massive],
            }

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
    # Reverse the members dictionary. PID: HID and PID: dph
    members_rev = defaultdict(list)
    dph_rev = defaultdict(list)

    members_keys = np.array([item for item in halo_memb_temp.keys()])
    for hid in tqdm(members_keys, ncols=100, colour='blue',
                    desc='Reversing dicts'):
        for i, sub_hid in enumerate([*halo_memb_temp[hid]['OHID']]):
            members_rev[sub_hid].append(hid)
            dph_rev[sub_hid].append(halo_memb_temp[hid]['dph'][i])

    # Look for repeated members
    repeated_members = []
    for sub_hid, elements in members_rev.items():
        if len(elements) > 1:
            repeated_members.append(sub_hid)

    # Create a dictionary with the objects marked for removal per halo. HID: PID
    sub_hids_to_remove = defaultdict(list)
    for sub_hid in repeated_members:
        current_sub_hid = np.array(members_rev[sub_hid])
        current_dph = np.array(dph_rev[sub_hid])
        loc_min = np.argmin(current_dph)
        mask_remove = current_dph != current_dph[loc_min]

        for hid in current_sub_hid[mask_remove]:
            sub_hids_to_remove[hid].append(sub_hid)

    # Create a new members catalogue.
    halo_memb = {}
    sub_hids_to_remove_keys = np.array([item for item in sub_hids_to_remove.keys()])
    for hid in tqdm(members_keys, ncols=100, colour='blue',
                    desc='Cleaning members'):
        # If HID has members maked for removal
        if hid in sub_hids_to_remove_keys:
            pid_remove = sub_hids_to_remove[hid]
            mask_keep = np.isin(
                halo_memb_temp[hid]['OHID'],
                pid_remove,
                assume_unique=True,
                invert=True
            )
            # If there are no members left move to the next item.
            if mask_keep.sum() == 0:
                continue

            halo_memb[hid] = {
                'OHID': halo_memb_temp[hid]['OHID'][mask_keep],
            }
        # If no members marked for removal, simply copy
        else:
            halo_memb[hid] = {
                'OHID': halo_memb_temp[hid]['OHID'],
            }

    # ==========================================================================
    #                               Parent halo IDs
    # ==========================================================================
    # Load parent candidates
    with h5.File(path + f'run_{dir_name}/catalogue_raw.hdf5', 'r') as hdf:
        ohids = hdf['OHID'][()]
    pids = np.full(ohids.shape[0], fill_value=-1, dtype=np.int32)
    
    # Set the parent halo ID for all members.
    for hid, value in tqdm(halo_memb.items(), ncols=100, colour='green',
                           desc='Tagging parents'):
        mask = np.isin(ohids, value['OHID'], assume_unique=True)
        pids[mask] = hid

    # Save results
    with h5.File(path + f'run_{dir_name}/halo_pids.hdf5', 'w') as hdf:
        hdf.create_dataset('PID', data=pids)
    
    return


@timer
def percolate_particles(
    path: str,
    min_num_part: int,
    part_mass: float,
    dir_name: str,
):
    # Load parent haloes
    with h5.File(path + f'run_{dir_name}/halo_pids.hdf5', 'r') as hdf:
        ohids_pids = hdf['PID'][()]
    mask_parents = ohids_pids == -1

    with h5.File(path + f'run_{dir_name}/catalogue_raw.hdf5', 'r') as hdf:
        ohids = hdf['OHID'][()]
    ohids = ohids[mask_parents]

    # Load members
    halo_memb_temp = {}
    with h5.File(path + f'run_{dir_name}/members_part_temp.hdf5', 'r') as hdf:
        file_keys = list(hdf.keys())
        for key in tqdm(file_keys, ncols=100, colour='blue',
                        desc='Loading members'):
            # Only load members of parent haloes.
            if int(key) in ohids:
                halo_memb_temp[int(key)] = {
                    'PID': hdf[key]['PID'][()],
                    'dph': hdf[key]['dph'][()],
                    'row_idx': hdf[key]['row_idx'][()],
                }
    
    # ==========================================================================
    #                                   Step 1
    # 
    #   Particles may orbit at most one halo.
    # 
    # ==========================================================================
    # By definition, any one particle may only orbit a single halo at a time.
    # Because the orbiting classification of a particle occurs irrespective of 
    # any previous associations, it can happen that a single particle is orbiting 
    # more than one halo at a time (multiple membership). We keep membership to 
    # the closest parent.
    # ==========================================================================
    # Reverse the members dictionary. PID: HID and PID: dph
    members_rev = defaultdict(list)
    dph_rev = defaultdict(list)

    members_keys = np.array([item for item in halo_memb_temp.keys()])
    for hid in tqdm(members_keys, ncols=100, colour='blue',
                    desc='Reversing dicts'):
        for i, sub_hid in enumerate([*halo_memb_temp[hid]['PID']]):
            members_rev[sub_hid].append(hid)
            dph_rev[sub_hid].append(halo_memb_temp[hid]['dph'][i])

    # Look for repeated members
    repeated_members = []
    for sub_hid, elements in members_rev.items():
        if len(elements) > 1:
            repeated_members.append(sub_hid)

    # Create a dictionary with the objects marked for removal per halo. HID: PID
    pids_to_remove = defaultdict(list)
    for pid in repeated_members:
        current_pid = np.array(members_rev[pid])
        current_dph = np.array(dph_rev[pid])
        loc_min = np.argmin(current_dph)
        mask_remove = current_dph != current_dph[loc_min]

        for hid in current_pid[mask_remove]:
            pids_to_remove[hid].append(pid)

    # Create a new members and non-members catalogues
    halo_memb = {}
    halo_non_memb = {}
    removed_haloes = []
    pids_to_remove_keys = np.array([item for item in pids_to_remove.keys()])
    for hid in tqdm(members_keys, ncols=100, colour='blue',
                    desc='Cleaning members'):
        # If HID has members maked for removal
        if hid in pids_to_remove_keys:
            pid_remove = pids_to_remove[int(hid)]
            mask_keep = np.isin(
                halo_memb_temp[hid]['PID'],
                pid_remove,
                assume_unique=True,
                invert=True,
            )
            
            # If it no longer has the minimum mass to be considered a halo
            if mask_keep.sum() < min_num_part:
                removed_haloes.append(hid)

            halo_memb[hid] = {
                'PID': halo_memb_temp[hid]['PID'][mask_keep],
                'row_idx': halo_memb_temp[hid]['row_idx'][mask_keep],
            }
            # Removed members are now non-members
            halo_non_memb[hid] = halo_memb_temp[hid]['row_idx'][~mask_keep]
        # If no members marked for removal, simply copy
        else:
            halo_memb[hid] = {
                'PID': halo_memb_temp[hid]['PID'],
                'row_idx': halo_memb_temp[hid]['row_idx'],
            }

    # ==========================================================================
    #                                   Step 2
    # 
    #   Mass recomputation
    # 
    # ==========================================================================
    # Now that the parents have been percolated, the orbiting mass has changed
    # and needs to be recomputed.
    # ==========================================================================
    # Load raw data to keep same ordering
    with h5.File(path + f'run_{dir_name}/catalogue_raw.hdf5', 'r') as hdf:
        ohids = hdf['OHID'][()]
        morb_new = hdf['Morb'][()]
    
    members_keys = np.array([item for item in halo_memb.keys()])
    for i, hid in enumerate(tqdm(ohids, ncols=100, colour='green',
                            desc='Computing Morb')):
        # Skip if object is no longer a halo or is not a parent
        if hid in removed_haloes or hid not in members_keys:
            continue
        # Compute the orbiting mass otherwise.
        morb_new[i] = part_mass * len(halo_memb[hid]['PID'])

    # ==========================================================================
    #                               Save results
    # ==========================================================================
    # Create new halo catalogue with the remaining haloes and subs (as long as 
    # Morb > 0)
    mask_morb = (morb_new > 0) & np.isin(ohids, removed_haloes, invert=True)

    with h5.File(path + f'run_{dir_name}/halo_pids.hdf5', 'r') as hdf:
        ohids_pids = hdf['PID'][()]
    mask_parents = ohids_pids == -1
    
    with h5.File(path + f'run_{dir_name}/catalogue_raw.hdf5', 'r') as hdf_raw, \
        h5.File(path + f'run_{dir_name}/catalogue.hdf5', 'w') as hdf:
        for key in hdf_raw.keys():
            hdf.create_dataset(key, data=hdf_raw[key][mask_morb])
        hdf.create_dataset('Morb_perc', data=morb_new[mask_morb])
        hdf.create_dataset('PID', data=ohids_pids[mask_morb])
    
    # Create members catalogue for parent haloes only.
    with h5.File(path + f'run_{dir_name}/members.hdf5', 'w') as hdf:
        # Only save members of parent haloes.
        for hid in tqdm(ohids[mask_morb & mask_parents], ncols=100, colour='green',
                        desc='Saving members'):
            hdf.create_dataset(f'{hid}/PID', data=halo_memb[hid]['PID'])
            hdf.create_dataset(f'{hid}/row_idx', data=halo_memb[hid]['row_idx'])

    # Non members catalogue
    # with h5.File(path + f'run_{dir_name}/non_members.hdf5', 'w') as hdf:
    #     pass

    return


if __name__ == "__main__":
    pass
