import os
from typing import Tuple

import h5py as h5
import numpy as np
from tqdm import tqdm

from dynhalo.utils import cartesian_product, get_np_unit_dytpe, timer
from dynhalo.finder.coordinates import relative_coordinates


def generate_sub_box_grid(
    boxsize: float,
    subsize: float,
) -> Tuple[np.ndarray]:
    """Generates a 3D grid of sub-boxes.

    Parameters
    ----------
    boxsize : float
        Size of simulation box
    subsize : float
        Size of sub-box

    Returns
    -------
    Tuple[np.ndarray]
        ID and centre coordinate for all sub-boxes
    """

    # Number of sub-boxes per side
    boxes_per_side = np.int_(np.ceil(boxsize / subsize))

    # Determine data type for integer arrays based on the maximum number of
    # elements
    uint_dtype = get_np_unit_dytpe(boxes_per_side)
    # Set of natural numbers from 0 to N-1
    n_range = np.arange(boxes_per_side, dtype=uint_dtype)

    # Shift in each dimension for numbering sub-boxes
    uint_dtype = get_np_unit_dytpe(boxes_per_side**2)
    shift = np.array(
        [1, boxes_per_side, boxes_per_side * boxes_per_side], dtype=uint_dtype)

    # Set of index vectors. Each vector points to the (i, j, k)-th sub-box
    n_pos = np.int_(cartesian_product([n_range, n_range, n_range]))

    # Set of all possible unique IDs for each sub-box
    ids = np.sum(n_pos * shift, axis=1)
    sort_order = np.argsort(ids)

    # Sort IDs so that the ID matches the row index.
    n_pos = n_pos[sort_order]
    ids = ids[sort_order]

    # Sub-box central coordinate. Populate each sub-box with one point at the
    # centre.
    centres = subsize * (n_pos + 0.5)
    return ids, centres


def get_sub_box_id(
    x: np.ndarray,
    boxsize: float,
    subsize: float,
) -> int:
    """Returns the sub-box ID to which the coordinates `x` fall into

    Parameters
    ----------
    x : np.ndarray
        Position in cartesian coordinates
    boxsize : float
        Size of simulation box
    subsize : float
        Size of sub-box

    Returns
    -------
    int
        ID of the sub-box
    """
    # Number of sub-boxes per side
    boxes_per_side = np.int_(np.ceil(boxsize / subsize))
    # Determine data type for integer arrays based on the maximum number of
    # elements
    uint_dtype = get_np_unit_dytpe(boxes_per_side**3)
    # Shift in each dimension for numbering sub-boxes
    shift = np.array(
        [1, boxes_per_side, boxes_per_side * boxes_per_side], dtype=uint_dtype)
    # In the rare case an object is located exactly at the edge of the box,
    # move it 'inwards' by a tiny amount so that the box id is correct.
    x[np.where(x==boxsize)] -= 1e-8
    x[np.where(x==0)] += 1e-8
    if x.ndim > 1:
        return np.int_(np.sum(shift * np.floor(x / subsize), axis=1))
    else:
        return np.int_(np.sum(shift * np.floor(x / subsize)))


def get_adjacent_sub_box_ids(
    sub_box_id: np.ndarray,
    sub_box_ids: np.ndarray,
    positions: np.ndarray,
    boxsize: float,
    subsize: float,
) -> np.ndarray:
    """Returns a list of all IDs that are adjacent to the specified sub-box ID.
    There are always 27 adjacent boxes in a 3D volume, including the specified ID.

    Parameters
    ----------
    sub_box_id : np.ndarray
        ID of the sub-box
    sub_box_ids : np.ndarray
        IDs of all sub-boxes
    positions : np.ndarray
        Positions of all the centres of the sub-boxes
    boxsize : float
        Size of simulation box
    subsize : float
        Size of sub-box

    Returns
    -------
    np.ndarray
        List of sub-box IDs adjacent to `id`

    Raises
    ------
    ValueError
        If `id` is not found in the allowed values in `ids`
    """
    if sub_box_id not in sub_box_ids:
        raise ValueError(f'ID {sub_box_id} is out of bounds')

    x0 = positions[sub_box_ids == sub_box_id]
    d = relative_coordinates(x0, positions, boxsize)
    d = np.sqrt(np.sum(np.square(d), axis=1))
    mask = d <= 1.01*np.sqrt(3)*subsize
    return sub_box_ids[mask]


@timer
def generate_sub_box_ids(
    positions: np.ndarray,
    boxsize: float,
    subsize: float,
    chunksize: float,
    path: str,
    name: str = None
) -> None:
    """Gets the sub-box ID for each position

    Parameters
    ----------
    positions : np.ndarray
        Cartesian coordinates
    boxsize : float
        Size of simulation box
    subsize : float
        Size of sub-box
    chunksize : float
        Number of items to process at a time in chunks
    path : str
        Where to save the IDs
    name : str, optional
        An additional name or identifier appended at the end of the file name, 
        by default None

    Returns
    -------
    None
    """
    n_items = positions.shape[0]
    n_iter = n_items // chunksize

    # Determine data type for integer arrays based on the maximum number of
    # elements
    boxes_per_side = np.int_(np.ceil(boxsize / subsize))
    uint_dtype = get_np_unit_dytpe(boxes_per_side**3)

    ids = np.zeros(n_items, dtype=uint_dtype)

    for chunk in tqdm(range(n_iter), desc='Chunk', ncols=100, colour='blue'):
        low = chunk * chunksize
        if chunk < n_iter - 2:
            upp = (chunk + 1) * chunksize
        else:
            upp = None
        ids[low:upp] = get_sub_box_id(positions[low:upp], boxsize, subsize)
        # if np.max(ids) > boxes_per_side**3:
            # print(chunk)

    if name:
        file_name = f'sub_box_id_{name}.hdf5'
    else:
        file_name = f'sub_box_id.hdf5'
    with h5.File(path + file_name, 'w') as hdf:
        hdf.create_dataset('SBID', data=ids, dtype=uint_dtype)

    return None


@timer
def split_simulation_into_sub_boxes(
    positions: np.ndarray,
    velocities: np.ndarray,
    ids: np.ndarray,
    boxsize: float,
    subsize: float,
    chunksize: float,
    dtypes: list,
    path: str,
    name: str = None,
) -> None:
    """Sorts all items into sub-boxes and saves them in disc.

    Parameters
    ----------
    positions : np.ndarray
        _description_
    velocities : np.ndarray
        _description_
    ids : np.ndarray
        Unique IDs for each position (e.g. PID, HID)
    boxsize : float
        Size of simulation box
    subsize : float
        Size of sub-box
    chunksize : float
        Number of items to process at a time in chunks
    dtypes : list
        Data types for positions and velocities
    path : str
        Where to save the IDs
    name : str, optional
        An additional name or identifier appended at the end of the file name, 
        by default None

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the chunksize is larger than the number of items
    """
    # Check chunksize is smaller than the number of items
    n_items = positions.shape[0]
    if chunksize > n_items:
        raise ValueError(
            f"The specified chunksize {chunksize} is larger than the number of items {n_items}")

    # Create directory if it does not exist
    save_path = path + 'sub_boxes/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(save_path):
        generate_sub_box_ids(positions, boxsize, subsize,
                             chunksize, path, name)

    if name:
        sub_box_ids_file = path + f'sub_box_id_{name}.hdf5'
    else:
        sub_box_ids_file = path + f'sub_box_id.hdf5'
    with h5.File(sub_box_ids_file, 'r') as hdf:
        sub_box_ids = hdf['SBID'][()]

    n_iter = n_items // chunksize

    uint_dtype_row = get_np_unit_dytpe(n_items)
    row_idx = np.arange(n_items, dtype=uint_dtype_row)

    uint_dtype_ids = get_np_unit_dytpe(np.max(ids))
    for chunk in tqdm(range(n_iter), desc='Chunk', ncols=100, colour='blue'):
        # Select chunk
        low = chunk * chunksize
        if chunk < n_iter - 2:
            upp = (chunk + 1) * chunksize
        else:
            upp = None
        sb_ids = sub_box_ids[low:upp]
        pos = positions[low:upp]
        vel = velocities[low:upp]
        pid = ids[low:upp]
        row = row_idx[low:upp]

        sb_unique = np.unique(sb_ids)
        order = np.argsort(sb_ids)
        # Save all items at each unique sub-box ID
        for sub_box in sb_unique:
            left = np.searchsorted(sb_ids, sub_box, side="left", sorter=order)
            right = np.searchsorted(
                sb_ids, sub_box, side="right", sorter=order)

            pos_item = pos[order][left:right]
            vel_item = vel[order][left:right]
            pid_item = pid[order][left:right]
            row_item = row[order][left:right]

            with h5.File(save_path + f"{sub_box}.hdf5", "a") as hdf:
                if not name in hdf.keys():
                    hdf.create_group(name)

                # If it is the first time opening this file, create datasets
                if not 'ID' in hdf[name].keys():
                    hdf.create_dataset(
                        name=f'{name}/ID',
                        data=pid_item,
                        maxshape=(None, ),
                        dtype=uint_dtype_ids,
                    )
                    hdf.create_dataset(
                        name=f'{name}/pos',
                        data=pos_item,
                        maxshape=(None, pos_item.shape[-1]),
                        dtype=dtypes[0],
                    )
                    hdf.create_dataset(
                        name=f'{name}/vel',
                        data=vel_item,
                        maxshape=(None, vel_item.shape[-1]),
                        dtype=dtypes[1],
                    )
                    hdf.create_dataset(
                        name=f'{name}/row_idx',
                        data=row_item,
                        maxshape=(None, ),
                        dtype=uint_dtype_row,
                    )
                # If it is not the first time opening the file, reshape the
                # datasets
                else:
                    last_item = pid_item.shape[0]
                    new_shape = hdf[f'{name}/ID'].shape[0] + last_item

                    hdf[f'{name}/ID'].resize((new_shape), axis=0)
                    hdf[f'{name}/ID'][-last_item:] = pid_item
                    hdf[f'{name}/pos'].resize((new_shape), axis=0)
                    hdf[f'{name}/pos'][-last_item:] = pos_item
                    hdf[f'{name}/vel'].resize((new_shape), axis=0)
                    hdf[f'{name}/vel'][-last_item:] = vel_item
                    hdf[f'{name}/row_idx'].resize((new_shape), axis=0)
                    hdf[f'{name}/row_idx'][-last_item:] = row_item

    return None


def _load_sub_box(
    sub_box_id: int,
    path: str,
    name: str = None,
) -> Tuple[np.ndarray]:
    """Load sub-box

    Parameters
    ----------
    sub_box_id : int
        Sub-box ID
    path : str
        Location from where to load the file
    name : str, optional
        Identifier within the file, by default None

    Returns
    -------
    Tuple[np.ndarray]
        Position, velocity, ID and row index
    """
    if name:
        prefix = f'{name}/'
    else:
        prefix = None
    try:
        with h5.File(path + f'sub_boxes/{sub_box_id}.hdf5', 'r') as hdf:
            pos = hdf[prefix + 'pos'][()]
            vel = hdf[prefix + 'vel'][()]
            pid = hdf[prefix + 'ID'][()]
            row = hdf[prefix + 'row_idx'][()]
    except:
        pos, vel, pid, row = None, None, None, None
    return pos, vel, pid, row


def load_particles(
    sub_box_id: int,
    boxsize: float,
    subsize: float,
    path: str,
    padding: float = 5.0,
) -> Tuple[np.ndarray]:
    """Load particles from a sub-box

    Parameters
    ----------
    sub_box_id : int
        Sub-box ID
    path : str
        Location from where to load the file
    boxsize : float
        Size of simulation box
    subsize : float
        Size of sub-box
    padding : float
        Only particles up to this distance from the sub-box edge are considered 
        for classification. Defaults to 5

    Returns
    -------
    Tuple[np.ndarray]
        Position, velocity, ID and row index
    """
    # Generate the IDs and positions of the sub-box grid
    grid_ids, grid_pos = generate_sub_box_grid(boxsize, subsize)
    # Get the adjacent sub-box IDs
    adj_sub_box_ids = get_adjacent_sub_box_ids(
        sub_box_id=sub_box_id,
        sub_box_ids=grid_ids,
        positions=grid_pos,
        boxsize=boxsize,
        subsize=subsize
    )

    # Create empty lists (containers) to save the data from file for each ID
    pos, vel, pid, row = ([[] for _ in range(len(adj_sub_box_ids))]
                          for _ in range(4))

    # Load all adjacent boxes
    for i, sub_box in enumerate(adj_sub_box_ids):
        pos[i], vel[i], pid[i], row[i] = _load_sub_box(
            sub_box, path, name='part')
    # Concatenate into a single array
    pos = np.concatenate(pos)
    vel = np.concatenate(vel)
    pid = np.concatenate(pid)
    row = np.concatenate(row)

    # Mask particles within a padding distance of the edge of the box in each
    # direction
    loc_id = grid_ids == sub_box_id
    padded_distance = 0.5 * subsize + padding
    rel_abs_position = np.abs(relative_coordinates(
        grid_pos[loc_id], pos, boxsize, periodic=True))
    # Probably a better way to create this mask
    mask_x = (rel_abs_position[:, 0] < padded_distance)
    mask_y = (rel_abs_position[:, 1] < padded_distance)
    mask_z = (rel_abs_position[:, 2] < padded_distance)
    mask = mask_x & mask_y & mask_z

    return pos[mask], vel[mask], pid[mask], row[mask]


def load_seeds(
    sub_box_id: int,
    boxsize: float,
    subsize: float,
    path: str,
    padding: float = 5.0,
    adjacent: bool = False,
) -> Tuple[np.ndarray]:
    """Load seeds from a sub-box

    Parameters
    ----------
    sub_box_id : int
        Sub-box ID
    boxsize : float
        Size of simulation box
    subsize : float
        Size of sub-box
    path : str
        Location from where to load the file
    padding : float
        Only particles up to this distance from the sub-box edge are considered 
        for classification. Defaults to 5
    adjacent : bool
        If True, returns only de adjacent seeds, by default False

    Returns
    -------
    Tuple[np.ndarray]
        Position, velocity, ID and row index
    """
    if adjacent:
        # Generate the IDs and positions of the sub-box grid
        grid_ids, grid_pos = generate_sub_box_grid(boxsize, subsize)
        # Get the adjacent sub-box IDs
        adj_sub_box_ids = get_adjacent_sub_box_ids(
            sub_box_id=sub_box_id,
            sub_box_ids=grid_ids,
            positions=grid_pos,
            boxsize=boxsize,
            subsize=subsize
        )
        # Create empty lists (containers) to save the data from file for each ID
        # pos, vel, pid, row = ([[] for _ in range(len(adj_sub_box_ids)-1)]
        pos, vel, pid, row = ([] for _ in range(4))

        # Load all adjacent boxes
        for sub_box in adj_sub_box_ids[adj_sub_box_ids!=sub_box_id]:
            if sub_box == sub_box_id:
                continue
            else:
                postemp, veltemp, pidtemp, rowtemp = _load_sub_box(
                    sub_box, path, name='seed')
                # If no seeds where found
                if any([p is None for p in (postemp, veltemp, pidtemp, rowtemp)]):
                    continue
                else:
                    pos.append(postemp)
                    vel.append(veltemp)
                    pid.append(pidtemp)
                    row.append(rowtemp)
        # Concatenate into a single array
        pos = np.concatenate(pos)
        vel = np.concatenate(vel)
        pid = np.concatenate(pid)
        row = np.concatenate(row)

        # Mask seeds within a padding distance of the edge of the box in each
        # direction
        loc_id = grid_ids == sub_box_id
        padded_distance = 0.5 * subsize + padding
        rel_abs_position = np.abs(relative_coordinates(
            grid_pos[loc_id], pos, boxsize, periodic=True))
        # Probably a better way to create this mask
        mask_x = (rel_abs_position[:, 0] < padded_distance)
        mask_y = (rel_abs_position[:, 1] < padded_distance)
        mask_z = (rel_abs_position[:, 2] < padded_distance)
        mask = mask_x & mask_y & mask_z

        return pos[mask], vel[mask], pid[mask], row[mask]

    else:
        return _load_sub_box(sub_box_id, path=path, name='seed')


if __name__ == '__main__':
    pass
