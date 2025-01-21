import os
from typing import Tuple

import h5py as h5
import numpy as np
from tqdm import tqdm

from dynhalo.finder.coordinates import relative_coordinates
from dynhalo.utils import cartesian_product, get_np_unit_dytpe, timer


def generate_mini_box_grid(
    boxsize: float,
    minisize: float,
) -> Tuple[np.ndarray]:
    """Generates a 3D grid of mini boxes.

    Parameters
    ----------
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box

    Returns
    -------
    Tuple[np.ndarray]
        ID and centre coordinate for all mini boxes
    """

    # Number of mini boxes per side
    boxes_per_side = np.int_(np.ceil(boxsize / minisize))

    # Determine data type for integer arrays based on the maximum number of
    # elements
    uint_dtype = get_np_unit_dytpe(boxes_per_side)
    # Set of natural numbers from 0 to N-1
    n_range = np.arange(boxes_per_side, dtype=uint_dtype)

    # Shift in each dimension for numbering mini boxes
    uint_dtype = get_np_unit_dytpe(boxes_per_side**2)
    shift = np.array(
        [1, boxes_per_side, boxes_per_side * boxes_per_side], dtype=uint_dtype)

    # Set of index vectors. Each vector points to the (i, j, k)-th mini box
    n_pos = np.int_(cartesian_product([n_range, n_range, n_range]))

    # Set of all possible unique IDs for each mini box
    ids = np.sum(n_pos * shift, axis=1)
    sort_order = np.argsort(ids)

    # Sort IDs so that the ID matches the row index.
    n_pos = n_pos[sort_order]
    ids = ids[sort_order]

    # Sub-box central coordinate. Populate each mini box with one point at the
    # centre.
    centres = minisize * (n_pos + 0.5)
    return ids, centres


def get_mini_box_id(
    x: np.ndarray,
    boxsize: float,
    minisize: float,
) -> int:
    """Returns the mini box ID to which the coordinates `x` fall into

    Parameters
    ----------
    x : np.ndarray
        Position in cartesian coordinates
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box

    Returns
    -------
    int
        ID of the mini box
    """
    # Number of mini boxes per side
    boxes_per_side = np.int_(np.ceil(boxsize / minisize))
    # Determine data type for integer arrays based on the maximum number of
    # elements
    uint_dtype = get_np_unit_dytpe(boxes_per_side**3)
    # Shift in each dimension for numbering mini boxes
    shift = np.array(
        [1, boxes_per_side, boxes_per_side * boxes_per_side], dtype=uint_dtype)
    # In the rare case an object is located exactly at the edge of the box,
    # move it 'inwards' by a tiny amount so that the box id is correct.
    x[np.where(x==boxsize)] -= 1e-8
    x[np.where(x==0)] += 1e-8
    if x.ndim > 1:
        return np.int_(np.sum(shift * np.floor(x / minisize), axis=1))
    else:
        return np.int_(np.sum(shift * np.floor(x / minisize)))


def get_adjacent_mini_box_ids(
    mini_box_id: np.ndarray,
    mini_box_ids: np.ndarray,
    positions: np.ndarray,
    boxsize: float,
    minisize: float,
) -> np.ndarray:
    """Returns a list of all IDs that are adjacent to the specified mini box ID.
    There are always 27 adjacent boxes in a 3D volume, including the specified ID.

    Parameters
    ----------
    mini_box_id : np.ndarray
        ID of the mini box
    mini_box_ids : np.ndarray
        IDs of all mini boxes
    positions : np.ndarray
        Positions of all the centres of the mini boxes
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box

    Returns
    -------
    np.ndarray
        List of mini box IDs adjacent to `id`

    Raises
    ------
    ValueError
        If `id` is not found in the allowed values in `ids`
    """
    if mini_box_id not in mini_box_ids:
        raise ValueError(f'ID {mini_box_id} is out of bounds')

    x0 = positions[mini_box_ids == mini_box_id]
    d = relative_coordinates(x0, positions, boxsize)
    d = np.sqrt(np.sum(np.square(d), axis=1))
    mask = d <= 1.01 * np.sqrt(3.) * minisize
    return mini_box_ids[mask]


@timer
def generate_mini_box_ids(
    positions: np.ndarray,
    boxsize: float,
    minisize: float,
    save_path: str,
    chunksize: int = 100_000,
    name: str = None
) -> None:
    """Gets the mini box ID for each position

    Parameters
    ----------
    positions : np.ndarray
        Cartesian coordinates
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
    save_path : str
        Where to save the IDs
    chunksize : int, optional
        Number of items to process at a time in chunks, by default 100_000
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
    boxes_per_side = np.int_(np.ceil(boxsize / minisize))
    uint_dtype = get_np_unit_dytpe(boxes_per_side**3)

    ids = np.zeros(n_items, dtype=uint_dtype)

    for chunk in tqdm(range(n_iter), desc='Chunk', ncols=100, colour='blue'):
        low = chunk * chunksize
        if chunk < n_iter - 2:
            upp = (chunk + 1) * chunksize
        else:
            upp = None
        ids[low:upp] = get_mini_box_id(positions[low:upp], boxsize, minisize)

    if name:
        file_name = f'mini_box_id_nside_{boxes_per_side}_{name}.hdf5'
    else:
        file_name = f'mini_box_id_nside_{boxes_per_side}.hdf5'
    with h5.File(save_path + file_name, 'w') as hdf:
        hdf.create_dataset('MBID', data=ids, dtype=uint_dtype)

    return None


@timer
def split_box_into_mini_boxes(
    positions: np.ndarray,
    velocities: np.ndarray,
    uid: np.ndarray,
    save_path: str,
    boxsize: float, 
    minisize: float,
    chunksize: int = 100_000,
    name: str = None,
    props: Tuple[list, list, list] = None
) -> None:
    """Sorts all items into mini boxes and saves them in disc.

    Parameters
    ----------
    positions : np.ndarray
        Cartesian coordinates
    velocities : np.ndarray
        Cartesian velocities
    pid : np.ndarray
        Unique IDs for each position (e.g. PID, HID)
    save_path : str
        Where to save the IDs
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
    chunksize : int, optional
        Number of items to process at a time in chunks, by default 100_000
    name : str, optional
        An additional name or identifier appended at the end of the file name, 
        by default None
    props : tuple[list(array), list(str), list(dtype)], optional
        Additional arrays to be sorted into mini boxes.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If `chunksize` is too small, the chunk-finding loop cannot properly 
        resolve all the mini box ids within a chunk.
    """
    # Determine number of partitions per side
    boxes_per_side = np.int_(np.ceil(boxsize / minisize))

    # Load mini box ids
    if name:
        mini_box_ids_file = save_path + \
            f'mini_box_id_nside_{boxes_per_side}_{name}.hdf5'
    else:
        mini_box_ids_file = save_path + \
            f'mini_box_id_nside_{boxes_per_side}.hdf5'
    # Load if file already exists
    try:
        with h5.File(mini_box_ids_file, 'r') as hdf:
            mini_box_ids = hdf['MBID'][()]
    # Create and file otherwise
    except:
        generate_mini_box_ids(
            positions=positions, 
            boxsize=boxsize, 
            minisize=minisize,
            path=save_path,
            chunksize=chunksize,
            name=name,
            )
        with h5.File(mini_box_ids_file, 'r') as hdf:
            mini_box_ids = hdf['MBID'][()]

    # Create target directory
    save_dir = save_path + f'mini_boxes_nside_{boxes_per_side}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Sort items by mini box id
    mb_order = np.argsort(mini_box_ids)

    # Get smallest data type to represent IDs
    uint_dtype_pid = get_np_unit_dytpe(np.max(uid))

    # Get smallest data type to represent the row index of each item
    n_items = mini_box_ids.shape[0]
    uint_dtype_row = get_np_unit_dytpe(n_items)
    row_idx = np.arange(n_items, dtype=uint_dtype_row)
    
    # Sort data by mini box id
    mb_order = np.argsort(mini_box_ids)
    mini_box_ids = mini_box_ids[mb_order]
    velocities = velocities[mb_order]
    positions = positions[mb_order]
    row_idx = row_idx[mb_order]
    uid = uid[mb_order]

    if props:
        props = props[0]
        labels = props[1]
        dtypes = props[2]
        for k, item in enumerate(props):
            props[k] = item[mb_order]

    # Get chunk slices
    n_items = mini_box_ids.shape[0]

    chunk_idx = [0,]
    i, upp = 0, 0
    i_max = int(1.1*(mini_box_ids.shape[0]//chunksize))
    while True:
        if i > i_max:
            print(chunk_idx, i, upp)
            raise RuntimeError(f'Maximum iterations reached i_max = {i_max}' + \
                               'Chunk size too small. Please increase it.')
        low = chunk_idx[-1]
        upp = low + chunksize
        if upp < n_items:
            idx = low + np.argmin(mini_box_ids[low:] - mini_box_ids[upp])
            chunk_idx.append(idx)
            i += 1
        else:
            idx = -1
            chunk_idx.append(idx)
            break

    if props:
        labels = ('ID', 'pos', 'vel', 'row_idx', *labels)
        dtypes = (uint_dtype_pid, np.float32, np.float32, uint_dtype_row, 
                  *dtypes)
    else:
        labels = ('ID', 'pos', 'vel', 'row_idx')
        dtypes = (uint_dtype_pid, np.float32, np.float32, uint_dtype_row)

    # For each chunk
    for chunk_i in tqdm(range(len(chunk_idx)-1), desc='Processing chunks',
                        ncols=100, colour='blue'):
        # Select chunk
        low = chunk_idx[chunk_i]
        upp = chunk_idx[chunk_i + 1]

        mb_chunk = mini_box_ids[low : upp]
        pos_chunk = positions[low : upp]
        vel_chunk = velocities[low : upp]
        pid_chunk = uid[low : upp]
        row_chunk = row_idx[low : upp]

        if props:
            props_chunks = [None for _ in range(len(props))]
            for k, item in enumerate(props):
                props_chunks[k] = item[low : upp]

        # Check which mini box ids are in the chunk
        mb_chunk_low = mb_chunk[0]
        mb_chunk_upp = mb_chunk[-1] + 1

        # Get index (search sorted style) of the first occurence of each distinct 
        # mini box id. Append a -1 at the end for completeness.
        indexed_slice = []
        for mb_id in range(mb_chunk_low, mb_chunk_upp):
            indexed_slice.append(np.argmin(mb_chunk - mb_id))
        indexed_slice.append(-1)

        # Save data per slice
        for i, mb_id in enumerate(range(mb_chunk_low, mb_chunk_upp)):
            if props:
                data = (
                    pid_chunk[indexed_slice[i] : indexed_slice[i+1]],
                    pos_chunk[indexed_slice[i] : indexed_slice[i+1]],
                    vel_chunk[indexed_slice[i] : indexed_slice[i+1]],
                    row_chunk[indexed_slice[i] : indexed_slice[i+1]],
                    *[
                        item_chunk[indexed_slice[i] : indexed_slice[i+1]] \
                            for item_chunk in props_chunks
                    ],
                )
            else:
                data = (
                    pid_chunk[indexed_slice[i] : indexed_slice[i+1]],
                    pos_chunk[indexed_slice[i] : indexed_slice[i+1]],
                    vel_chunk[indexed_slice[i] : indexed_slice[i+1]],
                    row_chunk[indexed_slice[i] : indexed_slice[i+1]],
                )
                
            with h5.File(save_dir + f'{mb_id}.hdf5', 'a') as hdf:
                if not name in hdf.keys():
                    hdf.create_group(name)

                for (label_i, data_i, dtype_i) in zip(labels, data, dtypes):
                    hdf.create_dataset(name=f'{name}/{label_i}', data=data_i,
                                       dtype=dtype_i)

    return None


def _load_mini_box(
    mini_box_id: int,
    load_path: str,
    boxes_per_side: int,
    name: str = None,
) -> Tuple[np.ndarray]:
    """Load seed or particle data in mini box

    Parameters
    ----------
    mini_box_id : int
        Sub-box ID
    path : str
        Location from where to load the file
    boxes_per_side : int
        Number of partitions per side of the simulation box
    name : str, optional
        Identifier within the file: seed or particle. If `None`, the method 
        returns None for each array. By default None

    Returns
    -------
    Tuple[np.ndarray]
        Position, velocity, ID and row index.

    None if mini box file does not exist or `name` was set to None.

    """
    if name:
        prefix = f'{name}/'
    else:
        prefix = None
    try:
        file_name = f'mini_boxes_nside_{boxes_per_side}/{mini_box_id}.hdf5'
        with h5.File(load_path + file_name, 'r') as hdf:
            pos = hdf[prefix + 'pos'][()]
            vel = hdf[prefix + 'vel'][()]
            pid = hdf[prefix + 'ID'][()]
            row = hdf[prefix + 'row_idx'][()]
    except:
        pos, vel, pid, row = None, None, None, None
    return pos, vel, pid, row


def load_particles(
    mini_box_id: int,
    boxsize: float,
    minisize: float,
    load_path: str,
    padding: float = 5.0,
) -> Tuple[np.ndarray]:
    """Load particles from a mini box including all particles in adjacent boxes 
    up to the `padding` distance.

    Parameters
    ----------
    mini_box_id : int
        Sub-box ID
    load_path : str
        Location from where to load the file
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
    padding : float
        Only particles up to this distance from the mini box edge are considered 
        for classification. Defaults to 5.

    Returns
    -------
    Tuple[np.ndarray]
        Position, velocity, ID and row index
    """
    # Determine number of partitions per side
    boxes_per_side = np.int_(np.ceil(boxsize / minisize))

    # Generate the IDs and positions of the mini box grid
    grid_ids, grid_pos = generate_mini_box_grid(boxsize, minisize)

    # Get the adjacent mini box IDs
    adj_mini_box_ids = get_adjacent_mini_box_ids(
        mini_box_id=mini_box_id,
        mini_box_ids=grid_ids,
        positions=grid_pos,
        boxsize=boxsize,
        minisize=minisize
    )

    # Create empty lists (containers) to save the data from file for each ID
    pos, vel, pid, row = ([[] for _ in range(len(adj_mini_box_ids))]
                          for _ in range(4))

    # Load all adjacent boxes
    for i, mini_box in enumerate(adj_mini_box_ids):
        pos[i], vel[i], pid[i], row[i] = _load_mini_box(
            mini_box, load_path, boxes_per_side, name='part')
    # Concatenate into a single array
    pos = np.concatenate(pos)
    vel = np.concatenate(vel)
    pid = np.concatenate(pid)
    row = np.concatenate(row)

    # Mask particles within a padding distance of the edge of the box in each
    # direction
    loc_id = grid_ids == mini_box_id
    padded_distance = 0.5 * minisize + padding
    rel_abs_position = np.abs(relative_coordinates(
        grid_pos[loc_id], pos, boxsize, periodic=True))
    # Probably a better way to create this mask
    mask_x = (rel_abs_position[:, 0] < padded_distance)
    mask_y = (rel_abs_position[:, 1] < padded_distance)
    mask_z = (rel_abs_position[:, 2] < padded_distance)
    mask = mask_x & mask_y & mask_z

    return pos[mask], vel[mask], pid[mask], row[mask]


def load_seeds(
    mini_box_id: int,
    boxsize: float,
    minisize: float,
    load_path: str,
    padding: float = 5.0,
    adjacent: bool = False,
) -> Tuple[np.ndarray]:
    """Load seeds from a mini box

    Parameters
    ----------
    mini_box_id : int
        Sub-box ID
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
    load_path : str
        Location from where to load the file
    padding : float
        Only particles up to this distance from the mini box edge are considered 
        for classification. Defaults to 5
    adjacent : bool
        If True, returns only de adjacent seeds, by default False

    Returns
    -------
    Tuple[np.ndarray]
        Position, velocity, ID and row index
    """
    # Determine number of partitions per side
    boxes_per_side = np.int_(np.ceil(boxsize / minisize))

    if adjacent:
        # Generate the IDs and positions of the mini box grid
        grid_ids, grid_pos = generate_mini_box_grid(boxsize, minisize)
        # Get the adjacent mini box IDs
        adj_mini_box_ids = get_adjacent_mini_box_ids(
            mini_box_id=mini_box_id,
            mini_box_ids=grid_ids,
            positions=grid_pos,
            boxsize=boxsize,
            minisize=minisize
        )

        # Create empty lists (containers) to save the data from file for each ID
        pos, vel, pid, row, r200, m200 = ([] for _ in range(6))

        # Load all adjacent boxes
        for mini_box in adj_mini_box_ids[adj_mini_box_ids!=mini_box_id]:
            if mini_box == mini_box_id:
                continue
            else:
                postemp, veltemp, pidtemp, rowtemp = _load_mini_box(
                    mini_box, load_path, boxes_per_side, name='seed')
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
        loc_id = grid_ids == mini_box_id
        padded_distance = 0.5 * minisize + padding
        rel_abs_position = np.abs(relative_coordinates(
            grid_pos[loc_id], pos, boxsize, periodic=True))
        # Probably a better way to create this mask
        mask_x = (rel_abs_position[:, 0] < padded_distance)
        mask_y = (rel_abs_position[:, 1] < padded_distance)
        mask_z = (rel_abs_position[:, 2] < padded_distance)
        mask = mask_x & mask_y & mask_z

        return pos[mask], vel[mask], pid[mask], row[mask]

    else:
        data = _load_mini_box(mini_box_id, load_path, boxes_per_side,name='seed')
        file_name = f'mini_boxes_nside_{boxes_per_side}/{mini_box_id}.hdf5'
        prefix = 'seed/'
        with h5.File(load_path + file_name, 'r') as hdf:
            r200 = hdf[prefix + 'R200b'][()]
            m200 = hdf[prefix + 'M200b'][()]

        return *data, r200, m200


if __name__ == '__main__':
    pass
