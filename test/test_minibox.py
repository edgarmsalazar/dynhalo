import os
import numpy
import h5py

from dynhalo.finder.minibox import (generate_mini_box_grid,
                                    generate_mini_box_ids,
                                    get_adjacent_mini_box_ids, get_mini_box_id)
from dynhalo import utils

l_box = 100
l_mb = 20

def test_generate_mini_box_grid():
    """Check if `generate_mini_box_grid` creates a regular grid."""
    ids, centres = generate_mini_box_grid(boxsize=l_box, minisize=l_mb)

    assert len(ids) == len(centres)  # Length of arrays is the same
    assert len(ids) == numpy.int_(numpy.ceil(l_box / l_mb))**3  # Number of elements is (l_box/l_mb)**3
    assert all(centres[0] == numpy.full(3, 0.5*l_mb))  # First position is shifted by l_mb/2


def test_get_mini_box_id():
    """Check if `get_mini_box_id` generates the right IDs for each particle."""
    _, centres = generate_mini_box_grid(boxsize=l_box, minisize=l_mb)
    n_mb = numpy.int_(numpy.ceil(l_box / l_mb))**3

    # Partition box and retrive subbox ID for each particle. Only one particle
    # per subbox.
    box_ids = get_mini_box_id(x=centres, boxsize=l_box, minisize=l_mb)

    assert box_ids[0] == 0  # First particle is in the first box with ID = 0
    assert box_ids[-1] == n_mb - 1  # Last particle is in the last box with ID = 999
    assert len(numpy.unique(box_ids)) == n_mb  # One particle per subbox


def test_get_adjacent_mini_box_ids():
    """Check if the number of adjacent miniboxes is in fact 27."""
    ids, centres = generate_mini_box_grid(boxsize=l_box, minisize=l_mb)

    adj_ids = get_adjacent_mini_box_ids(
        mini_box_id=0,
        mini_box_ids=ids,
        positions=centres,
        boxsize=l_box,
        minisize=l_mb,
    )
    assert len(adj_ids) == 27


def test_generate_mini_box_ids():
    """Sort items into miniboxes according to their positions."""
    n_samples = 1000
    chunk_size = 10
    seed = 1234
    pos = utils.gen_data_pos_random(l_box, n_samples, seed)

    path = os.getcwd() + '/test/tmp_dir_test/'
    utils.mkdir(path)

    generate_mini_box_ids(pos, l_box, l_mb, path, chunk_size, 'test')
    file_name = path + 'mini_box_id_nside_5_test.hdf5'

    hdf = h5py.File(file_name, 'r')
    assert 'MBID' in hdf.keys()
    
    file_ids = hdf['MBID'][()]
    assert min(file_ids) == 0
    assert max(file_ids) <= numpy.int_(numpy.ceil(l_box / l_mb))**3
    assert len(file_ids) == n_samples

    # Tidy up
    hdf.close()
    os.remove(file_name)
    
    # Check without `name`
    generate_mini_box_ids(pos, l_box, l_mb, path, chunk_size)
    file_name = path + 'mini_box_id_nside_5.hdf5'
    
    # Tidy up
    os.remove(file_name)
    os.removedirs(path)
