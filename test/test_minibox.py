import numpy as np
# import pytest
from dynhalo.finder.minibox import get_adjacent_sub_box_ids, get_sub_box_id
from dynhalo.utils import cartesian_product

# Generate 3D synthetic data. 100 u subvolumes in a 1000 u box.
l_box = 100
l_sv = 20
n_svps = np.int_(np.ceil(l_box / l_sv))
shift = np.array([1, n_svps, n_svps * n_svps], dtype=int)

# Populate coordinates with one particle per subbox at the centre in steps
# of l_sv between particles
n_range = np.arange(n_svps, dtype=int)
n_sv = n_svps * n_svps * n_svps
n_pos = np.int_(cartesian_product([n_range, n_range, n_range]))
id_sv = np.sum(n_pos * shift, axis=1)
x_sv = l_sv * (n_pos + 0.5)


def test_adjacent_svid():
    adj_ids = get_adjacent_sub_box_ids(
        sub_box_id=0,
        sub_box_ids=id_sv,
        positions=x_sv,
        boxsize=l_box,
        subsize=l_sv,
    )
    assert len(adj_ids) == 27
    # assert all([item in adj_ids[i] for i, item in enumerate(adj_ids)])


def test_fsvid():
    """Check if `fsvid` generates the right IDs for each particle"""
    # Partition box and retrive subbox ID for each particle. Only one particle
    # per subbox.
    box_ids = get_sub_box_id(x=x_sv, boxsize=l_box, subsize=l_sv)

    assert box_ids[0] == 0  # First particle is in the first box with ID = 0
    assert box_ids[-1] == n_sv - 1  # Last particle is in the last box with ID = 999
    assert len(np.unique(box_ids)) == n_sv  # One particle per subbox

