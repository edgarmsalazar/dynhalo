import os

import numpy
import pytest

from dynhalo import utils


def test_get_np_unit_dytpe():
    # Check for input type
    with pytest.raises(TypeError):
        utils.get_np_unit_dytpe(-1)
        utils.get_np_unit_dytpe(1.0)
    
    # Check that the correct type is returned
    assert utils.get_np_unit_dytpe(65_534) == numpy.uint16
    assert utils.get_np_unit_dytpe(4_294_967_294) == numpy.uint32
    assert utils.get_np_unit_dytpe(4_294_967_295) == numpy.uint64
    assert utils.get_np_unit_dytpe(18_446_744_073_709_551_614) == numpy.uint64

    # Check that it overflows if the number is too large for unit64
    with pytest.raises(OverflowError):
        utils.get_np_unit_dytpe(18_446_744_073_709_551_615)


def test_mkdir():
    dirpath_fail = '/zzz/tmp_dir_test/'
    with pytest.raises(FileNotFoundError):
        utils.mkdir(dirpath_fail, verbose=False)

    dirpath_pass = os.getcwd() + '/test/tmp_dir_test/'
    utils.mkdir(dirpath_pass, verbose=False)
    assert os.path.exists(dirpath_pass)
    os.removedirs(dirpath_pass)


def test_cartesian_product():
    # List [0, 1, 2]
    n_points = 3
    points = numpy.arange(n_points)
    points_float = numpy.linspace(0, n_points-1, n_points)

    # Repeat the list twice and thrice
    arrs_2 = 2 * [points]
    arrs_3 = 3 * [points]

    cart_prod_1 = utils.cartesian_product([points])
    cart_prod_2 = utils.cartesian_product(arrs_2)
    cart_prod_3 = utils.cartesian_product(arrs_3)
    cart_prod_float = utils.cartesian_product([points, points_float])

    # Cardinality of Nx...xN = N^n
    assert len(cart_prod_1) == n_points
    assert len(cart_prod_2) == n_points*n_points
    assert len(cart_prod_3) == n_points*n_points*n_points
    # Each element has shape (n,)
    assert cart_prod_1[0].shape == (1,)
    assert cart_prod_2[0].shape == (2,)
    assert cart_prod_3[0].shape == (3,)
    # Check dtypes
    assert type(cart_prod_1[0][0]) == numpy.int64
    assert type(cart_prod_float[0][0]) == numpy.float64


def test_gen_data_pos_regular():
    """Check if `gen_data_pos_regular` creates a regular grid."""
    l_box = 100.
    l_mb = 20.

    pos = utils.gen_data_pos_regular(l_box, l_mb)

    assert len(pos) == numpy.int_(numpy.ceil(l_box / l_mb))**3  # Number of elements is (l_box/l_mb)**3
    assert all(pos[0] == numpy.full(3, 0.5*l_mb))  # First position is shifted by l_mb/2


def test_gen_data_pos_random():
    """Check if `gen_data_pos_random` generates the right number of samples and
    within the box."""
    l_box = 100.
    n_samples = 1000
    seed = 1234

    pos = utils.gen_data_pos_random(l_box, n_samples, seed)

    assert pos.shape == (n_samples, 3)
    assert numpy.max(pos) <= l_box
    assert numpy.min(pos) >= 0