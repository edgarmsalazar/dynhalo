# -*- coding: utf-8 -*-
""" Some utility routines and constants
"""
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from time import time
from typing import Any, Callable, List

import numpy

__all__ = ["timer, mkdir"]

# Gravitational constant
G_gravity = 4.3e-09     # Mpc (km/s)^2 / M_sun

@dataclass(frozen=True)
class COLS:
    """
    """
    HEADER: str = "\033[95m"
    OKBLUE: str = "\033[94m"
    OKCYAN: str = "\033[96m"
    OKGREEN: str = "\033[92m"
    WARNING: str = "\033[93m"
    FAIL: str = "\033[91m"
    ENDC: str = "\033[0m"
    BOLD: str = "\033[1m"
    UNDERLINE: str = "\033[4m"
    BULLET: str = "\u25CF"


OKGOOD = f"{COLS.OKGREEN}{COLS.BULLET}{COLS.ENDC} "
FAIL = f"{COLS.FAIL}{COLS.BULLET}{COLS.ENDC} "


def get_np_unit_dytpe(num: Any) -> numpy.dtype:
    """Determines the minimum unsigned integer type to represent `num`.

    Parameters
    ----------
    num : Any
        Numerical value.

    Returns
    -------
    numpy.dtype
        Numpy data type class.

    Raises
    ------
    TypeError
        If `num` is not an integer or it is a negative value.
    OverflowError
        If `num` cannot be represented by any 16, 32 or 64 bit unsigned integer.
    """
    np_unit_dtypes = numpy.array([numpy.uint16, numpy.uint32, numpy.uint64])
    check = [num < numpy.iinfo(item).max for item in np_unit_dtypes]
    if num < 0:
        raise TypeError
    if any(check):
        return np_unit_dtypes[numpy.argmax(check)]
    else:
        raise OverflowError


def timer(procedure: Callable, *, fancy=False) -> Callable:
    """Decorator that prints the procedure's execution time

    Parameters
    ----------
    procedure : Callable
        Any callable

    Returns
    -------
    Callable
        Returns callable object/return value
    """

    def wrapper(*args, **kwargs):
        now_start = datetime.now()
        start = time()
        return_value = procedure(*args, **kwargs)
        now_end = datetime.now()
        if fancy:
            print(f"\t{COLS.BOLD}Process:{COLS.ENDC} {COLS.FAIL}{procedure.__name__}{COLS.ENDC}")
            print(f"\t Start:  " + \
                  f"{COLS.HEADER}{now_start.strftime('%Y-%m-%d %H:%M:%S')}{COLS.ENDC}")
            print(f"\t Finish: " + \
                  f"{COLS.OKCYAN}{now_end.strftime('%Y-%m-%d %H:%M:%S')}{COLS.ENDC}")
            print(
                f"\t{COLS.BULLET}{COLS.BOLD}{COLS.OKGREEN} Elapsed time:{COLS.ENDC} "
                + f"{COLS.WARNING}{timedelta(seconds=time()-start)}{COLS.ENDC} "
            )
        else:
            print(f"\t Process: {procedure.__name__}")
            print(f"\t Start:  {now_start.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"\t Finish: {now_end.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"\t {COLS.BULLET} Elapsed time: {timedelta(seconds=time()-start)}")
        return return_value

    return wrapper


def mkdir(path: str, verbose: bool = False) -> None:
    """Checks if a path exists and creates a directory in path if not.

    Parameters
    ----------
    path : str
        Path where directory should exist. 
    verbose : bool, optional
        Whether to print info on directory creation process, by default False

    Returns
    -------
    None
    """
    abspath = os.path.abspath(path)
    isdir = os.path.isdir(abspath)
    if isdir:
        if verbose:
            print(f"Directory exists at {abspath}")
        return None
    else:
        print("Creating directory")
        try:
            os.mkdir(os.path.abspath(path))
            if verbose:
                print(f"Directory created at {abspath}")
        except:
            print(f"Directory could not be created at {abspath}")
            raise
    return None


def cartesian_product(arrays: List[numpy.ndarray]) -> numpy.ndarray:
    """Generalized N-dimensional products
    Taken from https://stackoverflow.com/questions/11144513/
    Answer by Nico Schlömer
    Updated for numpy > 1.25

    Parameters
    ----------
    arrays : List[numpy.ndarray]
        _description_

    Returns
    -------
    _type_
        _description_
    """
    la = len(arrays)
    dtype = numpy.result_type(*[a.dtype for a in arrays])
    arr = numpy.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(numpy.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def gen_data_pos_regular(boxsize: float, gridsize: float) -> numpy.ndarray:
    """Populate coordinates with one particle per subbox at the centre in steps
    of nside between particles.

    Parameters
    ----------
    boxsize : float
        Length of the box.
    gridsize : float
        Length of the grid.

    Returns
    -------
    numpy.ndarray
        
    """
    # Number of cells per side.
    n_per_side = numpy.int_(numpy.ceil(boxsize / gridsize))
    
    # Determine data type for integer arrays based on the maximum number of
    # elements.
    uint_dtype = get_np_unit_dytpe(n_per_side)
    
    # Set of natural numbers from 0 to N-1.
    n_range = numpy.arange(n_per_side, dtype=uint_dtype)

    # Set of index vectors. Each vector points to the (i, j, k)-th cell.
    pos = numpy.int_(cartesian_product([n_range, n_range, n_range]))
    centre = gridsize * (pos + 0.5)
    return centre


def gen_data_pos_random(boxsize: float, nsamples: int, seed=None) -> numpy.ndarray:
    """Generate random data points inside a cubic box.

    Parameters
    ----------
    boxsize : float
        Length of the box.
    nsamples : int
        Number of points to sample.
    seed : _type_, optional
        Random gnerator seed, by default None

    Returns
    -------
    numpy.ndarray

    """
    numpy.random.seed(seed=seed)
    data_pos = boxsize * numpy.random.uniform(0, 1, (nsamples, 3))
    return data_pos


if __name__ == '__main__':
    pass
