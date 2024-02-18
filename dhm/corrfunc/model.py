import numpy as np
from scipy.special import erf as scierf


def error_func_pos_incr(x: float, c0: float, cm: float, cv: float) -> float:
    x = (x - cm) / (np.sqrt(2) * cv)
    return 0.5 * c0 * (1 + scierf(x))


def error_func_pos_decr(x: float, c0: float, cm: float, cv: float) -> float:
    x = (x - cm) / (np.sqrt(2) * cv)
    return 0.5 * c0 * (1 - scierf(x))


if __name__ == "__main__":
    pass
