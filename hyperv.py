import numpy as np
from pymoo.indicators.hv import HV

def hypervolume(ref_point, A):
    ref_point = np.array(ref_point)
    A = np.array(list(A))
    ind = HV(ref_point=ref_point)
    hpv = ind(A)
    volume = (np.prod(ref_point, dtype=np.float64))
    return (hpv / volume) * 100


