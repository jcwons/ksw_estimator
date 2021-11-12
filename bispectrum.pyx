import pywigxjpf as wig
import pythran
cimport cython 
from cython.parallel import parallel, prange

cython: np_pythran=True
cython: cxx=True
import math
import numpy as np
cimport numpy as np

cdef double pi = math.pi


cdef local_shape(np.ndarray[double,ndim=4] beta_s, long long ell1, long long ell2, long long ell3, int l_min, int pidx1, int pidx2, int pidx3, np.ndarray[double,ndim=1] shape_func):
    shape_func =  beta_s[ell1 - l_min, pidx1, 1, :] * beta_s[ell2 - l_min, pidx2, 1, :] * beta_s[ell3 - l_min, pidx3, 0, :] \
                + beta_s[ell2 - l_min, pidx1, 1, :] * beta_s[ell3 - l_min, pidx2, 1, :] * beta_s[ell1 - l_min, pidx3, 0, :] \
                + beta_s[ell3 - l_min, pidx1, 1, :] * beta_s[ell1 - l_min, pidx2, 1, :] * beta_s[ell2 - l_min, pidx3, 0, :]
    shape_func = 2 * shape_func
    return shape_func

cdef equil_shape(np.ndarray[double,ndim=4] beta_s, long long ell1, long long ell2, long long ell3, int l_min, int pidx1, int pidx2, int pidx3, np.ndarray[double,ndim=1] shape_func):
    shape_func = -beta_s[ell1 - l_min, pidx1, 1, :] * beta_s[ell2 - l_min, pidx2, 1, :] * beta_s[ell3 - l_min, pidx3, 0, :] \
                - beta_s[ell2 - l_min, pidx1, 1, :] * beta_s[ell3 - l_min, pidx2, 1, :] * beta_s[ell1 - l_min, pidx3, 0, :] \
                - beta_s[ell3 - l_min, pidx1, 1, :] * beta_s[ell1 - l_min, pidx2, 1, :] * beta_s[ell2 - l_min, pidx3, 0, :]
    # delta delta delta
    shape_func -= beta_s[ell1 - l_min, pidx1, 3, :] * beta_s[ell2 - l_min, pidx2, 3, :] * beta_s[ell3 - l_min, pidx3, 3, :]
    # beta gamma delta
    shape_func += beta_s[ell1 - l_min, pidx1, 1, :] * beta_s[ell2 - l_min, pidx2, 2, :] * beta_s[ell3 - l_min, pidx3, 3, :]
    # bdg
    shape_func += beta_s[ell1 - l_min, pidx1, 1, :] * beta_s[ell2 - l_min, pidx2, 3, :] * beta_s[ell3 - l_min, pidx3, 2, :]
    # gbd
    shape_func += beta_s[ell1 - l_min, pidx1, 3, :] * beta_s[ell2 - l_min, pidx2, 1, :] * beta_s[ell3 - l_min, pidx3, 2, :]
    # dgb
    shape_func += beta_s[ell1 - l_min, pidx1, 3, :] * beta_s[ell2 - l_min, pidx2, 2, :] * beta_s[ell3 - l_min, pidx3, 1, :]
    # gbd
    shape_func += beta_s[ell1 - l_min, pidx1, 2, :] * beta_s[ell2 - l_min, pidx2, 1, :] * beta_s[ell3 - l_min, pidx3, 3, :]
    # gdb
    shape_func += beta_s[ell1 - l_min, pidx1, 2, :] * beta_s[ell2 - l_min, pidx2, 3, :] * beta_s[ell3 - l_min, pidx3, 1, :]
    shape_func = 6 * shape_func
    return shape_func


cdef int delta_l1l2l3(int ell1, int ell2, int ell3):
    """
    Returns the permutation coefficient
    delta = 6 if ell1=ell2=ell3
    delta = 2 if 2 ell are the same
    delta = 1 otherwise
    """
    if ell1 == ell2 == ell3:
        delta = 6
    elif ell1 == ell2 or ell2 == ell3 or ell1 == ell3:
        delta = 2
    else:
        delta = 1
    return delta


def compute_bispec(ells, radii, beta_s, invcov, fisher_sub, pol_trpl, rank, shape):
    # ells are either all ells or in case of paralellising a sub set
    # radii are computed in main.py
    r2 = radii**2
    shape_func = np.zeros_like(radii)
    if shape == 'local':
        shape_temp = 0
    else:
        shape_temp = 1
    compute_bispec_worker(ells, radii, r2, beta_s, invcov, shape_func, fisher_sub, pol_trpl, rank, shape_temp)
    return fisher_sub



@cython.wraparound(False)
@cython.boundscheck(False)
cdef compute_bispec_worker(np.ndarray[long long,ndim=1] ells, np.ndarray[double,ndim=1] radii, np.ndarray[double,ndim=1] r2, np.ndarray[double,ndim=4] beta_s, np.ndarray[double,ndim=3] invcls, np.ndarray[double,ndim=1] shape_func, np.ndarray[double,ndim=1] fisher_sub, np.ndarray[int,ndim=2] pol_trpl, int rank, int shape_temp):
    cdef long long ell3, ell2, ell1
    cdef int shape_factor, lidx3, l_min, pidx1, pidx2, pidx3, pidx4, pidx5, pidx6
    cdef double gaunt, wig3f, fis, fisher, fourpi, bispec1, bispec2 
     
    fourpi = 4 * pi
    shape_factor = 2
    l_min = 2
    #wigner from the library

    # pol_trp
    lidx3=0
    for ell3 in ells:
        fisher = 0 
        if rank == 0: 
            print('lmax:', lidx3+2)#, flush=True)
        for ell2 in range(2, ell3 + 1):
            for ell1 in range(2, ell2 + 1):
                # Wig3j is only non-zero for even sums of ell and triangle equation
                if ((ell3 + ell2 + ell1) % 2) == 1 or ell3 > ell1 + ell2:
                    continue
                for pidx3, pidx2, pidx1 in pol_trpl:
                    gaunt = (2 * ell1 + 1) * (2 * ell2 + 1) * (2 * ell3 + 1) / fourpi *  wig.wig3jj(2 * ell1, 2 * ell2, 2 * ell3, 0, 0, 0)**2
                    # local shape (beta beta alpha + 2 perm)
                    if shape_temp == 0:
                        shape_func = local_shape(beta_s, ell1, ell2, ell3, l_min, pidx1, pidx2, pidx3, shape_func)
                    else:
                        shape_func = equil_shape(beta_s, ell1, ell2, ell3, l_min, pidx1, pidx2, pidx3, shape_func)
                    # calculate the B_(l1,l2,l3)^(X1,X2,X3) and multiply angular factor squared
                    bispec1 = np.trapz(shape_func * r2, radii)
                    bispec1 *= gaunt  
                    for pidx6, pidx5, pidx4 in pol_trpl:
                        if shape_temp == 0:
                            shape_func = local_shape(beta_s, ell1, ell2, ell3, l_min, pidx4, pidx5, pidx6, shape_func)
                        else:
                            shape_func = equil_shape(beta_s, ell1, ell2, ell3, l_min, pidx4, pidx5, pidx6, shape_func)
                        # calculate  B_(l1,l2,l3)^(X4,X5,X6)
                        bispec2 = np.trapz(shape_func*r2, radii)
                        fis = bispec1 * bispec2
                        
                        fis *= (invcls[ell1-l_min, pidx1, pidx4] * invcls[ell2-l_min, pidx2, pidx5] *
                                invcls[ell3-l_min, pidx3, pidx6])
                        fis /= delta_l1l2l3(ell3, ell2, ell1) 

                        fisher += fis
        fisher_sub[lidx3] = fisher
        lidx3 +=1
    return fisher_sub
