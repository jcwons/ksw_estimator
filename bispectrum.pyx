import pywigxjpf as wig
#import pythran
cimport cython 
from cython.parallel import parallel, prange

cython: np_pythran=True
cython: cxx=True
import math
import numpy as np
cimport numpy as np

cdef double pi = math.pi


#cdef double[:] array_mul(double[:] A, double [:] B):
##    cdef int i
#    for i in range(np.size(A))
#        double[i]

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


def compute_bispec(ells, radii, beta_s, cls_lensed, fisher_sub, rank):
    # ells are either all ells or in case of paralellising a sub set
    # radii are computed in main.py
    r2 = radii**2



    # beta_s = cosmo.
    shape_func = np.zeros_like(radii)
    bprim = np.zeros_like(radii)
    #fisher_lmax = np.zeros(ells.size, dtype= 'float64')
    #fisher_sub = np.zeros(ells.size, dtype= 'float64')
    compute_bispec_worker(ells, radii, r2, beta_s, cls_lensed, shape_func, bprim, fisher_sub, rank)
    #self.printmpi('Compute bispectrum')
    #fnl_file = path + '/fnl_lmax.txt'
    return fisher_sub



@cython.wraparound(False)
@cython.boundscheck(False)
cdef compute_bispec_worker(np.ndarray[long long,ndim=1] ells, np.ndarray[double,ndim=1] radii, np.ndarray[double,ndim=1] r2, np.ndarray[double,ndim=4] beta_s, np.ndarray[double,ndim=2] cls_lensed, np.ndarray[double,ndim=1] shape_func, np.ndarray[double,ndim=1] bprim, np.ndarray[double,ndim=1] fisher_sub, int rank):
    cdef long long ell3, ell2, ell1
    cdef int shape_factor, lidx3, l_min, pidx1, pidx2, pidx3
    cdef double gaunt, wig3f, fis, fisher, fourpi
     
    fourpi = 4 * pi
    shape_factor = 2
    l_min = 2
    pidx1, pidx2, pidx3 = [0,0,0]
    #wigner from the library
    lidx3=0
    for ell3 in ells:
        fisher = 0 
        if rank == 0:        
            print(lidx3)
        for ell2 in range(2, ell3 + 1):
            for ell1 in range(2, ell2 + 1):
                # Wig3j is only non-zero for even sums of ell and triangle equation
                if ((ell3 + ell2 + ell1) % 2) == 1 or ell3 > ell1 + ell2:
                    continue
                # for pidx in range(pol_trpl.shape[0]):
                #pidx1, pidx2, pidx3 = [1,1,1]
                gaunt = np.sqrt((2 * ell1 + 1) * (2 * ell2 + 1) * (2 * ell3 + 1) / fourpi) *  wig.wig3jj(2 * ell1, 2 * ell2, 2 * ell3, 0, 0, 0)
                shape_func = beta_s[ell1-l_min, pidx1, 1, :] * beta_s[ell2-l_min, pidx2, 1, :] * beta_s[ell3-l_min, pidx3, 0, :] \
                                + beta_s[ell2-l_min, pidx1, 1, :] * beta_s[ell3-l_min, pidx2, 1, :] * beta_s[ell1-l_min, pidx3, 0, :] \
                                + beta_s[ell3-l_min, pidx1, 1, :] * beta_s[ell1-l_min, pidx2, 1, :] * beta_s[ell2-l_min, pidx3, 0, :]
                bprim = shape_factor * gaunt * shape_func * r2
                #integrand = bprim* r2 
                bispec = np.trapz(bprim, radii)
                fis = bispec
                fis /= (cls_lensed[pidx1, ell1-l_min] * cls_lensed[pidx2, ell2-l_min] *
                        cls_lensed[pidx3, ell3-l_min])
                fis /= delta_l1l2l3(ell3, ell2, ell1)
                fis *= bispec   
                fisher += fis
        fisher_sub[lidx3] = fisher
        lidx3 +=1
    return fisher_sub
    #fisher_full = fisher_sub
    #fnl_max = np.zeros_like(fisher_full)
    #for idx, item in enumerate(fisher_full):
    #    fnl_max[idx] = item + fnl_max[idx-1]
    #fnl_max = 1 / np.sqrt(fnl_max)
    
























    ''' 
    if self.mpi_size > 1:
        if self.mpi_rank == 0:
            print('Combining fisher')
            fisher_full = np.zeros(ells.size)
            # Place root fisher_sub in fisher_full
            fisher_full[ells_per_rank[0] - 2] = fisher_sub
        else:
            fisher = None

        for rank in range(1, self.mpi_size):
            # Send fisher from other ranks over to root
            if self.mpi_rank == rank:
                self.comm.Send(fisher_sub, dest=0, tag=rank)
                #print(self.mpi_rank, 'sent')

            # Receive fisher on root
            if self.mpi_rank == 0:
                ell_size = ells_per_rank[rank].size
                fisher_sub = np.zeros(ell_size)
                self.comm.Recv(fisher_sub, source=rank, tag=rank)
                #print('root received fisher{}'.format(rank))
                fisher_full[ells_per_rank[rank] - 2] = fisher_sub
        fisher_full = self.comm.bcast(fisher_full, root=0)
    fisher_file = path + '/fisher_{}.pkl'.format(ells.size+1)
    with open(fisher_file, 'wb') as handle:
        pickle.dump(fisher_full, handle, protocol=pickle.HIGHEST_PROTOCOL)
    self.printmpi('Done: fisher')
    

    fnl_end = np.sum(fisher_full)
    fnl_end = 1/np.sqrt(fnl_end)

    fnl_max = np.zeros_like(fisher_full)
    for idx, item in enumerate(fisher_full):
        fnl_max[idx] = item + fnl_max[idx-1]
    fnl_max = 1 / np.sqrt(fnl_max)
    self.printmpi(fnl_end)
    np.savetxt(fnl_file, (fnl_max, ells))
    '''
