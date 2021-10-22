import numpy as np
import cosmo as cs
from scipy.special import spherical_jn
from scipy.integrate import trapz
import pywigxjpf as wig
import pickle
import os
import sys
# import itertools
import timeit
from mpi4py import MPI

path = os.getcwd()+'/Output'

def get_updated_radii():
    """
    Get the radii (in Mpc) that are more suitable for the sst case.
    """

    low = np.linspace(0, 9377, num=98, dtype=float, endpoint=False)
    re1 = np.linspace(9377, 10007, num=18, dtype=float, endpoint=False)
    re2 = np.linspace(10007, 12632, num=25, dtype=float, endpoint=False)
    rec = np.linspace(12632, 13682, num=50, dtype=float, endpoint=False)
    rec_new = np.linspace(13682, 15500, num=300, dtype=float, endpoint=False)
    rec_extra = np.linspace(15500, 18000, num=10, dtype=float, endpoint=False)

    radii = np.concatenate((low, re1, re2, rec, rec_new, rec_extra))

    return radii


def ps_idx(p1, p2):
    """
    Function to return correct power spectrum index
    cls are saved as (TT,EE,TE)
    Probably need to update this once we include proper inversion of cls
    """
    if p1 == p2:
        return p1
    else:
        return 3


def get_default_radii():
    """
    Get the radii (in Mpc) used in table 1 of liquori 2007
    Gives the wrong fn(ell) scaling because large radii are missing
    """

    low = np.linspace(0, 9377, num=98, dtype=float, endpoint=False)
    re1 = np.linspace(9377, 10007, num=18, dtype=float, endpoint=False)
    re2 = np.linspace(10007, 12632, num=25, dtype=float, endpoint=False)
    rec = np.linspace(12632, 13682, num=300, dtype=float, endpoint=False)

    radii = np.concatenate((low, re1, re2, rec))

    return radii


def delta_l1l2l3(ell1, ell2, ell3):
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


class PreCalc:
    def __init__(self):
        self.cosmo = {}
        self.bins = {}
        self.beta = {}
        self.bispec = {}

        self.comm = MPI.COMM_WORLD
        self.mpi_rank = self.comm.Get_rank()  # Assigns every core a rank (integer value)
        self.mpi_size = self.comm.Get_size()

        # Initialise wigner 3j tables
        wig.wig_table_init(2 * 3000, 9)
        wig.wig_temp_init(2 * 3000)

    def printmpi(self, text):
        if self.mpi_rank == 0:
            print(text, flush=True)

    def init_cosmo(self, lmax):
        """
        Collect transfer functions and cls to save in self.cosmo
        1) Run CAMB and get T,E,B
        2) Read in transfer and cls from kSZ and pSZ
        """
        cosmo_file = path + '/cosmo_{}.pkl'.format(lmax - 100)
        recompute_cosmo = False
        if self.mpi_rank == 0:
            try:
                pkl_file = open(cosmo_file, 'rb')
            except IOError:
                print('{} not found'.format(cosmo_file))
                recompute_cosmo = True
            else:
                print('loaded cosmo from {}'.format(cosmo_file))
                self.cosmo = pickle.load(pkl_file)
                pkl_file.close()
        # Tell all ranks if file was found
        recompute_cosmo = self.comm.bcast(recompute_cosmo, root=0)

        if recompute_cosmo is False:
            self.cosmo = self.comm.bcast(self.cosmo, root=0)

        if recompute_cosmo:
            self.printmpi('Runnnig CAMB for transfer, k, ells and cls ')
            transfer, cls = cs.run_camb(lmax=lmax, lSampleBoost=50)
            self.cosmo['transfer'] = transfer
            self.cosmo['cls'] = cls
            if self.mpi_rank == 0:
                print('Storing cosmo as: {}'.format(cosmo_file))
                # Store in pickle file.
                with open(cosmo_file, 'wb') as handle:
                    pickle.dump(self.cosmo, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
            self.printmpi('Computed cosmo')

    def alpha_prim(self, k):
        """
        Returns shape function
        For local:  alpha_prim = 1
        """
        if k is None:
            k = self.cosmo['scalar']['k']
        alpha_prim = np.ones(k.size)
        return alpha_prim

    def beta_prim(self, k):
        """
        Returns shape function
                    beta = norm * (k/k_pivot)^(ns-4)

        norm = (3/5)^2 * As * 2 * pi^2 /k_p^3
        """
        norm = np.sqrt(6.234181826176155e-15)  # Need to update this part
        kpivot = 0.05
        ns = 0.965
        if k is None:
            k = self.cosmo['scalar']['k']
        km3 = k ** -3
        km3 *= (k / kpivot) ** (ns - 1)
        beta_prim = km3 * norm
        return beta_prim

    def init_beta(self, prim_shape='local'):
        """
        Sets up calculation of beta:
        2/pi int k^2 dk f(k) j_l(k*r) T_{x,ell}(k)
        if precomputed skips calculation, read file and stores in dict
        """

        ells = self.cosmo['cls']['ells']
        lmax = ells.size + 1

        #        radii = get_default_radii()
        radii = get_updated_radii()
        self.beta['radii'] = radii

        pols_s = ['T', 'E']

        # Check if beta already has been computed
        beta_file = path + '/beta_{}.pkl'.format(lmax)
        try:
            pkl_file = open(beta_file, 'rb')
            recompute_beta = False
        except IOError:
            self.printmpi('{} not found'.format(beta_file))
            recompute_beta = True
        else:
            self.printmpi('loaded beta from {}'.format(beta_file))
            self.beta['beta_s'] = pickle.load(pkl_file)
            pkl_file.close()
        # Put this whole computation into a function
        if recompute_beta:
            self.compute_beta(radii, pols_s, ells, prim_shape)

    def compute_beta(self, radii, pols_s, ells, prim_shape):
        """
        Computes 2/pi int k^2 dk f(k) j_l(k*r) T_{x,ell}(k)
        Splits calculation among ells
        :param radii: array float
        :param pols_s: array str ['T','E']
        :param ells: array int, size lmax
        :param prim_shape: str at the moment only supports local
        :return:beta(ells_sub.size, len(pols_s), np.size(func, 0), radii.size)
                containing the K functional for every shape function
        """
        transfer_s = self.cosmo['transfer']['scalar']
        k = self.cosmo['transfer']['k']
        lmax = ells.size + 1
        # Get primordial shape function
        if prim_shape == 'local':  # Additional shapes have to be included
            func = np.zeros([2, k.size])
            func[0, :] = self.alpha_prim(k)
            func[1, :] = self.beta_prim(k)
        else:
            raise ValueError('Only local primordial shape support at the moment')

        k2 = k ** 2
        func *= k2

        # Distribute ells for even work load, large ell are slower
        ells_per_rank = []
        ells_sub = ells[self.mpi_rank::self.mpi_size]
        # Save size of rank for each rank to combine later
        for rank in range(self.mpi_size):
            ells_per_rank.append(ells[rank::self.mpi_size])
        # temporary beta will be combined with all ranks later
        beta_temp = np.zeros((ells_sub.size, len(pols_s), np.size(func, 0), radii.size))

        self.printmpi('Compute: beta')
        for lidx, ell in enumerate(ells_sub):
            if lidx % 10 ==0:
                self.printmpi(lidx)
            for ridx, radius in enumerate(radii):
                kr = k * radius
                # jl[lidx, ridx, :] = spherical_jn(ell, kr)
                jl = spherical_jn(ell, kr)
                # integrate
                for sidx, funcs in enumerate(func):
                    for pidx, pol in enumerate(pols_s):
                        integrand = np.zeros(k.size)
                        integrand += jl[:]
                        integrand *= transfer_s[pidx, ell - 2, :]
                        integrand *= func[sidx, :]
                        beta_temp[lidx, pidx, sidx, ridx] = trapz(integrand, k)
        beta_temp *= 2 / np.pi

        # Move beta if not parallel
        beta_full = beta_temp

        if self.mpi_size > 1:
            if self.mpi_rank == 0:
                print('Combining beta')
                beta_full = np.zeros((ells.size, len(pols_s), np.size(func, 0), radii.size))
                # Place root beta_temp in beta_full
                beta_full[ells_per_rank[0] - 2, :, :, :] = beta_temp[:, :, :, :]
            else:
                beta_full = None

            for rank in range(1, self.mpi_size):
                # Send beta from other ranks over to root
                if self.mpi_rank == rank:
                    self.comm.Send(beta_temp, dest=0, tag=rank)
                    #print(self.mpi_rank, 'sent')

                # Receive beta on root
                if self.mpi_rank == 0:
                    ell_size = ells_per_rank[rank].size
                    beta_sub = np.zeros((ell_size, len(pols_s), np.size(func, 0), radii.size))
                    self.comm.Recv(beta_sub, source=rank, tag=rank)
                    print('root received {}'.format(rank))
                    beta_full[ells_per_rank[rank] - 2, :, :, :] = beta_sub[:, :, :, :]
            beta_full = self.comm.bcast(beta_full, root=0)
        beta_file = path+'/beta_{}.pkl'.format(lmax)
        with open(beta_file, 'wb') as handle:
            pickle.dump(beta_full, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Done: beta')

        self.beta['beta_s'] = beta_full

    def init_bispec(self):
        """
        Calculate the bispectrum
        """
        ells = self.cosmo['cls']['ells']
        cls_lensed = self.cosmo['cls']['cls']['total']
        beta_s = self.beta['beta_s']

        radii = get_updated_radii()

        ells = np.array(ells, dtype='int64')

        self.compute_bispec(ells=ells, beta_s=beta_s, radii=radii, cls_lensed=cls_lensed)
        sys.stdout.flush()
        # pol_trpl = np.array(list(itertools.product([0, 1], repeat=3)))  # all combinations of TTT,TTE,...

    def compute_bispec(self, ells, beta_s, radii, cls_lensed):
        r2 = radii ** 2
        pol_trpl = np.array([[0, 0, 0], [1, 1, 1]])
        # shape_factor depends on primordial template, i.e. 2 for local, 4 for others
        shape_factor = 2
        fisher_lmax = np.zeros(ells.size)
        idx = 0
        fisher = 0
        self.printmpi('Compute bispectrum')
        fnl_file = path + '/fnl_lmax.txt'

        # Distribute ells for even work load, large ell are slower
        ells_per_rank = []
        ells_sub = ells[self.mpi_rank::self.mpi_size]
        # Save size of rank for each rank to combine later
        for rank in range(self.mpi_size):
            ells_per_rank.append(ells[rank::self.mpi_size])
        # temporary beta will be combined with all ranks later
        fisher_sub = np.zeros((ells_sub.size))
        l_min = int(2)

        for lidx3, ell3 in enumerate(ells_sub):
            fisher = 0
            if lidx3 % 10 == 0:
                self.printmpi(lidx3)
            for lidx2, ell2 in enumerate(np.arange(2, ell3 + 1)):
                for lidx1, ell1 in enumerate(np.arange(2, ell2 + 1)):
                    # Wig3j is only non-zero for even sums of ell and triangle equation
                    if ((ell3 + ell2 + ell1) % 2) == 1 or ell3 > ell1 + ell2:
                        continue
                    # for pidx in range(pol_trpl.shape[0]):
                    for pidx in range(1):
                        pidx1, pidx2, pidx3 = pol_trpl[pidx, :]
                        ang = np.sqrt((2 * ell1 + 1) * (2 * ell2 + 1) * (2 * ell3 + 1) / (4 * np.pi))
                        wig3j = wig.wig3jj_array([2 * ell1, 2 * ell2, 2 * ell3, 0, 0, 0])
                        ang *= wig3j
                        shape_func = beta_s[ell1-l_min, pidx1, 1, :] * beta_s[ell2-l_min, pidx2, 1, :] * beta_s[ell3-l_min, pidx3, 0, :] \
                                     + beta_s[ell2-l_min, pidx1, 1, :] * beta_s[ell3-l_min, pidx2, 1, :] * beta_s[lidx1, pidx3, 0, :] \
                                     + beta_s[ell3-l_min, pidx1, 1, :] * beta_s[ell1-l_min, pidx2, 1, :] * beta_s[ell2-l_min, pidx3, 0, :]
                        bprim = shape_factor * ang * shape_func
                        integrand = bprim * r2
                        bispec = trapz(integrand, radii)
                        '''
                        for piidx in range(pol_trpl.shape[0]):
                            pidx4, pidx5, pidx6 = pol_trpl[pidx, :]
                            ang = np.sqrt((2 * ell1 + 1) * (2 * ell2 + 1) * (2 * ell3 + 1))
                            ang /= np.sqrt(4 * np.pi)
                            ang *= wig3j
                            shape_func = beta_s[lidx1, pidx4, 1, :] * beta_s[lidx2, pidx5, 1, :] * beta_s[lidx3, pidx6,0, :] \
                                         + beta_s[lidx2, pidx4, 1, :] * beta_s[lidx3, pidx5, 1, :] * beta_s[lidx1, pidx6, 0, :] \
                                         + beta_s[lidx3, pidx4, 1, :] * beta_s[lidx1, pidx5, 1, :] * beta_s[lidx2, pidx6, 0, :]
                            bprim = shape_factor * ang * shape_func
                            integrand = bprim * r2
                            bispec2 = trapz(integrand, radii)
                        '''
                        # Now we have B_l1,l2,l3
                        # Get B_{l1 l2 l3} * B^*_{l1 l2 l3}/(C_l1 C_l2 C_l3)
                        fis = bispec
                        # fis /= (cl[pidx1] * cl[pidx2] * cl[pidx3])
                        fis /= (cls_lensed[pidx1, ell1-l_min] * cls_lensed[pidx2, ell2-l_min] *
                                cls_lensed[pidx3, ell3-l_min])
                        fis *= bispec
                        fis /= delta_l1l2l3(ell3, ell2, ell1)
                        if fis < 0:
                            print(fis)
                        fisher += fis
                        idx += 1
            fisher_lmax[lidx3] = 1 / np.sqrt(fisher)
            fisher_sub[lidx3] = fisher
        #print(fisher_lmax[-1])

        # fisher has been calculated at all ranks
        fisher_full = fisher_sub

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


F = PreCalc()
F.init_cosmo(lmax=200)  # Actually only calculates to lmax - 100
start = timeit.default_timer()
F.init_beta()
F.init_bispec()
stop = timeit.default_timer()
F.printmpi('Time: {}'.format(stop - start))
