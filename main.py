import numpy as np
import cosmo as cs
from scipy.special import spherical_jn
from scipy.integrate import trapz
from scipy.linalg import inv
import pywigxjpf as wig
import pickle
import os
import sys
import bispectrum
# import itertools
import timeit
from mpi4py import MPI

path = os.getcwd() + '/Output'

def init_pol_triplets(pol_opts):
    """
    initialise array with polarisation triplets
    So far only includes T=0 and E=1
    :return:
    """
    if pol_opts == 0:
        pol_trpl = np.zeros((1, 3), dtype=int)
        pol_trpl[0] = 0, 0, 0
    elif pol_opts == 1:
        pol_trpl = np.zeros((1, 3), dtype=int)
        pol_trpl[0] = 1, 1, 1
    elif pol_opts == 2:
        pol_trpl = np.zeros((8, 3), dtype=int)
        pol_trpl[0] = 0, 0, 0
        pol_trpl[1] = 1, 1, 1
        pol_trpl[2] = 0, 0, 1
        pol_trpl[3] = 0, 1, 0
        pol_trpl[4] = 1, 0, 0
        pol_trpl[5] = 0, 1, 1
        pol_trpl[6] = 1, 0, 1
        pol_trpl[7] = 1, 1, 0

    return(pol_trpl)


def get_updated_radii():
    """
    Get the radii (in Mpc) that are more suitable for the sst case.
    """
    vlow = np.linspace(0, 500, num=10, dtype=float, endpoint=False)
    low = np.linspace(500, 9377, num=98, dtype=float, endpoint=False)
    re1 = np.linspace(9377, 10007, num=18, dtype=float, endpoint=False)
    re2 = np.linspace(10007, 12632, num=25, dtype=float, endpoint=False)
    rec = np.linspace(12632, 13682, num=50, dtype=float, endpoint=False)
    rec_new = np.linspace(13682, 15500, num=300, dtype=float, endpoint=False)
    rec_extra = np.linspace(15500, 20000, num=100, dtype=float, endpoint=False)
    rec_extra2 = np.linspace(20000, 40000, num=50, dtype=float, endpoint=False)

    radii = np.concatenate((vlow, low, re1, re2, rec, rec_new, rec_extra,rec_extra2))

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

def local_shape(beta_s, ell1, ell2, ell3, l_min, pidx1, pidx2, pidx3):
    shape_func =  beta_s[ell1 - l_min, pidx1, 1, :] * beta_s[ell2 - l_min, pidx2, 1, :] * beta_s[ell3 - l_min, pidx3, 0, :] \
                + beta_s[ell2 - l_min, pidx1, 1, :] * beta_s[ell3 - l_min, pidx2, 1, :] * beta_s[ell1 - l_min, pidx3, 0, :] \
                + beta_s[ell3 - l_min, pidx1, 1, :] * beta_s[ell1 - l_min, pidx2, 1, :] * beta_s[ell2 - l_min, pidx3, 0, :]
    return shape_func

def equil_shape(beta_s, ell1, ell2, ell3, l_min, pidx1, pidx2, pidx3):
    shape_func = - local_shape(beta_s, ell1, ell2, ell3, l_min, pidx1, pidx2, pidx3)
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
    return shape_func

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

        norm = (3/5)^1/2 * As * 2 * pi^2 /k_p^3
        for scalar scalar tensor, the 3/5 factor gets replaced by a 2. Tensor perturbations are gauge invariant so no
        3/5. And probably comes with a 2 due to +,x components
        """
        scalar_amp = 2.1056e-9  # amplitude of scalar perturbations
        # I will probably move the 3/5 somewhere else and put a comment at both places. Maybe x3/5 at the end (fis)
        norm = 2 * (np.pi ** 2) * scalar_amp * (3/5)**(1/2)


        kpivot = 0.05
        ns = 0.965

        if k is None:
            k = self.cosmo['scalar']['k']
        km3 = k ** -3
        # Multiply the power spectrum to is As * (k/kp)^(n_s-1)
        km3 *= (k / kpivot) ** (ns - 1)
        #k / kpivot) ** (ns - 1)
        beta_prim = km3 * norm
        return beta_prim

    def gamma_prim(self, k):
            """
            Returns shape function
                        beta = norm * (k/k_pivot)^(ns-4)

            norm = (3/5)^1/2 * As * 2 * pi^2 /k_p^3
            for scalar scalar tensor, the 3/5 factor gets replaced by a 2. Tensor perturbations are gauge invariant so no
            3/5. And probably comes with a 2 due to +,x components
            """
            kpivot = 0.05
            ns = 0.965

            if k is None:
                k = self.cosmo['scalar']['k']
            km3 = k ** -3
            # Multiply the power spectrum scaling: (k/kp)^(n_s-1)
            km3 *= (k / kpivot) ** (ns - 1)
            gamma_prim = km3 ** (1 / 3)
            return gamma_prim

    def delta_prim(self, k):
            """
            Returns shape function
                        beta = norm * (k/k_pivot)^(ns-4)

            norm = (3/5)^1/2 * As * 2 * pi^2 /k_p^3
            for scalar scalar tensor, the 3/5 factor gets replaced by a 2. Tensor perturbations are gauge invariant so no
            3/5. And probably comes with a 2 due to +,x components
            """
            kpivot = 0.05
            ns = 0.965

            if k is None:
                k = self.cosmo['scalar']['k']
            km3 = k ** -3
            # Multiply the power spectrum scaling: (k/kp)^(n_s-1)
            km3 *= (k / kpivot) ** (ns - 1)
            delta_prim = km3 ** (2 / 3)
            return delta_prim

    def init_beta(self, prim_shape='local'):
        """
        Sets up calculation of beta:
        2/pi int k^2 dk f(k) j_l(k*r) T_{x,ell}(k)
        if precomputed skips calculation, read file and stores in dict
        """

        ells = self.cosmo['cls']['ells']
        lmax = ells.size + 1

        #radii = get_komatsu_radii()
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
            if lidx % 10 == 0:
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
                    # print(self.mpi_rank, 'sent')

                # Receive beta on root
                if self.mpi_rank == 0:
                    ell_size = ells_per_rank[rank].size
                    beta_sub = np.zeros((ell_size, len(pols_s), np.size(func, 0), radii.size))
                    self.comm.Recv(beta_sub, source=rank, tag=rank)
                    print('root received {}'.format(rank))
                    beta_full[ells_per_rank[rank] - 2, :, :, :] = beta_sub[:, :, :, :]
            beta_full = self.comm.bcast(beta_full, root=0)
        beta_file = path + '/beta_{}.pkl'.format(lmax)
        with open(beta_file, 'wb') as handle:
            pickle.dump(beta_full, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.printmpi('Done: beta')

        self.beta['beta_s'] = beta_full

    def init_bispec(self, shape='local'):
        """
        Calculate the bispectrum
        """
        ells = self.cosmo['cls']['ells']
        cls_lensed = self.cosmo['cls']['cls']['lensed_scalar']
        beta_s = self.beta['beta_s']

        radii = get_updated_radii()

        ells = np.array(ells, dtype='int64')


        self.compute_bispec(ells=ells, beta_s=beta_s, radii=radii, cls_lensed=cls_lensed, shape=shape)
        sys.stdout.flush()
        # pol_trpl = np.array(list(itertools.product([0, 1], repeat=3)))  # all combinations of TTT,TTE,...

    def invert_cls(self, ells, cls, pol_opts):
        cov = np.zeros((ells.size, 2, 2))
        invcov = np.zeros_like(cov)
        if pol_opts == 0:
            cov[:, 0, 0] = cls[0, :]
            cov[:, 1, 1] = cls[0, :]
            for lidx in range(ells.size):
                invcov[lidx,:,:] = inv(cov[lidx,:,:])
        elif pol_opts == 1:
            cov[:, 0, 0] = cls[1, :]
            cov[:, 1, 1] = cls[1, :]
            for lidx in range(ells.size):
                invcov[lidx, :, :] = inv(cov[lidx, :, :])
        elif pol_opts == 2:
            cov[:,0,0] = cls[0,:]
            cov[:,1,1] = cls[1,:]
            cov[:,1,0] = cls[3,:]
            cov[:,0,1] = -cls[3,:]
            for lidx in range(ells.size):
                invcov[lidx,:,:] = inv(cov[lidx,:,:])
        return invcov

    def compute_bispec(self, ells, beta_s, radii, cls_lensed, shape):
        # shape_factor depends on primordial template, i.e. 2 for local, 6 for others
        fisher_lmax = np.zeros(ells.size)
        self.printmpi('Compute bispectrum')
        fnl_file = path + '/fnl_lmax.txt'
        np.savetxt('myfile.txt', np.column_stack([ells, cls_lensed[0,:]]))
        # Distribute ells for even work load, large ell are slower
        ells_per_rank = []
        ells_sub = ells[self.mpi_rank::self.mpi_size]
        # Save size of rank for each rank to combine later
        for rank in range(self.mpi_size):
            ells_per_rank.append(ells[rank::self.mpi_size])
        # temporary beta will be combined with all ranks later

        fisher_sub = np.zeros((ells_sub.size))
        pol_opts = 2 #0: T only, 1: E only, 2: TE mixed
        pol_trpl = init_pol_triplets(pol_opts)
        invcov = self.invert_cls(ells, cls_lensed, pol_opts)
        # pol_trpl = init_pol_triplets(0)
        # Calculation performed in Cython. See bispectrum.pyx
<<<<<<< HEAD
        bispectrum.compute_bispec(ells_sub, radii, beta_s, cls_lensed, fisher_sub, self.mpi_rank)
=======
        bispectrum.compute_bispec(ells_sub, radii, beta_s, invcov, fisher_sub, pol_trpl, self.mpi_rank, shape)

>>>>>>> be3a9d457fcc2ce53c61fdca423959fa91cbd529
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
                    print(self.mpi_rank, 'sent')

                # Receive fisher on root
                if self.mpi_rank == 0:
                    ell_size = ells_per_rank[rank].size
                    fisher_sub = np.zeros(ell_size)
                    self.comm.Recv(fisher_sub, source=rank, tag=rank)
                    # print('root received fisher{}'.format(rank))
                    fisher_full[ells_per_rank[rank] - 2] = fisher_sub
            fisher_full = self.comm.bcast(fisher_full, root=0)
        fisher_file = path + '/fisher_{}.pkl'.format(ells.size + 1)
        with open(fisher_file, 'wb') as handle:
            pickle.dump(fisher_full, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.printmpi('Done: fisher')

        fnl_end = np.sum(fisher_full)
        fnl_end = 1 / np.sqrt(fnl_end)

        fnl_max = np.zeros_like(fisher_full)
        for idx, item in enumerate(fisher_full):
            fnl_max[idx] = item + fnl_max[idx - 1]
        fnl_max = 1 / np.sqrt(fnl_max)
        self.printmpi(fnl_end)
        np.savetxt(fnl_file, np.column_stack([ells, fnl_max]))


F = PreCalc()
F.init_cosmo(lmax=300)  # Actually only calculates to lmax - 100
start = timeit.default_timer()
F.init_beta()
F.init_bispec()
stop = timeit.default_timer()
F.printmpi('Time: {}'.format(stop - start))
