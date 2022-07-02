import numpy as np
# import cosmo as cs
from scipy.special import spherical_jn
from scipy.integrate import trapz, simps
from scipy.linalg import inv
import pywigxjpf as wig
import pickle
import os
import sys
import ksz
# import bispectrum
import itertools
import timeit
from mpi4py import MPI
# import py3nj
import cProfile
import pstats

path = os.getcwd() + '/Output'


def get_ksz_radii():
    """
    Get the radii (in Mpc) that are more suitable for the sst case.
    """

    # booster increases the number of samples points
    booster = 5
    vlow = np.linspace(0, 10, num=booster * 10, dtype=float, endpoint=False)
    ksz1 = np.linspace(10, 2000, num=booster * 80, dtype=float, endpoint=False)
    ksz2 = np.linspace(2000, 6000, num=booster * 300, dtype=float, endpoint=False)
    ksz3 = np.linspace(6000, 9377, num=booster * 200, dtype=float, endpoint=False)
    re1 = np.linspace(9377, 10007, num=booster * 18, dtype=float, endpoint=False)
    re2 = np.linspace(10007, 12632, num=booster * 25, dtype=float, endpoint=False)
    rec = np.linspace(12632, 13682, num=booster * 50, dtype=float, endpoint=False)
    rec_new = np.linspace(13682, 15500, num=booster * 300, dtype=float, endpoint=False)
    rec_extra = np.linspace(15500, 30000, num=booster * 50, dtype=float, endpoint=False)

    radii = np.concatenate((vlow, ksz1, ksz2, ksz3, re1, re2, rec, rec_new, rec_extra))
    return radii


def invert_sparse(cls_sparse, pols, ells):
    # Calculate all cls first, then keep only requested pols
    # pols are int pointing at non-zero entries in cls

    # Remove zeros from cls array
    cls_dense = np.zeros([ells.size, pols.shape[0], pols.shape[0]])
    for lidx, ell in enumerate(ells):
        for idx, i in enumerate(pols):
            for jdx, j in enumerate(pols):
                cls_dense[lidx, idx, jdx] = cls_sparse[lidx, i, j]

    # Now we can invert the cls_dense
    invcov_dense = np.zeros_like(cls_dense)
    for lidx, ell in enumerate(ells):
        try:
            invcov_dense[lidx, :, :] = np.linalg.pinv(cls_dense[lidx, :, :])
        except(np.linalg.LinAlgError):
            invcov_dense[lidx, :, :] = 0
    # For the rest of the code we use sparse invcov
    invcov_sparse = np.zeros_like(cls_sparse)
    for lidx, ell in enumerate(ells):
        for idx, i in enumerate(pols):
            for jdx, j in enumerate(pols):
                invcov_sparse[lidx, i, j] = invcov_dense[lidx, idx, jdx]
    return (invcov_sparse)


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
    # old version without dictionaries
    shape_func = beta_s[ell1 - l_min, pidx1, 1, :] * beta_s[ell2 - l_min, pidx2, 1, :] * beta_s[ell3 - l_min, pidx3, 0,
                                                                                         :] \
                 + beta_s[ell2 - l_min, pidx2, 1, :] * beta_s[ell3 - l_min, pidx3, 1, :] * beta_s[ell1 - l_min, pidx1,
                                                                                           0, :] \
                 + beta_s[ell3 - l_min, pidx3, 1, :] * beta_s[ell1 - l_min, pidx1, 1, :] * beta_s[ell2 - l_min, pidx2,
                                                                                           0, :]
    shape_func *= 2
    return shape_func


def local_shape2(beta_s, ell1, ell2, ell3, l_min, pidx1, pidx2, pidx3):
    # New version where polarisation index is a dictionary key
    s1 = beta_s[pidx1][ell1 - l_min, 1, :] * beta_s[pidx2][ell2 - l_min, 1, :] * beta_s[pidx3][ell3 - l_min, 0, :]
    s2 = beta_s[pidx2][ell2 - l_min, 1, :] * beta_s[pidx3][ell3 - l_min, 1, :] * beta_s[pidx1][ell1 - l_min, 0, :]
    s3 = beta_s[pidx3][ell3 - l_min, 1, :] * beta_s[pidx1][ell1 - l_min, 1, :] * beta_s[pidx2][ell2 - l_min, 0, :]
    shape_func = 2 * (s1 + s2 + s3)
    return shape_func


def equil_shape(beta_s, ell1, ell2, ell3, l_min, pidx1, pidx2, pidx3):
    """"

    This one should definitely be updated to newer version with polarisation dict key
    """
    shape_func = - local_shape(beta_s, ell1, ell2, ell3, l_min, pidx1, pidx2, pidx3)
    # delta delta delta
    shape_func -= beta_s[ell1 - l_min, pidx1, 3, :] * beta_s[ell2 - l_min, pidx2, 3, :] * beta_s[ell3 - l_min, pidx3, 3,
                                                                                          :]
    # beta gamma delta
    shape_func += beta_s[ell1 - l_min, pidx1, 1, :] * beta_s[ell2 - l_min, pidx2, 2, :] * beta_s[ell3 - l_min, pidx3, 3,
                                                                                          :]
    # bdg
    shape_func += beta_s[ell1 - l_min, pidx1, 1, :] * beta_s[ell2 - l_min, pidx2, 3, :] * beta_s[ell3 - l_min, pidx3, 2,
                                                                                          :]
    # gbd
    shape_func += beta_s[ell1 - l_min, pidx1, 3, :] * beta_s[ell2 - l_min, pidx2, 1, :] * beta_s[ell3 - l_min, pidx3, 2,
                                                                                          :]
    # dgb
    shape_func += beta_s[ell1 - l_min, pidx1, 3, :] * beta_s[ell2 - l_min, pidx2, 2, :] * beta_s[ell3 - l_min, pidx3, 1,
                                                                                          :]
    # gbd
    shape_func += beta_s[ell1 - l_min, pidx1, 2, :] * beta_s[ell2 - l_min, pidx2, 1, :] * beta_s[ell3 - l_min, pidx3, 3,
                                                                                          :]
    # gdb
    shape_func += beta_s[ell1 - l_min, pidx1, 2, :] * beta_s[ell2 - l_min, pidx2, 3, :] * beta_s[ell3 - l_min, pidx3, 1,
                                                                                          :]
    return shape_func


# (beam, noise, lmin)
class PreCalc:
    def __init__(self):
        self.cosmo = {}
        self.bins = {}
        self.beta = {}
        self.bispec = {}
        self.invcov = {}
        self.ksz = {}
        self.delta = {}

        self.comm = MPI.COMM_WORLD
        self.mpi_rank = self.comm.Get_rank()  # Assigns every core a rank (integer value)
        self.mpi_size = self.comm.Get_size()

        # Initialise wigner 3j tables
        wig.wig_table_init(2 * 3000, 9)
        wig.wig_temp_init(2 * 3000)
        scalar_amp = 2.1056e-9  # amplitude of scalar perturbations
        # I will probably move the 3/5 somewhere else and put a comment at both places. Maybe x3/5 at the end (fis)
        self.norm = 2 * (np.pi ** 2) * scalar_amp * (3 / 5) ** (1 / 2)

        # set noise values to 0
        self.beam = 0  # in arcmin
        self.noise = 0  # in muK * arcmin
        self.lmin = 1

        # Until which point ksz modes are included
        self.cutoff_ksz = 100

    def init_planck_setting(self, ):
        self.beam = 5  # in arcmin
        self.noise = 40  # in muK * arcmin
        self.lmin = 2
        # self.fsky = 0.75

    def init_CMBS4_setting(self, ):
        self.beam = 1  # in arcmin
        self.noise = 1  # in muK * arcmin
        self.lmin = 30
        # self.fsky = 0.40

    def printmpi(self, text):
        if self.mpi_rank == 0:
            print(text, flush=True)

    def init_cosmo(self, lmax, AccuracyBoost=1, kSampling=1):
        """
        Collect transfer functions and cls to save in self.cosmo
        1) Run CAMB and get T,E,B
        2) Read in transfer and cls from kSZ and pSZ
        """
        cosmo_file = path + '/cosmo_{}_8_3.pkl'.format(301)
        # cosmo_file = path + '/cosmo_300_8_3.pkl'
        # cosmo_file = path + '/Tests/cosmo_{}_2b_3k.pkl'.format(lmax - 100)
        # cosmo_file = path + '/cosmo.pkl'
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
        if self.mpi_rank == 0:
            self.cosmo['scalar'] = self.cosmo['scalar'][:2, :lmax, :]
        if recompute_cosmo is False:
            self.cosmo = self.comm.bcast(self.cosmo, root=0)

        if recompute_cosmo:
            self.printmpi('Runnnig CAMB for transfer, k, ells and cls ')
            transfer, cls = cs.run_camb(lmax=lmax, lSampleBoost=50, AccuracyBoost=AccuracyBoost, kSampling=kSampling)
            self.cosmo['transfer'] = transfer
            self.cosmo['cls'] = cls
            if self.mpi_rank == 0:
                print('Storing cosmo as: {}'.format(cosmo_file))
                # Store in pickle file.
                with open(cosmo_file, 'wb') as handle:
                    pickle.dump(self.cosmo, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
            self.printmpi('Computed cosmo')
        self.printmpi(self.cosmo['scalar'][0, 0, :])
    def init_ksz(self, lmax, ze, N_bins, AccuracyBoost=2):
        """
        Collect transfer functions and cls to save in self.ksz
        ze currently not used. file name is what determines number of bins and everything
        Precalculate ksz and use correct file name
        """
        self.N_bins = N_bins
        ksz_file = path + '/transfer_300_b_{}.pkl'.format(N_bins)
        recompute_ksz = False
        #if self.mpi_rank == 0:
        if True:
            try:
                pkl_file = open(ksz_file, 'rb')
            except IOError:
                self.printmpi('{} not found'.format(ksz_file))
                recompute_ksz = True
            else:
                self.printmpi('loaded ksz from {}'.format(ksz_file))
                self.ksz['transfer'] = pickle.load(pkl_file)
                pkl_file.close()
        if self.mpi_rank == 0:
            print(self.ksz['transfer']['velocity'].shape)
        self.ksz['transfer']['velocity'] = self.ksz['transfer']['velocity'][:, :lmax, :]
        delta_temp = self.ksz['transfer']['density'][:, :lmax, :]
        self.delta['transfer'] = {'density': delta_temp} 
        # Tell all ranks if file was found
        recompute_ksz = self.comm.bcast(recompute_ksz, root=0)
#        self.ksz = self.comm.bcast(self.ksz, root=0)
#        self.delta = self.comm.bcast(self.delta, root=0)
        if recompute_ksz:
            self.printmpi('Runnnig CAMB for transfer, k, ells and cls ')
            transfer = ksz.run_camb(lmax=lmax, ze=ze, AccuracyBoost=AccuracyBoost)
            self.ksz['transfer'] = transfer
            if self.mpi_rank == 0:
                print('Storing ksz as: {}'.format(ksz_file))
                # Store in pickle file.
                with open(ksz_file, 'wb') as handle:
                    pickle.dump(self.ksz, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
            self.printmpi('Computed ksz')

    def init_delta(self, lmax, ze, N_bins, AccuracyBoost=2):
        """
        Collect transfer functions and cls to save in self.ksz
        ze currently not used. file name is what determines number of bins and everything
        Precalculate ksz and use correct file name
        """
        self.N_bins = N_bins
        # ksz_file = path + '/transfer_300_b_{}.pkl'.format(N_bins)
        delta_file = path + '/transfer_300_sync_{}.pkl'.format(N_bins)
        recompute_delta = False
#        if self.mpi_rank == 0:
        if True:
            try:
                pkl_file = open(delta_file, 'rb')
            except IOError:
                self.printmpi('{} not found'.format(delta_file))
                recompute_ksz = True
            else:
                self.printmpi('loaded delta from {}'.format(delta_file))
                self.delta['transfer'] = pickle.load(pkl_file)
                pkl_file.close()
        # Tell all ranks if file was found
        self.delta['transfer']['density'] = self.delta['transfer']['density'][:, :lmax, :]
        recompute_delta = self.comm.bcast(recompute_delta, root=0)
#        self.delta = self.comm.bcast(self.delta, root=0)

        if recompute_delta:
            self.printmpi('Runnnig CAMB for transfer, k, ells and cls ')
            transfer = ksz.run_camb(lmax=lmax, ze=ze, AccuracyBoost=AccuracyBoost)
            self.delta['transfer'] = transfer
            if self.mpi_rank == 0:
                print('Storing ksz as: {}'.format(delta_file))
                # Store in pickle file.
                with open(delta_file, 'wb') as handle:
                    pickle.dump(self.ksz, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
            self.printmpi('Computed ksz')

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

        kpivot = 0.05
        ns = 0.965

        if k is None:
            k = self.cosmo['scalar']['k']
        km3 = k ** -3
        # Multiply the power spectrum to is As * (k/kp)^(n_s-1)

        km3 *= (k / kpivot) ** (ns - 1)

        beta_prim = km3 * self.norm
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
        gamma_prim = km3 * self.norm
        gamma_prim = gamma_prim ** (1 / 3)
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
        delta_prim = km3 * self.norm
        delta_prim = delta_prim ** (2 / 3)
        return delta_prim

    def init_beta(self, lmax, prim_shape='local', N_bins=2, density_cutoff=0):
        """
        Sets up calculation of beta:
        2/pi int k^2 dk f(k) j_l(k*r) T_{x,ell}(k)
        if precomputed skips calculation, read file and stores in dict
        """
        self.delta_lmin = density_cutoff
        self.N_bins = N_bins
        ells = np.arange(lmax) + 1
        self.ells = ells
        lmax = ells.size
        self.printmpi('lmax:{}'.format(lmax))
        # radii = get_komatsu_radii()
        radii = get_ksz_radii()
        self.beta['radii'] = radii
        self.delta['transfer']['density'][:, :self.delta_lmin, :] = self.ksz['transfer']['velocity'][:,
                                                                    :self.delta_lmin, :]

        # Check if beta already has been computed
        beta_file = path + '/beta_' + prim_shape + '_{}_{}.pkl'.format(lmax, self.N_bins)
        # beta_file = path + '/beta_' + prim_shape + '_{}.pkl'.format(51)
        # beta_file = path + '/Debug/beta_test_300.pkl'
        try:
            pkl_file = open(beta_file, 'rb')
            recompute_beta = False
        except IOError:
            self.printmpi('{} not found'.format(beta_file))
            recompute_beta = True
        else:
            self.printmpi('loaded beta from {}'.format(beta_file))
            self.beta = pickle.load(pkl_file)
            pkl_file.close()
        # Put this whole computation into a function
        if recompute_beta:
            # calculates alpha and beta for ksz until ell=self.cutoff_ksz
            # recompute always because fast
            self.compute_beta(radii=radii, ells=ells, prim_shape=prim_shape,
                              transfer_s=self.ksz['transfer']['velocity'][:, :, :], k=self.cosmo['k'],
                              mode='ksz')
            # calculates alpha and beta for delta until ell=self.cutoff_ksz
            self.compute_beta(radii=radii, ells=ells, prim_shape=prim_shape,
                              transfer_s=self.delta['transfer']['density'][:, :, :], k=self.cosmo['k'],
                              mode='delta')
            # calculates beta for T and E
            # search if beta file has been calculated before
            beta_cmb_file = path + '/beta_cmb_' + prim_shape + '_{}.pkl'.format(lmax)
            try:
                pkl_file = open(beta_cmb_file, 'rb')
            except IOError:
                self.printmpi('{} not found'.format(beta_cmb_file))
                self.compute_beta(radii=radii, ells=ells, prim_shape=prim_shape,
                                  transfer_s=self.cosmo['scalar'][:2, :, :], k=self.cosmo['k'],
                                  mode='CMB')
            else:
                self.printmpi('loaded beta from {}'.format(beta_file))
                self.beta.update(pickle.load(pkl_file))  # load cmb beta file and add to dictionary
                pkl_file.close()

            self.printmpi(self.beta.keys())
            beta_file = path + '/beta_' + prim_shape + '_{}_{}.pkl'.format(lmax, self.N_bins)
            with open(beta_file, 'wb') as handle:
                pickle.dump(self.beta, handle, protocol=pickle.HIGHEST_PROTOCOL)
            self.printmpi('Done: beta')
        for i in range(2, self.N_bins + 2):
            self.beta[i][:self.delta_lmin, :, :] = self.beta[i + self.N_bins][:self.delta_lmin, :, :]

    def compute_beta(self, radii, ells, prim_shape, transfer_s, k, mode):
        """
        Computes 2/pi int k^2 dk f(k) j_l(k*r) T_{x,ell}(k)
        Splits calculation among ells
        :param radii: array float
        :param ells: array int, size lmax
        :param prim_shape: str at the moment only supports local
        :return:beta(ells_sub.size, len(pols_s), np.size(func, 0), radii.size)
                containing the K functional for every shape function
        """

        lmax = ells.size + 1
        n_pol = transfer_s.shape[0]
        self.printmpi('number of modes is {}'.format(n_pol))
        # Get primordial shape function
        if prim_shape == 'local':  # Additional shapes have to be included
            func = np.zeros([2, k.size])
            func[0, :] = self.alpha_prim(k)
            func[1, :] = self.beta_prim(k)
        else:
            func = np.zeros([4, k.size])
            func[0, :] = self.alpha_prim(k)
            func[1, :] = self.beta_prim(k)
            func[2, :] = self.gamma_prim(k)
            func[3, :] = self.delta_prim(k)

        k2 = k ** 2
        func *= k2

        # Distribute ells for even work load, large ell are slower
        ells_per_rank = []
        ells_sub = ells[self.mpi_rank::self.mpi_size]
        # Save size of rank for each rank to combine later
        for rank in range(self.mpi_size):
            ells_per_rank.append(ells[rank::self.mpi_size])
        # temporary beta will be combined with all ranks later
        beta_temp = np.zeros((ells_sub.size, n_pol, np.size(func, 0), radii.size))

        self.printmpi('Compute: beta')
        for lidx, ell in enumerate(ells_sub):
            if lidx % 1 == 0:
                self.printmpi('{}/{}'.format(lidx, len(ells_sub) - 1))
            for ridx, radius in enumerate(radii):
                kr = k * radius
                # jl[lidx, ridx, :] = spherical_jn(ell, kr)
                jl = spherical_jn(ell, kr)
                # integrate
                for sidx, funcs in enumerate(func):
                    for pidx in range(n_pol):
                        # stop calculating beta for kSZ if above kSZ
                        integrand = np.zeros(k.size)
                        integrand += jl[:]
                        integrand *= transfer_s[pidx, ell - self.lmin, :]
                        integrand *= func[sidx, :]
                        beta_temp[lidx, pidx, sidx, ridx] = trapz(integrand, k)
            if ell == 1:
                print(' beta ell=1')
                print(beta_temp[lidx, 0, :, :])
                print(ell - self.lmin)
                print('transfer')
                print(transfer_s[0, ell - self.lmin, :20])

        beta_temp *= 2 / np.pi

        # Move beta if not parallel
        beta_full = beta_temp

        if self.mpi_size > 1:
            if self.mpi_rank == 0:
                print('Combining beta')
                beta_full = np.zeros((ells.size, n_pol, np.size(func, 0), radii.size))
                # Place root beta_temp in beta_full
                beta_full[ells_per_rank[0] - self.lmin, :, :, :] = beta_temp[:, :, :, :]
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
                    beta_sub = np.zeros((ell_size, n_pol, np.size(func, 0), radii.size))
                    self.comm.Recv(beta_sub, source=rank, tag=rank)
                    print('root received {}'.format(rank))
                    beta_full[ells_per_rank[rank] - self.lmin, :, :, :] = beta_sub[:, :, :, :]
            beta_full = self.comm.bcast(beta_full, root=0)

        '''
        Beta dictionary keys:           Example for N_bin = 4
        0                   T           0       T
        1                   E           1       E
        2       - N_bin+1   density     2-5     density
        N_bin+2 - 2*N_bin+1             6-10    velocity
        '''
        # Put the density beta function into dictionary
        if mode == 'CMB':
            beta_file_cmb = path + '/beta_cmb_' + prim_shape + '_{}.pkl'.format(lmax)
            beta_cmb = {0: beta_full[:, 0, :, :], 1: beta_full[:, 1, :, :]}
            with open(beta_file_cmb, 'wb') as handle:
                pickle.dump(beta_cmb, handle, protocol=pickle.HIGHEST_PROTOCOL)
            self.beta.update(beta_cmb)

        elif mode == 'delta':
            beta_file_delta = path + '/beta_delta_' + prim_shape + '_{}.pkl'.format(lmax)
            beta_delta = {}
            for i in range(n_pol):
                beta_delta[i + 2] = beta_full[:, i, :, :]  # first two entries will be from CMB
            with open(beta_file_delta, 'wb') as handle:
                pickle.dump(beta_delta, handle, protocol=pickle.HIGHEST_PROTOCOL)
                self.beta.update(beta_delta)

        # if loop handles naming. ksz index + 2
        elif mode == 'ksz':
            beta_file_ksz = path + '/beta_ksz_' + prim_shape + '_{}.pkl'.format(lmax)
            beta_ksz = {}
            for i in range(n_pol):
                beta_ksz[i + 2 + self.N_bins] = beta_full[:, i, :, :]  # first two entries will be from CMB
            with open(beta_file_ksz, 'wb') as handle:
                pickle.dump(beta_ksz, handle, protocol=pickle.HIGHEST_PROTOCOL)
                self.beta.update(beta_ksz)

    def pols_str_to_int(self, pol_opts, n_bins):
        # At this part we include number of bins via n_bins. I guess if we have additional bins and not change n_bins
        # we will only use the first 2 bins (2 = default number)
        self.n_bins = n_bins
        if pol_opts == 'T':
            pols = np.array([0])
        elif pol_opts == 'E':
            pols = np.array([1])
        elif pol_opts == 'TE':
            pols = np.array([0, 1])

        elif pol_opts == 'delta':
            pols = np.arange(n_bins) + 2
        elif pol_opts == 'delta1':
            pols = np.array([2])
        elif pol_opts == 'delta2':
            pols = np.array([3])
        elif pol_opts == 'delta3':
            pols = np.array([4])
        elif pol_opts == 'delta4':
            pols = np.array([5])

        elif pol_opts == 'ksz':
            pols = np.arange(n_bins) + 2 + n_bins
        elif pol_opts == 'ksz1':
            pols = np.array([n_bins + 1 + 1])
        elif pol_opts == 'ksz2':
            pols = np.array([n_bins + 2 + 1])
        elif pol_opts == 'ksz3':
            pols = np.array([n_bins + 3 + 1])
        elif pol_opts == 'ksz4':
            pols = np.array([n_bins + 4 + 1])


        elif pol_opts == 'deltaksz':
            pols = np.arange(2 * n_bins) + 2
        elif pol_opts == 'vd1':
            pols = np.array([2, n_bins + 1 + 1])
        elif pol_opts == 'vd2':
            pols = np.array([3, n_bins + 2 + 1])
        elif pol_opts == 'all':
            pols = np.arange(2 * n_bins + 2)

        elif pol_opts == 'Tdelta':
            pols = np.append(0, np.arange(n_bins) + 2)
        elif pol_opts == 'Edelta':
            pols = np.append(1, np.arange(n_bins) + 2)
        elif pol_opts == 'Tksz':
            pols = np.append(0, np.arange(n_bins) + 2 + n_bins)
        elif pol_opts == 'Eksz':
            pols = np.append(1, np.arange(n_bins) + 2 + n_bins)
        else:
            print('polarisation not matching any tags. Code will proceed with pol==deltaksz')
        self.pols = pols
        self.printmpi('number of modes: {}'.format(pols.size))
        self.printmpi('Modes: {}'.format(pols))
        self.init_pol_triplets()

    def init_pol_triplets(self):
        """
        initialise array with polarisation triplets
        :return:
        """
        pols = self.pols
        pol_trpl = np.array(list(itertools.product(pols, repeat=3)))
        self.pol_trpl = pol_trpl

        self.pol_state = {}
        pol_states = np.array(list(itertools.product([0, 1, 2], repeat=3)))
        for pol in pol_states:
            pol_trpl_loop = np.array(list(itertools.product(pols, repeat=3)))
            for i in range(3):
                if pol[i] == 1:
                    pol_trpl_loop = pol_trpl_loop[pol_trpl_loop[:, i] > 1]
                elif pol[i] == 2:
                    pol_trpl_loop = pol_trpl_loop[pol_trpl_loop[:, i] < 2]
            self.pol_state[str(pol)] = pol_trpl_loop

    def init_invcov_from_transfer(self, pols, density_cutoff=0, ksz_cutoff=-1):
        """
        Calculates the inverted cls from transfer fcts
        First calculates cls then inverts using np.linalg.pinv


        :param pols: array int
        :return:invcov(ells_sub.size, len(pol_opts), len(pol_opts))
                containing inverse cls
        """
        # Note for large number of bins:
        # cls matrix will get rather large for O(10) modes. It might be smarter to split into two different matrices
        # one containing all modes until self.cutoff_ksz and the other for larger ell containing only T and E

        # Get transfer, ells and k from dict
        ells = self.ells
        k = self.ksz['transfer']['k']
        # Load T,E transfer functions as they are
        transfer_cmb = self.cosmo['scalar'][:2, :, :]  # lmax is 100 higher, remove B pol
        # create array for ksz as T,E with n_bins
        transfer_ksz = np.zeros([self.N_bins, transfer_cmb.shape[1], transfer_cmb.shape[2]])
        # Add ksz transfer functions, have other entries 0 (different lmax)
        transfer_ksz[:, :self.ksz['transfer']['velocity'].shape[1], :] = self.ksz['transfer']['velocity'][:,
                                                                         self.lmin - 1:, :]
        # create array for delta as T,E with n_bins
        transfer_delta = np.zeros([self.N_bins, transfer_cmb.shape[1], transfer_cmb.shape[2]])
        transfer_delta[:, :self.delta['transfer']['density'].shape[1], :] = self.delta['transfer']['density'][:,
                                                                          self.lmin - 1:, :]

        # if different k has been used for cmb and ksz, error will pop up
        transfer_temp = np.zeros([2 * self.N_bins + 2, transfer_cmb.shape[1], transfer_cmb.shape[2]])
        transfer_temp[0:2, :, :] = transfer_cmb
        transfer_temp[2:self.N_bins + 2, :, :] = transfer_delta
        transfer_temp[self.N_bins + 2:, :, :] = transfer_ksz

        # Only get the requested polarisations. Set the rest to 0
        transfer = np.zeros_like(transfer_temp)
        transfer[pols, :, :] = transfer_temp[pols, :, :]
        # Create all duplets of polarisation
        # T : 0     E : 1
        # B1: 2     B2: 3

        n_pol = transfer.shape[0]
        pol_dup = np.array(list(itertools.product(range(n_pol), repeat=2)))
        # (0,0) (0,1) (0,2) ...
        # Primordial Power spectrum
        P_R = 2.1056e-9 * (k / 0.05) ** (0.9665 - 1)
        # Calculate Cls
        # 4*pi * int dk/k * transfer * transfer * P_R
        cls = np.zeros((ells.size, n_pol, n_pol))
        for i, j in pol_dup:
            for lidx, ell in enumerate(ells):
                if (ell > self.cutoff_ksz and i > 1) or (ell > self.cutoff_ksz and j > 1):
                    continue
                else:
                    I = 4 * np.pi / k * transfer[i, lidx, :] * transfer[j, lidx, :] * P_R
                    cls[lidx, i, j] = simps(I, k)
        # print(ells.shape)
        # print(cls.shape)
        cls_txt = ells
        for i in range(n_pol):
            cls_txt = np.column_stack([cls_txt, cls[:, i, i]])
            # print(cls_txt.shape)
        np.savetxt('cls.txt', cls_txt)
        # Add option for noise here
        # cls = self.add_noise_ksz(cls)

        # delta lmin is implemented by hand at this point removing all density cls up to lmin 
        #self.delta_lmin = density_cutoff
        #cls[:self.delta_lmin, 2:self.N_bins+2, :] = 0        
        #cls[:self.delta_lmin,: , 2:self.N_bins+2] = 0
        #ksz_lmax = ksz_cutoff
        #cls[self.ksz_lmax:, self.N_bins+2:, :] = 0
        #cls[self.ksz_lmax:, :, self.N_bins+2:] = 0
        
        # print(cls[:3,:,:])
        # Invert Cls to get inverse covariance matrix
        invcov = invert_sparse(cls_sparse=cls, pols=self.pols, ells=ells)
        for i in range(n_pol):
            cls_txt = np.column_stack([cls_txt, invcov[:, i, i]])
            # print(cls_txt.shape)
        np.savetxt('invcov.txt', cls_txt)
        # I think we don't need the dicts afterwards, can save some memory
        # Especially when every Core loads one copy
        self.invcov['invcov'] = invcov
        # pretty sure we don't use the cls afterwards
        # self.invcov['cls'] = cls
        self.invcov['ells'] = ells

        self.cosmo.clear()
        self.ksz.clear()

    def init_bispec(self, shape='local', output_name='/fnl_lmax.txt', polarisation='T'):
        """
        Calculate the bispectrum
        """
        ells = self.invcov['ells']
        # beta_s = self.beta

        radii = get_ksz_radii()
        ells = np.array(ells, dtype='int64')  # avoid overflow of int
        self.compute_bispec(ells=ells, beta_s=self.beta, radii=radii, shape=shape, output_name=output_name)
        sys.stdout.flush()

    def add_noise_ksz(self, cls):
        """
        Not used at the moment, but should be called before inverting the matrix
        I feel this should be further up in the code
        :param cls:
        :return:
        """
        nls = np.loadtxt("Compare/noise_vvold120.txt")
        nls = nls[2:, :]
        nls = np.append(nls, nls[-1:, :], axis=0)
        print(nls.shape)
        print(cls.shape)
        for idx, i in enumerate([-2, -1]):
            cls[:, i, i] = cls[:, i, i] + nls[:, idx]
        return cls

    def add_noise(self, cls, ells, noise_T, noise_E, beam):
        """
        adding noise to the Cls
        Cls.shape= (4,lmax)
        assuming no noise for off diagonal elemets (i.e. TE)
        :return: cls with noise on the diagonal
        """
        arc2rad = np.pi / 180 / 60
        Nls = np.zeros_like(cls)
        Nls[0, :] = (noise_T * arc2rad) ** 2 * np.exp(ells * (ells + 1) * (beam * arc2rad) ** 2 / (8 * np.log(2)))
        Nls[1, :] = (noise_E * arc2rad) ** 2 * np.exp(ells * (ells + 1) * (beam * arc2rad) ** 2 / (8 * np.log(2)))
        cls += Nls  # Add noise for T and E

    def compute_bispec(self, ells, beta_s, radii, shape, output_name):
        """
        :param ells: array of integers from lmin to lmax
        :param beta_s: dict with first key being the polarisation pointing beta[pidx][lidx,sidx,ridx]
        :param radii: radii if mismatched then booster for beta and current booster are different
        :param shape: string, default=local
        :param output_name: string
        :return:
        """
        self.printmpi('Compute bispectrum')
        fnl_file = path + '/' + output_name

        # Distribute ells for even work load, large ell are slower
        lmin = self.lmin  # call lmin from init, default=2
        ells = np.arange(lmax)+1
        #ells = np.arange(30)+1
        # ells = self.ells
        self.printmpi(ells)
        ells_per_rank = []
        ells_sub = ells[self.mpi_rank::self.mpi_size]

        # Save size of rank for each rank to combine later
        for rank in range(self.mpi_size):
            ells_per_rank.append(ells[rank::self.mpi_size])

        fisher_sub = np.zeros((ells_sub.size))
        pol_trpl = self.pol_trpl
        #        invcov = self.invcov['invcov']

        # Clear dictionaries to free memory
        # self.invcov.clear()
        self.ksz.clear()
        self.cosmo.clear()

        # introduce lmin for density field
        # self.delta_lmin = 10
        # self.invcov['invcov'][:self.delta_lmin, 2:self.N_bins+2, 2:self.N_bins+2] = 0

        # Calculation performed in Cython. See bispectrum.pyx
        # bispectrum.compute_bispec(ells_sub, radii, beta_s, invcov, fisher_sub, pol_trpl, self.mpi_rank, shape, lmin)
        # self.compute_bispec_python(ells_sub, radii, beta_s, invcov, fisher_sub, pol_trpl, self.mpi_rank, shape, lmin)

        self.compute_bispec_wig_pre(ells_sub, radii, beta_s, self.invcov['invcov'], fisher_sub, pol_trpl, self.mpi_rank,
                                    shape, lmin)

        # fisher has been calculated at all ranks and is now being combined
        fisher_full = fisher_sub
        if self.mpi_size > 1:
            if self.mpi_rank == 0:
                print('Combining fisher')
                fisher_full = np.zeros(ells.size)
                # Place root fisher_sub in fisher_full
                fisher_full[ells_per_rank[0] - lmin] = fisher_sub
            else:
                fisher = None

            for rank in range(1, self.mpi_size):
                # Send fisher from other ranks over to root
                if self.mpi_rank == rank:
                    self.comm.Send(fisher_sub, dest=0, tag=rank)
                    # print(self.mpi_rank, 'sent')

                # Receive fisher on root
                if self.mpi_rank == 0:
                    ell_size = ells_per_rank[rank].size
                    fisher_sub = np.zeros(ell_size)
                    self.comm.Recv(fisher_sub, source=rank, tag=rank)
                    print('root received fisher{}'.format(rank))
                    fisher_full[ells_per_rank[rank] - lmin] = fisher_sub
            fisher_full = self.comm.bcast(fisher_full, root=0)
        self.printmpi('Done: fisher')
        # saving output

        fnl_end = np.sum(fisher_full)
        fnl_end = 1 / np.sqrt(fnl_end)

        fnl_max = np.zeros_like(fisher_full)
        for idx, item in enumerate(fisher_full):
            fnl_max[idx] = item + fnl_max[idx - 1]
        if self.mpi_rank==0:
            fnl_max = 1 / np.sqrt(fnl_max)
            np.savetxt(fnl_file, np.column_stack([ells, fnl_max]))
        self.printmpi(fnl_end)

    def compute_bispec_wig_pre(self, ells_sub, radii, beta_s, invcov, fisher_sub, pol_trpl, rank, shape, lmin):
        """
        Actual calculation of the bispectrum happens here
        Cutoff is implemented in here as well.
        """
        r2 = radii ** 2
        # Initialising some integers
        lidx3 = 0
        idx = 0
        # Load precalculated gaunt
        wig_file = 'precalc/gaunt_lmin{}_lmax{}_rank{}_size{}.pkl'.format(lmin, lmax, self.mpi_rank, self.mpi_size)
        pkl_file = open(wig_file, 'rb')
        gaunt_array = pickle.load(pkl_file)
        lmin_array = lmin
        # lmin_array = 1
        self.printmpi('start bispec calculation')
        # print(invcov[:3,:,:])
        for ell3 in ells_sub:
            fisher = 0
            self.printmpi('{}/{}'.format(lidx3, len(ells_sub) - 1))
            for ell2 in range(lmin_array, ell3 + 1):
                for ell1 in range(lmin_array, ell2 + 1):
                    # Wig3j is only non-zero for even sums of ell and triangle equation
                    if ((ell3 + ell2 + ell1) % 2) == 1 or ell3 > ell1 + ell2:
                        continue
                    gaunt = gaunt_array[idx]
                    pidx = 0
                    # Get bispectra for all polarisation triplets
                    bispec = np.zeros(pol_trpl.shape[0])
                    for pidx3, pidx2, pidx1 in pol_trpl:
                        shape_func = local_shape2(beta_s, ell1, ell2, ell3, lmin, pidx1, pidx2, pidx3)
                        bispec[pidx] = np.trapz(shape_func * r2, radii)

                        pidx += 1

                    delta_123 = delta_l1l2l3(ell3, ell2, ell1)  # save here because ells don't change
                    # combine different bispectra
                    for i in range(pol_trpl.shape[0]):
                        for j in range(pol_trpl.shape[0]):
                            fis = bispec[i] * bispec[j] * gaunt
                            # print(ell1,ell2,ell3)
                            # print(fis)
                            # print(invcov[ell2 - lmin, :,:])
                            fis *= invcov[ell1 - lmin, pol_trpl[i, 2], pol_trpl[j, 2]]  # pol_trpl[,2] = pidx1
                            # print(invcov[ell1 - lmin, pol_trpl[i, 2], pol_trpl[j, 2]],invcov[ell2 - lmin, pol_trpl[i, 1], pol_trpl[j, 1]],invcov[ell3 - lmin, pol_trpl[i, 0], pol_trpl[j, 0]])
                            fis *= invcov[ell2 - lmin, pol_trpl[i, 1], pol_trpl[j, 1]]
                            fis *= invcov[ell3 - lmin, pol_trpl[i, 0], pol_trpl[j, 0]]
                            fis /= delta_123
                            fisher += fis
                    idx += 1

            fisher_sub[lidx3] = fisher
            lidx3 += 1
        return fisher_sub


try:
    lmax = int(sys.argv[1])
except(IndexError):
    lmax = int(100)
try:
    output_name = str(sys.argv[2])
except(IndexError):
    output_name = 'test.txt'
try:
    prim_shape = str(sys.argv[3])
except(IndexError):
    prim_shape = str('local')
try:
    polarisation = str(sys.argv[4])
except(IndexError):
    polarisation = 'TE'
try:
    N_bins = int(sys.argv[5])
except(IndexError):
    N_bins = int(4)
try: 
    density_cutoff = int(sys.argv[6])
except(IndexError):
    density_cutoff = int(0)
try:
    ksz_cutoff = int(sys.argv[7])
except(IndexError):
    ksz_cutoff = int(-1)
# This needs to be called in a different way
F = PreCalc()
# F.init_planck_setting()
# F.init_CMBS4_setting()

ze = [0.2, 3]
# N_bins = 1

# Calculate CMB and ksz data
F.init_cosmo(lmax=lmax, kSampling=12)  # Actually only calculates to lmax - 100
F.init_ksz(lmax=lmax, ze=ze, N_bins=N_bins)
#F.init_delta(lmax=lmax, ze=ze, N_bins=N_bins)
F.init_beta(lmax=lmax, prim_shape=prim_shape, N_bins=N_bins, density_cutoff=density_cutoff)
# Polarisation input transformed into array. Also gets pol_trpl for later


F.pols_str_to_int(polarisation, N_bins)
# Get invcov. Will need to add noise here probably
F.init_invcov_from_transfer(F.pols, density_cutoff=density_cutoff, ksz_cutoff=ksz_cutoff)
# Final calculating the fisher matrix element
F.init_bispec(shape=prim_shape, output_name=output_name, polarisation=polarisation)

# with cProfile.Profile() as pr:
#    F.init_bispec(shape=prim_shape, output_name=output_name, polarisation=polarisation)

# cProfile.run('F.init_bispec(shape=prim_shape, output_name=output_name, polarisation=polarisation)', filename=output_name+'.prof')
# stats = pstats.Stats
# stats.sort_stats(pstats.SortKey.TIME)
# stats.print_stats()

# stop = timeit.default_timer()
# F.printmpi('Time: {}'.format(stop - start))

# stats = pstats.Stats(pr)
# stats.sort_stats(pstats.SortKey.TIME)
# stats.print_stats()
# stats.dump_stats(filename='needs_profiling.prof')
