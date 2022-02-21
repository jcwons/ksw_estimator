import numpy as np
import cosmo as cs
from scipy.special import spherical_jn
from scipy.integrate import trapz, simps
from scipy.linalg import inv
import pywigxjpf as wig
import pickle5 as pickle
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
    booster = 2
    vlow = np.linspace(0, 10, num=booster*10, dtype=float, endpoint=False)
    ksz1 = np.linspace(10, 2000, num=booster*80, dtype=float, endpoint=False)
    ksz2 = np.linspace(2000, 6000, num=booster*300, dtype=float, endpoint=False)
    ksz3 = np.linspace(6000, 9377, num=booster*200, dtype=float, endpoint=False)
    low = np.linspace(10, 9377, num=booster*98, dtype=float, endpoint=False)
    re1 = np.linspace(9377, 10007, num=booster*18, dtype=float, endpoint=False)
    re2 = np.linspace(10007, 12632, num=booster*25, dtype=float, endpoint=False)
    rec = np.linspace(12632, 13682, num=booster*50, dtype=float, endpoint=False)
    rec_new = np.linspace(13682, 15500, num=booster*300, dtype=float, endpoint=False)
    rec_extra = np.linspace(15500, 30000, num=booster*50, dtype=float, endpoint=False)


    smith_1 = np.linspace(0, 9500, num=10*150, dtype=float, endpoint=False)
    smith_2 = np.linspace(9500, 11000, num=10*300, dtype=float, endpoint=False)
    smith_3 = np.linspace(11000, 13800, num=10*150, dtype=float, endpoint=False)
    smith_4 = np.linspace(13800, 14600, num=10*400, dtype=float, endpoint=False)
    smith_5 = np.linspace(14600, 16000, num=10*100, dtype=float, endpoint=False)
    smith_6 = np.logspace(np.log10(16000), np.log10(50000), num=10*100, dtype=float, endpoint=False)





    #radii = np.concatenate((vlow, low, re1, re2, rec, rec_new, rec_extra))
    radii = np.concatenate((vlow, ksz1, ksz2, ksz3, re1, re2, rec, rec_new, rec_extra))
    #radii = np.concatenate((smith_1, smith_2, smith_3, smith_4, smith_5, smith_6))

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
    shape_func = beta_s[ell1 - l_min, pidx1, 1, :] * beta_s[ell2 - l_min, pidx2, 1, :] * beta_s[ell3 - l_min, pidx3, 0,
                                                                                         :] \
                 + beta_s[ell2 - l_min, pidx1, 1, :] * beta_s[ell3 - l_min, pidx2, 1, :] * beta_s[ell1 - l_min, pidx3,
                                                                                           0, :] \
                 + beta_s[ell3 - l_min, pidx1, 1, :] * beta_s[ell1 - l_min, pidx2, 1, :] * beta_s[ell2 - l_min, pidx3,
                                                                                           0, :]
    shape_func *= 2
    return shape_func


def equil_shape(beta_s, ell1, ell2, ell3, l_min, pidx1, pidx2, pidx3):
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
        self.lmin = 2

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

    def init_cosmo(self, lmax, AccuracyBoost=2):
        """
        Collect transfer functions and cls to save in self.cosmo
        1) Run CAMB and get T,E,B
        2) Read in transfer and cls from kSZ and pSZ
        """
        #cosmo_file = path + '/cosmo_{}.pkl'.format(lmax - 100)
        cosmo_file = path + '/Tests/cosmo_{}_2b_3k.pkl'.format(lmax - 100)
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
            transfer, cls = cs.run_camb(lmax=lmax, lSampleBoost=50, AccuracyBoost=AccuracyBoost)
            self.cosmo['transfer'] = transfer
            self.cosmo['cls'] = cls
            if self.mpi_rank == 0:
                print('Storing cosmo as: {}'.format(cosmo_file))
                # Store in pickle file.
                with open(cosmo_file, 'wb') as handle:
                    pickle.dump(self.cosmo, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
            self.printmpi('Computed cosmo')
        ells = self.cosmo['cls']['ells']
        clss = self.cosmo['cls']['cls']['lensed_scalar']
        np.savetxt('cls.txt', np.column_stack([ells, clss[0, :], clss[1, :], clss[2, :], clss[3, :]]))

    def init_ksz(self, lmax, ze, AccuracyBoost=2):
        """
        Collect transfer functions and cls to save in self.ksz
        2) Read in transfer and cls from kSZ
        """
        #ksz_file = path + '/ksz_{}.pkl'.format(300)
        ksz_file = path + '/Tests/ksz_{}_2b_3k.pkl'.format(300)
        recompute_ksz = False
        if self.mpi_rank == 0:
            try:
                pkl_file = open(ksz_file, 'rb')
            except IOError:
                print('{} not found'.format(ksz_file))
                recompute_ksz = True
            else:
                print('loaded ksz from {}'.format(ksz_file))
                self.ksz = pickle.load(pkl_file)
                pkl_file.close()
        # Tell all ranks if file was found
        recompute_ksz = self.comm.bcast(recompute_ksz, root=0)

        if recompute_ksz is False:
            self.ksz = self.comm.bcast(self.ksz, root=0)

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

    def init_beta(self, prim_shape='local'):
        """
        Sets up calculation of beta:
        2/pi int k^2 dk f(k) j_l(k*r) T_{x,ell}(k)
        if precomputed skips calculation, read file and stores in dict
        """

        ells = self.cosmo['cls']['ells']
        lmax = ells.size + 1
        # radii = get_komatsu_radii()
        radii = get_ksz_radii()
        self.beta['radii'] = radii

        # Check if beta already has been computed
        beta_file = path + '/beta_' + prim_shape + '_{}.pkl'.format(lmax)
        #beta_file = path + '/Tests/beta_' + prim_shape + '_{}_2b_5k_3r.pkl'.format(lmax)
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
            self.compute_beta(radii, ells, prim_shape)

    def compute_beta(self, radii, ells, prim_shape):
        """
        Computes 2/pi int k^2 dk f(k) j_l(k*r) T_{x,ell}(k)
        Splits calculation among ells
        :param radii: array float
        :param ells: array int, size lmax
        :param prim_shape: str at the moment only supports local
        :return:beta(ells_sub.size, len(pols_s), np.size(func, 0), radii.size)
                containing the K functional for every shape function
        """
        transfer_cmb = self.cosmo['transfer']['scalar'][:2, :-100, :]
        transfer_ksz = self.ksz['transfer']['ksz'][:, :transfer_cmb.shape[1],:]
        self.printmpi(transfer_cmb.shape)
        self.printmpi(transfer_ksz.shape)
        #transfer_s = transfer_cmb
        k = self.cosmo['transfer']['k']
        transfer_s = np.append(transfer_cmb, transfer_ksz, axis=0)
        lmax = ells.size + 1
        n_pol = transfer_s.shape[0]
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
            if lidx % 5 == 0:
                self.printmpi('{}/{}'.format(lidx, len(ells_sub) - 1))
            for ridx, radius in enumerate(radii):
                kr = k * radius
                # jl[lidx, ridx, :] = spherical_jn(ell, kr)
                jl = spherical_jn(ell, kr)
                # integrate
                for sidx, funcs in enumerate(func):
                    for pidx in range(n_pol):
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
                beta_full = np.zeros((ells.size, n_pol, np.size(func, 0), radii.size))
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
                    beta_sub = np.zeros((ell_size, n_pol, np.size(func, 0), radii.size))
                    self.comm.Recv(beta_sub, source=rank, tag=rank)
                    print('root received {}'.format(rank))
                    beta_full[ells_per_rank[rank] - 2, :, :, :] = beta_sub[:, :, :, :]
            beta_full = self.comm.bcast(beta_full, root=0)
        beta_file = path + '/beta_' + prim_shape + '_{}.pkl'.format(lmax)
        with open(beta_file, 'wb') as handle:
            pickle.dump(beta_full, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.printmpi('Done: beta')
        self.printmpi(beta_full.shape)
        self.beta['beta_s'] = beta_full

    def pols_str_to_int(self, pol_opts, n_bins):
        if pol_opts == 'T':
            pols = np.array([0])
        elif pol_opts == 'E':
            pols = np.array([1])
        elif pol_opts == 'TE':
            pols = np.array([0, 1])
        elif pol_opts == 'ksz':
            pols = np.arange(n_bins) + 2
        elif pol_opts == 'ksz1':
            pols = np.array([2])
        elif pol_opts == 'ksz2':
            pols = np.array([3])
        elif pol_opts == 'ksz3':
            pols = np.array([4])
        elif pol_opts == 'Tksz':
            pols = np.append(0, np.arange(n_bins) + 2)
        elif pol_opts == 'Eksz':
            pols = np.append(1, np.arange(n_bins) + 2)
        else:
            pols = np.arange(n_bins + 2)
        self.pols = pols

        self.init_pol_triplets()

    def init_invcov_from_transfer(self, pols):
        """
        Calculates the inverted cls from transfer fcts
        First calculates cls then inverts
        :param pols: array int
        :return:invcov(ells_sub.size, len(pol_opts), len(pol_opts))
                containing inverse cls
        """

        # Get transfer, ells and k from dict
        ells = self.cosmo['transfer']['ells'][:-100]
        k = self.cosmo['transfer']['k']
        transfer_cmb = self.cosmo['transfer']['scalar'][:2, :ells.size, :]  # lmax is 100 higher, remove B pol
        transfer_ksz = self.ksz['transfer']['ksz'][:, :ells.size, :]
        # if different k has been used for cmb and ksz, error will pop up
        transfer = np.append(transfer_cmb, transfer_ksz, axis=0)

        # Only get the requested polarisations
        transfer = transfer[pols, :, :]

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
                I = 4 * np.pi / k * transfer[i, lidx, :] * transfer[j, lidx, :] * P_R
                cls[lidx, i, j] = simps(I, k)
        #print(ells.shape)
        #print(cls.shape)
        cls_txt = ells
        for i in range(n_pol):
            cls_txt = np.column_stack([cls_txt, cls[:, i, i]])
            #print(cls_txt.shape)
        np.savetxt('cls.txt', cls_txt)
        # Add option for noise here
        #cls = self.add_noise_ksz(cls)
        # Invert Cls to get inverse covariance matrix
        invcov = np.zeros_like(cls)
        for lidx in range(ells.size):
            invcov[lidx, :, :] = inv(cls[lidx, :, :])
        # I think we don't need the dicts afterwards, can save some memory
        # Especially when every Core loads one copy
        self.invcov['invcov'] = invcov
        self.invcov['cls'] = cls
        self.invcov['ells'] = ells

        # self.cosmo.clear()
        # self.ksz.clear()

    def init_pol_triplets(self):
        """
        initialise array with polarisation triplets
        So far only includes T=0 and E=1
        :return:
        """
        pols = self.pols
        pol_trpl = np.array(list(itertools.product(pols, repeat=3)))
        self.pol_trpl = pol_trpl

    def init_bispec(self, shape='local', output_name='/fnl_lmax.txt', polarisation=0):
        """
        Calculate the bispectrum
        """
        ells = self.cosmo['cls']['ells']
        #cls = self.cosmo['cls']['cls']['lensed_scalar']
        beta_s = self.beta['beta_s']

        radii = get_ksz_radii()
        ells = np.array(ells, dtype='int64')  # avoid overflow of int
        # Noise part needs to be adjusted for ksz cases
        # beam = self.beam
        # noise_T = self.noise
        # noise_E = np.sqrt(2) * noise_T
        # self.add_noise(cls, ells, noise_T, noise_E, beam)  # returns cls = cls + nls
        self.compute_bispec(ells=ells, beta_s=beta_s, radii=radii, shape=shape, output_name=output_name)
        sys.stdout.flush()
        # pol_trpl = np.array(list(itertools.product([0, 1], repeat=3)))  # all combinations of TTT,TTE,...

    def add_noise_ksz(self,cls):
        nls = np.loadtxt("Compare/noise_vvold120.txt")
        nls = nls[2:,:]
        nls = np.append(nls, nls[-1:,:],axis=0)
        print(nls.shape)
        print(cls.shape)
        for idx, i in enumerate([-2,-1]):
            cls[:,i,i] = cls[:,i,i] + nls[:,idx]
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

    # veraltet
    def invert_cls(self, ells, cls, pol_opts):
        cov = np.zeros((ells.size, 2, 2))
        invcov = np.zeros_like(cov)
        if pol_opts == 0:
            cov[:, 0, 0] = cls[0, :]
            cov[:, 1, 1] = cls[0, :]
        elif pol_opts == 1:
            cov[:, 0, 0] = cls[1, :]
            cov[:, 1, 1] = cls[1, :]
        elif pol_opts == 2:
            cov[:, 0, 0] = cls[0, :]  # TT
            cov[:, 1, 1] = cls[1, :]  # EE
            cov[:, 1, 0] = cls[3, :]  # TE
            cov[:, 0, 1] = cls[3, :]  # TE
        for lidx in range(ells.size):
            invcov[lidx, :, :] = inv(cov[lidx, :, :])
        return invcov

    def compute_bispec(self, ells, beta_s, radii, shape, output_name):
        # shape_factor depends on primordial template, i.e. 2 for local, 6 for others
        fisher_lmax = np.zeros(ells.size)
        self.printmpi('Compute bispectrum')
        fnl_file = path + '/' + output_name
        # Distribute ells for even work load, large ell are slower
        lmin = self.lmin  # call lmin from init, default=2
        # ells = [2...lmax]

        #ells = ells[lmin - 2:]  # ells = [lmin..lmax]. For lmin=2 nothing changes
        #cls = cls[:, lmin - 2:]  # remove ell>lmin as above

        ells_per_rank = []
        ells_sub = ells[self.mpi_rank::self.mpi_size]
        # Save size of rank for each rank to combine later
        for rank in range(self.mpi_size):
            ells_per_rank.append(ells[rank::self.mpi_size])
        # temporary beta will be combined with all ranks later

        fisher_sub = np.zeros((ells_sub.size))
        pol_opts = polarisation  # 0: T only, 1: E only, 2: TE mixed
        pol_trpl = self.pol_trpl
        invcov = self.invcov['invcov']

        # Calculation performed in Cython. See bispectrum.pyx
        # bispectrum.compute_bispec(ells_sub, radii, beta_s, invcov, fisher_sub, pol_trpl, self.mpi_rank, shape, lmin)
        # self.compute_bispec_python(ells_sub, radii, beta_s, invcov, fisher_sub, pol_trpl, self.mpi_rank, shape, lmin)
        self.compute_bispec_wig_pre(ells_sub, radii, beta_s, invcov, fisher_sub, pol_trpl, self.mpi_rank, shape, lmin)

        # fisher has been calculated at all ranks
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
                    # print('root received fisher{}'.format(rank))
                    fisher_full[ells_per_rank[rank] - lmin] = fisher_sub
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

    def compute_bispec_python(self, ells_sub, radii, beta_s, invcov, fisher_sub, pol_trpl, rank, shape, lmin):
        r2 = radii ** 2
        fourpi = 4 * np.pi
        shape_temp = 0
        invcls = invcov
        lidx3 = 0
        l_min = lmin
        for ell3 in ells_sub:
            fisher = 0
            self.printmpi(lidx3)
            for ell2 in range(lmin, ell3 + 1):
                for ell1 in range(lmin, ell2 + 1):
                    # Wig3j is only non-zero for even sums of ell and triangle equation
                    if ((ell3 + ell2 + ell1) % 2) == 1 or ell3 > ell1 + ell2:
                        continue

                    gaunt = (2 * ell1 + 1) * (2 * ell2 + 1) * (2 * ell3 + 1) / fourpi * wig.wig3jj(2 * ell1, 2 * ell2,
                                                                                                   2 * ell3, 0, 0,
                                                                                                   0) ** 2
                    for pidx3, pidx2, pidx1 in pol_trpl:
                        # gaunt = (2 * ell1 + 1) * (2 * ell2 + 1) * (2 * ell3 + 1) / fourpi * wig.wig3jj(2 * ell1, 2 * ell2, 2 * ell3, 0, 0, 0) ** 2
                        # gaunt = (2 * ell1 + 1) * (2 * ell2 + 1) * (2 * ell3 + 1) / fourpi * wig3j(2 * ell1, 2 * ell2, 2 * ell3, 0, 0, 0) ** 2
                        # local shape (beta beta alpha + 2 perm)
                        if shape_temp == 0:
                            shape_func = local_shape(beta_s, ell1, ell2, ell3, l_min, pidx1, pidx2, pidx3)
                        else:
                            shape_func = None
                        # calculate the B_(l1,l2,l3)^(X1,X2,X3) and multiply angular factor squared
                        bispec1 = np.trapz(shape_func * r2, radii)
                        for pidx6, pidx5, pidx4 in pol_trpl:
                            if (pidx3, pidx2, pidx1) == (pidx6, pidx5, pidx4):
                                bispec2 = bispec1
                            else:
                                if shape_temp == 0:
                                    shape_func = local_shape(beta_s, ell1, ell2, ell3, l_min, pidx4, pidx5, pidx6)
                                # calculate  B_(l1,l2,l3)^(X4,X5,X6)
                                bispec2 = np.trapz(shape_func * r2, radii)
                            fis = bispec1 * bispec2 * gaunt
                            fis *= (invcls[ell1 - lmin, pidx1, pidx4] * invcls[ell2 - lmin, pidx2, pidx5] *
                                    invcls[ell3 - lmin, pidx3, pidx6])
                            fis /= delta_l1l2l3(ell3, ell2, ell1)
                            fisher += fis
            fisher_sub[lidx3] = fisher
            lidx3 += 1
        return fisher_sub

    def compute_bispec_wig_pre(self, ells_sub, radii, beta_s, invcov, fisher_sub, pol_trpl, rank, shape, lmin):
        r2 = radii ** 2
        invcls = invcov
        lidx3 = 0
        idx = 0
        l_min = lmin

        pol_trpl_norm = np.array(list(itertools.product(np.arange(len(self.pols)), repeat=3)))
        # Load precalculated gaunt
        wig_file = 'precalc/gaunt_lmin{}_lmax{}_rank{}_size{}.pkl'.format(lmin, lmax, self.mpi_rank, self.mpi_size)
        pkl_file = open(wig_file, 'rb')
        gaunt_array = pickle.load(pkl_file)
        for ell3 in ells_sub:
            fisher = 0
            self.printmpi('{}/{}'.format(lidx3, len(ells_sub) - 1))
            for ell2 in range(lmin, ell3 + 1):
                for ell1 in range(lmin, ell2 + 1):
                    # Wig3j is only non-zero for even sums of ell and triangle equation
                    if ((ell3 + ell2 + ell1) % 2) == 1 or ell3 > ell1 + ell2:
                        continue
                    gaunt = gaunt_array[idx]
                    pidx = 0
                    # Get bispectra for all polarisation triplets
                    bispec = np.zeros(pol_trpl.shape[0])
                    for pidx3, pidx2, pidx1 in pol_trpl:
                        shape_func = local_shape(beta_s, ell1, ell2, ell3, l_min, pidx1, pidx2, pidx3)
                        bispec[pidx] = np.trapz(shape_func * r2, radii)
                        pidx += 1
                    delta_123 = delta_l1l2l3(ell3, ell2, ell1)  # save here because ells don't change
                    for i in range(pol_trpl.shape[0]):
                        for j in range(pol_trpl.shape[0]):
                            fis = bispec[i] * bispec[j] * gaunt
                            fis *= invcls[ell1 - lmin, pol_trpl_norm[i, 2], pol_trpl_norm[j, 2]]  # pol_trpl[,2] = pidx1
                            fis *= invcls[ell2 - lmin, pol_trpl_norm[i, 1], pol_trpl_norm[j, 1]]
                            fis *= invcls[ell3 - lmin, pol_trpl_norm[i, 0], pol_trpl_norm[j, 0]]
                            fis /= delta_123
                            fisher += fis
                    idx += 1

            fisher_sub[lidx3] = fisher
            lidx3 += 1
        return fisher_sub

    def combine_bispec(self, pol_trpl, pol_trpl_norm, bispec, delta_123, gaunt, invcls, ell1, ell2, ell3, lmin):
        fisher = 0
        for i in range(pol_trpl.shape[0]):
            for j in range(pol_trpl.shape[0]):
                fis = bispec[i] * bispec[j] * gaunt
                fis *= invcls[ell1 - lmin, pol_trpl_norm[i, 2], pol_trpl_norm[j, 2]]  # pol_trpl[,2] = pidx1
                fis *= invcls[ell2 - lmin, pol_trpl_norm[i, 1], pol_trpl_norm[j, 1]]
                fis *= invcls[ell3 - lmin, pol_trpl_norm[i, 0], pol_trpl_norm[j, 0]]
                fis /= delta_123
                fisher += fis
                return fisher
try:
    lmax = int(sys.argv[1])
except(IndexError):
    lmax = int(300)
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
    polarisation = 'all'



# This needs to be called in a different way
N_bins = 2
ze = [1, 2]

F = PreCalc()
# F.init_planck_setting()
# F.init_CMBS4_setting()

# Calculate CMB and ksz data
F.init_cosmo(lmax=lmax + 100, AccuracyBoost=1)  # Actually only calculates to lmax - 100
F.init_ksz(lmax=lmax+100, ze=ze, AccuracyBoost=1)
start = timeit.default_timer()
# Pre-calculate the primordial fcts
F.init_beta(prim_shape)
# Polarisation input transformed into array. Also gets pol_trpl for later
F.pols_str_to_int(polarisation, N_bins)
# Get invcov. Will need to add noise here probably
F.init_invcov_from_transfer(F.pols)
# Final calculating the fisher matrix element
F.init_bispec(shape=prim_shape, output_name=output_name, polarisation=polarisation)

#with cProfile.Profile() as pr:
#    F.init_bispec(shape=prim_shape, output_name=output_name, polarisation=polarisation)

#cProfile.run('F.init_bispec(shape=prim_shape, output_name=output_name, polarisation=polarisation)', filename=output_name+'.prof')
# stats = pstats.Stats
# stats.sort_stats(pstats.SortKey.TIME)
# stats.print_stats()

#stop = timeit.default_timer()
#F.printmpi('Time: {}'.format(stop - start))

#stats = pstats.Stats(pr)
#stats.sort_stats(pstats.SortKey.TIME)
#stats.print_stats()
#stats.dump_stats(filename='needs_profiling.prof')
