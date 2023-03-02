import numpy as np
# import cosmo as cs
from scipy.special import spherical_jn
from scipy.integrate import trapz, simps
from scipy.linalg import inv
import pywigxjpf as wig
import pickle
import os
import sys
# import ksz
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
    booster = 1
    vlow = np.linspace(0, 10, num=booster * 10, dtype=float, endpoint=False)
    ksz = np.linspace(10, 8000, num=booster * 7200, dtype=float, endpoint=False)

    radii = np.concatenate((vlow, ksz))
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

def local_shape(beta_s, ell1, ell2, ell3, lmin, p1, p2, p3):
    # New version where polarisation index is a dictionary key
    # ell index and pidx has to match
    # bba
    shape = beta_s[p1][ell1 - lmin, 1, :] * beta_s[p2][ell2 - lmin, 1, :] * beta_s[p3][ell3 - lmin, 0, :]
    # bab
    shape += beta_s[p1][ell1 - lmin, 1, :] * beta_s[p2][ell2 - lmin, 0, :] * beta_s[p3][ell3 - lmin, 1, :]
    # abb
    shape += beta_s[p1][ell1 - lmin, 0, :] * beta_s[p2][ell2 - lmin, 1, :] * beta_s[p3][ell3 - lmin, 1, :]
    shape *= 2
    return shape

def equil_shape(beta_s, ell1, ell2, ell3 ,lmin, p1, p2, p3):
    # equilateral, orthogonal and flattened shape consist of same shape functions, but with different prefactors
    # a: alpha, b: beta, g: gamma, d: delta
    # eq: -1,-2,+1
    # or: +1, +3, -1
    # fo: -3, -8, +3
    
    
    bba = 1.
    ddd = 3.
    bdg = -1.

    # bba
    shape = bba * local_shape(beta_s, ell1, ell2, ell3, lmin, p1, p2, p3) / 2 # local shape has 2 factor
    # ddd
    shape += ddd * beta_s[p1][ell1 - lmin, 3, :] * beta_s[p2][ell2 - lmin, 3, :] * beta_s[p3][ell3 - lmin, 3, :]
    # bgd
    shape += bdg * beta_s[p1][ell1 - lmin, 1, :] * beta_s[p2][ell2 - lmin, 2, :] * beta_s[p3][ell3 - lmin, 3, :]
    # bdg
    shape += bdg * beta_s[p1][ell1 - lmin, 1, :] * beta_s[p2][ell2 - lmin, 3, :] * beta_s[p3][ell3 - lmin, 2, :]
    # gbd
    shape += bdg * beta_s[p1][ell1 - lmin, 2, :] * beta_s[p2][ell2 - lmin, 1, :] * beta_s[p3][ell3 - lmin, 3, :]
    # gdb
    shape += bdg * beta_s[p1][ell1 - lmin, 2, :] * beta_s[p2][ell2 - lmin, 3, :] * beta_s[p3][ell3 - lmin, 1, :]
    # dbg
    shape += bdg * beta_s[p1][ell1 - lmin, 3, :] * beta_s[p2][ell2 - lmin, 1, :] * beta_s[p3][ell3 - lmin, 2, :]
    # dgb
    shape += bdg * beta_s[p1][ell1 - lmin, 3, :] * beta_s[p2][ell2 - lmin, 2, :] * beta_s[p3][ell3 - lmin, 1, :]

    shape *= 6
    return shape

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
        self.transfer = {}

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

        pkl_file = open('k_and_z.pkl', 'rb')
        kz = pickle.load(pkl_file)
        self.k = kz['k']
        self.z = kz['z']

    def printmpi(self, text):
        if self.mpi_rank == 0:
            print(text, flush=True)

    def init_transfer(self, lmax, N_bins):
        """
        Collect transfer functions and cls to save in self.ksz
        ze currently not used. file name is what determines number of bins and everything
        Precalculate ksz and use correct file name
        """
        self.printmpi(self.pols)
        for i in self.pols:
            self.transfer[i] = {}
            self.N_bins = N_bins
            if i < N_bins:
                ksz_file = path + '/{}/transfer_{}_d_l100_z2.pkl'.format(N_bins, i)
            else:
                ksz_file = path + '/{}/transfer_100_n{}_b{}.pkl'.format(N_bins, N_bins, i-N_bins)
            try:
                pkl_file = open(ksz_file, 'rb')
            except IOError:
                self.printmpi('{} not found'.format(ksz_file))
                recompute_ksz = True
            else:
                self.printmpi('loaded ksz from {}'.format(ksz_file))
                self.ksz['transfer'] = pickle.load(pkl_file)
                pkl_file.close()

            if i < N_bins:
                self.transfer[i] = self.ksz['transfer'][:100,:]
                # Option above used to make problems, if still not working use option below
#                self.transfer = {i: {'density': self.ksz['transfer']['density']}}
            else:
                self.transfer[i] = self.ksz['transfer']['velocity']
            #self.printmpi(self.transfer)
    def alpha_prim(self, k):
        """
        Returns shape function
        For local:  alpha_prim = 1
        """
        if k is None:
            k = self.k
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
            k = self.k
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
            k = self.k
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
            k = self.k
        km3 = k ** -3
        # Multiply the power spectrum scaling: (k/kp)^(n_s-1)
        km3 *= (k / kpivot) ** (ns - 1)
        delta_prim = km3 * self.norm
        delta_prim = delta_prim ** (2 / 3)
        return delta_prim

    def init_beta(self, lmax, prim_shape='local', N_bins=2):
        """
        Sets up calculation of beta:
        2/pi int k^2 dk f(k) j_l(k*r) T_{x,ell}(k)
        if precomputed skips calculation, read file and stores in dict
        """
        self.N_bins = N_bins
        ells = np.arange(lmax) + 1
        self.ells = ells
        lmax = ells.size
        self.printmpi('lmax:{}'.format(lmax))
        # radii = get_komatsu_radii()
        radii = get_ksz_radii()
        self.beta['radii'] = radii
        for i in self.pols:
            # Check if beta already has been computed
            # if prim_shape == 'local':
            #     beta_file = path + '/{}/beta_{}_l{}_n{}_b{}.pkl'.format(self.N_bins, prim_shape, lmax, self.N_bins, i)
            # else:
            #     beta_file = path + '/{}/beta_{}_l{}_n{}_b{}.pkl'.format(self.N_bins, 'equil', lmax, self.N_bins, i)
            beta_file = path + '/{}/beta_{}_l{}_n{}_b{}.pkl'.format(self.N_bins, 'equil', lmax, self.N_bins, i)
            try:
                pkl_file = open(beta_file, 'rb')
                recompute_beta = False
            except IOError:
                self.printmpi('{} not found'.format(beta_file))
                recompute_beta = True
            else:
                self.printmpi('loaded beta from {}'.format(beta_file))
                beta_temp = pickle.load(pkl_file)
                pkl_file.close()
            if prim_shape == 'local':
                for j in beta_temp.keys():
                    beta_temp[j]=beta_temp[j][:,:2,:]
            if recompute_beta:
                if i < N_bins:
                    self.compute_beta(radii=radii, ells=ells, prim_shape=prim_shape, transfer_s=self.transfer[i], idx=i)
                else:
                    self.compute_beta(radii=radii, ells=ells, prim_shape=prim_shape, transfer_s=self.transfer[i]['velocity'], idx=i)
            self.beta.update(beta_temp)
            self.printmpi('Done beta bin: {}'.format(i))
        self.printmpi('Done: beta')

    def compute_beta(self, radii, ells, prim_shape, transfer_s,idx):
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
        k = self.k
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
        beta_temp = np.zeros((ells_sub.size, np.size(func, 0), radii.size))

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
                    # stop calculating beta for kSZ if above kSZ
                    integrand = np.zeros(k.size)
                    integrand += jl[:]
                    integrand *= transfer_s[ell - self.lmin, :]
                    integrand *= func[sidx, :]
                    beta_temp[lidx, sidx, ridx] = trapz(integrand, k)
        beta_temp *= 2 / np.pi

        # Move beta if not parallel
        beta_full = beta_temp

        if self.mpi_size > 1:
            if self.mpi_rank == 0:
                print('Combining beta')
                beta_full = np.zeros((ells.size, np.size(func, 0), radii.size))
                # Place root beta_temp in beta_full
                beta_full[ells_per_rank[0] - self.lmin, :, :] = beta_temp[:, :, :]
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
                    beta_sub = np.zeros((ell_size, np.size(func, 0), radii.size))
                    self.comm.Recv(beta_sub, source=rank, tag=rank)
                    print('root received {}'.format(rank))
                    beta_full[ells_per_rank[rank] - self.lmin, :, :] = beta_sub[:, :, :]
            beta_full = self.comm.bcast(beta_full, root=0)

        '''
        Beta dictionary keys:           Example for N_bin = 4
        0      - N_bin-1   density     0-3     density
        N_bin - 2*N_bin-1              4-7    velocity
        '''
        # Put the density beta function into dictionary
        beta_file_calc = path + '/{}/beta_{}_l{}_n{}_b{}.pkl'.format(self.N_bins, prim_shape, lmax-1, self.N_bins, idx)
        beta_calc = {}
        beta_calc[idx] = beta_full
        with open(beta_file_calc, 'wb') as handle:
            pickle.dump(beta_calc, handle, protocol=pickle.HIGHEST_PROTOCOL)
            self.beta.update(beta_calc)

    def pols_str_to_int(self, pol_opts, n_bins, distance):
        # At this part we include number of bins via n_bins. I guess if we have additional bins and not change n_bins
        # we will only use the first 2 bins (2 = default number)
        self.n_bins = n_bins
        self.distance = distance
        if pol_opts == 'delta':
            pols = np.arange(n_bins)

        elif 'delta_' in pol_opts:
            pols = pol_opts.split('_')[1]
            if '-' in pols:
                start = pols.split('-')[0]
                end = pols.split('-')[1]
                pols = np.arange(int(start), int(end) + 1) - 1
                pols = np.ravel(pols)
            else:
                pols = int(pols) - 1

        elif pol_opts == 'ksz':
            pols = np.arange(n_bins) + n_bins

        elif 'ksz_' in pol_opts:
            pols = pol_opts.split('_')[1]
            if '-' in pols:
                start = pols.split('-')[0]
                end = pols.split('-')[1]
                pols = np.arange(int(start), int(end) + 1) + n_bins - 1
            else:
                pols = int(pols) + n_bins - 1
        elif pol_opts == 'deltaksz':
            pols = np.arange(2*n_bins)
        else:
            print('polarisation input not supported. Use "delta", "ksz"  for all bins. "delta_a" for bin a. "delta_a-b" for bin a-b')
        pols=np.array([pols])
        self.pols = np.ravel(pols)
        self.printmpi('number of modes: {}'.format(pols.size))
        self.printmpi('Modes: {}'.format(pols))
        self.printmpi('Delta bins go from {} to {}'.format(0,n_bins-1))
        self.printmpi('ksz bins go from {} to {}'.format(n_bins, 2*n_bins-1))
        self.init_pol_triplets()

    def init_pol_triplets(self):
        """
        initialise array with polarisation triplets
        :return:
        """
        pols = self.pols
        # counts triplets with actual numbers example pol = 3,4,5
        # trpl: 3,3,3  3,3,4  3,3,5  3,4,3 ...
        pol_trpl = np.array(list(itertools.product(pols, repeat=3)))
        pol_trpl_norm = np.array(list(itertools.product(np.arange(pols.size), repeat=3)))

        nearest_neighbour = False # Only keep nearest neighbours, change to False for all polariations
        neighbours = self.distance # in 3D, nearest: 2, next to nearest 6, 8, 14
        if nearest_neighbour:
            self.printmpi(pol_trpl.shape[0] ** 2)

            to_remove = np.zeros(pol_trpl.shape[0])
            for i in range(pol_trpl.shape[0]):
                distance = (pol_trpl[i, 0] - pol_trpl[i, 1]) ** 2 + (pol_trpl[i, 0] - pol_trpl[i, 2]) ** 2 + (
                            pol_trpl[i, 1] - pol_trpl[i, 2]) ** 2
                if distance > neighbours:
                    to_remove[i] = 1
            pol_trpl = pol_trpl[np.where(to_remove == 0)]
            self.printmpi(pol_trpl.shape[0] ** 2)
            pol_trpl_norm = pol_trpl_norm[np.where(to_remove == 0)]

        self.pol_trpl = pol_trpl
        # trpl_norm: 0,0,0  0,0,1  0,0,2  0,1,0 ...
        self.pol_trpl_norm = pol_trpl_norm


    def init_invcov_from_transfer(self, pols, density_cutoff=0, ksz_cutoff=-1):
        """
        Calculates the inverted cls from transfer fcts
        First calculates cls then inverts using np.linalg.inv


        :param pols: array int
        :return:invcov(ells_sub.size, len(pol_opts), len(pol_opts))
                containing inverse cls
        """
        # Get transfer, ells and k from dict
        ells = self.ells
        k = self.k
        # Combine transfer dictionary into array
        transfer = np.zeros([self.pols.size, ells.size, k.size])
        for idx, i in enumerate(self.pols):
            #print(self.transfer[i].shape)
            #print(transfer[idx, :, :].shape)
            transfer[idx, :, :] = self.transfer[i]
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
        cls_txt = ells
        for i in range(n_pol):
            cls_txt = np.column_stack([cls_txt, cls[:, i, i]])
            # print(cls_txt.shape)
        np.savetxt('cls.txt', cls_txt)
        # delta lmin is implemented by hand at this point removing all density cls up to lmin 
        self.delta_lmin = density_cutoff
        cls[:self.delta_lmin, :self.N_bins, :] = 0
        cls[:self.delta_lmin, :, :self.N_bins] = 0
        # similar ksz_lmax
        self.ksz_lmax = ksz_cutoff
        cls[self.ksz_lmax:, self.N_bins:, :] = 0
        cls[self.ksz_lmax:, :, self.N_bins:] = 0
        

        # Invert Cls to get inverse covariance matrix
        # invcov = invert_sparse(cls_sparse=cls, pols=self.pols, ells=ells)
        invcov = np.zeros_like(cls)
        for lidx, ell in enumerate(ells):
            try:
                invcov[lidx, :, :] = np.linalg.inv(cls[lidx, :, :])
            except(np.linalg.LinAlgError):
                invcov[lidx, :, :] = np.linalg.pinv(cls[lidx, :, :])
        for i in range(n_pol):
            cls_txt = np.column_stack([cls_txt, invcov[:, i, i]])
            # print(cls_txt.shape)
        np.savetxt('invcov.txt', cls_txt)
        # I think we don't need the dicts afterwards, can save some memory
        # Especially when every Core loads one copy
        self.invcov['invcov'] = invcov
        self.invcov['ells'] = ells

        self.transfer.clear()

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

    def compute_bispec(self, ells, beta_s, radii, shape, output_name):
        """
        :param ells: array of integers from lmin to lmax
        :param beta_s: dict with first key being the polarisation pointing beta[pidx][lidx,sidx,ridx]
        :param radii: radii if mismatched then booster for beta and current booster are different
        :param shape: string, default=local
        :param output_name: string
        :return:

        Calls the function to calculate the bispectrum and then combines
        There used to be different implementation (basic, cython, prewig), but now only prewig is implemented
        Could switch back to basic because up to ell=100 wigner doesn't take much time, but no need
        """
        self.printmpi('Compute bispectrum')
        fnl_file = path + '/' + output_name

        # Distribute ells for even work load, large ell are slower
        lmin = self.lmin  # call lmin from init, default=2
        ells_per_rank = []
        ells_sub = ells[self.mpi_rank::self.mpi_size]

        # Save size of rank for each rank to combine later
        for rank in range(self.mpi_size):
            ells_per_rank.append(ells[rank::self.mpi_size])

        fisher_sub = np.zeros((ells_sub.size))
        pol_trpl = self.pol_trpl
        invcov = self.invcov['invcov']
        # Calculation performed in Cython. See bispectrum.pyx
        # bispectrum.compute_bispec(ells_sub, radii, beta_s, invcov, fisher_sub, pol_trpl, self.mpi_rank, shape, lmin)
        # self.compute_bispec_python(ells_sub, radii, beta_s, invcov, fisher_sub, pol_trpl, self.mpi_rank, shape, lmin)

        self.compute_bispec_wig_pre(ells_sub, radii, beta_s, invcov, fisher_sub, pol_trpl, self.mpi_rank, shape, lmin)

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
                    # print('root received fisher{}'.format(rank))
                    fisher_full[ells_per_rank[rank] - lmin] = fisher_sub
            fisher_full = self.comm.bcast(fisher_full, root=0)
        self.printmpi('Done: fisher')
        # saving output
        if self.mpi_rank == 0:
            fnl_end = np.sum(fisher_full)
            fnl_end = 1 / np.sqrt(fnl_end)

            fnl_max = np.zeros_like(fisher_full)
            for idx, item in enumerate(fisher_full):
               fnl_max[idx] = item + fnl_max[idx - 1]
            fnl_max = 1 / np.sqrt(fnl_max)
            self.printmpi(fnl_end)
            np.savetxt(fnl_file, np.column_stack([ells, fnl_max]))


    def compute_bispec_wig_pre(self, ells_sub, radii, beta_s, invcov, fisher_sub, pol_trpl, rank, shape, lmin):
        """
        Actual calculation of the bispectrum happens here
        Cutoff is implemented in here as well.
        """
        pol_trpl_inv = self.pol_trpl_norm
        self.printmpi(pol_trpl)
        self.printmpi(pol_trpl_inv)
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
                        if shape == 'local':
                            shape_func = local_shape(beta_s, ell1, ell2, ell3, lmin, pidx1, pidx2, pidx3)
                        elif shape == 'equil':
                            shape_func = equil_shape(beta_s, ell1, ell2, ell3, lmin, pidx1, pidx2, pidx3)
                        bispec[pidx] = np.trapz(shape_func * r2, radii)

                        pidx += 1

                    delta_123 = delta_l1l2l3(ell3, ell2, ell1)  # save here because ells don't change
                    # combine different bispectra
                    for i in range(pol_trpl_inv.shape[0]):
                        for j in range(pol_trpl_inv.shape[0]):
                            fis = bispec[i] * bispec[j] * gaunt
                            fis *= invcov[ell1 - lmin, pol_trpl_inv[i, 2], pol_trpl_inv[j, 2]]  # pol_trpl[,2] = pidx1
                            fis *= invcov[ell2 - lmin, pol_trpl_inv[i, 1], pol_trpl_inv[j, 1]]
                            fis *= invcov[ell3 - lmin, pol_trpl_inv[i, 0], pol_trpl_inv[j, 0]]
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
try:
    distance = int(sys.argv[8])
except(IndexError):
    distance = int(0)
# This needs to be called in a different way
F = PreCalc()
# F.init_planck_setting()
# F.init_CMBS4_setting()


# Calculate CMB and ksz data
# Polarisation input transformed into array. Decides which bins to load.
F.pols_str_to_int(polarisation, N_bins, distance)
# Get the requested transfer functions from file
F.init_transfer(lmax=lmax, N_bins=N_bins)
# Get/calculate the requested beta functions
F.init_beta(lmax=lmax, prim_shape=prim_shape, N_bins=N_bins)
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
