import numpy as np
from scipy import special
import camb
from camb import model
from mpi4py import MPI
import pickle
import sys

def az(z):
    """Scale factor at a given redshift"""
    az = 1.0 / (1.0 + z)
    return az

class CalcKSZ:
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.mpi_rank = self.comm.Get_rank()  # Assigns every core a rank (integer value)
        self.mpi_size = self.comm.Get_size()

def Chi_bin_boundaries(z_min, z_max, pars, data, N) :
    """
    Get comoving distances (chi) of boundaries of N bins from z_min to z_max,
    equally spaced in comoving distance
    """
    Chi_min = data.comoving_radial_distance(z_min)
    Chi_max = data.comoving_radial_distance(z_max)
    Chi_boundaries = np.linspace(Chi_min, Chi_max, N+1)
    return Chi_boundaries

def Chi_bin_centers(z_min, z_max, pars, data, N) :
    """
    Get comoving distances at center of of bins from Chi_bin_boundaries()
    """
    Chi_boundaries = Chi_bin_boundaries(z_min, z_max, pars, data, N)
    # Get center of bins in comoving distance, convert to redshift
    Chis = ( Chi_boundaries[:-1] + Chi_boundaries[1:] ) / 2.0
    return Chis

def Z_bin(z_min, z_max, N, data, pars) :
    """
    Get redshifts corresponding to bin centers from Chi_bin_centers
    """
    Chis = Chi_bin_centers(z_min, z_max, pars, data, N)
    return data.redshift_at_comoving_radial_distance(Chis)

def Z_bin_samples(z_min, z_max, N_bins, Bin_num, N_samples_in_bin, data, pars):
    """
    Get redshifts of samples in a "bin" between conf.z_min and conf.z_max,
    uniformly distributed in chi, and at bin centers (so excluding boundaries.)

    N = number of bins between z_min and z_max
    B = bin number to get samples in
    N_samples = number of samples in bin
    """
    # Get boundaries of bins
    Chi_boundaries = Chi_bin_boundaries(z_min, z_max, pars, data, N_bins)
    Z_boundaries = data.redshift_at_comoving_radial_distance(Chi_boundaries)

    # Generate redshift samples inside bin
    Chi_samples = np.linspace(Chi_boundaries[Bin_num], Chi_boundaries[Bin_num + 1], N_samples_in_bin)

    # Translate this to redshift boundaries
    z_samples = data.redshift_at_comoving_radial_distance(Chi_samples)
    return z_samples

def get_transfer_bin_v(Delta_bin, z_bin, k, ells, data):
    if MPI_rank == 0:
        print('Redshift range of bin:{} {}'.format(z_bin[0],z_bin[-1]))
    delta_v_lk = np.zeros([ells.size, k.size])

    if MPI_rank == 0:
        ave, res = divmod(z_bin.size, MPI_size)
        count = [ave + 1 if p < res else ave for p in range(MPI_size)]
        count = np.array(count)

        displ = [sum(count[:p]) for p in range(MPI_size)]
        displ = np.array(displ)
    else:
        z_bin = None
        # initialize count on worker processes
        count = np.zeros(MPI_size, dtype=np.int)
        displ = None
    comm.Bcast(count, root=0)
    z_rank = np.zeros(count[MPI_rank])
    comm.Scatterv([z_bin, count, displ, MPI.DOUBLE], z_rank, root=0)
    comm.Barrier()

    for zidx, z in enumerate(z_rank):
        if MPI_rank == 0:
            print("z:  {}/{} ".format(zidx + 1, len(z_rank)), end='', flush=True)
        Hubble = data.h_of_z(z)
        v_cambs = - Delta_bin[:,zidx]  # delta_bin is split so that zidx works
        # camb gives Dv*k^2/(a*H)*Tk
        v_cambs *= az(z) * Hubble
        v_cambs = np.squeeze(v_cambs)
        Dv_Tk = v_cambs / k  # camb gives Dv * k^2
#        print("{}/{}".format(zidx + 1, len(ze)), end='')

        chie = data.comoving_radial_distance(z)
        for lidx, ell in enumerate(ells):
            const = 1 / (2 * ell + 1)
            bessel = (ell * special.spherical_jn(ell - 1, k * chie) - (ell + 1) * special.spherical_jn(ell + 1,
                                                                                                       k * chie))
            delta_v = const * Dv_Tk[:] * bessel
            delta_v_lk[lidx, :] += delta_v
    comm.Barrier()
    delta_v_lk_total = np.zeros_like(delta_v_lk)
    # the 'total' array will hold the sum of each 'delta_v_lk' array
    comm.Reduce(
        [delta_v_lk, MPI.DOUBLE],
        [delta_v_lk_total, MPI.DOUBLE],
        op=MPI.SUM,
        root=0
    )
    return delta_v_lk_total

def get_transfer_bin_d(Delta_bin, z_bin, k, ells, data):
    if MPI_rank == 0:
        print('Redshift range of bin:{} {}'.format(z_bin[0], z_bin[-1]))
    delta_d_lk = np.zeros([ells.size, k.size])

    if MPI_rank == 0:
        ave, res = divmod(z_bin.size, MPI_size)
        count = [ave + 1 if p < res else ave for p in range(MPI_size)]
        count = np.array(count)

        displ = [sum(count[:p]) for p in range(MPI_size)]
        displ = np.array(displ)
    else:
        z_bin = None
        # initialize count on worker processes
        count = np.zeros(MPI_size, dtype=np.int)
        displ = None
    comm.Bcast(count, root=0)
    z_rank = np.zeros(count[MPI_rank])
    comm.Scatterv([z_bin, count, displ, MPI.DOUBLE], z_rank, root=0)
    comm.Barrier()

    for zidx, z in enumerate(z_rank):
        if MPI_rank == 0:
            print("z:  {}/{} ".format(zidx + 1, len(z_rank)), end='', flush=True)
        Hubble = data.h_of_z(z)
        v_cambs = - Delta_bin[:, zidx]  # delta_bin is split so that zidx works
        # camb gives Dv*k^2/(a*H)*Tk
        v_cambs *= az(z) * Hubble
        v_cambs = np.squeeze(v_cambs)
        Dv_Tk = v_cambs / k  # camb gives Dv * k^2
        #        print("{}/{}".format(zidx + 1, len(ze)), end='')
        chie = data.comoving_radial_distance(z)
        for lidx, ell in enumerate(ells):
            bessel = special.spherical_jn(ell, k * chie)
            delta_d = Dv_Tk[:] * bessel
            delta_d_lk[lidx, :] += delta_d

    comm.Barrier()
    delta_d_lk_total = np.zeros_like(delta_d_lk)
    # the 'total' array will hold the sum of each 'delta_d_lk' array
    comm.Reduce(
        [delta_d_lk, MPI.DOUBLE],
        [delta_d_lk_total, MPI.DOUBLE],
        op=MPI.SUM,
        root=0
    )
    return delta_d_lk_total

def run_camb(lmax=100, N_bins=4,pol='both',input='camb'):
    acc_opts = dict(AccuracyBoost=10,
                    lSampleBoost=50,
                    lAccuracyBoost=7,
                    DoLateRadTruncation=False)

    pars = camb.CAMBparams()
    pars.set_accuracy(**acc_opts)

    pars.set_cosmology(H0=67.66,
                       TCMB=2.7255,
                       YHe=0.24,
                       standard_neutrino_neff=True,
                       ombh2=0.02242,
                       omch2=0.11933,
                       tau=0.0561,
                       mnu=0.06,
                       omk=0)

    pars.InitPower.set_params(ns=0.9665,
                              pivot_scalar=0.05,
                              As=2.1056e-9)

    pars.set_dark_energy()
    pars.NonLinear = model.NonLinear_none

    k_eta_fac=10

    k_acc = 2

    pars.Accuracy.SourcekAccuracyBoost =k_acc
    pars.Accuracy.IntkAccuracyBoost = k_acc
    pars.Accuracy.BesselBoost = k_acc


    #lmax = max(300, lmax)
    max_eta_k = k_eta_fac * lmax
    max_eta_k = max(max_eta_k, 50000)
    
    pars.max_l = lmax
    pars.max_l_tensor = lmax
    pars.max_eta_k = max_eta_k
    pars.max_eta_k_tensor = max_eta_k
    pars.max_l_evolve = lmax + 300

    pars.AccurateBB = True
    pars.AccurateReionization = True
    pars.AccuratePolarization = True

    # pars.set_for_lmax(2500, lens_potential_accuracy=3)

    pars.set_for_lmax(7000)
    # calculate results for these parameters
    # print('Calculate transfer functions')
    data = camb.get_background(pars)

    ells = np.arange(1, lmax + 1)
    #ells = np.arange(1, 10 + 1)
    # Broadcast CAMB results to other cores
    # Transfer data from dict to local variables
    # CAMB Initialisation DONE
    # Get the velocity growth function
    if MPI_rank==0:
        print('Get transfer functions for velocity', flush=True)
    pkl_file = open('k_and_z.pkl', 'rb')
    kz = pickle.load(pkl_file)
    k = kz['k']
    z = kz['z']

    if MPI_rank==0:
        print('Number of bins: {}'.format(N_bins))
        print('Sampling between the redshifts: {} - {}'.format(z[0],z[-1]))
    if input=='sync':
        if MPI_rank==0:
            print('Using synchronous gauge')
        pkl_file = open('Output/transfer_delta_sync.pkl', 'rb')
    elif input=='new':
        if MPI_rank==0:
            print('Using newtonian gauge')
        pkl_file = open('Output/transfer_delta_new.pkl', 'rb')
    else:
        if MPI_rank==0:
            print('Using camb with for delta synchronous gauge')
        pkl_file = open('Output/transfer_delta_v_camb_all.pkl', 'rb')
    deltas = pickle.load(pkl_file)


    #delta_v_zlk = np.zeros([N_bins, ells.size, k.size])
    #delta_d_zlk = np.zeros([N_bins, ells.size, k.size])
    if pol == 'velocity':
        delta_v_zlk = np.zeros([N_bins, ells.size, k.size])
        if MPI_rank == 0:
            print('Calculating velocity transfer function')
    elif pol == 'density':
        delta_d_zlk = np.zeros([N_bins, ells.size, k.size])
        if MPI_rank == 0:
            print('Calculating density transfer function')
    else:
        delta_v_zlk = np.zeros([ells.size, k.size])
        delta_d_zlk = np.zeros([ells.size, k.size])
        if MPI_rank == 0:
            print('Calculating velocity and density transfer function')


    for i in range(N_bins):
        start = np.int32((z.size / N_bins) * i)
        end = np.int32((z.size / N_bins) * (i + 1))
        z_bin = z[start:end]
        if (input=='sync' or input=='new'):
            Delta_bin = deltas[:, start:end]
            delta_d_zlk[i, :, :] = get_transfer_bin_d(Delta_bin, z_bin, k, ells, data)
            delta_d_zlk[i, :, :] /= z_bin.size
        else:
            if pol == 'velocity':
                Delta_bin = deltas[:,start:end,1]
                delta_v_zlk[i, :, :] = get_transfer_bin_v(Delta_bin, z_bin, k, ells, data)
                delta_v_zlk[i, :, :] /= z_bin.size
            elif pol == 'density':
                Delta_bin = deltas[:, start:end, 0]
                delta_d_zlk[i, :, :] = get_transfer_bin_d(Delta_bin, z_bin, k, ells, data)
                delta_d_zlk[i, :, :] /= z_bin.size
            else:
                #Delta_bin = deltas[:, start:end, 1]
                #delta_v_zlk[i, :, :] = get_transfer_bin_v(Delta_bin, z_bin, k, ells, data)
                #delta_v_zlk[i, :, :] /= z_bin.size
                #if MPI_rank == 0:
                #    tag = './Output/{}/transfer_{}_v_l100_z2.pkl'.format(N_bins,i)
                #    with open(tag, 'wb') as handle:
                #        pickle.dump(delta_v_zlk[i, :, :], handle,
                #            protocol=pickle.HIGHEST_PROTOCOL)
                Delta_bin = deltas[:, start:end, 0]
                delta_d_zlk[:, :] = get_transfer_bin_d(Delta_bin, z_bin, k, ells, data)
                delta_d_zlk[:, :] /= z_bin.size
                if MPI_rank == 0:
                    tag = './Output/{}/transfer_{}_d_l100_z2.pkl'.format(N_bins,i)
                    with open(tag, 'wb') as handle:
                        pickle.dump(delta_d_zlk[:, :], handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
        if MPI_rank == 0:
            print('Bin done')

    transfer_ksz={}
    if (input=='sync'or input=='new'):
        transfer_ksz['density'] = delta_d_zlk
    else:
        transfer_ksz['velocity'] = delta_v_zlk
        transfer_ksz['density'] = delta_d_zlk
    transfer_ksz['k'] = k
    transfer_ksz['redshift'] = z
    transfer_ksz['ells'] = ells
    if input=='sync':
        tag = './Output/{}/transfer_300_sync_{}.pkl'.format(N_bins)
    elif input=='new':
        tag = './Output/transfer_300_new_{}.pkl'.format(N_bins)
    else:
        tag = './Output/{}/transfer_l100_z2.pkl'.format(N_bins)
    if MPI_rank == 0:
        with open(tag, 'wb') as handle:
            pickle.dump(transfer_ksz, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
    return transfer_ksz

comm = MPI.COMM_WORLD
MPI_size = comm.Get_size()
MPI_rank = comm.Get_rank()

try:
    N_bins = int(sys.argv[1])
except(IndexError):
    N_bins = 4
try:
    pol = str(sys.argv[2])
except(IndexError):
    pol = 'both'
try:
    input = str(sys.argv[3])
except(IndexError):
    input = 'camb'

    
run_camb(lmax=100, N_bins=N_bins,pol=pol,input=input)


