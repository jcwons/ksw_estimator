# get delta_v_zlk
# for each bin call get_transfer_bin
# Generate z_samples --> split z_samples between cores
# Loop over ell and z to get all delta_v_zlk
# sum up different z's for each ell --> combine delta_v_zlk of each core



import numpy as np
from scipy import special
import camb
from camb import model
from mpi4py import MPI
import pickle

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

def Ker(ze, k, data):
    Dv_Tk = np.zeros([ze.size, k.size])
    for zidx, z in enumerate(ze):
        ae = az(z)
        etas = data.conformal_time(0.0) - data.comoving_radial_distance(z)
        Hubble = data.h_of_z(z)
        # Rtopsi = 5/3 # don't need this. We work with R
        v_cambs = - data.get_time_evolution(k, etas, ['v_newtonian_cdm'])[:, 0]
        # camb gives Dv*k^2/(a*H)*Tk
        v_cambs *= ae * Hubble
        v_cambs = np.squeeze(v_cambs)
        Ker_Tk_camb = v_cambs / k  # camb gives Dv * k^2
        Dv_Tk[zidx, :] = Ker_Tk_camb
        print("{}/{}".format(zidx+1 ,len(ze)), end='')
    return Dv_Tk

def get_transfer(ze, k, ells, data):
    print('Get the velocity growth function')
    #Dv_Tk = Ker(ze, k, data)
    delta_v_zlk = np.zeros([ze.size, ells.size, k.size])
    # Get transfer functions
    for zidx, z in enumerate(ze):
        for lidx, ell in enumerate(ells):
            const = 1 / (2 * ell + 1)
            chie = data.comoving_radial_distance(z)
            bessel = (ell * special.spherical_jn(ell - 1, k * chie) - (ell + 1) * special.spherical_jn(ell + 1,
                                                                                                       k * chie))
            bessel = special.spherical_jn(ell, k * chie)
            #delta_v = const * Dv_Tk[zidx, :] * bessel
            delta_v = const * k * bessel
            delta_v_zlk[zidx, lidx, :] = delta_v
    return delta_v_zlk

def get_transfer_bin(z_min, z_max, N_bins, Bin_num, N_samples_in_bin, k, ells, data, pars, delta_v_zlk):
    delta_v_zlk_rank = np.zeros([N_bins, ells.size, k.size])

    if MPI_rank == 0:
        z_samples = Z_bin_samples(z_min, z_max, N_bins, Bin_num, N_samples_in_bin, data, pars)
        ave, res = divmod(z_samples.size, MPI_size)
        count = [ave + 1 if p < res else ave for p in range(MPI_size)]
        count = np.array(count)

        displ = [sum(count[:p]) for p in range(MPI_size)]
        displ = np.array(displ)
        print('Samples are:', z_samples)
    else:
        z_samples = None
        # initialize count on worker processes
        count = np.zeros(MPI_size, dtype=np.int)
        displ = None
    comm.Bcast(count, root=0)
    z_rank = np.zeros(count[MPI_rank])
    comm.Scatterv([z_samples, count, displ, MPI.DOUBLE], z_rank, root=0)
    print(MPI_rank, z_rank)
    comm.Barrier()
    for zidx, z in enumerate(z_rank):
        print("{}  {}/{}".format(MPI_rank,zidx + 1, len(z_rank)), end='', flush=True)
        etas = data.conformal_time(0.0) - data.comoving_radial_distance(z)
        Hubble = data.h_of_z(z)
        v_cambs = - data.get_time_evolution(k, etas, ['v_newtonian_cdm'])[:, 0]
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
            delta_v_zlk_rank[Bin_num, lidx, :] += delta_v
    comm.Barrier()

    # the 'totals' array will hold the sum of each 'data' array
    if comm.rank == 0:
        # only processor 0 will actually get the data
        delta_v_zlk = np.zeros_like(delta_v_zlk_rank)
    else:
        totals = None
    comm.Reduce(
        [delta_v_zlk_rank, MPI.DOUBLE],
        [delta_v_zlk, MPI.DOUBLE],
        op=MPI.SUM,
        root=0
    )

    return delta_v_zlk

def run_camb(lmax=300, k_eta_fac=5, AccuracyBoost=0.8, lSampleBoost=50, lAccuracyBoost=1, ze=np.array([1, 2])):
    transfer_ksz = {}
        # Initialising CAMB with Planck 2018 Cosmology
    print("Initialising CAMB with Planck 2018 Cosmology", flush=True)
    acc_opts = dict(AccuracyBoost=AccuracyBoost,
                    lSampleBoost=lSampleBoost,
                    lAccuracyBoost=lAccuracyBoost,
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

    max_eta_k = k_eta_fac * lmax
    max_eta_k = max(max_eta_k, 50000)

    pars.max_l = lmax+700
    pars.max_eta_k = max_eta_k
    pars.max_l_evolve = lmax * 10   # NOt sure if that does anything at all

    pars.AccurateBB = True
    pars.AccurateReionization = True
    pars.AccuratePolarization = True

    if MPI_rank == 0:
        print('Getting the ks',flush=True)
        k = camb.get_transfer_functions(pars).get_cmb_transfer_data('scalar').q
        ksize = np.array(k.size, dtype=np.int)
    else:
        ksize=np.zeros(1, dtype=np.int)
    comm.Barrier()
    comm.Bcast(ksize, root=0)
    if MPI_rank != 0:
        k = np.zeros(ksize, dtype='d')
    comm.Bcast([k,MPI.DOUBLE], root=0)
    comm.Barrier()

    #Increase lmax to improve fit for ells close to lmax
    pars.set_for_lmax(7000)
    data = camb.get_background(pars)

    transfer_ksz['k'] = k
    transfer_ksz['redshift'] = ze
    ells = np.arange(2, lmax + 1)
    transfer_ksz['ells'] = ells
# Broadcast CAMB results to other cores
    # Transfer data from dict to local variables
    # CAMB Initialisation DONE
    # Get the velocity growth function
    if MPI_rank==0:
        print('Get transfer functions for velocity', flush=True)

    z_min = 0.2
    z_max = 2
    N_bins = 1
    N_samples_in_bin = 8

    delta_v_zlk = np.zeros([N_bins, ells.size, k.size])
    for i in range(N_bins):
        delta_v_zlk = get_transfer_bin(z_min, z_max, N_bins, i, N_samples_in_bin, k, ells, data, pars, delta_v_zlk)
        print('Bin done')
    delta_v_zlk /= N_samples_in_bin

    transfer_ksz['ksz'] = delta_v_zlk
    transfer_ksz['k'] = k
    transfer_ksz['redshift'] = ze
    transfer_ksz['ells'] = ells
    if MPI_rank == 0:
        with open('ksz_test_MPI.pkl', 'wb') as handle:
            pickle.dump(delta_v_zlk, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
    return transfer_ksz

comm = MPI.COMM_WORLD
MPI_size = comm.Get_size()
MPI_rank = comm.Get_rank()

run_camb()
#ksz_dict = {}
#ksz_dict['cls'] = cls_ksz
#ksz_dict['transfer'] = transfer_ksz

#ksz_file = 'ksz.pkl'
#with open(ksz_file, 'wb') as handle:
#    pickle.dump(ksz_dict, handle,
#                protocol=pickle.HIGHEST_PROTOCOL)


