import numpy as np
import pickle5 as pickle
import camb
from camb import model

# Define functions needed for to calculate the redshift/comoving distances at which we need the transfer function
def Chi_bin_boundaries(z_min, z_max, pars, data, N_bins):
    """
    Get comoving distances (chi) of boundaries of N bins from z_min to z_max,
    equally spaced in comoving distance
    """
    Chi_min = data.comoving_radial_distance(z_min)
    Chi_max = data.comoving_radial_distance(z_max)
    Chi_boundaries = np.linspace(Chi_min, Chi_max, N_bins + 1)
    return Chi_boundaries

def Chi_bin_centers(z_min, z_max, N):
    """
    Get comoving distances at center of of bins from Chi_bin_boundaries()
    """
    Chi_boundaries = Chi_bin_boundaries(z_min, z_max, pars, data, N)
    # Get center of bins in comoving distance, convert to redshift
    Chis = (Chi_boundaries[:-1] + Chi_boundaries[1:]) / 2.0
    return Chis

def Z_bin(z_min, z_max, N, data, pars):
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

# Set parameters and run CAMB
acc_opts = dict(AccuracyBoost=3,
                lSampleBoost=50,
                lAccuracyBoost=3,
                DoLateRadTruncation=False)

pars = camb.CAMBparams()
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
pars.set_accuracy(**acc_opts)
pars.set_dark_energy()
pars.NonLinear = model.NonLinear_none
data = camb.get_background(pars)

# Load cosmo file to get k and calculate Z

beta_file = './k_and_z.pkl'
pkl_file = open(beta_file, 'rb')
beta = pickle.load(pkl_file)
ks = beta['k']
print('Number of k-samples:  {}'.format(ks.size))
z_min = 0.2
z_max = 2
N_bins = 1
Bin_num = 0
N_samples_in_bin = np.int(20000)
z_samples = Z_bin_samples(z_min, z_max, N_bins, Bin_num, N_samples_in_bin, data, pars)

print('Number of z - samples between z_min={} and z_max={}:  {}'.format(z_min,z_max,z_samples.size))

# Save ks and zs in dict to pickle
out = {}
out['k']=ks
out['z']=z_samples
with open('k_and_z.pkl', 'wb') as handle:
    pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Calculate the transfer function for matter and velocity
N = 50
Ns = int(ks.size/N)
etas = data.conformal_time(0.0) - data.comoving_radial_distance(z_samples)
for i in range(N+1):
    print(i)
    transfer = data.get_time_evolution(ks[Ns*i:Ns*(i+1)], etas, ['delta_tot','v_newtonian_cdm'])
    
    with open('Output/transfer_delta_v_camb_{}.pkl'.format(i), 'wb') as handle:
       pickle.dump(transfer, handle, protocol=pickle.HIGHEST_PROTOCOL)

