import numpy as np
import pywigxjpf as wig
import pickle
from mpi4py import MPI

def get_wig_size(ells_sub):
    index = ells_sub - lmin + 1
    size_wigj = np.sum((index + 1) * index / 2)
    size_wigj = int(round(size_wigj))
    return(size_wigj)


wig.wig_table_init(2 * 3000, 9)
wig.wig_temp_init(2 * 3000)

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()  # Assigns every core a rank (integer value)
mpi_size = comm.Get_size()

lmax = 200
lmin = 1
ells = np.arange(lmin,lmax+1)

ells_sub = ells[mpi_rank::mpi_size]
size_wigj =get_wig_size(ells_sub)
print(ells_sub)
print(size_wigj)
wigs = np.zeros(size_wigj)

fourpi = 4 * np.pi

count = 0

for ell3 in ells_sub:
        for ell2 in range(lmin,ell3+1):
            for ell1 in range(lmin,ell2+1):
                if ((ell3 + ell2 + ell1) % 2) == 1 or ell3 > ell1 + ell2:
                    continue
                else:
                    count += 1
wigs = np.zeros(count)

idx = 0
for ell3 in ells_sub:
    print(ell3)
    for ell2 in range(lmin,ell3+1):
        for ell1 in range(lmin,ell2+1):
            if ((ell3 + ell2 + ell1) % 2) == 1 or ell3 > ell1 + ell2:
                continue
            wigs[idx] = (2 * ell1 + 1) * (2 * ell2 + 1) * (2 * ell3 + 1) / fourpi * wig.wig3jj(2 * ell1, 2 * ell2, 2 * ell3, 0, 0, 0) ** 2
            idx += 1
wig_file = 'precalc/gaunt_lmin{}_lmax{}_rank{}_size{}.pkl'.format(lmin, lmax, mpi_rank, mpi_size)
with open(wig_file, 'wb') as handle:
    pickle.dump(wigs, handle, protocol=pickle.HIGHEST_PROTOCOL)
wig_file = 'precalc/gaunt_lmin{}_lmax{}_rank{}_size{}.pkl'.format(lmin, lmax, mpi_rank, mpi_size)
pkl_file = open(wig_file, 'rb')
gaunt_array = pickle.load(pkl_file)
print('precalc/gaunt_lmin{}_lmax{}_rank{}_size{}.pkl'.format(lmin, lmax, mpi_rank, mpi_size))
