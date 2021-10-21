import matplotlib.pyplot as plt
import numpy as np

data_test = np.loadtxt('radii_komatsu.txt')
#print(data_test[:,1])



data = np.loadtxt("fnl_lmax.txt")
#data = np.loadtxt("fnl_lmax=600E.txt")
ell = data[1, :]
fnl = data[0, :]
norm=6.234181826176155e-15
plt.plot(ell, fnl)
plt.yscale('log')
plt.xscale('log')

data = np.loadtxt("wmap5baosn_max_likelihood_lmax=1500_fNLerror.txt")
ell_kom = data[:,0]
fnl_kom = data[:,1]
plt.plot(ell_kom, fnl_kom)

'''plt.show()
lmax=350
plt.plot(ell[10:lmax], (fnl[10:lmax] / fnl_kom[10:lmax])**(-1))

plt.show()'''
data = np.loadtxt("fnl_500T.txt")
ellT = data[1,:]
fnlT = data[0,:]
fac = 3/5
plt.plot(ellT, fnlT)#/fac**4)


plt.show()
