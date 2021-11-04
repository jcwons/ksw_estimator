import matplotlib.pyplot as plt
import numpy as np



data_komm = np.loadtxt("wmap5baosn_max_likelihood_lmax=1500_fNLerror.txt")
ell_kom = data_komm[:,0]
fnl_kom = data_komm[:,1]
plt.plot(ell_kom, fnl_kom)
data_E = np.loadtxt('./Output/fnl_lmax=900T.txt')
data_T = np.loadtxt('./Output/fnl_lmax=1000T.txt')
ell_E = data_E[1, :]
fnl_E = data_E[0, :]
ell_T = data_T[1, :]
fnl_T = data_T[0, :]

plt.plot(ell_T, fnl_T)
plt.plot(ell_E, fnl_E)

plt.yscale('log')
plt.xscale('log')
plt.xlim([100,1100])
plt.ylim([0.9,200])
plt.show()
#print(data_test[:,1])




data = np.loadtxt("wmap5baosn_max_likelihood_lmax=1500_fNLerror.txt")
ell_kom = data[:,0]
fnl_kom = data[:,1]
plt.plot(ell_kom, fnl_kom)

'''plt.show()
lmax=350
plt.plot(ell[10:lmax], (fnl[10:lmax] / fnl_kom[10:lmax])**(-1))

plt.show()'''
data = np.loadtxt("./Output/fnl_lmax_myradius.txt")
ellT1 = data[1,:]
fnlT1 = data[0,:]
fac = 3/5
plt.plot(ellT1, fnlT1)#/fac**4)

data = np.loadtxt("./Output/fnl_lmax.txt")
ellT = data[1,:]
fnlT = data[0,:]
plt.plot(ellT, fnlT)

#print(fnl_kom[0:300:5]/fnlT[0:300:5])
#print(fnlT1[0:300:5]/fnlT[0:300:5])
plt.yscale('log')
plt.xscale('log')

plt.show()
