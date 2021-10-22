import matplotlib.pyplot as plt
import numpy as np

print((3/5)**(-4)*0.81)



data_komm = np.loadtxt("wmap5baosn_max_likelihood_lmax=1500_fNLerror.txt")
ell_kom = data_komm[:,0]
fnl_kom = data_komm[:,1]
plt.plot(ell_kom, fnl_kom)
data_E = np.loadtxt('./Output/fnl_lmax=900E.txt')
data_T = np.loadtxt('./Output/fnl_lmax=900T.txt')
ell_E = data_E[1, :]
fnl_E = data_E[0, :]*6.5
ell_T = data_T[1, :]
fnl_T = data_T[0, :]*6.5

print(fnl_T[100:800:10]/fnl_E[100:800:10])
print(fnl_T[100:800:10]-fnl_E[100:800:10])
print((fnl_T[800]/fnl_kom[800])**(-1))

plt.plot(ell_T, fnl_T)
plt.plot(ell_E, fnl_E)

plt.yscale('log')
plt.xscale('log')
plt.xlim([100,1100])
plt.ylim([0.9,200])
plt.show()
#print(data_test[:,1])



data = np.loadtxt("fnl_lmax.txt")
#data = np.loadtxt("fnl_lmax=600E.txt")
ell = data[1, :]
fnl = data[0, :]
norm=6.234181826176155e-15
plt.plot(ell, fnl)

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


#plt.show()
