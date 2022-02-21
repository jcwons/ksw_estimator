import matplotlib.pyplot as plt
import numpy as np

def plot_from_data(file):
    data = np.loadtxt(file)
    ell = data[:, 0]
    fnl = data[:, 1]
    plt.plot(ell, fnl)

plt.yscale('log')
plt.xscale('log')
plot_from_data('./Output/equil/fnl_eqT.txt')
plot_from_data('./Output/equil/fnl_eqE.txt')

plt.show()
print(1/np.sqrt((1/4.8**2)+(1/16**2)))
plt.yscale('log')
plt.xscale('log')
plot_from_data('./Output/local/fnl_local_T_planck.txt')
#plot_from_data('./Output/local/fnl_local_E.txt')
plot_from_data('./Output/local/fnl_local_E_planck.txt')

plt.show()

data_komm = np.loadtxt("./Output/fnl_Tmixed.txt")
ell_kom = data_komm[:,0]
fnl_kom = data_komm[:,1]
data_E = np.loadtxt('./Output/fnl_Emixed.txt')
data_T = np.loadtxt('./Output/fnl_TEnew.txt')
ell_E = data_E[:,0]
fnl_E = data_E[:,1]
ell_T = data_T[:,0]
fnl_T = data_T[:,1]

plt.plot(ell_kom, fnl_kom)
plt.plot(ell_T, fnl_T)
plt.plot(ell_E, fnl_E)
#plt.plot(ell_kom, fnl_kom*ell_kom)
#plt.plot(ell_T, fnl_T*ell_T)
#plt.plot(ell_E, fnl_E*ell_T)

plt.yscale('log')
plt.xscale('log')
plt.xlim([100,1100])
plt.ylim([1,100])
plt.show()
#print(data_test[:,1])




data = np.loadtxt("./Output/fnl_E2000.txt")
ell_kom = data[1,:]
fnl_kom = data[0,:]
plt.plot(ell_kom, fnl_kom)

'''plt.show()
lmax=350
plt.plot(ell[10:lmax], (fnl[10:lmax] / fnl_kom[10:lmax])**(-1))

plt.show()'''
data = np.loadtxt("./Output/fnl_T2000.txt")
ellT1 = data[1,:]
fnlT1 = data[0,:]

data = np.loadtxt("./Output/fnl_Emixed.txt")
ell_kom = data[:,0]
fnl_kom = data[:,1]
plt.plot(ell_kom, fnl_kom)

print(fnlT1[197])
plt.plot(ellT1, fnlT1)#/fac**4)

#print(fnl_kom[0:300:5]/fnlT[0:300:5])
#print(fnlT1[0:300:5]/fnlT[0:300:5])
plt.yscale('log')
plt.xscale('log')

plt.show()
