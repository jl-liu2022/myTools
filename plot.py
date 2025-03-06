import numpy as np
import matplotlib.pyplot as plt

data_aper_BV = np.loadtxt('2022HRS_aper_cal.dat')
data_aper_gri = np.loadtxt('2022HRS_aper_sloan_cal.dat')
data_psf_BV = np.loadtxt('2022HRS_psf_cal.dat')
data_psf_gri = np.loadtxt('2022HRS_psf_sloan_cal.dat')

fig, ax = plt.subplots()
ax.set_ylim(20,2)
#ax.errorbar(data_aper_BV[:,0], data_aper_BV[:,4], yerr=data_aper_BV[:,5], label='B', linestyle='', fmt = 'o', capsize = 7)
#ax.errorbar(data_aper_BV[:,0], data_aper_BV[:,7] - 2, yerr=data_aper_BV[:,8], label='V', linestyle='', fmt = 'o', capsize = 7)
#ax.errorbar(data_aper_gri[:,0], data_aper_gri[:,1] - 4, yerr=data_aper_BV[:,2],label='g', linestyle='', fmt = 'o', capsize = 7)
#ax.errorbar(data_aper_gri[:,0], data_aper_gri[:,4] - 6, yerr=data_aper_BV[:,5],label='r', linestyle='', fmt = 'o', capsize = 7)
#ax.errorbar(data_aper_gri[:,0], data_aper_gri[:,7] - 8, yerr=data_aper_BV[:,8],label='i', linestyle='', fmt = 'o', capsize = 7)
ax.errorbar(data_psf_BV[:,0], data_aper_BV[:,4], yerr=data_aper_BV[:,5], label='B', linestyle='', fmt = 'o', capsize = 7)
ax.errorbar(data_psf_BV[:,0], data_aper_BV[:,7] - 2, yerr=data_aper_BV[:,8], label='V', linestyle='', fmt = 'o', capsize = 7)
ax.errorbar(data_psf_gri[:,0], data_aper_gri[:,1] - 4, yerr=data_aper_BV[:,2],label='g', linestyle='', fmt = 'o', capsize = 7)
ax.errorbar(data_psf_gri[:,0], data_aper_gri[:,4] - 6, yerr=data_aper_BV[:,5],label='r', linestyle='', fmt = 'o', capsize = 7)
ax.errorbar(data_psf_gri[:,0], data_aper_gri[:,7] - 8, yerr=data_aper_BV[:,8],label='i', linestyle='', fmt = 'o', capsize = 7)



ax.legend()
plt.show()