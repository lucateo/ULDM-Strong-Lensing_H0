import numpy as np
import matplotlib.pyplot as plt
import pickle, h5py
# Try with putting prior in non_H0_prior posterior
#  file_name = 'mock_corner_PLFraction_noH0Prior_uldm2uldm.h5'
#
#  h5file = h5py.File(file_name, 'r')
#  mcmc_new_list = h5file['dataset_mock'][:]
#  mcmc_new_list2 = h5file['dataset_mock_masses'][:]
#  h5file.close()
#
#
#  plt.hist(mcmc_new_list[:,5], bins=30, weights= np.exp(-(67.4 - mcmc_new_list[:,5])**2/0.4**2/2), range=(66, 69) )
#  plt.show()


# Try with flat prior on theta_c, correction on masses
#  file_name = 'mock_corner_PLFraction_NEW_largetheta_c_uldm2uldm.h5'
#
#  h5file = h5py.File(file_name, 'r')
#  mcmc_new_list = h5file['dataset_mock'][:]
#  mcmc_new_list2 = h5file['dataset_mock_masses'][:]
#  h5file.close()
#  plt.hist(mcmc_new_list2[:,3], bins=30, weights = 1/mcmc_new_list[:,3] )
#  plt.show()

noH0priorFlag = "_NEW_Inverse_theta_c_"
file_name = 'mock_corner_PLFraction'+noH0priorFlag+'uldm2uldm.h5'
h5file = h5py.File(file_name, 'r')
mcmc_new_list = h5file['dataset_mock'][:]
mcmc_new_list2 = h5file['dataset_mock_masses'][:]
mcmc_new_list3 = h5file['dataset_mock_sampled_theta_c'][:]
h5file.close()

# theta_c, not corrected
plt.xlabel(r'$ \theta_{\rm c} $')
plt.hist(mcmc_new_list[:,3], bins=30, range = (0,80))
plt.savefig('theta_c_Prior_Uniform_Inverse_theta_c.pdf')
plt.show()

# theta_c
plt.xlabel(r'$ \theta_{\rm c} $')
plt.hist(mcmc_new_list[:,3], bins=30, range = (0,80), weights = mcmc_new_list[:,3]* mcmc_new_list[:,3])
plt.savefig('theta_c_Prior_corrected_from_Inverse_theta_c.pdf')
plt.show()

# log10 m
plt.xlabel(r'$ \log_{10} m $')
plt.hist(mcmc_new_list2[:,2], bins=30, range=(-26,-24), weights = mcmc_new_list[:,3])
plt.savefig('log_m_Prior_corrected_from_Inverse_theta_c.pdf')
plt.show()

# log10 M
plt.xlabel(r'$ \log_{10} M $')
plt.hist(mcmc_new_list2[:,3], bins=20, range=(10,14), weights = mcmc_new_list[:,3])
plt.savefig('log10_M_Prior_corrected_from_Inverse_theta_c.pdf')
plt.show()

# log10 m, try correction for uniform theta_c
plt.xlabel(r'$ \log_{10} m $')
plt.hist(mcmc_new_list2[:,2], bins=30, range=(-26,-24), weights = mcmc_new_list[:,3]* mcmc_new_list[:,3] )
plt.savefig('log_m_Prior_Uniform_theta_c_from_Inverse_theta_c.pdf')
plt.show()

# log10 M, try correction for uniform theta_c
plt.xlabel(r'$ \log_{10} M $')
plt.hist(mcmc_new_list2[:,3], bins=20, range=(10,14), weights = mcmc_new_list[:,3]*mcmc_new_list[:,3])
plt.savefig('log10_M_Prior_theta_c_Uniform_from_Inverse_theta_c.pdf')
plt.show()
