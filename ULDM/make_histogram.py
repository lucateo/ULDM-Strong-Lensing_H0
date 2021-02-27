import numpy as np
import matplotlib.pyplot as plt
import pickle, h5py

noH0priorFlag = "_NEW_Inverse_theta_c_"
file_name = 'mock_corner_PLFraction'+noH0priorFlag+'uldm2uldm.h5'
h5file = h5py.File(file_name, 'r')
mcmc_new_list = h5file['dataset_mock'][:]
mcmc_new_list2 = h5file['dataset_mock_masses'][:]
mcmc_new_list3 = h5file['dataset_mock_sampled_theta_c'][:]
h5file.close()

# theta_c
plt.hist(mcmc_new_list[:,3], bins=30, range = (0,40), weights = mcmc_new_list[:,3]* mcmc_new_list[:,3])
plt.show()

# log10 m
plt.hist(mcmc_new_list2[:,2], bins=30, range=(-25.7,-24), weights = mcmc_new_list[:,3])
plt.show()

# log10 M
plt.hist(mcmc_new_list2[:,3], bins=20, range=(10,13.5), weights = mcmc_new_list[:,3])
plt.show()

