import emcee
import numpy as np
import os
import h5py

# Check the mcmc samples directly, useful to see whether some hypothesis are true
#  noH0priorFlag = "_NEW_Inverse_theta_c_noH0Prior"
#  file_name = 'mock_corner_PLFraction'+noH0priorFlag+'uldm2uldm.h5'
#  h5file = h5py.File(file_name, 'r')
#  samples = h5file['dataset_mock_sampled_theta_c'][:]
#  h5file.close()
#  for i in range(len(samples)):
#      if samples[i,2] > 0.2 and samples[i,4] > 72:
#          print(samples[i])


#  filename = "mock_results_PLFraction_NEW_Inverse_theta_c_uldm2uldm.h5"
filename = "mock_results_PLFraction_NEW_Inverse_theta_c_noH0Prioruldm2uldm.h5"
#  filename = "mock_results_PLFraction_uldm2pl.h5"
#  print(os.path.exists(filename))
# IMPORTANT! Put the name keyword, otherwise it will give error
reader = emcee.backends.HDFBackend(filename, name="lenstronomy_mcmc_emcee", read_only=True)
print("Number of evaluations: {0}".format(reader.iteration))

#  print(reader.initialized)
tau = reader.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
log_prior_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)

print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
print("flat log prob shape: {0}".format(log_prob_samples.shape))
print("flat log prior shape: {0}".format(log_prior_samples.shape))
