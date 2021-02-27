import emcee
import numpy as np
import os

filename = "mock_results_PLFraction_NEW_Inverse_theta_c_uldm2uldm.h5"
#  print(os.path.exists(filename))
# IMPORTANT! Put the name keyword, otherwise it will give error
reader = emcee.backends.HDFBackend(filename, name="lenstronomy_mcmc_emcee", read_only=True)
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
