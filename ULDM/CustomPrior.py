import numpy as np

class CustomPrior(object):
    """! Class for defining the prior to use in the hierarchical sampling
    """
    def __init__(self, planck_prior=True, log_scatter=False):
        """! Initialize the class
        """
        self._planck_prior = planck_prior
        self._log_scatter = log_scatter
        self.om_mean = 0.298
        self.sigma_om = 0.022
        self.h0_mean = 67.4
        self.h0_sigma = 0.5

    def __call__(self, kwargs_cosmo, kwargs_lens, kwargs_kin):
        """! Used to make the class behave as a function
        @param kwargs_cosmo Keyword arguments for cosmological model
        @param kwargs_lens Keyword arguments for lens model
        @param kwargs_kin Keyword arguments for kinematics
        @return log prior
        """
        return self.log_likelihood(kwargs_cosmo, kwargs_lens, kwargs_kin)

    def log_likelihood(self, kwargs_cosmo, kwargs_lens, kwargs_kin):
        """! The log prior
        @param kwargs_cosmo Keyword arguments for cosmological model
        @param kwargs_lens Keyword arguments for lens model
        @param kwargs_kin Keyword arguments for kinematics
        @return log prior
        """
        logL = 0
        # a prior on Omega_m helps in constraining the MST parameter as the kinematics becomes less cosmology dependent...
        if self._planck_prior is True:
            om = kwargs_cosmo.get('om', self.om_mean)
            logL += -(om - self.om_mean)**2 / self.sigma_om**2 / 2
            ## Change here, h0_mean is the CMB value (gaussian prior)
            h0 = kwargs_cosmo.get('h0', self.h0_mean)
            logL += -(h0 - self.h0_mean)**2 / self.h0_sigma**2 / 2
        ## Uniform priors
        if self._log_scatter is True:
            #  lambda_mst_sigma = kwargs_lens.get('lambda_mst_sigma', 1)
            #  logL += np.log(1/lambda_mst_sigma)
            #  lambda_ifu_sigma = kwargs_lens.get('lambda_ifu_sigma', 1)
            #  logL += np.log(1/lambda_ifu_sigma)
            a_ani_sigma = kwargs_kin.get('a_ani_sigma', 1)
            logL += np.log(1/a_ani_sigma)
            sigma_v_sys_error = kwargs_kin.get('sigma_v_sys_error', 1)
            logL += np.log(1/sigma_v_sys_error)

        a_ani = kwargs_kin.get('a_ani', 1)
        logL += np.log(1/a_ani)
        return logL



