__author__ = 'sibirrer'

import numpy as np

LIKELIHOOD_TYPES = ['DdtGaussian', 'DdtDdKDE', 'DdtDdGaussian', 'DsDdsGaussian', 'DdtLogNorm', 'IFUKinCov', 'DdtHist',
                    'DdtHistKDE', 'DdtHistKin', 'DdtGaussKin']


class LensLikelihoodBase(object):
    """
    master class containing the likelihood definitions of different analysis
    """
    def __init__(self, z_lens, z_source, likelihood_type, name='name', **kwargs_likelihood):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param name: string (optional) to name the specific lens
        :param likelihood_type: string to specify the likelihood type
        :param kwargs_likelihood: keyword arguments specifying the likelihood function,
        see individual classes for their use
        """
        self._name = name
        self._z_lens = z_lens
        self._z_source = z_source
        self.likelihood_type = likelihood_type
        if likelihood_type in ['DdtGaussian']:
            from hierarc.Likelihood.LensLikelihood.ddt_gauss_likelihood import DdtGaussianLikelihood
            self._lens_type = DdtGaussianLikelihood(z_lens, z_source, **kwargs_likelihood)
        elif likelihood_type in ['DdtDdKDE']:
            from hierarc.Likelihood.LensLikelihood.ddt_dd_kde_likelihood import DdtDdKDELikelihood
            self._lens_type = DdtDdKDELikelihood(z_lens, z_source, **kwargs_likelihood)
        elif likelihood_type == 'DdtDdGaussian':
            from hierarc.Likelihood.LensLikelihood.ddt_dd_gauss_likelihood import DdtDdGaussian
            self._lens_type = DdtDdGaussian(z_lens, z_source, **kwargs_likelihood)
        elif likelihood_type in ['DsDdsGaussian']:
            from hierarc.Likelihood.LensLikelihood.ds_dds_gauss_likelihood import DsDdsGaussianLikelihood
            self._lens_type = DsDdsGaussianLikelihood(z_lens, z_source, **kwargs_likelihood)
        elif likelihood_type == 'DdtLogNorm':
            from hierarc.Likelihood.LensLikelihood.ddt_lognorm_likelihood import DdtLogNormLikelihood
            self._lens_type = DdtLogNormLikelihood(z_lens, z_source, **kwargs_likelihood)
        elif likelihood_type == 'IFUKinCov':
            from hierarc.Likelihood.LensLikelihood.kin_likelihood import KinLikelihood
            self._lens_type = KinLikelihood(z_lens, z_source, **kwargs_likelihood)
        elif likelihood_type == 'DdtHist':
            from hierarc.Likelihood.LensLikelihood.ddt_hist_likelihood import DdtHistLikelihood
            self._lens_type = DdtHistLikelihood(z_lens, z_source, **kwargs_likelihood)
        elif likelihood_type == 'DdtHistKDE':
            from hierarc.Likelihood.LensLikelihood.ddt_hist_likelihood import DdtHistKDELikelihood
            self._lens_type = DdtHistKDELikelihood(z_lens, z_source, **kwargs_likelihood)
        elif likelihood_type == 'DdtHistKin':
            from hierarc.Likelihood.LensLikelihood.ddt_hist_kin_likelihood import DdtHistKinLikelihood
            self._lens_type = DdtHistKinLikelihood(z_lens, z_source, **kwargs_likelihood)
        elif likelihood_type == 'DdtGaussKin':
            from hierarc.Likelihood.LensLikelihood.ddt_gauss_kin_likelihood import DdtGaussKinLikelihood
            self._lens_type = DdtGaussKinLikelihood(z_lens, z_source, **kwargs_likelihood)
        else:
            raise ValueError('likelihood_type %s not supported! Supported are %s.' % (likelihood_type, LIKELIHOOD_TYPES))

    def num_data(self):
        """
        number of data points across the lens sample

        :return: integer
        """
        return self._lens_type.num_data

    def log_likelihood(self, ddt, dd, aniso_scaling=None, sigma_v_sys_error=None):
        """

        :param ddt: time-delay distance [physical Mpc]
        :param dd: angular diameter distance to the lens [physical Mpc]
        :param aniso_scaling: array of size of the velocity dispersion measurement or None, scaling of the predicted
         dimensionless quantity J (proportional to sigma_v^2) of the anisotropy model in the sampling relative to the
         anisotropy model used to derive the prediction and covariance matrix in the init of this class.
        :param sigma_v_sys_error: unaccounted uncertainty in the velocity dispersion measurement
        :return: natural logarithm of the likelihood of the data given the model
        """
        if self.likelihood_type in ['DdtGaussian', 'DdtLogNorm', 'DdtHist', 'DdtHistKDE']:
            return self._lens_type.log_likelihood(ddt, dd)
        elif self.likelihood_type in ['DdtDdKDE', 'DdtDdGaussian', 'DsDdsGaussian']:
            return self._lens_type.log_likelihood(ddt, dd, aniso_scaling=aniso_scaling)
        elif self.likelihood_type in ['DdtHistKin', 'IFUKinCov', 'DdtGaussKin']:
            return self._lens_type.log_likelihood(ddt, dd, aniso_scaling=aniso_scaling,
                                                  sigma_v_sys_error=sigma_v_sys_error)
        else:
            raise ValueError('likelihood type %s not fully supported.' % self.likelihood_type)

    def ddt_measurement(self):
        """

        :return: ddt measurement median, 1-sigma (without lambda correction factor)
        """
        if self.likelihood_type in ['DdtGaussian', 'DdtHist', 'DdtHistKDE', 'DdtHistKin', 'DdtGaussKin']:
            return self._lens_type.ddt_measurement()
        return None, None

    def sigma_v_measurement(self, sigma_v_sys_error=None):
        """


        :return: data vector, measurement covariance matrix for velocity dispersion
        """
        if self.likelihood_type in ['DdtHistKin', 'IFUKinCov', 'DdtGaussKin']:
            return self._lens_type.sigma_v_measurement(sigma_v_sys_error=sigma_v_sys_error)
        return None, None

    def sigma_v_prediction(self, ddt, dd, aniso_scaling=None):
        """

        :param ddt: ddt in physical Mpc
        :param dd: dd in physical Mpc
        :param aniso_scaling: anisotropy scaling in J
        :return: model predicted velocity dispersion (vector) and model covariance matrix thereof
        """
        if self.likelihood_type in ['DdtHistKin', 'IFUKinCov', 'DdtGaussKin']:
            return self._lens_type.sigma_v_prediction(ddt, dd, aniso_scaling)
        return None, None
