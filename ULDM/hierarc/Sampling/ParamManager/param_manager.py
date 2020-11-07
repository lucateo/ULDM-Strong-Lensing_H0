from hierarc.Sampling.ParamManager.kin_param import KinParam
from hierarc.Sampling.ParamManager.cosmo_param import CosmoParam
from hierarc.Sampling.ParamManager.lens_param import LensParam


class ParamManager(object):
    """
    class for managing the parameters involved
    """
    def __init__(self, cosmology, ppn_sampling=False, lambda_mst_sampling=False, lambda_mst_distribution='NONE',
                 anisotropy_sampling=False, anisotropy_model='OM', anisotropy_distribution='NONE',
                 kappa_ext_sampling=False, kappa_ext_distribution='NONE', lambda_ifu_sampling=False,
                 lambda_ifu_distribution='NONE', alpha_lambda_sampling=False, sigma_v_systematics=False,
                 log_scatter=False,
                 kwargs_lower_cosmo=None, kwargs_upper_cosmo=None,
                 kwargs_fixed_cosmo={}, kwargs_lower_lens=None, kwargs_upper_lens=None, kwargs_fixed_lens={},
                 kwargs_lower_kin=None, kwargs_upper_kin=None, kwargs_fixed_kin={}):
        """

        :param cosmology: string describing cosmological model
        :param ppn_sampling: post-newtonian parameter sampling
        :param lambda_mst_sampling: bool, if True adds a global mass-sheet transform parameter in the sampling
        :param lambda_mst_distribution: string, distribution function of the MST transform
        :param lambda_ifu_sampling: bool, if True samples a separate lambda_mst for a second (e.g. IFU) data set
        independently
        :param alpha_lambda_sampling: bool, if True samples a parameter alpha_lambda, which scales lambda_mst linearly
         according to a predefined quantity of the lens
        :param lambda_ifu_distribution: string, distribution function of the lambda_ifu parameter
        :param anisotropy_sampling: bool, if True adds a global stellar anisotropy parameter that alters the single lens
        kinematic prediction
        :param anisotropy_distribution: string, indicating the distribution function of the anisotropy model
        :param sigma_v_systematics: bool, if True samples paramaters relative to systematics in the velocity dispersion
         measurement
        :param log_scatter: boolean, if True, samples the Gaussian scatter amplitude in log space (and thus flat prior in log)
        """
        self._kin_param = KinParam(anisotropy_sampling=anisotropy_sampling, anisotropy_model=anisotropy_model,
                                   distribution_function=anisotropy_distribution, log_scatter=log_scatter,
                                   sigma_v_systematics=sigma_v_systematics, kwargs_fixed=kwargs_fixed_kin)
        self._cosmo_param = CosmoParam(cosmology=cosmology, ppn_sampling=ppn_sampling, kwargs_fixed=kwargs_fixed_cosmo)
        self._lens_param = LensParam(lambda_mst_sampling=lambda_mst_sampling,
                                     lambda_mst_distribution=lambda_mst_distribution,
                                     lambda_ifu_sampling=lambda_ifu_sampling,
                                     lambda_ifu_distribution=lambda_ifu_distribution,
                                     kappa_ext_sampling=kappa_ext_sampling,
                                     kappa_ext_distribution=kappa_ext_distribution,
                                     alpha_lambda_sampling=alpha_lambda_sampling,
                                     log_scatter=log_scatter,
                                     kwargs_fixed=kwargs_fixed_lens)
        self._kwargs_upper_cosmo, self._kwargs_lower_cosmo = kwargs_upper_cosmo, kwargs_lower_cosmo
        self._kwargs_upper_lens, self._kwargs_lower_lens = kwargs_upper_lens, kwargs_lower_lens
        self._kwargs_upper_kin, self._kwargs_lower_kin = kwargs_upper_kin, kwargs_lower_kin

    @property
    def num_param(self):
        """
        number of parameters being sampled

        :return: integer
        """
        return len(self.param_list())

    def param_list(self, latex_style=False):
        """

        :param latex_style: bool, if True returns strings in latex symbols, else in the convention of the sampler
        :return: list of the free parameters being sampled in the same order as the sampling
        """
        list = []
        list += self._cosmo_param.param_list(latex_style=latex_style)
        list += self._lens_param.param_list(latex_style=latex_style)
        list += self._kin_param.param_list(latex_style=latex_style)
        return list

    def args2kwargs(self, args):
        """

        :param args: sampling argument list
        :return: keyword argument list with parameter names
        """
        i = 0
        kwargs_cosmo, i = self._cosmo_param.args2kwargs(args, i=i)
        kwargs_lens, i = self._lens_param.args2kwargs(args, i=i)
        kwargs_kin, i = self._kin_param.args2kwargs(args, i=i)
        return kwargs_cosmo, kwargs_lens, kwargs_kin

    def kwargs2args(self, kwargs_cosmo=None, kwargs_lens=None, kwargs_kin=None):
        """

        :param kwargs_cosmo: keyword argument list of parameters for cosmology sampling
        :param kwargs_lens: keyword argument list of parameters for lens model sampling
        :param kwargs_kin: keyword argument list of parameters for kinematic sampling
        :return: sampling argument list in specified order
        """
        args = []
        args += self._cosmo_param.kwargs2args(kwargs_cosmo)
        args += self._lens_param.kwargs2args(kwargs_lens)
        args += self._kin_param.kwargs2args(kwargs_kin)
        return args

    def cosmo(self, kwargs_cosmo):
        """

        :param kwargs_cosmo: keyword arguments of parameters (can include others not used for the cosmology)
        :return: astropy.cosmology instance
        """
        return self._cosmo_param.cosmo(kwargs_cosmo)

    @property
    def param_bounds(self):
        """

        :return: argument list of the hard bounds in the order of the sampling
        """
        lower_limit = self.kwargs2args(kwargs_cosmo=self._kwargs_lower_cosmo, kwargs_lens=self._kwargs_lower_lens,
                                       kwargs_kin=self._kwargs_lower_kin)
        upper_limit = self.kwargs2args(kwargs_cosmo=self._kwargs_upper_cosmo, kwargs_lens=self._kwargs_upper_lens,
                                       kwargs_kin=self._kwargs_upper_kin)
        return lower_limit, upper_limit
