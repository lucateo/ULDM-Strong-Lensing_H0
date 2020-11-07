__author__ = 'sibirrer'

import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase


class Convergence(LensProfileBase):
    """
    a single mass sheet (external convergence)
    """
    param_names = ['kappa_ext', 'ra_0', 'dec_0']
    lower_limit_default = {'kappa_ext': -10, 'ra_0': -100, 'dec_0': -100}
    upper_limit_default = {'kappa_ext': 10, 'ra_0': 100, 'dec_0': 100}

    def function(self, x, y, kappa_ext, ra_0=0, dec_0=0):
        """
        lensing potential

        :param x: x-coordinate
        :param y: y-coordinate
        :param kappa_ext: external convergence
        :return: lensing potential
        """
        theta, phi = param_util.cart2polar(x - ra_0, y - dec_0)
        f_ = 1./2 * kappa_ext * theta**2
        return f_

    def derivatives(self, x, y, kappa_ext, ra_0=0, dec_0=0):
        """
        deflection angle

        :param x: x-coordinate
        :param y: y-coordinate
        :param kappa_ext: external convergence
        :return: deflection angles (first order derivatives)
        """
        x_ = x - ra_0
        y_ = y - dec_0
        f_x = kappa_ext * x_
        f_y = kappa_ext * y_
        return f_x, f_y

    def hessian(self, x, y, kappa_ext, ra_0=0, dec_0=0):
        """
        Hessian matrix

        :param x: x-coordinate
        :param y: y-coordinate
        :param kappa_ext: external convergence
        :return: second order derivatives f_xx, f_yy, f_xy
        """
        gamma1 = 0
        gamma2 = 0
        kappa = kappa_ext
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_yy, f_xy
