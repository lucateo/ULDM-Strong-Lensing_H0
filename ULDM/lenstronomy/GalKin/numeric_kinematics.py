import numpy as np
from scipy.interpolate import interp1d

import lenstronomy.Util.constants as const
from lenstronomy.GalKin.light_profile import LightProfile
from lenstronomy.GalKin.anisotropy import Anisotropy
from lenstronomy.GalKin.cosmo import Cosmo
from lenstronomy.LensModel.single_plane import SinglePlane
import lenstronomy.GalKin.velocity_util as util


class NumericKinematics(Anisotropy):

    def __init__(self, kwargs_model, kwargs_cosmo, interpol_grid_num=100, log_integration=False, max_integrate=100,
                 min_integrate=0.001):
        """

        :param interpol_grid_num:
        :param log_integration:
        :param max_integrate:
        :param min_integrate:
        """
        mass_profile_list = kwargs_model.get('mass_profile_list')
        light_profile_list = kwargs_model.get('light_profile_list')
        anisotropy_model = kwargs_model.get('anisotropy_model')
        self._interp_grid_num = interpol_grid_num
        self._log_int = log_integration
        self._max_integrate = max_integrate  # maximal integration (and interpolation) in units of arcsecs
        self._min_integrate = min_integrate  # min integration (and interpolation) in units of arcsecs
        self._max_interpolate = max_integrate  # we chose to set the interpolation range to the integration range
        self._min_interpolate = min_integrate  # we chose to set the interpolation range to the integration range
        self.lightProfile = LightProfile(light_profile_list, interpol_grid_num=interpol_grid_num,
                                         max_interpolate=max_integrate, min_interpolate=min_integrate)
        Anisotropy.__init__(self, anisotropy_type=anisotropy_model)
        self.cosmo = Cosmo(**kwargs_cosmo)
        self._mass_profile = SinglePlane(mass_profile_list)

    def sigma_s2(self, r, R, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """
        returns unweighted los velocity dispersion for a specified projected radius

        :param r: 3d radius (not needed for this calculation)
        :param R: 2d projected radius (in angular units of arcsec)
        :param kwargs_mass: mass model parameters (following lenstronomy lens model conventions)
        :param kwargs_light: deflector light parameters (following lenstronomy light model conventions)
        :param kwargs_anisotropy: anisotropy parameters, may vary according to anisotropy type chosen.
            We refer to the Anisotropy() class for details on the parameters.
        :return: line-of-sight projected velocity dispersion at projected radius R
        """
        I_R_sigma2 = self._I_R_sigma2_interp(R, kwargs_mass, kwargs_light, kwargs_anisotropy)
        I_R = self.lightProfile.light_2d(R, kwargs_light)
        return np.nan_to_num(I_R_sigma2 / I_R)

    def sigma_r2(self, r, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """
        computes numerically the solution of the Jeans equation for a specific 3d radius
        E.g. Equation (A1) of Mamon & Lokas https://arxiv.org/pdf/astro-ph/0405491.pdf
        l(r) \sigma_r(r) ^ 2 =  1/f(r) \int_r^{\infty} f(s) l(s) G M(s) / s^2 ds
        where l(r) is the 3d light profile
        M(s) is the enclosed 3d mass
        f is the solution to
        d ln(f)/ d ln(r) = 2 beta(r)

        :param r: 3d radius
        :param kwargs_mass: mass model parameters (following lenstronomy lens model conventions)
        :param kwargs_light: deflector light parameters (following lenstronomy light model conventions)
        :param kwargs_anisotropy: anisotropy parameters, may vary according to anisotropy type chosen.
            We refer to the Anisotropy() class for details on the parameters.
        :return: sigma_r**2
        """
        l_r = self.lightProfile.light_3d_interp(r, kwargs_light)
        f_r = self.anisotropy_solution(r, **kwargs_anisotropy)
        return 1 / f_r / l_r * self._jeans_solution_integral(r, kwargs_mass, kwargs_light, kwargs_anisotropy) * const.G / (const.arcsec * self.cosmo.dd * const.Mpc)

    def mass_3d(self, r, kwargs):
        """
        mass enclosed a 3d radius

        :param r: in arc seconds
        :param kwargs: lens model parameters in arc seconds
        :return: mass enclosed physical radius in kg
        """
        mass_dimless = self._mass_profile.mass_3d(r, kwargs)
        mass_dim = mass_dimless * const.arcsec ** 2 * self.cosmo.dd * self.cosmo.ds / self.cosmo.dds * const.Mpc * \
                   const.c ** 2 / (4 * np.pi * const.G)
        return mass_dim

    def grav_potential(self, r, kwargs_mass):
        """
        Gravitational potential in SI units

        :param r: radius (arc seconds)
        :param kwargs_mass:
        :return: gravitational potential
        """
        mass_dim = self.mass_3d(r, kwargs_mass)
        grav_pot = -const.G * mass_dim / (r * const.arcsec * self.cosmo.dd * const.Mpc)
        return grav_pot

    def draw_light(self, kwargs_light):
        """

        :param kwargs_light: keyword argument (list) of the light model
        :return: 3d radius (if possible), 2d projected radius, x-projected coordinate, y-projected coordinate
        """
        R = self.lightProfile.draw_light_2d(kwargs_light, n=1)[0]
        x, y = util.draw_xy(R)
        r = None
        return r, R, x, y

    def delete_cache(self):
        """
        delete interpolation function for a specific mass and light profile as well as for a specific anisotropy model

        :return:
        """
        if hasattr(self, '_log_mass_3d'):
            del self._log_mass_3d
        if hasattr(self, '_interp_jeans_integral'):
            del self._interp_jeans_integral
        if hasattr(self, '_interp_I_R_sigma2'):
            del self._interp_I_R_sigma2
        self.lightProfile.delete_cache()
        self.delete_anisotropy_cache()

    def _I_R_sigma2(self, R, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """
        equation A15 in Mamon&Lokas 2005 as a logarithmic numerical integral (if option is chosen)

        :param R: 2d projected radius (in angular units)
        :param kwargs_mass: mass model parameters (following lenstronomy lens model conventions)
        :param kwargs_light: deflector light parameters (following lenstronomy light model conventions)
        :param kwargs_anisotropy: anisotropy parameters, may vary according to anisotropy type chosen.
            We refer to the Anisotropy() class for details on the parameters.
        :return: integral of A15 in Mamon&Lokas 2005
        """
        R = max(R, self._min_integrate)
        if self._log_int is True:
            min_log = np.log10(R+0.001)
            max_log = np.log10(self._max_integrate)
            r_array = np.logspace(min_log, max_log, self._interp_grid_num)
            dlog_r = (np.log10(r_array[2]) - np.log10(r_array[1])) * np.log(10)
            IR_sigma2_dr = self._integrand_A15(r_array, R, kwargs_mass, kwargs_light, kwargs_anisotropy) * dlog_r * r_array
        else:
            r_array = np.linspace(R+0.001, self._max_integrate, self._interp_grid_num)
            dr = r_array[2] - r_array[1]
            IR_sigma2_dr = self._integrand_A15(r_array, R, kwargs_mass, kwargs_light, kwargs_anisotropy) * dr
        IR_sigma2 = np.sum(IR_sigma2_dr) # integral from angle to physical scales
        return IR_sigma2 * 2 * const.G / (const.arcsec * self.cosmo.dd * const.Mpc)

    def _I_R_sigma2_interp(self, R, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """
        quation A15 in Mamon&Lokas 2005 as interpolation in log space

        :param R: projected radius
        :param kwargs_mass: mass profile keyword arguments
        :param kwargs_light: light model keyword arguments
        :param kwargs_anisotropy: stellar anisotropy keyword arguments
        :return:
        """
        if not hasattr(self, '_interp_I_R_sigma2'):
            min_log = np.log10(self._min_integrate)
            max_log = np.log10(self._max_integrate)
            R_array = np.logspace(min_log, max_log, self._interp_grid_num)
            I_R_sigma2_array = []
            for R_i in R_array:
                I_R_sigma2_array.append(self._I_R_sigma2(R_i, kwargs_mass, kwargs_light, kwargs_anisotropy))
            self._interp_I_R_sigma2 = interp1d(np.log(R_array), np.array(I_R_sigma2_array), fill_value="extrapolate")
        return self._interp_I_R_sigma2(np.log(R))

    def _integrand_A15(self, r, R, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """
        integrand of A15 (in log space) in Mamon&Lokas 2005

        :param r: 3d radius in arc seconds
        :param R: 2d projected radius
        :param kwargs_mass: mass model parameters (following lenstronomy lens model conventions)
        :param kwargs_light: deflector light parameters (following lenstronomy light model conventions)
        :param kwargs_anisotropy: anisotropy parameters, may vary according to anisotropy type chosen.
            We refer to the Anisotropy() class for details on the parameters.
        :return:
        """
        k_r = self.K(r, R, **kwargs_anisotropy)
        l_r = self.lightProfile.light_3d_interp(r, kwargs_light)
        m_r = self._mass_3d_interp(r, kwargs_mass)
        out = k_r * l_r * m_r / r
        return out

    def _jeans_solution_integral(self, r, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """
        interpolated solution of the integral \int_r^{\infty} f(s) l(s) G M(s) / s^2 ds

        :param r:
        :param kwargs_mass:
        :param kwargs_light:
        :param kwargs_anisotropy:
        :return:
        """
        if not hasattr(self, '_interp_jeans_integral'):
            min_log = np.log10(self._min_integrate)
            max_log = np.log10(self._max_integrate)
            r_array = np.logspace(min_log, max_log, self._interp_grid_num)
            dlog_r = (np.log10(r_array[2]) - np.log10(r_array[1])) * np.log(10)
            integrand_jeans = self._integrand_jeans_solution(r_array, kwargs_mass, kwargs_light, kwargs_anisotropy) * dlog_r * r_array
            #flip array from inf to finite
            integral_jeans_r = np.cumsum(np.flip(integrand_jeans))
            #flip array back
            integral_jeans_r = np.flip(integral_jeans_r)
            #call 1d interpolation function
            self._interp_jeans_integral = interp1d(np.log(r_array), integral_jeans_r, fill_value="extrapolate")
        return self._interp_jeans_integral(np.log(r))

    def _integrand_jeans_solution(self, r, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """
        integrand of A1 (in log space) in Mamon&Lokas 2005 to calculate the Jeans equation numerically
        f(s) l(s) M(s) / s^2

        :param r:
        :param kwargs_mass:
        :param kwargs_light:
        :param kwargs_anisotropy:
        :return:
        """
        f_r = self.anisotropy_solution(r, **kwargs_anisotropy)
        l_r = self.lightProfile.light_3d_interp(r, kwargs_light)
        m_r = self._mass_3d_interp(r, kwargs_mass)
        out = f_r * l_r * m_r / r**2
        return out

    def _mass_3d_interp(self, r, kwargs, new_compute=False):
        """

        :param r: in arc seconds
        :param kwargs: lens model parameters in arc seconds
        :param new_compute: bool, if True, recomputes the interpolation
        :return: mass enclosed physical radius in kg
        """
        if not hasattr(self, '_log_mass_3d') or new_compute is True:
            r_array = np.logspace(np.log10(self._min_interpolate), np.log10(self._max_interpolate), self._interp_grid_num)
            mass_3d_array = self.mass_3d(r_array, kwargs)
            mass_3d_array[mass_3d_array < 10. ** (-10)] = 10. ** (-10)
            #mass_dim_array = mass_3d_array * const.arcsec ** 2 * self.cosmo.dd * self.cosmo.ds \
            #                 / self.cosmo.dds * const.Mpc * const.c ** 2 / (4 * np.pi * const.G)
            self._log_mass_3d = interp1d(np.log(r_array), np.log(mass_3d_array/r_array), fill_value="extrapolate")
        return np.exp(self._log_mass_3d(np.log(r))) * r
