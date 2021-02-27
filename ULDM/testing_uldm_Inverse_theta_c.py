import numpy as np
import copy
import lenstronomy.Util.constants as const
import matplotlib.pyplot as plt
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from astropy.cosmology import FlatLambdaCDM
# import the LensModel class #
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Profiles.uldm import Uldm
from lenstronomy.LensModel.Profiles.pemd import PEMD
from lenstronomy.LensModel.Profiles.uldm_pl import Uldm_PL
from scipy.special import hyp2f1
from mpmath import hyp3f2
from scipy.special import gamma as gamma_func

import scipy.integrate as integrate

# specify the choice of lens models #
lens_model_list = ['ULDM_PL']
cosmo = FlatLambdaCDM(H0 = 67.4, Om0=0.30, Ob0=0.00)
cosmo2 = FlatLambdaCDM(H0 = 74.3, Om0=0.30, Ob0=0.00)

b = 1.69
z_lens = 0.5
# specify source redshift
z_source = 1.5
# setup lens model class with the list of lens models
lensModel = LensModel(lens_model_list=lens_model_list, z_lens = z_lens, z_source=z_source, cosmo=cosmo)
lens_cosmo = LensCosmo(z_lens = z_lens, z_source = z_source, cosmo = cosmo)
lens_cosmo2 = LensCosmo(z_lens = z_lens, z_source = z_source, cosmo = cosmo2)

#################################### MASS FUNCTIONS ##########################
def phys2ModelParam(m_log10, lambda_factor, theta_E):
    eV2Joule = 1.6021*10**(-19)
    hbar = 6.62 * 10**(-34) / (2* np.pi)
    pc2meter = 3.086 * 10**(16)
    clight = 3*10**8
    G_const = 6.67 * 10**(-11)
    m_sun = 1.989 * 10**(30)
    m = 10**m_log10 * eV2Joule # in Joule
    lens_cosmo = LensCosmo(z_lens, z_source, cosmo)
    D_Lens = lens_cosmo.dd * 10**6 * pc2meter # in meter
    Sigma_c = lens_cosmo.sigma_crit * 10**(-12) * m_sun / pc2meter**2 # in kg/m^2

    A_Factor = 2 * G_const / clight**2 * Sigma_c * D_Lens * theta_E * const.arcsec
    z_fit = A_Factor / lambda_factor**2
    a_fit = 0.23 * np.sqrt(1 + 7.5 * z_fit * np.tanh( 1.5 * z_fit**(0.24)) )
    b_fit = 1.69 + 2.23/(1 + 2.2 * z_fit)**(2.47)
    slope = 2*b_fit
    core_half_factor = np.sqrt(0.5**(-1/slope) -1)
    theta_c = core_half_factor * clight * hbar / (lambda_factor * a_fit * m * D_Lens * const.arcsec )
    kappa_0 = lambda_factor**3 * m * np.sqrt(np.pi) * gamma_func(slope - 0.5)
    kappa_0 = kappa_0 * clight  /(4 * np.pi * Sigma_c * G_const * hbar * a_fit * gamma_func(slope) )
    M_sol = lambda_factor * clight**3 * hbar * np.sqrt(np.pi) * gamma_func(slope - 1.5)
    M_sol = M_sol / (G_const * m * a_fit**3 * 4 * gamma_func(slope) ) # in kg
    M_log10 = np.log10( M_sol/m_sun)
    return kappa_0, theta_c/core_half_factor, z_fit, slope, M_log10

def angles2phys(theta_c, slope, theta_E):
    eV2Joule = 1.6021*10**(-19)
    hbar = 6.62 * 10**(-34) / (2* np.pi)
    pc2meter = 3.086 * 10**(16)
    clight = 3*10**8
    G_const = 6.67 * 10**(-11)
    m_sun = 1.989 * 10**(30)
    lens_cosmo = LensCosmo(z_lens, z_source, cosmo)
    D_Lens = lens_cosmo.dd * 10**6 * pc2meter # in meter
    Sigma_c = lens_cosmo.sigma_crit * 10**(-12) * m_sun / pc2meter**2 # in kg/m^2

    A_Factor = 2 * G_const / clight**2 * Sigma_c * D_Lens * theta_E * const.arcsec
    lambda_factor = ((2.23 / (slope/2 - 1.69))**(1/2.47) - 1)/2.2
    lambda_factor = A_Factor/ lambda_factor
    lambda_factor = np.sqrt(lambda_factor)

    z_fit = A_Factor / lambda_factor**2
    a_fit = 0.23 * np.sqrt(1 + 7.5 * z_fit * np.tanh( 1.5 * z_fit**(0.24)) )
    core_half_factor = np.sqrt(0.5**(-1/slope) -1)
    m_particle = clight * hbar * theta_c / (lambda_factor * a_fit * D_Lens * const.arcsec)
    m_log10 = np.log10( m_particle/ eV2Joule)

    M_sol = lambda_factor * clight**3 * hbar * np.sqrt(np.pi) * gamma_func(slope - 1.5)
    M_sol = M_sol / (G_const * m_particle * a_fit**3 * 4 * gamma_func(slope) ) # in kg
    M_log10 = np.log10( M_sol/m_sun)

    kappa_0 = lambda_factor**3 * np.sqrt(np.pi) * m_particle * gamma_func(slope - 0.5) * clight / (4 * np.pi * G_const * hbar * Sigma_c * a_fit *  gamma_func(slope))
    return kappa_0, z_fit, m_log10, M_log10, lambda_factor
##############################################################################

# define parameter values of lens models
kappa_tilde = 0.067
sampled_theta_c = 0.15
theta_E = 1.485
gamma = 1.98
e1 = 0.1
e2 = 0.1
center_x = 0
center_y = 0
kwargs_lens = {'theta_E': theta_E, 'gamma': gamma, 'e1': e1, 'e2': e2, 'kappa_tilde': kappa_tilde, 'sampled_theta_c' : sampled_theta_c, 'center_x': center_x, 'center_y': center_y}
kwargs_lens = [kwargs_lens]

composite_lens = Uldm_PL()
uldm_lens = Uldm()
PL_lens = PEMD()

slope = composite_lens._slope(theta_E, kappa_tilde, sampled_theta_c)
kappa_0 = composite_lens._kappa_0_real(theta_E, kappa_tilde, sampled_theta_c)
theta_c = composite_lens._half_density_thetac(theta_E, kappa_tilde, sampled_theta_c)
theta_cTilde = composite_lens._theta_c_true(sampled_theta_c)
z_factor = composite_lens._z_factor(theta_E, kappa_tilde, sampled_theta_c)
a_fit = composite_lens._a_fit(theta_E, kappa_tilde, sampled_theta_c)
print('slope: ', slope, ' kappa_0 ', kappa_0,  'theta_c ', 1/theta_c, 'z factor ', z_factor, 'afit ', a_fit)
print(np.sqrt(0.5**(-1/slope) -1))

_, _, m_log10, _, lambda_factor = angles2phys(sampled_theta_c, slope, theta_E )
print(angles2phys(sampled_theta_c, slope, theta_E ))
print(phys2ModelParam(m_log10, lambda_factor, theta_E))

def lensing_analytic(x,y, kappa_0, theta_c, slope):
    R_squared = x**2 + y**2
    return kappa_0/2 * R_squared * hyp3f2(1, 1, slope - 0.5, 2, 2, -(R_squared*theta_c**2))
print('hypgeom trial ', lensing_analytic(2,1, kappa_0, theta_c, slope), uldm_lens.function(2, 1, kappa_0, theta_c, slope))

#  a_factor = (0.5)**(-1./slope) -1
#  print('a factor corresponding to selected slope ', a_factor)
#
alpha = lensModel.alpha(2.1, 1.2, kwargs_lens)
print('try deviation angle  ', alpha)

#  potential = uldm_lens.function(1.1, 1.1, kappa_0, theta_c, slope)
#  print('try potential ', potential)
#
MSD_potential = composite_lens.function(0.4, 0, theta_E, gamma, e1, e2, kappa_tilde, sampled_theta_c) - composite_lens.function(0.4, 0, theta_E, gamma, e1, e2, 0.0000001,10)
MSD_potential2 = composite_lens.function(0.5, 0, theta_E, gamma, e1, e2, kappa_tilde, sampled_theta_c) - composite_lens.function(0.5, 0, theta_E, gamma, e1, e2, 0.0000001,10)
print('Try MSD, uldm for large theta_c, potential ', MSD_potential - MSD_potential2, ' Pure MSD expected result ', kappa_0* 0.4**2 /2 - kappa_0* 0.5**2 /2)

# This works only if you put theta_E = 0 also at the beginning
print(uldm_lens.derivatives(1,1,kappa_0,theta_c,slope), composite_lens.derivatives(1,1,0, 1, 0, 0, kappa_tilde, sampled_theta_c))

fermat_lens = lensModel.fermat_potential(1,2,kwargs_lens)
print('fermat potential: ',fermat_lens)

dt = lensModel.arrival_time(.9, .4, kwargs_lens)
print('arrival time ',dt)

#  def log_prob(x):
#      if 0 < composite_lens._kappa_0_real(theta_E, x[0], x[1]) < 0.5 and 0<x[1]<10:
#          epsilon = 10**(-6)
#          deriv1 = (composite_lens._kappa_0_real(theta_E, x[0] + epsilon, x[1]) - composite_lens._kappa_0_real(theta_E, x[0] - epsilon, x[1]) )/(2*epsilon)
#          #  deriv2 = (composite_lens._kappa_0_real(theta_E, x[0], x[1] + epsilon) - composite_lens._kappa_0_real(theta_E, x[0], x[1] - epsilon) )/(2*epsilon)
#          deriv = np.abs(deriv1 )
#          return np.log(deriv) #- 2 * np.log(x[1])
#      else:
#          return -np.inf
#
#  nwalkers = 20
#  ndim = 2
#  p0 = 0.05 * np.random.rand(nwalkers, ndim)
#  import emcee
#  sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
#  state = sampler.run_mcmc(p0, 500)
#  sampler.reset()
#  sampler.run_mcmc(state, 2000)
#
#  samples = sampler.get_chain(flat=True)
#  plt.hist(composite_lens._kappa_0_real(theta_E, samples[:,0], samples[:,1]), 100, range=(0,0.5), color="k", histtype="step")
#
#  # Trying uniform distribution
#  #  samples = np.random.uniform(0.001, 0.5, 2000)
#  #  epsilon = 10**(-4) * np.ones(2000)
#  #  deriv = (composite_lens._kappa_0_real(theta_E, samples + epsilon, sampled_theta_c) - composite_lens._kappa_0_real(theta_E, samples - epsilon, sampled_theta_c) )*0.5*10**4
#  #  samples_kappa0 = composite_lens._kappa_0_real(theta_E, samples/deriv, sampled_theta_c)
#  #  plt.hist(samples_kappa0, range=(0,0.5))
#  plt.show()


