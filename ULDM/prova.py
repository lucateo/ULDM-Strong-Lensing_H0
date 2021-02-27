# import of standard python libraries
import numpy as np
import os
import time
import corner
import astropy.io.fits as pyfits
import copy

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pylab import rc

np.random.seed(42)
# 0.5, 1, 1.5 and 2 sigma of 2D Gaussian
ones = np.ones(1000)
#  ones = np.append(ones, [2,2,3,3,4])
percentile = np.percentile(np.cumsum(ones), [16,50,84])
print(percentile)

levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)
print(levels)

#  x = np.random.randn(1000, 1)
x = np.arange(1, 100,1)
y = np.arange(1,100,1)
samples = np.random.randn(100,2)
samples = []
for a in range(1000):
    samples.append([(a+1)%4/(a+1), (a-1)%3/(a+1)])

H, X, Y = np.histogram2d(x.flatten(),y.flatten(), bins=20)
plt = corner.corner(samples)
plt.savefig('prova.pdf')
# X is bin edges in x direction, Y in y direction
#  Hflat = H.flatten()
#  print(H)
#  inds = np.argsort(Hflat)[::-1]
#  print(inds)
#  Hflat = Hflat[inds]
#  print('New H_flat', Hflat)
#  sm = np.cumsum(Hflat)
#  sm /= sm[-1]
#  # You start from the most probable one and then do cumulative sum from biggest to smallest, until you reach one
#  print('sm', sm)
#  V = np.empty(len(levels))
#  for i, v0 in enumerate(levels):
#      try:
#          V[i] = Hflat[sm <= v0][-1]
#      except IndexError:
#          V[i] = Hflat[0]
#  V.sort()
#  print(V)
#  # So you put the square bins in decreasing order of how many occurrence are in that bin, the
#  # cumulative sum is just the normalized integral, starting from most probable square bin
#  m = np.diff(V) == 0
#  print(m)
#  if np.any(m) and not quiet:
#      logging.warning("Too few points to create valid contours")
#  while np.any(m):
#      V[np.where(m)[0][0]] *= 1.0 - 1e-4
#      m = np.diff(V) == 0
#  V.sort()

#  plt.hist(x, bins=20)
#  plt.show()



#  rc('axes', linewidth=2)
#  rc('xtick',labelsize=15)
#  rc('ytick',labelsize=15)
#
#  np.random.seed(42)
#
#  # import lenstronomy modules
#  from lenstronomy.LensModel.lens_model import LensModel
#  from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
#  from lenstronomy.LightModel.light_model import LightModel
#  from lenstronomy.PointSource.point_source import PointSource
#  from lenstronomy.ImSim.image_model import ImageModel
#  from  lenstronomy.Util import param_util
#  import lenstronomy.Util.simulation_util as sim_util
#  from lenstronomy.Util import image_util
#  from lenstronomy.Util import kernel_util
#  from lenstronomy.Data.imaging_data import ImageData
#  from lenstronomy.Data.psf import PSF
#  import lenstronomy.Util.constants as const
#  from scipy.special import gamma as gamma_func
#
#  # define lens configuration and cosmology (not for lens modelling)
#  z_lens = 0.5
#  z_source = 1.5
#  from astropy.cosmology import FlatLambdaCDM
#  cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.)
#  from lenstronomy.Cosmo.lens_cosmo import LensCosmo
#
#  cosmo_reference = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.)
#  lens_cosmo_reference = LensCosmo(z_lens = z_lens, z_source = z_source, cosmo = cosmo_reference)
#  Ddt_reference = lens_cosmo_reference.ddt
#  def logL_addition(kwargs_lens, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None, kwargs_special=None, kwargs_extinction=None):
#      """
#      a definition taking as arguments (kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_special, kwargs_extinction)
#              and returns a logL (punishing) value.
#      """
#      pc2meter = 3.086 * 10**(16)
#      clight = 3*10**8
#      G_const = 6.67 * 10**(-11)
#      m_sun = 1.989 * 10**(30)
#
#      Ddt_sampled =  kwargs_special['D_dt']
#      h0_sampled = 70 * Ddt_reference / Ddt_sampled
#      cosmo_current = FlatLambdaCDM(H0=h0_sampled, Om0=0.3, Ob0=0.)
#      lens_cosmo = LensCosmo(z_lens, z_source, cosmo_current)
#      D_Lens = lens_cosmo.dd * 10**6 * pc2meter # in meter
#      Sigma_c = lens_cosmo.sigma_crit * 10**(-12) * m_sun / pc2meter**2 # in kg/m^2
#
#      h0_mean = 67.4
#      h0_sigma = 0.5
#      logL = - (h0_sampled - h0_mean)**2 / h0_sigma**2 / 2 + np.log(h0_sampled/Ddt_sampled)
#
#      kappa_0 = kwargs_lens[2]['kappa_0']
#      theta_c = kwargs_lens[2]['theta_c']
#      slope = kwargs_lens[2]['slope']
#      theta_E = kwargs_lens[0]['theta_E']
#
#      A_Factor = 2 * G_const / clight**2 * Sigma_c * D_Lens * theta_E * const.arcsec
#      lambda_factor = ((2.23 / (slope/2 - 1.69))**(1/2.47) - 1)/2.2
#      lambda_factor = A_Factor/ lambda_factor
#      lambda_factor = np.sqrt(lambda_factor)
#      z_fit = A_Factor / lambda_factor**2
#      a_fit = 0.23 * np.sqrt(1 + 7.5 * z_fit * np.tanh( 1.5 * z_fit**(0.24)) )
#      constraint = lambda_factor**2 * clight**2 * np.sqrt(np.pi) * gamma_func(slope - 0.5)
#      constraint = constraint / (4 * np.pi * G_const * Sigma_c * D_Lens * a_fit**2 * gamma_func(slope))
#      core_half_factor = np.sqrt(0.5**(-1/slope) -1)
#      sample_constraint = kappa_0 * theta_c * const.arcsec / core_half_factor
#
#      logL = logL - (sample_constraint - constraint)**2 / 10**(-15) / 2
#      return logL, sample_constraint, constraint
#
#
#
#
#  kappa_0 = 0.1069
#  theta_E = 1.44 # 1.66 * (1 - kappa_0)
#  slope = 3.85
#  theta_c = 4.9
#  kwargs_pemd = {'theta_E': theta_E, 'gamma': 1.98, 'center_x': 0.0, 'center_y': 0.0, 'e1': -0.2, 'e2': 0.05}  # parameters of the deflector lens model
#  kwargs_shear = {'gamma1': 0.05, 'gamma2': -0.02}  # shear values to the source plane
#  kwargs_uldm = {'kappa_0': kappa_0, 'theta_c': theta_c, 'slope': slope, 'center_x': 0.0, 'center_y': 0.0, }  # parameters of the deflector lens model
#  kwargs_lens = [kwargs_pemd, kwargs_shear, kwargs_uldm]
#  kwargs_special = {'D_dt': 3458}
#
#
#  print(logL_addition(kwargs_lens = kwargs_lens, kwargs_special = kwargs_special))


