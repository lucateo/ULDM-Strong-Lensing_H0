import numpy as np
import copy
import lenstronomy.Util.constants as const
import matplotlib.pyplot as plt
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from astropy.cosmology import FlatLambdaCDM
# import the LensModel class #
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Profiles.uldm import Uldm

# specify the choice of lens models #
lens_model_list = ['ULDM']

cosmo = FlatLambdaCDM(H0 = 67, Om0=0.31, Ob0=0.05)

z_lens = 0.295
# specify source redshift
z_source = 0.658
# setup lens model class with the list of lens models
lensModel = LensModel(lens_model_list=lens_model_list, z_lens = z_lens, z_source=z_source, cosmo=cosmo)
lens_cosmo = LensCosmo(z_lens = z_lens, z_source = z_source, cosmo = cosmo)

Ddt = lens_cosmo.ddt
print(Ddt, " time delay surface")

# define parameter values of lens models
kwargs_uldm = {'m_noCosmo_log10': -8, 'M_noCosmo_log10' : -9, 'center_x': 0.1, 'center_y': 0}
kwargs_lens = [kwargs_uldm]

alpha = lensModel.alpha(2.1, 1.2, kwargs_lens)
print('try deviation angle  ', alpha)

flexion_lens = lensModel.flexion(1,2,kwargs_lens)
print('flexion  ', flexion_lens)

uldm_lens = Uldm()
integ = uldm_lens.lensing_Integral(0)
print('try lensing integral ', integ)

potential = uldm_lens.function(1.1, 1.1, -8.5, -10.5)
print('try potential ', potential)

abba = uldm_lens.derivatives(0.4, 0.7, -8.5, -10.5)
print('Try direct derivatives ', abba)

#  fermat_lens = lensModel.fermat_potential(1,2,kwargs_lens)
#  print(fermat_lens)

dt = lensModel.arrival_time(.9,.4,kwargs_lens)
print(dt)

D_Lens = lens_cosmo.dd * 10**6
Sigma_c = lens_cosmo.sigma_crit * 10**(-12)

print('D_lens ' , D_Lens, ' Sigma_crit ', Sigma_c)

m_noCosmo = -8.4
M_noCosmo = -11.66

mphys, Mphys = lens_cosmo.m_noCosmo2m_phys(m_noCosmo, M_noCosmo)
print("masses mock", mphys, Mphys)
theta_c = uldm_lens.theta_cRad(m_noCosmo, M_noCosmo)/const.arcsec
print("theta c = ", theta_c)
convergence = uldm_lens.density_2d(0.01, 0.01, m_noCosmo, M_noCosmo)
print("convergence = ", convergence)

Mass = uldm_lens.mass_3d(100,m_noCosmo, M_noCosmo)*(Sigma_c * D_Lens**2)
print("Mass trial check ", np.log10(Mass))


####### Plotting stuff
radius = np.arange(0.01, 10, 0.01)
D_Lens = lens_cosmo.dd * 1000 # in kpc
plt.xscale('log')
plt.yscale('log')
# rho is in M_sun/parsec^3
plt.plot(radius, (Sigma_c / D_Lens) * uldm_lens.density(radius/D_Lens /const.arcsec, m_noCosmo, M_noCosmo))
plt.show()





