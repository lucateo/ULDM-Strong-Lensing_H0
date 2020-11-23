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

thetac, alphac, rho_0 = lens_cosmo.uldm_phys2angles_rho0(-24,11.3)
print('rho0 ', rho_0,  'alphac ', alphac, 'thetac ', thetac)

# define parameter values of lens models
kwargs_uldm = {'m_noCosmo_log10': -8, 'M_noCosmo_log10' : -9, 'center_x': 0.1, 'center_y': 0}
kwargs_lens = [kwargs_uldm]

alpha = lensModel.alpha(0.1, 0.2, kwargs_lens)
print('try deviation angle  ', alpha)

flexion_lens = lensModel.flexion(1,2,kwargs_lens)
print('flexion  ', flexion_lens)

uldm_lens = Uldm()
integ = uldm_lens.lensing_Integral(0)
print('try lensing integral ', integ)

potential = uldm_lens.function(0.4, 0.7, -8.4, -9.5)
print('try potential ', potential)

abba = uldm_lens.derivatives(0.4, 0.7, -8.1, -8.4)
print('Try direct derivatives ', abba)

#  fermat_lens = lensModel.fermat_potential(1,2,kwargs_lens)
#  print(fermat_lens)

dt = lensModel.arrival_time(.9,.4,kwargs_lens)
print(dt)

D_Lens = lens_cosmo.dd * 10**6
Sigma_c = lens_cosmo.sigma_crit * 10**(-12)

print('D_lens ' , D_Lens, ' Sigma_crit ', Sigma_c)

def m_noCosmo2m_phys(m_noCosmo_log10, M_noCosmo_log10):
    D_Lens = lens_cosmo.dd * 10**6
    Sigma_c = lens_cosmo.sigma_crit * 10**(-12)
    m = -0.5*np.log10(Sigma_c * D_Lens**3) + m_noCosmo_log10
    M = np.log10(Sigma_c * D_Lens**2) + M_noCosmo_log10
    return m, M

def mPhys2noCosmo(m_log10, M_log10):
    D_Lens = lens_cosmo.dd * 10**6
    Sigma_c = lens_cosmo.sigma_crit * 10**(-12)
    m_noCosmo_log10 = 0.5*np.log10(Sigma_c * D_Lens**3) + m_log10
    M_noCosmo_log10 = -np.log10(Sigma_c * D_Lens**2) + M_log10
    return m_noCosmo_log10, M_noCosmo_log10

print(m_noCosmo2m_phys(-6.5,-11.9))


m_mock, M_mock = lens_cosmo.uldm_angles2phys(5.97,0.12)
print('m of mock ' , m_mock, 'M of mock ', M_mock)


####### Plotting stuff
radius = np.arange(0.01, 10, 0.1)
D_Lens = lens_cosmo.dd * 1000 # in kpc
plt.xscale('log')
plt.yscale('log')
theta_c, alpha_c, rho0 = lens_cosmo.uldm_phys2angles_rho0(-23,11)
# rho is in M_sun/parsec^3
plt.plot(radius, uldm_lens.density(radius/D_Lens /const.arcsec, rho_0, theta_c))
plt.show()





