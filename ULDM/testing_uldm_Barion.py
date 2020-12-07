import numpy as np
import copy
import lenstronomy.Util.constants as const
import matplotlib.pyplot as plt
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from astropy.cosmology import FlatLambdaCDM
# import the LensModel class #
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Profiles.uldm_bar import Uldm_Bar

# specify the choice of lens models #
lens_model_list = ['ULDM-BAR']

cosmo = FlatLambdaCDM(H0 = 67, Om0=0.31, Ob0=0.05)
cosmo2 = FlatLambdaCDM(H0 = 77, Om0=0.31, Ob0=0.05)

z_lens = 0.5
# specify source redshift
z_source = 1.5
# setup lens model class with the list of lens models
lensModel = LensModel(lens_model_list=lens_model_list, z_lens = z_lens, z_source=z_source, cosmo=cosmo)
lens_cosmo = LensCosmo(z_lens = z_lens, z_source = z_source, cosmo = cosmo)
lens_cosmo2 = LensCosmo(z_lens = z_lens, z_source = z_source, cosmo = cosmo2)

# define parameter values of lens models
kappa_0 = 0.09
theta_c = 3.10
theta_E = 1.423
gamma = 1.96
e1 = 0.1
e2 = 0.1
center_xULDM = 0.1
center_yULDM = 0
center_x = 0
center_y = 0
kwargs_uldm = {'kappa_0': kappa_0, 'theta_c' : theta_c, 'theta_E': theta_E, 'gamma': gamma, 'e1': e1, 'e2': e2, 'center_xULDM': center_xULDM, 'center_yULDM': center_yULDM, 'center_x': center_x, 'center_y': center_y}
kwargs_lens = [kwargs_uldm]

alpha = lensModel.alpha(2.1, 1.2, kwargs_lens)
print('try deviation angle  ', alpha)

flexion_lens = lensModel.flexion(1,2,kwargs_lens)
print('flexion  ', flexion_lens)

uldm_lens = Uldm_Bar()

potential = uldm_lens.function(1.1, 1.1, kappa_0, theta_c, theta_E, gamma, e1, e2)
print('try potential ', potential)

MSD_potential = uldm_lens.function(0.4, 0, kappa_0, 5, 0.00001, gamma, e1, e2)
MSD_potential2 = uldm_lens.function(0.5, 0, kappa_0, 5, 0.00001, gamma, e1, e2)
print('Try MSD, uldm for large theta_c, potential ', MSD_potential - MSD_potential2, ' Pure MSD expected result ', kappa_0* 0.4**2 /2 - kappa_0* 0.5**2 /2)

abba = uldm_lens.derivatives(0.4, 0.7, kappa_0, theta_c, theta_E, gamma, e1, e2)
print('Try direct derivatives ', abba)

MSD_trial = uldm_lens.derivatives(0.4, 0.0, kappa_0, 5, 0.00001, gamma, e1, e2)
print('Try MSD, uldm for large theta_c ', MSD_trial, ' Pure MSD expected result ', kappa_0*0.4)

fermat_lens = lensModel.fermat_potential(1,2,kwargs_lens)
print('fermat potential: ',fermat_lens)

dt = lensModel.arrival_time(.9, .4, kwargs_lens)
print('arrival time ',dt)

D_Lens = lens_cosmo.dd * 10**6
Sigma_c = lens_cosmo.sigma_crit * 10**(-12)

print('D_lens ' , D_Lens, ' Sigma_crit ', Sigma_c)

mass3d = uldm_lens.mass_3d(10000, kappa_0, theta_c) * const.arcsec**2 * Sigma_c * D_Lens**2
print('Mass lens: ', np.log10(mass3d))

mass3dLens = uldm_lens.mass_3d_lens(20, kappa_0, theta_c, theta_E, gamma) * const.arcsec**2 * Sigma_c * D_Lens**2
print('Mass lens PL + ULDM: ', np.log10(mass3dLens))

m, M, rho0, lambda_sol = lens_cosmo.ULDM_BAR_angles2phys(kappa_0, theta_c, theta_E)
print('mass ', m, 'Mass Soliton ', M, 'rho0 phys ', rho0, 'lambda ', lambda_sol)

####### Plotting stuff
radius = np.arange(0.01, 10, 0.01)
D_Lens = lens_cosmo.dd * 1000 # in kpc
plt.xscale('log')
plt.yscale('log')
# rho is in M_sun/parsec^3
plt.plot(radius, (Sigma_c / D_Lens) /const.arcsec * uldm_lens.density(radius/D_Lens /const.arcsec, kappa_0, theta_c))
plt.show()
