import numpy as np
import copy
import lenstronomy.Util.constants as const
import matplotlib.pyplot as plt
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from astropy.cosmology import FlatLambdaCDM
# import the LensModel class #
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Profiles.uldm import Uldm
from lenstronomy.LensModel.Profiles.spep import SPEP
from scipy.special import hyp2f1
from scipy.special import gamma as gamma_func

import scipy.integrate as integrate

# specify the choice of lens models #
lens_model_list = ['ULDM']
cosmo = FlatLambdaCDM(H0 = 67.4, Om0=0.30, Ob0=0.00)
cosmo2 = FlatLambdaCDM(H0 = 74.3, Om0=0.30, Ob0=0.00)

z_lens = 0.5
# specify source redshift
z_source = 1.5
# setup lens model class with the list of lens models
lensModel = LensModel(lens_model_list=lens_model_list, z_lens = z_lens, z_source=z_source, cosmo=cosmo)
lens_cosmo = LensCosmo(z_lens = z_lens, z_source = z_source, cosmo = cosmo)
lens_cosmo2 = LensCosmo(z_lens = z_lens, z_source = z_source, cosmo = cosmo2)

print(67* lens_cosmo.ddt/4000)
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
    return kappa_0, theta_c, slope, M_log10

def angles2phys(kappa_0, theta_c, slope, theta_E):
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
    m_particle = clight * hbar * core_half_factor / (lambda_factor * a_fit * D_Lens * theta_c * const.arcsec)
    m_log10 = np.log10( m_particle/ eV2Joule)

    M_sol = lambda_factor * clight**3 * hbar * np.sqrt(np.pi) * gamma_func(slope - 1.5)
    M_sol = M_sol / (G_const * m_particle * a_fit**3 * 4 * gamma_func(slope) ) # in kg
    M_log10 = np.log10( M_sol/m_sun)
    return m_log10, M_log10, lambda_factor
##############################################################################

# define parameter values of lens models
kappa_0 = 0.09
theta_c = 5.0
slope = 3.84
theta_E = 1.66
gamma = 1.98
e1 = 0.1
e2 = 0.1
center_x = 0
center_y = 0
kwargs_uldm = {'kappa_0': kappa_0, 'theta_c' : theta_c, 'slope': slope, 'center_x': center_x, 'center_y': center_y}
kwargs_lens = [kwargs_uldm]
print(angles2phys(kappa_0, theta_c, slope, theta_E * (1 - kappa_0)))
print(phys2ModelParam(-25.9,0.0016,1.44))




a_factor = (0.5)**(-1./slope) -1
print('a factor corresponding to selected slope ', a_factor)

alpha = lensModel.alpha(2.1, 1.2, kwargs_lens)
print('try deviation angle  ', alpha)

flexion_lens = lensModel.flexion(1,2,kwargs_lens)
print('flexion  ', flexion_lens)

uldm_lens = Uldm()
PL_lens = SPEP()

potential = uldm_lens.function(1.1, 1.1, kappa_0, theta_c, slope)
print('try potential ', potential)

MSD_potential = uldm_lens.function(0.4, 0, kappa_0, 8, slope )
MSD_potential2 = uldm_lens.function(0.5, 0, kappa_0, 8, slope)
print('Try MSD, uldm for large theta_c, potential ', MSD_potential - MSD_potential2, ' Pure MSD expected result ', kappa_0* 0.4**2 /2 - kappa_0* 0.5**2 /2)

abba = uldm_lens.derivatives(0.4, 0.7, kappa_0, theta_c, slope)
print('Try direct derivatives ', abba)

MSD_trial = uldm_lens.derivatives(0.4, 0.0, kappa_0, 5, slope)
print('Try MSD, uldm for large theta_c ', MSD_trial, ' Pure MSD expected result ', kappa_0*0.4)

fermat_lens = lensModel.fermat_potential(1,2,kwargs_lens)
print('fermat potential: ',fermat_lens)

dt = lensModel.arrival_time(.9, .4, kwargs_lens)
print('arrival time ',dt)

D_Lens = lens_cosmo.dd * 10**6
Sigma_c = lens_cosmo.sigma_crit * 10**(-12)

print('D_lens ' , D_Lens, ' Sigma_crit ', Sigma_c)

mass3d = uldm_lens.mass_3d(100, kappa_0, theta_c, slope) * const.arcsec**2 * Sigma_c * D_Lens**2
print('Mass lens: ', np.log10(mass3d))

mass3dPL = PL_lens.mass_3d_lens(theta_E*2, theta_E, gamma) * const.arcsec**2 * Sigma_c * D_Lens**2
mass3dPL_MSD = PL_lens.mass_3d_lens(theta_E*2, theta_E*(1-kappa_0), gamma) * const.arcsec**2 * Sigma_c * D_Lens**2
print('Mass lens PL: ', np.log10(mass3dPL), 'Mass lens PL MSD ', np.log10(mass3dPL_MSD))


hyperMass = hyp2f1(3./2, slope, 5./2, - 100**2)*100**3
print('Hyperfunction mass trial ', np.real(hyperMass))

####### Plotting stuff
radius = np.arange(0.01, 10, 0.01)*10
plt.rcParams.update({'font.size': 16, 'figure.autolayout': True})
plt.figure(figsize=(8,10))
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$\rho[ M_\odot /\mathrm{pc}^3]$', fontsize=16)
plt.xlabel(r'r [kpc]', fontsize=16)
print('Expected core ', theta_c * const.arcsec * D_Lens)
#  plt.xlim(100, 10000)
#  rho is in M_sun/parsec^3, I put the radius in kpc
plt.plot(radius, (Sigma_c / D_Lens) /const.arcsec * uldm_lens.density(1000*radius/D_Lens /const.arcsec, kappa_0, theta_c, slope))
plt.show()




