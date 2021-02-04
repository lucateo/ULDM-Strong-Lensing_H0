import numpy as np
import sympy
import copy
import lenstronomy.Util.constants as const
import matplotlib.pyplot as plt
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from astropy.cosmology import FlatLambdaCDM
# import the LensModel class #
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Profiles.cored_density_exp import CoredDensityExp
from lenstronomy.LensModel.Profiles.spep import SPEP

# specify the choice of lens models #
lens_model_list = ['CORED_DENSITY_EXP']

cosmo = FlatLambdaCDM(H0 = 67.4, Om0=0.30, Ob0=0.00)
cosmo2 = FlatLambdaCDM(H0 = 74.3, Om0=0.30, Ob0=0.00)

z_lens = 0.5
# specify source redshift
z_source = 1.5
# setup lens model class with the list of lens models
lensModel = LensModel(lens_model_list=lens_model_list, z_lens = z_lens, z_source=z_source, cosmo=cosmo)
lens_cosmo = LensCosmo(z_lens = z_lens, z_source = z_source, cosmo = cosmo)
lens_cosmo2 = LensCosmo(z_lens = z_lens, z_source = z_source, cosmo = cosmo2)

# define parameter values of lens models
kappa_0 = 0.09
theta_c = 5.0
theta_E = 1.66
gamma = 1.98
e1 = 0.1
e2 = 0.1
center_x = 0
center_y = 0
kwargs_uldm = {'kappa_0': kappa_0, 'theta_c' : theta_c, 'center_x': center_x, 'center_y': center_y}
kwargs_lens = [kwargs_uldm]

alpha = lensModel.alpha(2.1, 1.2, kwargs_lens)
print('try deviation angle  ', alpha)

flexion_lens = lensModel.flexion(1,2,kwargs_lens)
print('flexion  ', flexion_lens)

uldm_lens = CoredDensityExp()
PL_lens = SPEP()

potential = uldm_lens.function(1.1, 1.1, kappa_0, theta_c)
print('try potential ', potential)

MSD_potential = uldm_lens.function(0.4, 0, kappa_0, 8 )
MSD_potential2 = uldm_lens.function(0.5, 0, kappa_0, 8)
print('Try MSD, uldm for large theta_c, potential ', MSD_potential - MSD_potential2, ' Pure MSD expected result ', kappa_0* 0.4**2 /2 - kappa_0* 0.5**2 /2)

abba = uldm_lens.derivatives(0.4, 0.7, kappa_0, theta_c)
print('Try direct derivatives ', abba)

MSD_trial = uldm_lens.derivatives(0.4, 0.0, kappa_0, 5)
print('Try MSD, uldm for large theta_c ', MSD_trial, ' Pure MSD expected result ', kappa_0*0.4)

fermat_lens = lensModel.fermat_potential(1,2,kwargs_lens)
print('fermat potential: ',fermat_lens)

dt = lensModel.arrival_time(.9, .4, kwargs_lens)
print('arrival time ',dt)

D_Lens = lens_cosmo.dd * 10**6
Sigma_c = lens_cosmo.sigma_crit * 10**(-12)

print('D_lens ' , D_Lens, ' Sigma_crit ', Sigma_c)

mass3d = uldm_lens.mass_3d(100, kappa_0, theta_c) * const.arcsec**2 * Sigma_c * D_Lens**2
print('Mass lens: ', np.log10(mass3d))

mass3dPL = PL_lens.mass_3d_lens(theta_E, theta_E, gamma) * const.arcsec**2 * Sigma_c * D_Lens**2
mass3dPL_MSD = PL_lens.mass_3d_lens(theta_E, theta_E*(1-kappa_0), gamma) * const.arcsec**2 * Sigma_c * D_Lens**2
print('Mass lens PL: ', np.log10(mass3dPL), 'Mass lens PL MSD ', np.log10(mass3dPL_MSD))

################################## CONVERSION FUNCTIONS ########################################
def ULDM_BAR_angles2phys(kappa_0, theta_c, theta_E, h0, z_lens, z_source):
    """
    converts angular units entering ULDM-BAR class in physical ULDM mass which are
    - m_log10: it is \log_10 m , m in eV
    - M_noCosmo_log10: it is \log_10 ( M ), M in M_sun
    :param kappa_0: central convergence of soliton
    :param theta_c: core radius (in arcsec)
    :param theta_E: Einstein radius of power law model (in arcsec)
    :return: mass_particle (log10, eV), Mass_soliton (log10, M_sun), rho0_physical (M_sun/pc^3), lambda_soliton
    """
    cosmo_current = FlatLambdaCDM(H0 = h0, Om0=0.30, Ob0=0.0)
    lens_cosmo_current = LensCosmo(z_lens = z_lens, z_source = z_source, cosmo = cosmo_current)
    D_Lens = lens_cosmo_current.dd * 10**6 # in pc
    Sigma_c = lens_cosmo_current.sigma_crit * 10**(-12) # in M_sun/pc^2
    rhotilde = kappa_0 / (np.sqrt(np.pi) * theta_c) # in 1/arcsec
    rho_phys = rhotilde * Sigma_c / (D_Lens * const.arcsec)
    A_factor = 4.64096 * 10**(-19) * D_Lens * Sigma_c * theta_E
    B_factor = 1.71468 * 10**(17) / (theta_c**2 * rhotilde * Sigma_c * D_Lens)
    lambda_factor = np.sqrt( (0.18 + np.sqrt(0.034 + 1.8 * A_factor * B_factor))/ (2*B_factor) )
    mass = 2.25221 * 10**(-27) * np.sqrt(Sigma_c * rhotilde /D_Lens)/ lambda_factor**2
    a_fit = 0.18 + 0.45 * A_factor/lambda_factor**2
    Mass_sol = 2.09294 * 10**(-11) * lambda_factor / mass  * a_fit**(-1.5)
    mass = np.log10(mass)
    Mass_sol = np.log10(Mass_sol)
    return mass, Mass_sol, rho_phys, lambda_factor

def ULDM_BAR_phys2angles(m_log10, M_log10, theta_E, h0, z_lens, z_source):
    """
    converts physical ULDM mass in angles entering ULDM-BAR class
    :param m_log10: it is \log_10 m , m in eV
    :param M_noCosmo_log10: it is \log_10 ( M ), M in M_sun
    :param theta_E: Einstein radius of power law model (in arcsec)
    :return: kappa_0, the central convergence, theta_c, the core radius (in arcseconds), and lambda_soliton
    """
    cosmo_current = FlatLambdaCDM(H0 = h0, Om0=0.30, Ob0=0.0)
    lens_cosmo_current = LensCosmo(z_lens = z_lens, z_source = z_source, cosmo = cosmo_current)
    m = 10**m_log10
    M = 10**M_log10
    D_Lens = lens_cosmo_current.dd * 10**6 # in pc
    Sigma_c = lens_cosmo_current.sigma_crit * 10**(-12) # in M_sun/pc^2
    A_factor = 4.64096 * 10**(-19) * D_Lens * Sigma_c * theta_E
    A_tilde = 7.48536 * 10**9 * m * M
    lambda_factor = 1.2 * A_tilde**(1/4) * A_factor**(3/8) * np.sqrt(1 + 0.3 * ( (A_tilde**2) /A_factor)**(1/4)
            + 0.17 * (A_tilde**2 / A_factor)**(3/4) )
    ## Old method by solving with nsolve the equation
    #  x = Symbol('x')
    #  eq1 = (0.18 + 0.45*A_factor/x**2)**3 * A_tilde**2 * 128/np.pi  - x**2
    #  lambda_factor = nsolve(eq1, x, 0.001)
    #  lambda_factor = float(lambda_factor)
    a_fit = 0.18 + 0.45 * A_factor/lambda_factor**2
    theta_c = 1.31891 * 10**(-18) / (lambda_factor * m * D_Lens * np.sqrt(2*a_fit))
    kappa_0 = 1.97143 * 10**53 * np.sqrt(np.pi) * theta_c * m**2 * lambda_factor**4 * D_Lens / Sigma_c
    return kappa_0, theta_c, lambda_factor

###################################################################################################


m, M, rho0, lambda_sol = ULDM_BAR_angles2phys(kappa_0, theta_c, theta_E*(1-kappa_0), 67.4, z_lens, z_source)
print('mass ', m, 'Mass Soliton ', M, 'rho0 phys ', rho0, 'lambda ', lambda_sol)

kappa_0Trial, theta_cTrial, lambda_Trial = ULDM_BAR_phys2angles(m, M, theta_E*(1 - kappa_0), 67.4, z_lens, z_source)
print('kappa0 ', kappa_0Trial, ' theta_c ', theta_cTrial, 'lambda ', lambda_Trial)

############## PRIOR
cosmo_reference = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.)
lens_cosmo_reference = LensCosmo(z_lens = z_lens, z_source = z_source, cosmo = cosmo_reference)
Ddt_reference = lens_cosmo_reference.ddt

print('Ddt-h0 relation check ', 70 * Ddt_reference / lens_cosmo.ddt)
print('Ddt-h0 prior check ', lens_cosmo.ddt)
def logL_addition(Ddt_sampled):
    """
    a definition taking as arguments (kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_special, kwargs_extinction)
            and returns a logL (punishing) value.
    """
    h0_mean = 67.4
    h0_sigma = 0.5

    Ddt_sampled =  Ddt_sampled
    h0_sampled = 70 * Ddt_reference / Ddt_sampled
    logL = - (h0_sampled - h0_mean)**2 / h0_sigma**2 / 2 + np.log(h0_sampled/Ddt_sampled)
    return logL



ddt_array = np.arange(3000, 4000, 10)
plt.plot(ddt_array, np.exp(logL_addition(ddt_array)))
plt.show()
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
plt.plot(radius, (Sigma_c / D_Lens) /const.arcsec * uldm_lens.density(1000*radius/D_Lens /const.arcsec, kappa_0, theta_c))
plt.show()
