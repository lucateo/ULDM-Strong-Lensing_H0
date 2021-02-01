# import of standard python libraries
import numpy as np
import os
import time
import copy
import corner
import astropy.io.fits as pyfits
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.Util.param_util as param_util
import lenstronomy.Util.simulation_util as sim_util
import lenstronomy.Util.image_util as image_util
from lenstronomy.Util import kernel_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF

import lenstronomy.Util.constants as const
from lenstronomy.LensModel.Profiles.pemd import PEMD


# define lens configuration and cosmology (not for lens modelling)
z_lens = 0.5
z_source = 1.5
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=74.3, Om0=0.31, Ob0=0.)
cosmo_cmb = FlatLambdaCDM(H0=67.4, Om0=0.31, Ob0=0.)
lens_cosmo = LensCosmo(z_lens = z_lens, z_source = z_source, cosmo = cosmo)
lens_cosmo_cmb = LensCosmo(z_lens = z_lens, z_source = z_source, cosmo = cosmo_cmb)

# data specifics
sigma_bkg = .05  #  background noise per pixel (Gaussian)
exp_time = 100.  #  exposure time (arbitrary units, flux per pixel is in units # photons/exp_time unit)
numPix = 100  #  cutout pixel size
deltaPix = 0.05  #  pixel size in arcsec (area per pixel = deltaPix**2)
fwhm = 0.1  # full width half max of PSF (only valid when psf_type='gaussian')
psf_type = 'GAUSSIAN'  # 'GAUSSIAN', 'PIXEL', 'NONE'; point spread function
kernel_size = 91
#kernel_cut = kernel_util.cut_psf(kernel, kernel_size)

# initial input simulation
# generate the coordinate grid and image properties
kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exp_time, sigma_bkg)
data_class = ImageData(**kwargs_data)
# generate the psf variables

# Point spread function things
kwargs_psf = {'psf_type': psf_type, 'pixel_size': deltaPix, 'fwhm': fwhm}
#kwargs_psf = sim_util.psf_configure_simple(psf_type=psf_type, fwhm=fwhm, kernelsize=kernel_size, deltaPix=deltaPix, kernel=kernel)
psf_class = PSF(**kwargs_psf)

########################### CHOOSING THE LENS MODELLING STUFF FOR THE MOCK IMAGE #################
# lensing quantities
gamma1, gamma2 = param_util.shear_polar2cartesian(phi=-0.5, gamma=0.09) # shear quantities
kwargs_shear = {'gamma1': gamma1, 'gamma2': gamma2}  # shear values

theta_E = 1.66
kappa_0 = 0.1
theta_c = 5.0
gamma = 1.98

kwargs_spemd = {'theta_E': theta_E, 'gamma': gamma, 'center_x': 0.0, 'center_y': 0.0, 'e1': 0.05, 'e2': 0.05}  # parameters of the deflector lens model
kwargs_spemd_MST = {'theta_E': theta_E*(1 - kappa_0), 'gamma': 1.98, 'center_x': 0.0, 'center_y': 0.0, 'e1': 0.05, 'e2': 0.05}  # parameters of the deflector lens model
kwargs_uldm = {'kappa_0': kappa_0, 'theta_c': theta_c, 'center_x': 0.0, 'center_y': 0.0}  # parameters of the deflector lens model

# the lens model is a superposition of an elliptical lens model with external shear
lens_model_list = ['PEMD', 'SHEAR']
lens_model_list_uldm = ['PEMD', 'SHEAR', 'ULDM-BAR']

kwargs_lens = [kwargs_spemd, kwargs_shear]
kwargs_lens_uldm = [kwargs_spemd_MST, kwargs_shear, kwargs_uldm]

lens_model_class = LensModel(lens_model_list=lens_model_list, z_lens=z_lens, z_source=z_source, cosmo=cosmo)
lens_model_class_uldm = LensModel(lens_model_list=lens_model_list_uldm, z_lens=z_lens, z_source=z_source, cosmo=cosmo_cmb)

m, M, rho0, lambda_sol = lens_cosmo_cmb.ULDM_BAR_angles2phys(kappa_0, theta_c, theta_E*(1-kappa_0))
print('mass ', m, 'Mass Soliton ', M, 'rho0 phys ', rho0, 'lambda ', lambda_sol)

D_Lens = lens_cosmo.dd * 10**6
Sigma_c = lens_cosmo.sigma_crit * 10**(-12)
D_Lens_cmb = lens_cosmo_cmb.dd * 10**6
Sigma_c_cmb = lens_cosmo_cmb.sigma_crit * 10**(-12)

PL_lens = PEMD()
mass3dPL = PL_lens.mass_3d_lens(5, theta_E, gamma) * const.arcsec**2 * Sigma_c * D_Lens**2
mass3dPL_MSD = PL_lens.mass_3d_lens(5, theta_E*(1 - kappa_0), gamma) * const.arcsec**2 * Sigma_c_cmb * D_Lens_cmb**2

print('Mass lens PL: ', np.log10(mass3dPL), ' Mass lens PL with MSD', np.log10(mass3dPL_MSD), ' Mass PL with MSD + ULDM', np.log10(mass3dPL_MSD + 10**M))


# choice of source type
source_type = 'SERSIC'  # 'SERSIC' or 'SHAPELETS'
source_x = 0.
source_y = 0.1
# Sersic parameters in the initial simulation for the source
phi_G, q = 0.5, 0.8
e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
kwargs_sersic_source = {'amp': 2000, 'R_sersic': 0.1, 'n_sersic': 1, 'e1': e1, 'e2': e2, 'center_x': source_x, 'center_y': source_y}
#kwargs_else = {'sourcePos_x': source_x, 'sourcePos_y': source_y, 'quasar_amp': 400., 'gamma1_foreground': 0.0, 'gamma2_foreground':-0.0}
source_model_list = ['SERSIC_ELLIPSE']
kwargs_source = [kwargs_sersic_source]
source_model_class = LightModel(light_model_list=source_model_list)

# lens light model
phi_G, q = 0.9, 0.9
e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
kwargs_sersic_lens = {'amp': 4000, 'R_sersic': 0.4, 'n_sersic': 2., 'e1': e1, 'e2': e2, 'center_x': 0.0, 'center_y': 0}
lens_light_model_list = ['SERSIC_ELLIPSE']
kwargs_lens_light = [kwargs_sersic_lens]
lens_light_model_class = LightModel(light_model_list=lens_light_model_list)

point_source_list = ['LENSED_POSITION']
point_source_class = PointSource(point_source_type_list=point_source_list, fixed_magnification_list=[False])

kwargs_numerics = {'supersampling_factor': 1}

### KINEMATICS
# observational conditions of the spectroscopic campagne
R_slit = 1. # slit length in arcsec
dR_slit = 1.  # slit width in arcsec
psf_fwhm = 0.7

kwargs_aperture = {'aperture_type': 'slit', 'length': R_slit, 'width': dR_slit, 'center_ra': 0.05, 'center_dec': 0, 'angle': 0}
anisotropy_model = 'OM' # Osipkov-Merritt
aperture_type = 'slit'

kwargs_numerics_galkin = {'interpol_grid_num': 1000,  # numerical interpolation, should converge -> infinity
                          'log_integration': True,  # log or linear interpolation of surface brightness and mass models
                           'max_integrate': 100, 'min_integrate': 0.001}  # lower/upper bound of numerical integrals

r_ani = 1.
r_eff = 0.2
kwargs_anisotropy = {'r_ani': r_ani}
kwargs_seeing = {'psf_type': 'GAUSSIAN', 'fwhm': psf_fwhm}
kwargs_model = {'lens_model_list': lens_model_list,
                 'lens_light_model_list': lens_light_model_list,
                 'source_light_model_list': source_model_list,
                'point_source_model_list': point_source_list
                 }
kwargs_model_uldm = {'lens_model_list': lens_model_list_uldm,
                 'lens_light_model_list': lens_light_model_list,
                 'source_light_model_list': source_model_list,
                'point_source_model_list': point_source_list
                 }

from lenstronomy.Analysis.kinematics_api import KinematicsAPI
kin_api = KinematicsAPI(z_lens, z_source, kwargs_model, cosmo=cosmo,
                        # Put the appropriate bools here if you change the number of lenses!!
                        lens_model_kinematics_bool=[True, False], light_model_kinematics_bool=[True],
                        kwargs_aperture=kwargs_aperture, kwargs_seeing=kwargs_seeing,
                        anisotropy_model=anisotropy_model, kwargs_numerics_galkin=kwargs_numerics_galkin,
                        sampling_number=10000,  # numerical ray-shooting, should converge -> infinity
                        Hernquist_approx=True)

vel_disp = kin_api.velocity_dispersion(kwargs_lens, kwargs_lens_light, kwargs_anisotropy, r_eff=r_eff, theta_E=None, kappa_ext=0)

kin_api_uldm = KinematicsAPI(z_lens, z_source, kwargs_model_uldm, cosmo=cosmo_cmb,
                        # Put the appropriate bools here if you change the number of lenses!!
                        lens_model_kinematics_bool=[True, False, True], light_model_kinematics_bool=[True],
                        kwargs_aperture=kwargs_aperture, kwargs_seeing=kwargs_seeing,
                        anisotropy_model=anisotropy_model, kwargs_numerics_galkin=kwargs_numerics_galkin,
                        sampling_number=10000,  # numerical ray-shooting, should converge -> infinity
                        Hernquist_approx=True)

vel_disp_uldm = kin_api_uldm.velocity_dispersion(kwargs_lens_uldm, kwargs_lens_light, kwargs_anisotropy, r_eff=r_eff, theta_E=None, kappa_ext=0)

print(vel_disp, 'velocity dispersion in km/s', vel_disp * np.sqrt((1 - kappa_0)), 'velocity dispersion PL with MSD')
print(vel_disp_uldm, 'velocity dispersion ULDM in km/s')

kin_api_uldm = KinematicsAPI(z_lens, z_source, kwargs_model_uldm, cosmo=cosmo_cmb,
                        # Put the appropriate bools here if you change the number of lenses!!
                        lens_model_kinematics_bool=[False, False, True], light_model_kinematics_bool=[True],
                        kwargs_aperture=kwargs_aperture, kwargs_seeing=kwargs_seeing,
                        anisotropy_model=anisotropy_model, kwargs_numerics_galkin=kwargs_numerics_galkin,
                        sampling_number=10000,  # numerical ray-shooting, should converge -> infinity
                        Hernquist_approx=True)

vel_disp_uldm_only = kin_api_uldm.velocity_dispersion(kwargs_lens_uldm, kwargs_lens_light, kwargs_anisotropy, r_eff=r_eff, theta_E=None, kappa_ext=0)
print(vel_disp_uldm_only, 'Only ULDM in km/s')
print(np.sqrt((vel_disp * np.sqrt(1 - kappa_0))**2 + vel_disp_uldm_only**2 ), 'expected result by summing')



### Velocity dispersion dependence with core
plot_vel = False
plot_masses = False # put to True if you want the plot with the masses
if plot_vel == True:
    def velocity_dependence(kwargs_lens, kappa0_list, theta_core):

        kwargs_lens_result = copy.deepcopy(kwargs_lens)

        vel_disp_list = []

        kwargs_lens_kin = copy.deepcopy(kwargs_lens)
        kwargs_lens_kin[2]['kappa_0'] = 0
        kwargs_lens_kin[2]['theta_c'] = theta_core

        vel_disp_0 = kin_api.velocity_dispersion(kwargs_lens_kin, kwargs_lens_light, kwargs_anisotropy, r_eff=r_eff, theta_E=None,)

        for kappa0 in kappa0_list:
            kappa_ext = 0
            kwargs_lens_kin = copy.deepcopy(kwargs_lens)
            #del kwargs_lens_kin[2]['lambda_approx']
            kwargs_lens_kin[2]['kappa_0'] = kappa0
            kwargs_lens_kin[2]['theta_c'] = theta_core
            kwargs_lens_kin[0]['theta_E'] = kwargs_lens[0]['theta_E'] * (1 - kappa0)

            vel_disp = kin_api_uldm.velocity_dispersion(kwargs_lens_kin, kwargs_lens_light, kwargs_anisotropy, r_eff=r_eff, theta_E=None, kappa_ext=kappa_ext)
            vel_disp_list.append(vel_disp)
        return np.array(vel_disp_list), vel_disp_0

    # Plot with theta_c, kappa_0
    kappa0_list = np.linspace(0.01, 0.2, 20)
    theta_core_list = [0.5, 5, 10]
    vel_disp_list_r = []
    vel_disp_0_r = []
    for theta_core in theta_core_list:
        vel_disp_list, vel_disp_0 = velocity_dependence(kwargs_lens_uldm, kappa0_list, theta_core)
        vel_disp_list_r.append(vel_disp_list)
        vel_disp_0_r.append(vel_disp_0)
        print(theta_core)

    f, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 2]}, sharex=True)

    for i in range(len(theta_core_list)):
        vel_disp_list = vel_disp_list_r[i]
        vel_disp_0 = vel_disp_0_r[i]

        axes[0].plot(kappa0_list, vel_disp_list, label=r'$\sigma^{\rm P}$ for $\theta_{\rm core} =$'+str(theta_core_list[i]))
    #  axes[0].plot(kappa0_list, vel_disp_0*np.ones(20), 'k--', label=r"$\sigma^{\rm P} = (1 - \kappa_0)^{1/2} \sigma^{\rm P}(\kappa_0 = 0)$")
    axes[0].plot(kappa0_list, vel_disp_0*np.ones(20), 'k--', label=r"$\sigma^{\rm P} = \sigma^{\rm P}(\kappa_0 = 0)$")
    axes[0].set_ylim([220, 270])
    axes[0].set_ylabel(r'$\sigma^{\rm P}$ [km/s]', fontsize=20)
    axes[0].legend(fontsize=12)

    for i in range(len(theta_core_list)):
        vel_disp_list = vel_disp_list_r[i]
        vel_disp_0 = vel_disp_0_r[i]

        axes[1].plot(kappa0_list, (vel_disp_list - vel_disp_0) / vel_disp_0 , label=r'$\sigma^{\rm P}(\kappa_{\lambda_{\rm c}})$ for $\theta_{\rm c} =$'+str(theta_core_list[i]))
    axes[1].set_ylabel(r'$\Delta \sigma^{\rm P} / \sigma^{\rm P}_0 $', fontsize=20)
    axes[1].plot([0, 2], [0, 0], ':k')
    axes[1].set_ylim([-0.10, 0.10])
    plt.xlim([0.01, 0.2])
    plt.subplots_adjust(hspace = -0.2)
    plt.xlabel(r'$\kappa_0$', fontsize=20)
    plt.tight_layout()
    plt.savefig('Sigma_Dispersion2.pdf')
#  plt.show()
#
#  plt.clf()
# Plot with masses
############## A lot of things are probably wrong here, leave it be for now
elif plot_masses == True:
    def velocity_dependence_masses(kwargs_lens, M_list, m):

        kwargs_lens_result = copy.deepcopy(kwargs_lens)

        vel_disp_list = []

        kwargs_lens_kin = copy.deepcopy(kwargs_lens)
        kwargs_lens_kin[2]['kappa_0'] = 0

        vel_disp_0 = kin_api.velocity_dispersion(kwargs_lens_kin, kwargs_lens_light, kwargs_anisotropy, r_eff=r_eff, theta_E=None)

        for M in M_list:
            kappa_ext = 0
            kwargs_lens_kin = copy.deepcopy(kwargs_lens)
            #### Possible error here, using theta_E and not theta_E,MSD; probably the recursive correction on second line is not enough
            kappa0, theta_c, lambda_factor = lens_cosmo_cmb.ULDM_BAR_phys2angles(m, M, kwargs_lens[0]['theta_E'])
            kappa0, theta_c, lambda_factor = lens_cosmo_cmb.ULDM_BAR_phys2angles(m, M, kwargs_lens[0]['theta_E']* (1 - kappa0) )
            kwargs_lens_kin[2]['kappa_0'] = kappa0
            kwargs_lens_kin[2]['theta_c'] = theta_c
            kwargs_lens_kin[0]['theta_E'] = kwargs_lens[0]['theta_E'] * (1 - kappa0)

            vel_disp = kin_api.velocity_dispersion(kwargs_lens_kin, kwargs_lens_light, kwargs_anisotropy, r_eff=r_eff, theta_E=None, kappa_ext=kappa_ext)
            vel_disp_list.append([vel_disp,kappa0])
        return np.array(vel_disp_list), vel_disp_0
    M_list = np.linspace(9, 11.5, 20)
    m_list = [-25, -24.75, -24.5]

    vel_disp_list_r = []
    vel_disp_0_r = []
    kappa0_list_r = []
    for m in m_list:
        vel_disp_list, vel_disp_0 = velocity_dependence_masses(kwargs_lens_uldm, M_list, m)
        vel_disp_list_r.append(vel_disp_list[:,0])
        kappa0_list_r.append(vel_disp_list[:,1])
        vel_disp_0_r.append(vel_disp_0)
        print(m)

    f, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 2]}, sharex=True)

    for i in range(len(m_list)):
        vel_disp_list = vel_disp_list_r[i]
        vel_disp_0 = vel_disp_0_r[i]
        kappa0_list = kappa0_list_r[i]

        axes[0].plot(M_list, vel_disp_list, label=r'$\sigma^{\rm P}$ for $ m =$'+str(m_list[i]))
    axes[0].plot(M_list, vel_disp_0 *np.sqrt(1 - kappa0_list)
            , 'k--', label=r"$\sigma^{\rm P} =  \sigma^{\rm P}(\kappa_0 = 0)})$")
    axes[0].set_ylim([220, 270])
    axes[0].set_ylabel(r'$\sigma^{\rm P}$ [km/s]', fontsize=20)
    axes[0].legend(fontsize=12)

    for i in range(len(m_list)):
        vel_disp_list = vel_disp_list_r[i]
        vel_disp_0 = vel_disp_0_r[i]
        kappa0_list = kappa0_list_r[i]

        axes[1].plot(M_list, (vel_disp_list - vel_disp_0 *np.sqrt(1 - kappa0_list)
            ) / vel_disp_list , label=r'$\sigma^{\rm P}(\kappa_0)$ for $ m =$'+str(m_list[i]))
    axes[1].set_ylabel(r'$\Delta \sigma^{\rm P} / \sigma^{\rm P}$', fontsize=20)
    axes[1].plot([0, 2], [0, 0], ':k')
    axes[1].set_ylim([-0.01, 0.01])
    plt.xlim([9, 11.5])
    plt.subplots_adjust(hspace = -0.2)
    plt.xlabel(r'$ \log_{10} M \ [M_\odot]$', fontsize=20)
    plt.tight_layout()
    plt.savefig('Sigma_Dispersion_masses.pdf')


