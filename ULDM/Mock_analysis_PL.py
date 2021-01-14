# import of standard python libraries
import numpy as np
import os
import time
import copy
import corner
import astropy.io.fits as pyfits
import pickle, h5py

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

np.random.seed(42)

## Trying to simulate RXJ1131
# define lens configuration and cosmology (not for lens modelling)
z_lens = 0.5
z_source = 1.5
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=74, Om0=0.3, Ob0=0.)

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
kwargs_spemd = {'theta_E': 1.57, 'gamma': 1.98, 'center_x': 0.0, 'center_y': 0.0, 'e1': 0.05, 'e2': 0.05}  # parameters of the deflector lens model

# the lens model is a superposition of an elliptical lens model with external shear
lens_model_list = ['SPEP', 'SHEAR']
kwargs_lens = [kwargs_spemd, kwargs_shear]
lens_model_class = LensModel(lens_model_list=lens_model_list, z_lens=z_lens, z_source=z_source, cosmo=cosmo)

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

lensEquationSolver = LensEquationSolver(lens_model_class) # arguments with redshift and lens model
x_image, y_image = lensEquationSolver.findBrightImage(source_x, source_y, kwargs_lens, numImages=4,
                                                      min_distance=deltaPix, search_window=numPix * deltaPix)
mag = lens_model_class.magnification(x_image, y_image, kwargs=kwargs_lens)
kwargs_ps = [{'ra_image': x_image, 'dec_image': y_image,
                           'point_amp': np.abs(mag)*1000}]  # quasar point source position in the source plane and intrinsic brightness
point_source_list = ['LENSED_POSITION']
point_source_class = PointSource(point_source_type_list=point_source_list, fixed_magnification_list=[False])

kwargs_numerics = {'supersampling_factor': 1}

# Class to make the image
imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class,
                                lens_light_model_class, point_source_class, kwargs_numerics=kwargs_numerics)

# generate image
image_sim = imageModel.image(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
# Add noise to the image
poisson = image_util.add_poisson(image_sim, exp_time=exp_time)
bkg = image_util.add_background(image_sim, sigma_bkd=sigma_bkg)
# Sum model + noise
image_sim = image_sim + bkg + poisson

data_class.update_data(image_sim)
kwargs_data['image_data'] = image_sim

lens_model_list = ['SPEP', 'SHEAR']
kwargs_model = {'lens_model_list': lens_model_list,
                 'lens_light_model_list': lens_light_model_list,
                 'source_light_model_list': source_model_list,
                'point_source_model_list': point_source_list
                 }
# display the initial simulated image
#  cmap_string = 'gray'
#  #  cmap = plt.get_cmap(cmap_string)
#  cmap = copy.copy(plt.get_cmap(cmap_string))
#  cmap.set_bad(color='k', alpha=1.)
#  cmap.set_under('k')
#
#  v_min = -4
#  v_max = 2
#
#  f, axes = plt.subplots(1, 1, figsize=(6, 6), sharex=False, sharey=False)
#  ax = axes
#  im = ax.matshow(np.log10(image_sim), origin='lower', vmin=v_min, vmax=v_max, cmap=cmap, extent=[0, 1, 0, 1])
#  ax.get_xaxis().set_visible(False)
#  ax.get_yaxis().set_visible(False)
#  ax.autoscale(False)
#  plt.show()

########################## EXTRACT VALUES LIKE TIME DELAYS, VELOCITY DISPERSIONS; THESE ARE THE DATA OF A REAL OBSERVATION
from lenstronomy.Analysis.td_cosmography import TDCosmography
td_cosmo = TDCosmography(z_lens, z_source, kwargs_model, cosmo_fiducial=cosmo)

# time delays, the unit [days] is matched when the lensing angles are in arcsec
t_days = td_cosmo.time_delays(kwargs_lens, kwargs_ps, kappa_ext=0)
print("the time delays for the images at position ", kwargs_ps[0]['ra_image'], kwargs_ps[0]['dec_image'], "are: ", t_days)

# relative delays (observable). The convention is relative to the first image
dt_days =  t_days[1:] - t_days[0]
# and errors can be assigned to the measured relative delays (full covariance matrix not yet implemented)
dt_sigma = [3, 5, 10]  # Gaussian errors
# and here a realisation of the measurement with the quoted error bars
dt_measured = np.random.normal(dt_days, dt_sigma)
print("the measured relative delays are: ", dt_measured)

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

from lenstronomy.Analysis.kinematics_api import KinematicsAPI
kin_api = KinematicsAPI(z_lens, z_source, kwargs_model, cosmo=cosmo,
                        lens_model_kinematics_bool=[True, False], light_model_kinematics_bool=[True],
                        kwargs_aperture=kwargs_aperture, kwargs_seeing=kwargs_seeing,
                        anisotropy_model=anisotropy_model, kwargs_numerics_galkin=kwargs_numerics_galkin,
                        sampling_number=10000,  # numerical ray-shooting, should converge -> infinity
                        Hernquist_approx=True)

vel_disp = kin_api.velocity_dispersion(kwargs_lens, kwargs_lens_light, kwargs_anisotropy, r_eff=r_eff, theta_E=None, kappa_ext=0)
print(vel_disp, 'velocity dispersion in km/s')


############################# NOW THAT WE HAVE THE IMAGE AND THE TIME DELAYS + VELOCITY DISPERSION,
############################# TIME TO DO THE PARAMETER INFERENCE
# lens model choicers: initial guess and upper-lower bounds
fixed_lens = []
kwargs_lens_init = []
kwargs_lens_sigma = []
kwargs_lower_lens = []
kwargs_upper_lens = []

## ULDM model
## You have to put this, this means that the fixed parameters in this case are zero
## SPEP model
fixed_lens.append({})
kwargs_lens_init.append({'theta_E': 1.6, 'gamma': 2, 'center_x': 0.0, 'center_y': 0, 'e1': 0, 'e2': 0.})
kwargs_lens_sigma.append({'theta_E': .2, 'e1': 0.1, 'e2': 0.1, 'gamma': 0.1, 'center_x': 0.01, 'center_y': 0.01})
kwargs_lower_lens.append({'theta_E': 0.01, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.5, 'center_x': -10, 'center_y': -10})
kwargs_upper_lens.append({'theta_E': 10, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.5, 'center_x': 10, 'center_y': 10})

## SHEAR model
fixed_lens.append({'ra_0': 0, 'dec_0': 0})
kwargs_lens_init.append({'gamma1': 0, 'gamma2': 0})
#kwargs_lens_init.append(kwargs_shear)
kwargs_lens_sigma.append({'gamma1': 0.1, 'gamma2': 0.1})
kwargs_lower_lens.append({'gamma1': -0.5, 'gamma2': -0.5})
kwargs_upper_lens.append({'gamma1': 0.5, 'gamma2': 0.5})

## ULDM model
## You have to put this, this means that the fixed parameters in this case are zero
fixed_lens.append({})
#  kwargs_lens_init.append({'kappa_0': 0.1, 'theta_c': 7, 'center_x': 0.0, 'center_y': 0})
#  kwargs_lens_sigma.append({'kappa_0': 0.05, 'theta_c': 5, 'center_x': 0.01, 'center_y': 0.01})
#  kwargs_lower_lens.append({'kappa_0': 0.01, 'theta_c': 0.1, 'center_x': -10, 'center_y': -10})
#  kwargs_upper_lens.append({'kappa_0': 1.0, 'theta_c': 10, 'center_x': 10.0, 'center_y': 10.0})
kwargs_lens_init.append({'lambda_approx': 0.9, 'r_core': 7, 'center_x': 0.0, 'center_y': 0})
kwargs_lens_sigma.append({'lambda_approx': 0.1, 'r_core': 5, 'center_x': 0.01, 'center_y': 0.01})
kwargs_lower_lens.append({'lambda_approx': 0.5, 'r_core': 0.1, 'center_x': -10, 'center_y': -10})
kwargs_upper_lens.append({'lambda_approx': 1.0, 'r_core': 10, 'center_x': 10.0, 'center_y': 10.0})

lens_params = [kwargs_lens_init, kwargs_lens_sigma, fixed_lens, kwargs_lower_lens, kwargs_upper_lens]


# lens light model choices
fixed_lens_light = []
kwargs_lens_light_init = []
kwargs_lens_light_sigma = []
kwargs_lower_lens_light = []
kwargs_upper_lens_light = []

fixed_lens_light.append({})
kwargs_lens_light_init.append({'R_sersic': 0.5, 'n_sersic': 1, 'e1': 0, 'e2': 0., 'center_x': 0, 'center_y': 0})
#kwargs_lens_light_init.append(kwargs_sersic_lens)
kwargs_lens_light_sigma.append({'n_sersic': 0.5, 'R_sersic': 0.1, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1})
kwargs_lower_lens_light.append({'e1': -0.5, 'e2': -0.5, 'R_sersic': 0.01, 'n_sersic': 0.5, 'center_x': -10, 'center_y': -10})
kwargs_upper_lens_light.append({'e1': 0.5, 'e2': 0.5, 'R_sersic': 10, 'n_sersic': 8, 'center_x': 10, 'center_y': 10})

lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, fixed_lens_light, kwargs_lower_lens_light, kwargs_upper_lens_light]

# Source choices, seems to need only light profile stuff
fixed_source = []
kwargs_source_init = []
kwargs_source_sigma = []
kwargs_lower_source = []
kwargs_upper_source = []

fixed_source.append({})
kwargs_source_init.append({'R_sersic': 0.1, 'n_sersic': 1, 'e1': 0, 'e2': 0., 'center_x': 0, 'center_y': 0})
#kwargs_source_init.append(kwargs_sersic_source)
kwargs_source_sigma.append({'n_sersic': 0.5, 'R_sersic': 0.05, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1})
kwargs_lower_source.append({'e1': -0.5, 'e2': -0.5, 'R_sersic': 0.001, 'n_sersic': .5, 'center_x': -10, 'center_y': -10})
kwargs_upper_source.append({'e1': 0.5, 'e2': 0.5, 'R_sersic': 10, 'n_sersic': 5., 'center_x': 10, 'center_y': 10})

source_params = [kwargs_source_init, kwargs_source_sigma, fixed_source, kwargs_lower_source, kwargs_upper_source]

# Point source model, technical
fixed_ps = [{}]
kwargs_ps_init = kwargs_ps
kwargs_ps_sigma = [{'ra_image': 0.01 * np.ones(len(x_image)), 'dec_image': 0.01 * np.ones(len(x_image))}]
kwargs_lower_ps = [{'ra_image': -10 * np.ones(len(x_image)), 'dec_image': -10 * np.ones(len(y_image))}]
kwargs_upper_ps = [{'ra_image': 10* np.ones(len(x_image)), 'dec_image': 10 * np.ones(len(y_image))}]

fixed_cosmo = {'z_lens' : z_lens, 'z_source' : z_source, 'Om' : 0.3}
kwargs_cosmo_init = {'h0': 67.4}
# Find out its interpretation
kwargs_cosmo_sigma = {'h0': 1.}
kwargs_lower_cosmo = {'h0': 60}
kwargs_upper_cosmo = {'h0': 80}
cosmo_params = [kwargs_cosmo_init, kwargs_cosmo_sigma, fixed_cosmo, kwargs_lower_cosmo, kwargs_upper_cosmo]

ps_params = [kwargs_ps_init, kwargs_ps_sigma, fixed_ps, kwargs_lower_ps, kwargs_upper_ps]

lens_model_list_uldm = ['SPEP', 'SHEAR', 'CORED_DENSITY_EXP_MST']
# Just names of the various models used, like ULDM, SERSIC etc.
kwargs_model_uldm = {'lens_model_list': lens_model_list_uldm,
                 'lens_light_model_list': lens_light_model_list,
                 'source_light_model_list': source_model_list,
                'point_source_model_list': point_source_list
                 }
# Init, upper and lower bound values for all parameters of the model
kwargs_params = {'lens_model': lens_params,
                'source_model': source_params,
                'lens_light_model': lens_light_params,
                'point_source_model': ps_params,
                'special': cosmo_params}

# numerical options and fitting sequences
num_source_model = len(source_model_list)
kwargs_constraints = {'joint_source_with_point_source': [[0, 0]],
                      'num_point_source_list': [4],
                      'joint_lens_with_lens': [[0, 2, ['center_x', 'center_y']]],
                      'solver_type': 'PROFILE_SHEAR',  # 'PROFILE', 'PROFILE_SHEAR', 'ELLIPSE', 'CENTER'
                      'h0_sampling' : True,
                              }

# Defining parameters for H0 prior, IMPORTANT the double brackets!
#  prior_special = [['h0', 67.4, 0.5]]

# Defining prior for h0 * lambda_MST (Since I am subtracting the MST core on the lens model I am using,
# the h0 coming out from this model is actually h0/lambda_MST
class LikelihoodAddition(object):
    import numpy as np

    def __init__(self):
        pass

    def __call__(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None, kwargs_special=None, kwargs_extinction=None):
        return self.logL_addition(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_special, kwargs_extinction)

    def logL_addition(self, kwargs_lens, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None, kwargs_special=None, kwargs_extinction=None):
        """
        a definition taking as arguments (kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_special, kwargs_extinction)
                and returns a logL (punishing) value.
        """
        h0_mean = 67.4
        h0_sigma = 0.5

        h0_measured = kwargs_lens[2]['lambda_approx'] * kwargs_special['h0']
        logL = - (h0_measured - h0_mean)**2 / h0_sigma**2 / 2
        return logL
logL_addition = LikelihoodAddition()

kwargs_likelihood = {'check_bounds': True,
                     'force_no_add_image': False,
                     'source_marg': False,
                     'image_position_uncertainty': 0.004,
                     'check_matched_source_position': True,
                     'source_position_tolerance': 0.001,
                     'source_position_sigma': 0.001,
                     'h0_likelihood': True,
                     #  'prior_special' : prior_special,
                     'custom_logL_addition': logL_addition
                             }
# kwargs_data contains the image arraay
image_band = [kwargs_data, kwargs_psf, kwargs_numerics]
multi_band_list = [image_band]
kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear',
                    'time_delays_measured': dt_measured,
                    'time_delays_uncertainties': dt_sigma,}

from lenstronomy.Workflow.fitting_sequence import FittingSequence

mpi = False  # MPI possible, but not supported through that notebook.

from lenstronomy.Workflow.fitting_sequence import FittingSequence

run_sim = True

if run_sim == True:
    fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model_uldm, kwargs_constraints, kwargs_likelihood, kwargs_params)
    # Do before the PSO to reach a good starting value for MCMC
    fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': 200, 'n_iterations': 200}],
            ['MCMC', {'n_burn': 100, 'n_run': 500, 'walkerRatio': 10, 'sigma_scale': .2}]
    ]

    start_time = time.time()
    chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
    kwargs_result = fitting_seq.best_fit()

    file_name = 'mock_results_uldm_PL_MST.pkl'
    filedata = open(file_name, 'wb')
    pickle.dump(kwargs_result, filedata)
    filedata.close()

    file_name = 'mock_results_uldm_chain_PL_MST.pkl'
    filedata = open(file_name, 'wb')
    pickle.dump(chain_list, filedata)
    filedata.close()
    end_time = time.time()

    print(end_time - start_time, 'total time needed for computation')
    print('============ CONGRATULATION, YOUR JOB WAS SUCCESSFUL ================ ')
else:
    file_name = 'mock_results_uldm_PL_MST.pkl'
    filedata = open(file_name, 'rb')
    kwargs_result = pickle.load(filedata)
    filedata.close()

    file_name = 'mock_results_uldm_chain_PL_MST.pkl'
    filedata = open(file_name, 'rb')
    chain_list = pickle.load(filedata)
    filedata.close()

print('Final parameters given by MCMC: ', kwargs_result)

from lenstronomy.Plots import chain_plot
from lenstronomy.Plots.model_plot import ModelPlot

make_figures = False
if make_figures == True:
    modelPlot = ModelPlot(multi_band_list, kwargs_model_uldm, kwargs_result, arrow_size=0.02, cmap_string="gist_heat")
    f, axes = modelPlot.plot_main()
    f.savefig('Plot_main_PL.png')
    f, axes = modelPlot.plot_separate()
    f.savefig('Plot_separate_PL.png')
    f, axes = modelPlot.plot_subtract_from_data_all()
    f.savefig('Plot_subtract_PL.png')

make_chainPlot = False
make_cornerPlot = True
reprocess_corner = True
if make_chainPlot == True:
    # Plot the MonteCarlo
    for i in range(len(chain_list)):
        chain_plot.plot_chain_list(chain_list, i)
    chain_plot.plt.show()
    chain_plot.plt.savefig('chainPlot_PL.png')


if make_cornerPlot == True:
    sampler_type, samples_mcmc, param_mcmc, dist_mcmc  = chain_list[1]

    print("number of non-linear parameters in the MCMC process: ", len(param_mcmc))
    print("parameters in order: ", param_mcmc)
    print("number of evaluations in the MCMC process: ", np.shape(samples_mcmc)[0])

    import corner
    # import the parameter handling class #
    from lenstronomy.Sampling.parameters import Param
    # make instance of parameter class with given model options, constraints and fixed parameters #

    param = Param(kwargs_model_uldm, fixed_lens, fixed_source, fixed_lens_light, fixed_ps, fixed_cosmo,
                  kwargs_lens_init=kwargs_result['kwargs_lens'], **kwargs_constraints)
    # the number of non-linear parameters and their names #
    num_param, param_list = param.num_param()

    kwargs_corner = {'bins': 20, 'plot_datapoints': False, 'show_titles': True,
                     'label_kwargs': dict(fontsize=20), 'smooth': 0.5, 'levels': [0.68,0.95],
                    'fill_contours': True, 'alpha': 0.8}

    mcmc_new_list = []
    labels_new = [r"$\gamma$", r"$ \theta_{\rm E,MSD} $", r"$ \kappa_0 $", r"$ \theta_c $", r"$ h0 $"]
    if reprocess_corner == True:
        for i in range(len(samples_mcmc)):
            # transform the parameter position of the MCMC chain in a lenstronomy convention with keyword arguments #
            kwargs_result = param.args2kwargs(samples_mcmc[i])
            h0 = kwargs_result['kwargs_special']['h0']
            gamma = kwargs_result['kwargs_lens'][0]['gamma']
            theta_E = kwargs_result['kwargs_lens'][0]['theta_E']
            e1, e2 = kwargs_result['kwargs_lens'][0]['e1'], kwargs_result['kwargs_lens'][0]['e2']
            phi_G, q = param_util.ellipticity2phi_q(e1, e2)
            lambda_approx, theta_c = kwargs_result['kwargs_lens'][2]['lambda_approx'], kwargs_result['kwargs_lens'][2]['r_core']
            # Remember that the h0 coming out from this model is actually h0/lambda of the non-MSD subtract model
            h0 = h0 * lambda_approx
            #  cosmo_current = FlatLambdaCDM(H0 = h0, Om0=0.30, Ob0=0.0)
            #  lens_cosmo_current = LensCosmo(z_lens = z_lens, z_source = z_source, cosmo = cosmo_current)
            #  m_log10, M_log10, rho0_phys, lambda_soliton = lens_cosmo_current.ULDM_BAR_angles2phys(kappa_0, theta_c, theta_E)
            theta_E_MSD = theta_E * (lambda_approx)**(1/(gamma -1))
            mcmc_new_list.append([gamma, theta_E_MSD, lambda_approx, theta_c, h0])
        file_name = 'mock_corner_PL_MST.h5'
        try:
            h5file = h5py.File(file_name, 'w')
            h5file.create_dataset("dataset_mock", data=mcmc_new_list)
            h5file.create_dataset("dataset_mock_masses", data=mcmc_new_list2)
            h5file.close()
        except:
            print("The h5py stuff went wrong...")
    else:
        file_name = 'mock_corner_PL_MST.h5'
        h5file = h5py.File(file_name, 'r')
        mcmc_new_list = h5file['dataset_mock'][:]
        mcmc_new_list2 = h5file['dataset_mock_masses'][:]
        h5file.close()

    plot = corner.corner(mcmc_new_list, labels=labels_new, **kwargs_corner)
    plot.savefig('cornerPlot_PL_MST.png')


