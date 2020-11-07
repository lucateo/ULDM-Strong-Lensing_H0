# some standard python imports #
import copy
import numpy as np
import corner
import emcee
import pickle
import os
import csv
from os.path import dirname, abspath
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# %matplotlib inline

# we set a random seed
np.random.seed(seed=10)

# matplotlib configs
from pylab import rc
rc('axes', linewidth=2)
rc('xtick',labelsize=15)
rc('ytick',labelsize=15)

# hierArc and lenstronomy imports
from hierarc.Sampling.mcmc_sampling import MCMCSampler
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from hierarc.Diagnostics.goodness_of_fit import GoodnessOfFit
from lenstronomy.Plots import plot_util

# Local imports
from CustomPrior import CustomPrior # class with the options for the prior in hierarchical sampling
from Functions import properties

# relevant relative paths to other aspects of the analysis
path2Analysis = dirname(abspath(os.getcwd()))
path2tdcosmo = 'TDCOSMO_sample'
path2slacs = 'SLACS_sample'
path2kappa = os.path.join(path2Analysis, path2slacs, 'kappaSLACS')
file_name_tdcosmo = os.path.join(path2Analysis, path2tdcosmo, 'tdcosmo_sample.csv')
file_name_slacs = os.path.join(path2Analysis, path2slacs, 'slacs_all_params.csv')
file_name_slacs_los = os.path.join(path2Analysis, path2slacs, 'LineOfSightData.csv')

# Anisotropy model choice and distribution of parameters
anisotropy_model = 'OM' # 'OM', 'GOM' or 'const'; OM = Ossikptov-Merrit
anisotropy_distribution = 'GAUSSIAN'  # 'NONE' or 'GAUSSIAN'; this is the ditribution followed by the scatter in the posterior of a_ani

all_slacs_sample = True  # bool, if True, uses all SLACS lenses, also those without individual lensing-only power-law slope measurements and applies to those the population distribution prior

num_distribution_draw = 100  # number of draws from the hyper-parameter distribution in computing the Monte Carlo integral marginalization

############# IT WAS TRUE BEFORE
ifu_mst_sampling_separate = False  # if True, samples a separate MST parameter for the IFU lenses, this is due to the fact that the IFU observations are processed with a single star spectral template only.
#############

sigma_v_systematics = True  # bool, if True, samples an additional uncorrelated Gaussian uncertainty on all the velocity dispersion measurements of the SDSS data

############# IT WAS TRUE BEFORE
lambda_slope = False  # allow for linear scaling of lambda_int with r_eff/theta_E
#############

## On the original file it is called differently
planck_prior = True  # if True, sets a Gaussian prior with CMB value
log_scatter = True  # scatter parameters sampled in log space with linear prior in log space

############## Change this once you have the data for the plot in order not to make the chain run again
run_chains = False # boolean, if True, runs the chains, else uses the stored chains to do the plotting. ATTENTION: If you re-run the chains, they overwrite those currently in the folder (except continue_from_backend=True)
continue_from_backend = False  # boolean, if True, continues sampling the emcee chain from a backend (if exists), otherwise deletes previous chains and starts from scratch
##############

########### GOOD PARAMETERS ARE OF THE ORDER OF 1000 for n_burn and n_run
n_walkers =100  # number of walkers in the emcee sampling
n_burn = 10  # number of burn-in iterations ignored in the posteriors
## The number of n_burn and n_run are higher (the latter way higher) in the original, be careful on this
n_run = 10  # number of iterations (+n_burn) in the emcee sampling
###########

# pre-processed likelihood instances for SLACS SDSS and IFU data
############# THESE LIKELIHOODS SHOULD BE RE-PROCESSED
if anisotropy_model == 'OM':
    ifu_sample_file_name = 'slacs_ifu_om_processed.pkl'
    sdss_sample_file_name = 'slacs_slit_om_processed.pkl'
elif anisotropy_model == 'GOM':
    ifu_sample_file_name = 'slacs_ifu_gom_processed.pkl'
    sdss_sample_file_name = 'slacs_ifu_gom_processed.pkl'
##############

# names of the 7 TDCOSMO lenses in the sample (string needs to match pre-processed files)
TDCOSMO_lenses = ['B1608+656', 'RXJ1131-1231', 'HE0435-1223', 'SDSS1206+4332', 'WFI2033-4723', 'PG1115+080', 'DES0408-5354']

# select SLACS sample (selection criteria are processed in separate notebook)
# selected slit lenses
# all SLACS lenses with power-law slopes measured from the quality sample of Shajib et al. 2020 passing our selection criteria (see sample_selection notebook) (excluding the lenses with additonal IFU data)
sdss_names_quality = ['SDSSJ1402+6321', 'SDSSJ1630+4520', 'SDSSJ0330-0020', 'SDSSJ0029-0055',
                      'SDSSJ0728+3835', 'SDSSJ1112+0826', 'SDSSJ1306+0600', 'SDSSJ1531-0105',
                      'SDSSJ1621+3931']
# all SLACS lenses including those with population prior from Shajib et al. 2020 passing our selection criteria (see sample_selection notebook) (excluding the lenses with additonal IFU data)
sdss_names_all = ['SDSSJ1402+6321', 'SDSSJ1630+4520', 'SDSSJ0330-0020', 'SDSSJ0029-0055',
                  'SDSSJ0728+3835', 'SDSSJ1112+0826', 'SDSSJ1306+0600', 'SDSSJ1531-0105',
                  'SDSSJ1621+3931', 'SDSSJ1153+4612', 'SDSSJ0008-0004', 'SDSSJ0044+0113',
                  'SDSSJ0959+4416', 'SDSSJ1016+3859', 'SDSSJ1020+1122', 'SDSSJ1134+6027',
                  'SDSSJ1142+1001', 'SDSSJ1213+6708', 'SDSSJ1218+0830', 'SDSSJ1432+6317',
                  'SDSSJ1644+2625', 'SDSSJ2347-0005', 'SDSSJ1023+4230', 'SDSSJ1403+0006']

# selected IFU lenses within the criteria
ifu_names_quality = ['SDSSJ1627-0053', 'SDSSJ2303+1422', 'SDSSJ1250+0523', 'SDSSJ1204+0358', 'SDSSJ0037-0942']
# all including population prior
ifu_names_all = ['SDSSJ1627-0053', 'SDSSJ2303+1422', 'SDSSJ1250+0523', 'SDSSJ1204+0358', 'SDSSJ0037-0942', 'SDSSJ0912+0029', 'SDSSJ2321-0939', 'SDSSJ0216-0813', 'SDSSJ1451-0239']

# here we add the IFU lenses to the sdss sample. We don't double count unless we allow the IFU data to constrain the MST
#if ifu_mst_sampling_separate is True:
sdss_names_quality += ifu_names_quality
sdss_names_all += ifu_names_all

# this is for saving the chains under different names
if ifu_mst_sampling_separate is True:
    model_prefix = 'ifu_separate'
else:
    model_prefix = 'ifu_joint'
if lambda_slope is True:
    lambda_prefix = '_slope'
else:
    lambda_prefix = ''

if planck_prior is True:
    lambda_prefix += '_planck'
if log_scatter is True:
    lambda_prefix += '_log_scatter'

# Function to extract parameters from pre-computed csv file, we should have our
# own parameter file
def read_kappa_pdf(name, kappa_bins):
    """! Function that takes the values of \f$ k_{text{ext}} \f$ from file

    @param name The name of the galaxy; the function redirects to the relevant file
    @param kappa_bins Number of bins for the histogram
    @return the \f$ k_{text{ext}} \f$ histogram
    """
    filepath = os.path.join(path2kappa, 'kappahist_'+name+kappa_choice_ending)
    try:
        output = np.loadtxt(filepath, delimiter=' ', skiprows=1)
        kappa_sample = output[:, 0]
        kappa_weights = output[:, 1]
        kappa_pdf, kappa_bin_edges = np.histogram(kappa_sample, weights=kappa_weights, bins=kappa_bins, density=True)
        return kappa_pdf, kappa_bin_edges
    except:
        print('individual kappa_distribution for %s not available, using the population distribution instead.' %name)
        return kappa_pdf_tot, kappa_bin_edges_tot

# import TDCOSMO sample
tdcosmo_posterior_list = []
# This has all the single lens posteriors
for tdcosmo_lens in TDCOSMO_lenses:
    td_cosmo_file_name = os.path.join(path2Analysis, path2tdcosmo, tdcosmo_lens+'_processed.pkl')
    file = open(td_cosmo_file_name, 'rb')
    posterior = pickle.load(file)
    file.close()
    posterior['num_distribution_draws'] = num_distribution_draw
    posterior['error_cov_measurement'] = np.array(posterior['error_cov_measurement'], dtype='float')
    ## properties is the function that simply assigns the Galaxy properties
    ## from the file given; posterior['something'] loads the column 'something' from
    ## the table posterior
    ######################### PUT M SOLITON ONCE YOU HAVE IT
    theta_E, r_eff, gamma_pl, z_lens, z_source, sigma_sis = properties(tdcosmo_lens, file_name_tdcosmo, partial_name=True)
    #########################
    posterior['lambda_scaling_property'] = r_eff/theta_E - 1
    tdcosmo_posterior_list.append(posterior)

# Number of bins for the k_ext histogram
kappa_bins = np.linspace(-0.05, 0.2, 50)

names_selected = sdss_names_all

# Options for the plots
percentiles = [16, 50, 84]
quantiles = [0.16, 0.5, 0.84]
title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
fmt = "{{0:{0}}}".format(".3f").format

# option with file ending
with_shear = False  # bool, if True, adds shear constraints in LOS estimate (only available for a subset of the sample, not recommended and not the choice of Birrer et al. 2020)
kappa_choice_shear_ending = '_computed_1innermask_nobeta_zgap-1.0_-1.0_fiducial_120_gal_120_gamma_120_oneoverr_23.0_med_increments2_16_2_emptymsk_shearwithoutprior.cat'
kappa_choice_no_shear_ending = '_computed_1innermask_nobeta_zgap-1.0_-1.0_fiducial_120_gal_120_oneoverr_23.0_med_increments2_2_emptymsk.cat'

if with_shear is True:
    kappa_choice_ending = kappa_choice_shear_ending
else:
    kappa_choice_ending = kappa_choice_no_shear_ending

kwargs_ifu_all_list = []
kwargs_ifu_quality_list = []

file = open(os.path.join(path2Analysis, path2slacs, ifu_sample_file_name) , 'rb')
posterior_ifu_list = pickle.load(file)
file.close()
for kwargs_posterior in posterior_ifu_list:
    name = kwargs_posterior['name']
    #if name == 'SDSSJ0216-0813 ':
    #    name  = 'SDSSJ0216-0813'
    if name in ifu_names_all:
        kwargs_posterior_copy = copy.deepcopy(kwargs_posterior)
        if 'flag_ifu' in kwargs_posterior_copy:
            del kwargs_posterior_copy['flag_ifu']
        if 'flag_imaging' in kwargs_posterior_copy:
            del kwargs_posterior_copy['flag_imaging']
        kwargs_posterior_copy['num_distribution_draws'] = num_distribution_draw
        kappa_pdf, kappa_bin_edges = read_kappa_pdf(name, kappa_bins)
        kwargs_posterior_copy['kappa_pdf'] = kappa_pdf
        kwargs_posterior_copy['kappa_bin_edges'] = kappa_bin_edges
        kwargs_posterior_copy['mst_ifu'] = ifu_mst_sampling_separate
        ############# PUT M SOLITON ONCE YOU HAVE IT
        theta_E, r_eff, gamma_pl, z_lens, z_source, sigma_sis = properties(name, file_name_slacs, partial_name=False)
        #############
        kwargs_posterior_copy['lambda_scaling_property'] = r_eff/theta_E - 1
        if not ifu_mst_sampling_separate:  # if we sample the amplitude normalization of the IFU spectra separately, we do not need to include a systematic error on top
            kwargs_posterior_copy['sigma_sys_error_include'] = sigma_v_systematics

        if name in ifu_names_quality:
            kwargs_ifu_quality_list.append(kwargs_posterior_copy)
        if name in ifu_names_all:
            kwargs_ifu_all_list.append(kwargs_posterior_copy)

kwargs_sdss_all_list = []
kwargs_sdss_quality_list = []

file = open(os.path.join(path2Analysis, path2slacs, sdss_sample_file_name) , 'rb')
posterior_sdss_list = pickle.load(file)
file.close()
for kwargs_posterior in posterior_sdss_list:
    name = kwargs_posterior['name']
    #if name == 'SDSSJ0216-0813 ':
    #    name  = 'SDSSJ0216-0813'
    if name in sdss_names_all:
        kwargs_posterior_copy = copy.deepcopy(kwargs_posterior)
        if 'flag_ifu' in kwargs_posterior_copy:
            del kwargs_posterior_copy['flag_ifu']
        if 'flag_imaging' in kwargs_posterior_copy:
            del kwargs_posterior_copy['flag_imaging']
        kwargs_posterior_copy['num_distribution_draws'] = num_distribution_draw
        kappa_pdf, kappa_bin_edges = read_kappa_pdf(name, kappa_bins)
        kwargs_posterior_copy['kappa_pdf'] = kappa_pdf
        kwargs_posterior_copy['kappa_bin_edges'] = kappa_bin_edges
        kwargs_posterior_copy['sigma_sys_error_include'] = sigma_v_systematics
        ################### INSERT M SOLITON ONCE YOU HAVE IT
        theta_E, r_eff, gamma_pl, z_lens, z_source, sigma_sis = properties(name, file_name_slacs, partial_name=False)
        ###################
        kwargs_posterior_copy['lambda_scaling_property'] = r_eff/theta_E - 1

        if name in sdss_names_quality:
            kwargs_sdss_quality_list.append(kwargs_posterior_copy)
        if name in sdss_names_all:
            kwargs_sdss_all_list.append(kwargs_posterior_copy)

if all_slacs_sample is True:
    num_slacs = len(kwargs_sdss_all_list)
    num_ifu = len(kwargs_ifu_all_list)
else:
    num_slacs = len(kwargs_sdss_quality_list)
    num_ifu = len(kwargs_ifu_quality_list)

# FLCDM = Flat Lambda CDM
cosmology = 'FLCDM'  # available models: 'FLCDM', "FwCDM", "w0waCDM", "oLCDM"
# parameters are: 'h0', 'om', 'ok', 'w', 'w0', 'wa'
# w0 is static DE eq. state parameter (typically w0 = -1) whereas wa is the (possible)
# dynamical eq. state parameter
# there is also the option to sample the post-newtonian parameter gamma_ppn

beta_inf_min, beta_inf_max = 0, 1
a_ani_min, a_ani_max, a_an_mean = 0.1, 5, 1

## This should be the values used for the flat prior, these are the population parameters
# We don't use them all in our project
kwargs_lower_cosmo = {'h0': 0, 'om': 0.05}
kwargs_lower_lens = {'kappa_ext': -0.1, 'kappa_ext_sigma': 0.}
############### Original (with also lambda MSD
#  kwargs_lower_lens = {'lambda_mst': 0.5, 'lambda_mst_sigma': 0.001, 'kappa_ext': -0.1, 'kappa_ext_sigma': 0., 'lambda_ifu': 0.5, 'lambda_ifu_sigma': 0.01, 'alpha_lambda': -1}
################
kwargs_lower_kin = {'a_ani': a_ani_min, 'a_ani_sigma': 0.01, 'beta_inf': beta_inf_min, 'beta_inf_sigma': 0.001, 'sigma_v_sys_error': 0.01}

kwargs_upper_cosmo = {'h0': 150, 'om': 0.5}
kwargs_upper_lens = {'kappa_ext': 0.5, 'kappa_ext_sigma': 0.5}
############# Original
#  kwargs_upper_lens = {'lambda_mst': 1.5, 'lambda_mst_sigma': .5, 'kappa_ext': 0.5, 'kappa_ext_sigma': 0.5, 'lambda_ifu': 1.5, 'lambda_ifu_sigma': 0.5, 'alpha_lambda': 1}
############
kwargs_upper_kin = {'a_ani': a_ani_max, 'a_ani_sigma': 1. ,'beta_inf': beta_inf_max, 'beta_inf_sigma': 1, 'sigma_v_sys_error': 0.5}

# these values are held fixed throughout the entire sampling (optinal to add here)
kwargs_fixed_cosmo = {}
############ Originally there was nothing, I added fix MSD parameters
kwargs_fixed_lens = {'lambda_mst' : 1, 'lambda_mst_sigma' : 0, 'lambda_ifu' : 1, 'lambda_ifu_sigma' : 0, 'alpha_lambda' : 0}
############
# for the 'GOM' model is beta_inf should be free or fixed=1
kwargs_fixed_kin = {}

kwargs_mean_start = {'kwargs_cosmo': {'h0': 70, 'om': 0.3},
                     #'kwargs_lens': {'lambda_mst': 1., 'lambda_mst_sigma': .05, 'lambda_ifu': 1, 'lambda_ifu_sigma': 0.05, 'alpha_lambda': 0},
                     'kwargs_kin': {'a_ani': 1, 'a_ani_sigma': 0.1, 'beta_inf': 0.8, 'beta_inf_sigma': 0.1, 'sigma_v_sys_error': 0.05}}

kwargs_sigma_start = {'kwargs_cosmo': {'h0': 10, 'om': 0.1},
                     #'kwargs_lens': {'lambda_mst': .1, 'lambda_mst_sigma': .05, 'lambda_ifu': 0.1, 'lambda_ifu_sigma': 0.05, 'alpha_lambda': 0.1},
                     'kwargs_kin': {'a_ani': 0.3, 'a_ani_sigma': 0.1, 'beta_inf': 0.5, 'beta_inf_sigma': 0.1, 'sigma_v_sys_error': 0.05}}

kwargs_bounds = {'kwargs_lower_cosmo': kwargs_lower_cosmo,
                'kwargs_lower_lens': kwargs_lower_lens,
                'kwargs_lower_kin': kwargs_lower_kin,
                'kwargs_upper_cosmo': kwargs_upper_cosmo,
                'kwargs_upper_lens': kwargs_upper_lens,
                'kwargs_upper_kin': kwargs_upper_kin,
                'kwargs_fixed_cosmo': kwargs_fixed_cosmo,
                'kwargs_fixed_lens': kwargs_fixed_lens,
                'kwargs_fixed_kin': kwargs_fixed_kin}
############## Change the mst sampling here (Done, changed to False and None from True and GAUSSIAN
############## and comment alpha_lambda_sampling
kwargs_sampler = {'lambda_mst_sampling': False,
                 'lambda_mst_distribution': 'None',
                 'anisotropy_sampling': True,
                 'kappa_ext_sampling': False,  # we do not globally sample a external convergence distribution but instead are using the individually derived p(kappa) products.
                 'kappa_ext_distribution': 'GAUSSIAN',
                 'sigma_v_systematics': sigma_v_systematics,
                 'anisotropy_model': anisotropy_model,
                 'anisotropy_distribution': anisotropy_distribution,
                 #'alpha_lambda_sampling': lambda_slope,
                 #'log_scatter': log_scatter,  # you can either sample the parameters in log-space or chose a 1/x prior in the CustomPrior() class. We chose the CustomPrior to specify which parameters have a U(log()) prior.
                 'interpolate_cosmo': True, 'num_redshift_interp': 100,
                 'custom_prior': CustomPrior(planck_prior=planck_prior, log_scatter=log_scatter)
                 }
#################

# here we sample the tdcosmo data only
kwargs_sampler_tdcosmo = copy.deepcopy(kwargs_sampler)
kwargs_sampler_tdcosmo['sigma_v_systematics'] = False  # we do not add a systematic uncertainty on top of the data products for the TDCOSMO sample and thus do not need to sample this parameter.

## With the tight prior on H0
# cosmology is just FLCDM, kwargs are just dictionaries with instructions for sampling (CustomPrior is on kwargs_sampler_tdcosmo)
# the posterior list is just the data for every galaxy, with also the Ddt single lens posterior histogram
mcmc_sampler_tdcosmo = MCMCSampler(kwargs_likelihood_list=tdcosmo_posterior_list,
                                   cosmology=cosmology, kwargs_bounds=kwargs_bounds, **kwargs_sampler_tdcosmo)
# Set up the backend
# Don't forget to clear it in case the file already exists
# Here is where the MCMC is performed, be careful!

########## Uncomment the following to make the TDCOSMO only analysis
# Put the file in the current directory
filename = "tdcosmo_chain"+lambda_prefix+".h5"
backend = emcee.backends.HDFBackend(filename)
if run_chains is True:
    mcmc_samples_tdcosmo, log_prob_cosmo = mcmc_sampler_tdcosmo.mcmc_emcee(n_walkers, n_burn, n_run,
                                                                               kwargs_mean_start, kwargs_sigma_start,
                                                                               continue_from_backend=continue_from_backend, backend=backend)
else:
    mcmc_samples_tdcosmo = backend.get_chain(discard=n_burn, flat=True, thin=1)
    log_prob_cosmo = backend.get_log_prob(discard=n_burn, flat=True, thin=1)

## corner is a matplotlib used for multiple plots
corner.corner(mcmc_samples_tdcosmo, show_titles=True, labels=mcmc_sampler_tdcosmo.param_names(latex_style=True))
plt.show()
####################

# This is the full analysis with all the data
if all_slacs_sample is True:
    lens_list = tdcosmo_posterior_list + kwargs_sdss_all_list + kwargs_ifu_all_list
else:
    lens_list = tdcosmo_posterior_list + kwargs_sdss_quality_list + kwargs_ifu_quality_list

mcmc_sampler_slacs_ifu = MCMCSampler(kwargs_likelihood_list=lens_list,
                                   cosmology=cosmology, kwargs_bounds=kwargs_bounds,
                                   ######### Changed GAUSSIAN to None in lambda_ifu_distribution
                                     lambda_ifu_sampling=ifu_mst_sampling_separate, lambda_ifu_distribution='None',
                                     **kwargs_sampler)

ndim = mcmc_sampler_slacs_ifu.param.num_param
# Set up the backend
# Don't forget to clear it in case the file already exists

# Here you make the chain run for all lenses, uncomment

filename = "tdcosmo_slacs_ifu_chain"+model_prefix+lambda_prefix+".h5"
backend = emcee.backends.HDFBackend(filename)
if run_chains is True:
    mcmc_samples_slacs_ifu, log_prob_slacs_ifu = mcmc_sampler_slacs_ifu.mcmc_emcee(n_walkers, n_burn, n_run,
                                                                               kwargs_mean_start, kwargs_sigma_start,
                                                                               continue_from_backend=continue_from_backend, backend=backend)
else:
    mcmc_samples_slacs_ifu = backend.get_chain(discard=n_burn, flat=True, thin=1)
    log_prob_slacs_ifu = backend.get_log_prob(discard=n_burn, flat=True, thin=1)

mcmc_samples_slacs_ifu_plot = copy.deepcopy(mcmc_samples_slacs_ifu)
#  if blinded_plot is True:
#      mcmc_samples_slacs_ifu_plot[:, 0] *= 73 / np.mean(mcmc_samples_slacs_ifu_plot[:, 0])
#      mcmc_samples_slacs_ifu_plot[:, 2] *= 1 / np.mean(mcmc_samples_slacs_ifu_plot[:, 2])

corner.corner(mcmc_samples_slacs_ifu_plot, show_titles=True, labels=mcmc_sampler_slacs_ifu.param_names(latex_style=True))
plt.show()
