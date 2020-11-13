#########################
# Trying ULDM class with random things
#
#########################

# some standard python imports #
import numpy as np
import copy
import matplotlib.pyplot as plt

# import the LensModel class #
from lenstronomy.LensModel.lens_model import LensModel

# specify the choice of lens models #
lens_model_list = ['SIS','SHEAR']
redshift_list= [0.5, 0.5]
z_source = 1.5

# setup lens model class with the list of lens models #
lensModel = LensModel(lens_model_list=lens_model_list, z_source=z_source, lens_redshift_list=redshift_list, multi_plane=True)

# define parameter values of lens models #
kwargs_uldm = {'theta_c': 1.1, 'alpha_c' : 0.5, 'center_x': 0.1, 'center_y': 0}
kwargs_shear = {'gamma1': -0.01, 'gamma2': .03}
kwargs_sis = {'theta_E': 0.1, 'center_x': 1., 'center_y': -0.1}
kwargs_lens = [kwargs_sis, kwargs_shear]

# image plane coordinate #
theta_ra, theta_dec = .9, .4

# source plane coordinate #
beta_ra, beta_dec = lensModel.ray_shooting(theta_ra, theta_dec, kwargs_lens)
# Fermat potential #
fermat_pot = lensModel.fermat_potential(x_image=theta_ra, y_image=theta_dec, x_source=beta_ra, y_source=beta_dec, kwargs_lens=kwargs_lens)

# Magnification #
mag = lensModel.magnification(theta_ra, theta_dec, kwargs_lens)
print(mag)
# arrival time relative to a straight path through (0,0) #
dt = lensModel.arrival_time(theta_ra, theta_dec, kwargs_lens)

print(dt)
