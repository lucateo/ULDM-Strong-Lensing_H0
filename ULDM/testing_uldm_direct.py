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

z_lens = 0.5
# specify source redshift
z_source = 1.5
# setup lens model class with the list of lens models
lensModel = LensModel(lens_model_list=lens_model_list, z_lens = z_lens, z_source=z_source, cosmo=cosmo)
lens_cosmo = LensCosmo(z_lens = z_lens, z_source = z_source, cosmo = cosmo)

thetac, alphac, rho_0 = lens_cosmo.uldm_phys2angles_rho0(-25,12)
print(rho_0, alphac, thetac)

# define parameter values of lens models
kwargs_uldm = {'theta_c': 1.1, 'alpha_c' : 0.5, 'center_x': 0.1, 'center_y': 0}
kwargs_lens = [kwargs_uldm]

alpha = lensModel.alpha(0.1, 0.2, kwargs_lens)
print(alpha)

flexion_lens = lensModel.flexion(1,2,kwargs_lens)
print(flexion_lens)

uldm_lens = Uldm()
integ = uldm_lens.lensing_Integral(0)
print(integ)

potential = uldm_lens.function(0.4, 0.7, 0.1, 1)
print(potential)

abba = uldm_lens.derivatives(0.4, 0.7, 0.1, 1.)
print(abba)

#  fermat_lens = lensModel.fermat_potential(1,2,kwargs_lens)
#  print(fermat_lens)

dt = lensModel.arrival_time(.9,.4,kwargs_lens)
print(dt)

# Compute the mass
mass = uldm_lens.mass_3d(100,thetac,rho_0)
# Convert the mass that is in angular units in M_sun units, recall that arcsec = arcsec2rad
# and that dd is in Mpc, whereas rho_0 is in M_sun/pc^3
mass = (const.arcsec * lens_cosmo.dd * 10**6)**3 * mass
mass = np.log10(mass)
print('mass = ', mass)

####### Plotting stuff
radius = np.arange(0.1, 100, 5.)
D_Lens = lens_cosmo.dd * 1000 # in kpc
plt.plot(radius, uldm_lens.density(radius/D_Lens /const.arcsec, rho_0, thetac))
plt.show()





