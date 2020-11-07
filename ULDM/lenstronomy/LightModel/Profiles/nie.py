import numpy as np
import lenstronomy.Util.param_util as param_util
from lenstronomy.Util import util
from lenstronomy.LightModel.Profiles.profile_base import LightProfileBase


class NIE(LightProfileBase):
    """
    non-divergent isothermal ellipse (projected)
    This is effectively the convergence profile of the NIE lens model with an amplitude 'amp' rather than an Einstein
    radius 'theta_E'
    """
    param_names = ['amp', 'e1', 'e2', 's_scale', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'e1': -0.5, 'e2': -0.5, 's_scale': 0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 100, 'e1': 0.5, 'e2': 0.5, 's_scale': 100, 'center_x': 100, 'center_y': 100}

    def function(self, x, y, amp, e1, e2, s_scale, center_x=0, center_y=0):
        """

        :param x: x-coordinate
        :param y: y-coordinate
        :param amp: surface brightness normalization
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param s_scale: smoothing scale (square averaged of minor and major axis)
        :param center_x: center of profile
        :param center_y: center of profile
        :return: surface brightness of NIE profile
        """

        x_ = x - center_x
        y_ = y - center_y
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_G)
        s = s_scale * np.sqrt((1 + q ** 2) / (2 * q ** 2))
        f_ = amp/2. * (q**2 * (s**2 + x__**2) + y__**2)**(-1./2)
        return f_
