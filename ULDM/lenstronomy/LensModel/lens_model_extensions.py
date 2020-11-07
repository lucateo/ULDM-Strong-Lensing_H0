import numpy as np
import lenstronomy.Util.util as util
import lenstronomy.Util.mask_util as mask_util
import lenstronomy.Util.param_util as param_util


class LensModelExtensions(object):
    """
    class with extension routines not part of the LensModel core routines
    """
    def __init__(self, lensModel):
        """

        :param lensModel: instance of the LensModel() class, or with same functionalities.
        In particular, the following definitions are required to execute all functionalities presented in this class:
        def ray_shooting()
        def magnification()
        def kappa()
        def alpha()
        def hessian()

        """
        self._lensModel = lensModel

    def magnification_finite(self, x_pos, y_pos, kwargs_lens, source_sigma=0.003, window_size=0.1, grid_number=100,
                             shape="GAUSSIAN", polar_grid=False, aspect_ratio=0.5):
        """
        returns the magnification of an extended source with Gaussian light profile
        :param x_pos: x-axis positons of point sources
        :param y_pos: y-axis position of point sources
        :param kwargs_lens: lens model kwargs
        :param source_sigma: Gaussian sigma in arc sec in source
        :param window_size: size of window to compute the finite flux
        :param grid_number: number of grid cells per axis in the window to numerically comute the flux
        :return: numerically computed brightness of the sources
        """

        mag_finite = np.zeros_like(x_pos)
        deltaPix = float(window_size)/grid_number
        if shape == 'GAUSSIAN':
            from lenstronomy.LightModel.Profiles.gaussian import Gaussian
            quasar = Gaussian()
        elif shape == 'TORUS':
            import lenstronomy.LightModel.Profiles.ellipsoid as quasar
        else:
            raise ValueError("shape %s not valid for finite magnification computation!" % shape)
        x_grid, y_grid = util.make_grid(numPix=grid_number, deltapix=deltaPix, subgrid_res=1)

        if polar_grid is True:
            a = window_size*0.5
            b = window_size*0.5*aspect_ratio
            ellipse_inds = (x_grid*a**-1) **2 + (y_grid*b**-1) **2 <= 1
            x_grid, y_grid = x_grid[ellipse_inds], y_grid[ellipse_inds]

        for i in range(len(x_pos)):
            ra, dec = x_pos[i], y_pos[i]

            center_x, center_y = self._lensModel.ray_shooting(ra, dec, kwargs_lens)

            if polar_grid is True:
                theta = np.arctan2(dec,ra)
                xcoord, ycoord = util.rotate(x_grid, y_grid, theta)
            else:
                xcoord, ycoord = x_grid, y_grid

            betax, betay = self._lensModel.ray_shooting(xcoord + ra, ycoord + dec, kwargs_lens)

            I_image = quasar.function(betax, betay, 1., source_sigma, center_x, center_y)
            mag_finite[i] = np.sum(I_image) * deltaPix**2
        return mag_finite

    def zoom_source(self, x_pos, y_pos, kwargs_lens, source_sigma=0.003, window_size=0.1, grid_number=100,
                             shape="GAUSSIAN"):
        """
        computes the surface brightness on an image with a zoomed window

        :param x_pos: angular coordinate of center of image
        :param y_pos: angular coordinate of center of image
        :param kwargs_lens: lens model parameter list
        :param source_sigma: source size (in angular units)
        :param window_size: window size in angular units
        :param grid_number: number of grid points per axis
        :param shape: string, shape of source, supports 'GAUSSIAN' and 'TORUS
        :return: 2d numpy array
        """
        deltaPix = float(window_size) / grid_number
        if shape == 'GAUSSIAN':
            from lenstronomy.LightModel.Profiles.gaussian import Gaussian
            quasar = Gaussian()
        elif shape == 'TORUS':
            import lenstronomy.LightModel.Profiles.ellipsoid as quasar
        else:
            raise ValueError("shape %s not valid for finite magnification computation!" % shape)
        x_grid, y_grid = util.make_grid(numPix=grid_number, deltapix=deltaPix, subgrid_res=1)
        center_x, center_y = self._lensModel.ray_shooting(x_pos, y_pos, kwargs_lens)
        betax, betay = self._lensModel.ray_shooting(x_grid + x_pos, y_grid + y_pos, kwargs_lens)
        image = quasar.function(betax, betay, 1., source_sigma, center_x, center_y)
        return util.array2image(image)

    def critical_curve_tiling(self, kwargs_lens, compute_window=5, start_scale=0.5, max_order=10):
        """

        :param kwargs_lens: lens model keyword argument list
        :param compute_window: total window in the image plane where to search for critical curves
        :param start_scale: float, angular scale on which to start the tiling from (if there are two distinct curves in
         a region, it might only find one.
        :param max_order: int, maximum order in the tiling to compute critical curve triangles
        :return: list of positions representing coordinates of the critical curve (in RA and DEC)
        """
        numPix = int(compute_window / start_scale)
        x_grid_init, y_grid_init = util.make_grid(numPix, deltapix=start_scale, subgrid_res=1)
        mag_init = util.array2image(self._lensModel.magnification(x_grid_init, y_grid_init, kwargs_lens))
        x_grid_init = util.array2image(x_grid_init)
        y_grid_init = util.array2image(y_grid_init)

        ra_crit_list = []
        dec_crit_list = []
        # iterate through original triangles and return ra_crit, dec_crit list
        for i in range(numPix-1):
            for j in range(numPix-1):
                edge1 = [x_grid_init[i, j], y_grid_init[i, j], mag_init[i, j]]
                edge2 = [x_grid_init[i+1, j+1], y_grid_init[i+1, j+1], mag_init[i+1, j+1]]
                edge_90_1 = [x_grid_init[i, j+1], y_grid_init[i, j+1], mag_init[i, j+1]]
                edge_90_2 = [x_grid_init[i+1, j], y_grid_init[i+1, j], mag_init[i+1, j]]
                ra_crit, dec_crit = self._tiling_crit(edge1, edge2, edge_90_1, max_order=max_order,
                                                      kwargs_lens=kwargs_lens)
                ra_crit_list += ra_crit  # list addition
                dec_crit_list += dec_crit  # list addition
                ra_crit, dec_crit = self._tiling_crit(edge1, edge2, edge_90_2, max_order=max_order,
                                                      kwargs_lens=kwargs_lens)
                ra_crit_list += ra_crit  # list addition
                dec_crit_list += dec_crit  # list addition
        return np.array(ra_crit_list), np.array(dec_crit_list)

    def _tiling_crit(self, edge1, edge2, edge_90, max_order, kwargs_lens):
        """
        tiles a rectangular triangle and compares the signs of the magnification

        :param edge1: [ra_coord, dec_coord, magnification]
        :param edge2: [ra_coord, dec_coord, magnification]
        :param edge_90: [ra_coord, dec_coord, magnification]
        :param max_order: maximal order to fold triangle
        :param kwargs_lens: lens model keyword argument list
        :return:
        """
        ra_1, dec_1, mag_1 = edge1
        ra_2, dec_2, mag_2 = edge2
        ra_3, dec_3, mag_3 = edge_90
        sign_list = np.sign([mag_1, mag_2, mag_3])
        if sign_list[0] == sign_list[1] and sign_list[0] == sign_list[2]:  # if all signs are the same
            return [], []
        else:
            # split triangle along the long axis
            # execute tiling twice
            # add ra_crit and dec_crit together
            # if max depth has been reached, return the mean value in the triangle
            max_order -= 1
            if max_order <= 0:
                return [(ra_1 + ra_2 + ra_3)/3], [(dec_1 + dec_2 + dec_3)/3]
            else:
                # split triangle
                ra_90_ = (ra_1 + ra_2)/2  # find point in the middle of the long axis to split triangle
                dec_90_ = (dec_1 + dec_2)/2
                mag_90_ = self._lensModel.magnification(ra_90_, dec_90_, kwargs_lens)
                edge_90_ = [ra_90_, dec_90_, mag_90_]
                ra_crit, dec_crit = self._tiling_crit(edge1=edge_90, edge2=edge1, edge_90=edge_90_, max_order=max_order,
                                                      kwargs_lens=kwargs_lens)
                ra_crit_2, dec_crit_2 = self._tiling_crit(edge1=edge_90, edge2=edge2, edge_90=edge_90_, max_order=max_order,
                                                          kwargs_lens=kwargs_lens)
                ra_crit += ra_crit_2
                dec_crit += dec_crit_2
                return ra_crit, dec_crit

    def critical_curve_caustics(self, kwargs_lens, compute_window=5, grid_scale=0.01):
        """

        :param kwargs_lens: lens model kwargs
        :param compute_window: window size in arcsec where the critical curve is computed
        :param grid_scale: numerical grid spacing of the computation of the critical curves
        :return: lists of ra and dec arrays corresponding to different disconnected critical curves and their caustic counterparts

        """
        numPix = int(compute_window / grid_scale)
        x_grid_high_res, y_grid_high_res = util.make_grid(numPix, deltapix=grid_scale, subgrid_res=1)
        mag_high_res = util.array2image(self._lensModel.magnification(x_grid_high_res, y_grid_high_res, kwargs_lens))

        ra_crit_list = []
        dec_crit_list = []
        ra_caustic_list = []
        dec_caustic_list = []

        import matplotlib.pyplot as plt
        cs = plt.contour(util.array2image(x_grid_high_res), util.array2image(y_grid_high_res), mag_high_res, [0],
                         alpha=0.0)
        paths = cs.collections[0].get_paths()
        for i, p in enumerate(paths):
            v = p.vertices
            ra_points = v[:, 0]
            dec_points = v[:, 1]
            ra_crit_list.append(ra_points)
            dec_crit_list.append(dec_points)
            ra_caustics, dec_caustics = self._lensModel.ray_shooting(ra_points, dec_points, kwargs_lens)
            ra_caustic_list.append(ra_caustics)
            dec_caustic_list.append(dec_caustics)
        plt.cla()
        return ra_crit_list, dec_crit_list, ra_caustic_list, dec_caustic_list

    def hessian_eigenvectors(self, x, y, kwargs_lens, diff=None):
        """
        computes magnification eigenvectors at position (x, y)

        :param x: x-position
        :param y: y-position
        :param kwargs_lens: lens model keyword arguments
        :return: radial stretch, tangential stretch
        """
        f_xx, f_xy, f_yx, f_yy = self._lensModel.hessian(x, y, kwargs_lens, diff=diff)
        if isinstance(x, int) or isinstance(x, float):
            A = np.array([[1-f_xx, f_xy], [f_yx, 1-f_yy]])
            w, v = np.linalg.eig(A)
            v11, v12, v21, v22 = v[0, 0], v[0, 1], v[1, 0], v[1, 1]
            w1, w2 = w[0], w[1]
        else:
            w1, w2, v11, v12, v21, v22 = np.empty(len(x), dtype=float), np.empty(len(x), dtype=float), np.empty_like(x), np.empty_like(x), np.empty_like(x), np.empty_like(x)
            for i in range(len(x)):
                A = np.array([[1 - f_xx[i], f_xy[i]], [f_yx[i], 1 - f_yy[i]]])
                w, v = np.linalg.eig(A)
                w1[i], w2[i] = w[0], w[1]
                v11[i], v12[i], v21[i], v22[i] = v[0, 0], v[0, 1], v[1, 0], v[1, 1]
        return w1, w2, v11, v12, v21, v22

    def radial_tangential_stretch(self, x, y, kwargs_lens, diff=None, ra_0=0, dec_0=0,
                                  coordinate_frame_definitions=False):
        """
        computes the radial and tangential stretches at a given position

        :param x: x-position
        :param y: y-position
        :param kwargs_lens: lens model keyword arguments
        :param diff: float or None, finite average differential scale
        :return: radial stretch, tangential stretch
        """
        w1, w2, v11, v12, v21, v22 = self.hessian_eigenvectors(x, y, kwargs_lens, diff=diff)
        v_x, v_y = x - ra_0, y - dec_0

        prod_v1 = v_x*v11 + v_y*v12
        prod_v2 = v_x*v21 + v_y*v22
        if isinstance(x, int) or isinstance(x, float):
            if (coordinate_frame_definitions is True and abs(prod_v1) >= abs(prod_v2)) or (coordinate_frame_definitions is False and w1 >= w2):
            #if w1 > w2:
            #if abs(prod_v1) > abs(prod_v2):  # radial vector has larger scalar product to the zero point
                lambda_rad = 1. / w1
                lambda_tan = 1. / w2
                v1_rad, v2_rad = v11, v12
                v1_tan, v2_tan = v21, v22
                prod_r = prod_v1
            else:
                lambda_rad = 1. / w2
                lambda_tan = 1. / w1
                v1_rad, v2_rad = v21, v22
                v1_tan, v2_tan = v11, v12
                prod_r = prod_v2
            if prod_r < 0:  # if radial eigenvector points towards the center
                v1_rad, v2_rad = -v1_rad, -v2_rad
            if v1_rad * v2_tan - v2_rad * v1_tan < 0:  # cross product defines orientation of the tangential eigenvector
                v1_tan *= -1
                v2_tan *= -1

        else:
            lambda_rad, lambda_tan, v1_rad, v2_rad, v1_tan, v2_tan = np.empty(len(x), dtype=float), np.empty(len(x), dtype=float), np.empty_like(x), np.empty_like(x), np.empty_like(x), np.empty_like(x)
            for i in range(len(x)):
                if (coordinate_frame_definitions is True and abs(prod_v1[i]) >= abs(prod_v2[i])) or (
                        coordinate_frame_definitions is False and w1[i] >= w2[i]):
                #if w1[i] > w2[i]:
                    lambda_rad[i] = 1. / w1[i]
                    lambda_tan[i] = 1. / w2[i]
                    v1_rad[i], v2_rad[i] = v11[i], v12[i]
                    v1_tan[i], v2_tan[i] = v21[i], v22[i]
                    prod_r = prod_v1[i]
                else:
                    lambda_rad[i] = 1. / w2[i]
                    lambda_tan[i] = 1. / w1[i]
                    v1_rad[i], v2_rad[i] = v21[i], v22[i]
                    v1_tan[i], v2_tan[i] = v11[i], v12[i]
                    prod_r = prod_v2[i]
                if prod_r < 0:  # if radial eigenvector points towards the center
                    v1_rad[i], v2_rad[i] = -v1_rad[i], -v2_rad[i]
                if v1_rad[i] * v2_tan[i] - v2_rad[i] * v1_tan[i] < 0:  # cross product defines orientation of the tangential eigenvector
                    v1_tan[i] *= -1
                    v2_tan[i] *= -1

        return lambda_rad, lambda_tan, v1_rad, v2_rad, v1_tan, v2_tan

    def radial_tangential_differentials(self, x, y, kwargs_lens, center_x=0, center_y=0, smoothing_3rd=0.001,
                                        smoothing_2nd=None):
        """
        computes the differentials in stretches and directions

        :param x: x-position
        :param y: y-position
        :param kwargs_lens: lens model keyword arguments
        :param center_x: x-coord of center towards which the rotation direction is defined
        :param center_y: x-coord of center towards which the rotation direction is defined
        :param smoothing_3rd: finite differential length of third order in units of angle
        :param smoothing_2nd: float or None, finite average differential scale of Hessian
        :return:
        """
        lambda_rad, lambda_tan, v1_rad, v2_rad, v1_tan, v2_tan = self.radial_tangential_stretch(x, y, kwargs_lens,
                                                                                                diff=smoothing_2nd,
                                                                                                ra_0=center_x, dec_0=center_y,
                                                                                                coordinate_frame_definitions=True)
        x0 = x - center_x
        y0 = y - center_y

        # computing angle of tangential vector in regard to the defined coordinate center
        cos_angle = (v1_tan * x0 + v2_tan * y0) / np.sqrt((x0 ** 2 + y0 ** 2) * (v1_tan ** 2 + v2_tan ** 2))# * np.sign(v1_tan * y0 - v2_tan * x0)
        orientation_angle = np.arccos(cos_angle) - np.pi / 2

        # computing differentials in tangential and radial directions
        dx_tan = x + smoothing_3rd * v1_tan
        dy_tan = y + smoothing_3rd * v2_tan
        lambda_rad_dtan, lambda_tan_dtan, v1_rad_dtan, v2_rad_dtan, v1_tan_dtan, v2_tan_dtan = self.radial_tangential_stretch(dx_tan, dy_tan, kwargs_lens, diff=smoothing_2nd,
                                                                                                                              ra_0=center_x, dec_0=center_y, coordinate_frame_definitions=True)
        dx_rad = x + smoothing_3rd * v1_rad
        dy_rad = y + smoothing_3rd * v2_rad
        lambda_rad_drad, lambda_tan_drad, v1_rad_drad, v2_rad_drad, v1_tan_drad, v2_tan_drad = self.radial_tangential_stretch(
            dx_rad, dy_rad, kwargs_lens, diff=smoothing_2nd, ra_0=center_x, dec_0=center_y, coordinate_frame_definitions=True)

        # eigenvalue differentials in tangential and radial direction
        dlambda_tan_dtan = (lambda_tan_dtan - lambda_tan) / smoothing_3rd# * np.sign(v1_tan * y0 - v2_tan * x0)
        dlambda_tan_drad = (lambda_tan_drad - lambda_tan) / smoothing_3rd# * np.sign(v1_rad * x0 + v2_rad * y0)
        dlambda_rad_drad = (lambda_rad_drad - lambda_rad) / smoothing_3rd# * np.sign(v1_rad * x0 + v2_rad * y0)
        dlambda_rad_dtan = (lambda_rad_dtan - lambda_rad) / smoothing_3rd# * np.sign(v1_rad * x0 + v2_rad * y0)

        # eigenvector direction differentials in tangential and radial direction
        cos_dphi_tan_dtan = v1_tan * v1_tan_dtan + v2_tan * v2_tan_dtan #/ (np.sqrt(v1_tan**2 + v2_tan**2) * np.sqrt(v1_tan_dtan**2 + v2_tan_dtan**2))
        norm = np.sqrt(v1_tan**2 + v2_tan**2) * np.sqrt(v1_tan_dtan**2 + v2_tan_dtan**2)
        cos_dphi_tan_dtan /= norm
        arc_cos_dphi_tan_dtan = np.arccos(np.abs(np.minimum(cos_dphi_tan_dtan, 1)))
        dphi_tan_dtan = arc_cos_dphi_tan_dtan / smoothing_3rd

        cos_dphi_tan_drad = v1_tan * v1_tan_drad + v2_tan * v2_tan_drad  # / (np.sqrt(v1_tan ** 2 + v2_tan ** 2) * np.sqrt(v1_tan_drad ** 2 + v2_tan_drad ** 2))
        norm = np.sqrt(v1_tan ** 2 + v2_tan ** 2) * np.sqrt(v1_tan_drad ** 2 + v2_tan_drad ** 2)
        cos_dphi_tan_drad /= norm
        arc_cos_dphi_tan_drad = np.arccos(np.abs(np.minimum(cos_dphi_tan_drad, 1)))
        dphi_tan_drad = arc_cos_dphi_tan_drad / smoothing_3rd

        cos_dphi_rad_drad = v1_rad * v1_rad_drad + v2_rad * v2_rad_drad #/ (np.sqrt(v1_rad**2 + v2_rad**2) * np.sqrt(v1_rad_drad**2 + v2_rad_drad**2))
        norm = np.sqrt(v1_rad**2 + v2_rad**2) * np.sqrt(v1_rad_drad**2 + v2_rad_drad**2)
        cos_dphi_rad_drad /= norm
        cos_dphi_rad_drad = np.minimum(cos_dphi_rad_drad, 1)
        dphi_rad_drad = np.arccos(cos_dphi_rad_drad) / smoothing_3rd

        cos_dphi_rad_dtan = v1_rad * v1_rad_dtan + v2_rad * v2_rad_dtan # / (np.sqrt(v1_rad ** 2 + v2_rad ** 2) * np.sqrt(v1_rad_dtan ** 2 + v2_rad_dtan ** 2))
        norm = np.sqrt(v1_rad ** 2 + v2_rad ** 2) * np.sqrt(v1_rad_dtan ** 2 + v2_rad_dtan ** 2)
        cos_dphi_rad_dtan /= norm
        cos_dphi_rad_dtan = np.minimum(cos_dphi_rad_dtan, 1)
        dphi_rad_dtan = np.arccos(cos_dphi_rad_dtan) / smoothing_3rd

        return lambda_rad, lambda_tan, orientation_angle, dlambda_tan_dtan, dlambda_tan_drad, dlambda_rad_drad, dlambda_rad_dtan, dphi_tan_dtan, dphi_tan_drad, dphi_rad_drad, dphi_rad_dtan

    def curved_arc_estimate(self, x, y, kwargs_lens, smoothing=None, smoothing_3rd=0.001):
        """
        performs the estimation of the curved arc description at a particular position of an arbitrary lens profile

        :param x: float, x-position where the estimate is provided
        :param y: float, y-position where the estimate is provided
        :param kwargs_lens: lens model keyword arguments
        :return: keyword argument list corresponding to a CURVED_ARC profile at (x, y) given the initial lens model
        """
        radial_stretch, tangential_stretch, v_rad1, v_rad2, v_tang1, v_tang2 = self.radial_tangential_stretch(x, y, kwargs_lens, diff=smoothing)
        dx_tang = x + smoothing_3rd * v_tang1
        dy_tang = y + smoothing_3rd * v_tang2
        rad_dt, tang_dt, v_rad1_dt, v_rad2_dt, v_tang1_dt, v_tang2_dt = self.radial_tangential_stretch(dx_tang, dy_tang,
                                                                                                       kwargs_lens,
                                                                                                       diff=smoothing)
        d_tang1 = v_tang1_dt - v_tang1
        d_tang2 = v_tang2_dt - v_tang2
        delta = np.sqrt(d_tang1**2 + d_tang2**2)
        if delta > 1:
            d_tang1 = v_tang1_dt + v_tang1
            d_tang2 = v_tang2_dt + v_tang2
            delta = np.sqrt(d_tang1 ** 2 + d_tang2 ** 2)
        curvature = delta / smoothing_3rd
        direction = np.arctan2(v_rad2 * np.sign(v_rad1 * x + v_rad2 * y), v_rad1 * np.sign(v_rad1 * x + v_rad2 * y))
        #direction = np.arctan2(v_rad2, v_rad1)
        kwargs_arc = {'radial_stretch': radial_stretch,
                      'tangential_stretch': tangential_stretch,
                      'curvature': curvature,
                      'direction': direction,
                      'center_x': x, 'center_y': y}
        return kwargs_arc
