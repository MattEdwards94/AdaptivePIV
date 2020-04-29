import numpy as np
import PIV.distribution as distribution
import PIV.utilities as utilities
import math
import PIV.corr_window as corr_window
import PIV.dense_predictor as dense_predictor
import PIV.ensemble_solution as es
import PIV.multiGrid as multi_grid
import PIV
import matplotlib.pyplot as plt
from PIV.utilities import vprint, WS_for_iter
from mpl_toolkits import axes_grid1

ESSENTIAL, BASIC, TERSE = 1, 2, 3


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def amg_refinement(img, settings, plotting=False):

    init_h = settings.init_spacing
    min_thr = settings.min_thr
    n_iter = settings.n_iter_main
    n_iter_ref = settings.n_iter_ref

    # calc seeding
    img.calc_seed_density()

    min_sd = np.minimum(img.sd_IA, img.sd_IB)
    # if ends up being 0 or NaN, just assume it is some low value
    min_sd[min_sd == 0] = 0.0021
    min_sd[np.isnan(min_sd)] = 0.0021

    # Create initial grid
    mg = multi_grid.MultiGrid(img.dim, init_h, mask=img.mask)

    if plotting:
        mg.plot_grid(mask=img.mask)
        plt.gca().set_title("Initial Grid")

    # perform AIW to get window sizes and initial displacement grid
    mg.AIW(img)
    ws_first_iter = mg.interp_WS_unstructured(img.mask)*img.mask

    if plotting:
        fig, ax1, ax2 = utilities.plot_adjacent_images(ws_first_iter,
                                                       img.mask,
                                                       "WS",
                                                       "Dist after AIW",
                                                       cmap_a=None,
                                                       figsize=(35, 35),
                                                       axes_pad=0.5,
                                                       )
        ax1.set_xlim((0, 1280))
        ax2.set_xlim((0, 1280))
        ax1.set_ylim((0, 640))
        ax2.set_ylim((0, 640))
        mg.plot_distribution(handle=ax2)

    # Validate vectors
    mg.validation_NMT_8NN(idw=True)
    if plotting:
        fig, ax1, ax2 = utilities.plot_adjacent_images(ws_first_iter,
                                                       img.mask,
                                                       "WS",
                                                       "Validated dist after AIW",
                                                       cmap_a=None,
                                                       vminmax_a=[15, 30],
                                                       vminmax_b=[0, 1],
                                                       figsize=(35, 35),
                                                       axes_pad=0.5,
                                                       )
        ax1.set_xlim((0, 1280))
        ax2.set_xlim((0, 1280))
        ax1.set_ylim((0, 640))
        ax2.set_ylim((0, 640))
        mg.plot_distribution(handle=ax2)

    # interpolate for cubic and linear
    dp_cub = mg.interp_to_densepred(method='cubic')
    dp_lin = mg.interp_to_densepred(method='linear')
    dp_splint = dp_cub - dp_lin
    dp_splint.mask = img.mask
    dp_splint.apply_mask()

    nanmask = np.copy(img.mask)
    nanmask[img.mask == 0] = np.nan

    mx = np.max(np.maximum(dp_splint.u, dp_splint.v))

    # plot splint in u and v
    if plotting:
        fig, ax1, ax2 = utilities.plot_adjacent_images(np.abs(dp_splint.u*nanmask),
                                                       np.abs(
                                                           dp_splint.v*nanmask),
                                                       "Splint - U",
                                                       "Splint - V",
                                                       cmap_a="RdYlGn_r",
                                                       cmap_b="RdYlGn_r",
                                                       vminmax_a=[0, mx],
                                                       vminmax_b=[0, mx],
                                                       figsize=(35, 35),
                                                       axes_pad=0.5
                                                       )
        ax1.set_facecolor('k')
        ax2.set_facecolor('k')

    # plot norm of splint
    splint_norm = dp_splint.magnitude()
    if plotting:
        fig, ax1, ax2 = utilities.plot_adjacent_images(np.abs(dp_splint.u*nanmask),
                                                       nanmask,
                                                       "",
                                                       "",
                                                       cmap_a="RdYlGn_r",
                                                       cmap_b="gray",
                                                       vminmax_a=[0, 0.8],
                                                       vminmax_b=[0, 1],
                                                       figsize=(40, 40),
                                                       axes_pad=1,
                                                       cbar_mode='each',
                                                       cbar_pad=0.05
                                                       )
        ax1.set_facecolor('k')
        ax2.set_facecolor('k')
        im = ax2.images
        cb = im[-1].colorbar
        a = cb.ax.figure
        a.delaxes(cb.ax)

    peak_values = []
    for cell in mg.get_all_leaf_cells():

        bl, tr = cell.coordinates[0], cell.coordinates[2]
        if bl[0] >= img.dim[1] or bl[1] >= img.dim[0]:
            continue
        # print(bl, tr)
        # get view into splint
        sub_region = splint_norm[bl[1]:tr[1], bl[0]:tr[0]]
        pval = np.max(sub_region)
        peak_values.append(pval)

        # split cells above min threshold
        if pval > min_thr:
            cell.split()

    # plot histogram of peak values
    # if plotting:
    #     fig, ax1 = plt.subplots()
    #     ax1.hist(peak_values)
    #     ax1.grid()

    # deform image
    img_def = img.deform_image(dp_cub)

    # Now loop over the remaining iters
    for _iter in range(n_iter-1):
        if plotting:
            # plot new grid
            mg.plot_grid(ax=ax2, mask=img.mask)
            #plt.gca().set_title(f"Grid at start of iter {_iter+2}")

        # work out window size
        if settings.final_WS == 'auto':
            ws_final = utilities.round_to_odd(np.sqrt(15 / min_sd))
        else:
            ws_final = settings.final_WS * np.ones(img_def.dim)

        ws = ws_first_iter + ((_iter) / (n_iter - 1)) * \
            (ws_final - ws_first_iter)
        ws = utilities.round_to_odd(ws)
        ws[np.isnan(ws)] = 5

        for win in mg:
            win.WS = ws[win.y, win.x]

        # correlate windows
        mg.correlate_all_windows(img_def, dp_cub)

        if plotting:
            fig, ax1, ax2 = utilities.plot_adjacent_images(ws,
                                                           img.mask,
                                                           "WS",
                                                           f"Dist iter {_iter+2}",
                                                           cmap_a=None,
                                                           figsize=(35, 35),
                                                           axes_pad=0.5
                                                           )
            ax1.set_xlim((0, 1280))
            ax2.set_xlim((0, 1280))
            ax1.set_ylim((0, 640))
            ax2.set_ylim((0, 640))
            mg.plot_distribution(handle=ax2)

        # Validate vectors
        mg.validation_NMT_8NN(idw=True)
        if plotting:
            fig, ax1, ax2 = utilities.plot_adjacent_images(ws-25,
                                                           img.mask,
                                                           "WS",
                                                           f"Validated dist {_iter+2}",
                                                           cmap_a="RdYlGn",
                                                           vminmax_a=[-5, +5],
                                                           vminmax_b=[0, 1],
                                                           figsize=(35, 35),
                                                           axes_pad=0.5
                                                           )
            ax1.set_xlim((0, 1280))
            ax2.set_xlim((0, 1280))
            ax1.set_ylim((0, 640))
            ax2.set_ylim((0, 640))
            mg.plot_distribution(handle=ax2)

        # interpolate for cubic and linear
        dp_cub = mg.interp_to_densepred(method='cubic')
        dp_lin = mg.interp_to_densepred(method='linear')
        dp_splint = dp_cub - dp_lin
        dp_splint.mask = img.mask
        dp_splint.apply_mask()

        nanmask = np.copy(img.mask)
        nanmask[img.mask == 0] = np.nan

        mx = np.max(np.maximum(dp_splint.u, dp_splint.v))

        # plot splint in u and v
        if plotting:
            a = np.abs(dp_splint.u*nanmask)
            b = np.abs(dp_splint.v*nanmask)
            a = np.abs(np.gradient(dp_cub.magnitude(), axis=0)*nanmask)
            b = np.abs(np.gradient(dp_cub.magnitude(), axis=1)*nanmask)
            mx = 0.05
            fig, ax1, ax2 = utilities.plot_adjacent_images(a, b,
                                                           "Splint - U",
                                                           "Splint - V",
                                                           cmap_a="RdYlGn_r",
                                                           cmap_b="RdYlGn_r",
                                                           vminmax_a=[0, mx],
                                                           vminmax_b=[0, mx],
                                                           figsize=(35, 35),
                                                           axes_pad=0.5
                                                           )
            ax1.set_facecolor('k')
            ax2.set_facecolor('k')

        # plot norm of splint
        splint_norm = dp_splint.magnitude()
        if plotting:
            fig, ax1, ax2 = utilities.plot_adjacent_images(np.abs(dp_splint.u*nanmask),
                                                           nanmask,
                                                           "",
                                                           "",
                                                           cmap_a="RdYlGn_r",
                                                           cmap_b="gray",
                                                           vminmax_a=[0, 0.8],
                                                           vminmax_b=[0, 1],
                                                           figsize=(40, 40),
                                                           axes_pad=1,
                                                           cbar_mode='each',
                                                           cbar_pad=0.05
                                                           )
            ax1.set_facecolor('k')
            ax2.set_facecolor('k')
            im = ax2.images
            cb = im[-1].colorbar
            a = cb.ax.figure
            a.delaxes(cb.ax)

        if _iter < n_iter-2:
            peak_values = []
            for cell in mg.get_all_leaf_cells():
                bl, tr = cell.coordinates[0], cell.coordinates[2]
                if bl[0] >= img.dim[1] or bl[1] >= img.dim[0]:
                    continue

                # print(bl, tr)
                # get view into splint
                sub_region = splint_norm[bl[1]:tr[1], bl[0]:tr[0]]
                pval = np.max(sub_region)
                peak_values.append(pval)

                # split cells above min threshold
                if pval > min_thr:
                    cell.split()

        # plot histogram of peak values
        # if plotting:
            # fig, ax1 = plt.subplots()
            # ax1.hist(peak_values)
            # ax1.grid()

            # deform image
        img_def = img.deform_image(dp_cub)

    if n_iter_ref > 0:
        for _iter in range(n_iter_ref):
            # correlate
            mg.correlate_all_windows(img_def, dp_cub)
            # validate
            mg.validation_NMT_8NN(idw=True)
            # interpolate
            dp_cub = mg.interp_to_densepred()
            # deform image
            img_def = img.deform_image(dp_cub)

    if plotting:
        p_cub = mg.interp_to_densepred(method='cubic')
        dp_lin = mg.interp_to_densepred(method='linear')
        dp_splint = dp_cub - dp_lin
        dp_splint.mask = img.mask
        dp_splint.apply_mask()
        splint_norm = dp_splint.magnitude()
        peak_values = []
        for cell in mg.get_all_leaf_cells():
            bl, tr = cell.coordinates[0], cell.coordinates[2]
            if bl[0] >= img.dim[1] or bl[1] >= img.dim[0]:
                continue

            # print(bl, tr)
            # get view into splint
            sub_region = splint_norm[bl[1]:tr[1], bl[0]:tr[0]]
            pval = np.max(sub_region)
            peak_values.append(pval)

            # split cells above min threshold
            if pval > min_thr:
                cell.split()
        mg.plot_grid(ax=ax2, mask=img.mask)

    return dp_cub, mg


def adapt_multi_grid(img):
    """Analyses an image using the multi_grid approach
    """

    init_WS = 129
    final_WS = 65

    dp = PIV.DensePredictor(
        np.zeros(img.dim), np.zeros(img.dim), img.mask)

    amg = multi_grid.MultiGrid(img.dim, spacing=64, WS=init_WS)
    print("Grid created")

    # correlate all windows
    print("Correlating windows")
    amg.correlate_all_windows(img, dp)

    print("Validate vectors")
    amg.validation_NMT_8NN()

    print("Interpolating")
    dp = amg.interp_to_densepred()
    dp.mask = img.mask
    dp.apply_mask()

    print("Deforming image")
    img_def = img.deform_image(dp)

    print("Spitting all cells")
    amg.split_all_cells()
    print(amg.grids[1].x_vec)
    print(amg.grids[1].y_vec)
    print("Setting all windows to 65 pixel windows")
    for window in amg.windows:
        window.WS = final_WS

    # correlate all windows
    print("Correlating windows")
    amg.correlate_all_windows(img_def, dp)

    print("Validate vectors")
    amg.validation_NMT_8NN()

    print("Interpolating")
    dp = amg.interp_to_densepred()
    dp.mask = img.mask
    dp.apply_mask()

    return dp, amg


class MultiGridSettings():
    def __init__(self,
                 init_spacing=64,
                 final_WS=None,
                 min_thr=0.05,
                 n_iter_main=3, n_iter_ref=1,
                 target_init_NI=20, target_fin_NI=8,
                 verbosity=2):
        """

        Parameters
        ----------
        final_WS : int or str, optional
            Final window size, must be odd and 5 <= final_WS <= 245
            Otheriwse 'auto', where the window size will be calculated
            according to the seeding density
            Default 'auto'
        n_iter_main : int, optional
            Number of main iterations, wherein the WS and spacing will reduce
            from init_WS to final_WS Must be 1 <= n_iter_main <= 10
            If the number of main iterations is 1 then the final_WS and
            final_N_windows is ignored
            Default 3
        n_iter_ref : int, optional
            Number of refinement iterations, where the WS and locations remain
            fixed, however, subsequent iterations are performed to improve
            the solution. Must be 0 <= n_iter_ref <= 10
            Default 2
        target_init_NI : int, optional
            The number of particles to target per correlation window in the
            first iteration. Considering AIW, it is possible the resulting
            window will be significantly larger depending on the underlying
            displacement.
            Default = 20.
        target_fin_NI (int, optional
            The number of particles to target per correlation window in the last
            iteration. Unlike the initial target, the final WS should contain
            approximately this many particles, depending on the accuracy of
            particle detection and seeding density estimation
            Default = 8.
        """

        self.init_spacing = init_spacing
        self.min_thr = min_thr
        self.final_WS = final_WS
        self.n_iter_main = n_iter_main
        self.n_iter_ref = n_iter_ref
        self.target_init_NI = target_init_NI
        self.target_fin_NI = target_fin_NI
        self.verbosity = verbosity

    def __eq__(self, other):
        """
        Allow for comparing equality between settings classes

        Parameters
        ----------
        other : WidimSettings
            The other WidimSettings to be compared to

        Returns
        -------
            Bool:
                Whether the two WidimSettings match
        """

        if not isinstance(other, MultiGridSettings):
            return NotImplemented

        for s, o in zip(self.__dict__.values(), other.__dict__.values()):
            if s != o:
                if not np.all(np.isnan((s, o))):
                    return False

        return True

    def __repr__(self):
        output = f" final_WS: {self.final_WS}\n"
        output += f" n_iter_main: {self.n_iter_main}\n"
        output += f" n_iter_ref: {self.n_iter_ref}\n"
        output += f" target_init_NI: {self.target_init_NI}\n"
        output += f" target_fin_NI: {self.target_fin_NI}\n"
        output += f" verbosity: {self.verbosity}\n"
        return output

    @property
    def final_WS(self):
        return self._final_WS

    @final_WS.setter
    def final_WS(self, value):
        """Sets the value of the final window size, checking validity

        Parameters
        ----------
        value : int or str
            Final window size, must be odd and 5 <= final_WS <= 245
            Otheriwse 'auto', where the window size will be calculated
            according to the seeding density
            Default 'auto'
        """

        if value is None or value == 'auto':
            self._final_WS = 'auto'
        elif type(value) is str and value != 'auto':
            raise ValueError("If non-numeric input, must be 'auto'")
        elif int(value) != value:
            raise ValueError("Final WS must be integer")
        elif (value < 5) or (value > 245):
            raise ValueError("Final WS must be 5 <= WS <= 245")
        elif value % 2 != 1:
            raise ValueError("Final WS must be odd")
        else:
            self._final_WS = value

    @property
    def n_iter_main(self):
        return self._n_iter_main

    @n_iter_main.setter
    def n_iter_main(self, value):
        """Sets the number of main iterations, checking validity

        Parameters
        ----------
        value : int
            Number of main iterations, wherein the WS and spacing will reduce
            from init_WS to final_WS 1 <= n_iter_main <= 10
            If the number of main iterations is 1 then the final_WS is ignored
        """
        if int(value) != value:
            raise ValueError("Number of iterations must be integer")
        if value < 1:
            raise ValueError("Number of iterations must be at least 1")
        if value > 10:
            raise ValueError(
                "Number of main iterations must be at most 10")

        self._n_iter_main = value

    @property
    def n_iter_ref(self):
        return self._n_iter_ref

    @n_iter_ref.setter
    def n_iter_ref(self, value):
        """Sets the number of refinement iterations, checking validity

        Parameters
        ----------
        value : int
            Number of refinement iterations, where the WS and locations remain
            fixed, but subsequent iterations are performed to improve the soln
            0 <= n_iter_ref <= 10
        """

        if int(value) != value:
            msg = "Number of refinement iterations must be integer"
            raise ValueError(msg)
        if value < 0:
            msg = "Number of refinement iterations must be at least 0"
            raise ValueError(msg)
        if value > 10:
            msg = "Number of refinement iterations must be at most 10"
            raise ValueError(msg)

        self._n_iter_ref = value
