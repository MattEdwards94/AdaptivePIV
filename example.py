import numpy as np
import PIV
import matplotlib.pyplot as plt


def experimental_example(im_number=1, settings=None):
    """Demonstrates the PIV algorithm on an actual image from an experiment
    The experiment is the flow over a backwards facing step.
    """
    # load a PIV image
    img = PIV.piv_image.PIVImage.from_flowtype(flowtype=1, im_number=im_number)

    # show the images
    img.plot_images_animation()

    # define settings if not passed in
    if settings is None:
        settings = PIV.analysis.WidimSettings(init_WS=97,
                                              final_WS=33,
                                              WOR=0.5,
                                              n_iter_main=3,
                                              n_iter_ref=1,
                                              vec_val='NMT',
                                              interp='struc_cub')

    # analyse the image
    disp_field = PIV.analysis.widim(img, settings)

    # plot displacement field
    disp_field.plot_displacement_field(width=0.001, minshaft=2)
    plt.show()


def synthetic_example(dim=(500, 500), settings=None):
    """Demonstrates the ability to generate synthetic images with arbitrary
    displacement fields, and then analyses the images.
    """

    # define flow field
    u, v = np.empty(dim), np.empty(dim)
    xx, yy = np.meshgrid(np.arange(dim[1]), np.arange(dim[0]))

    u = 15 * (np.cos((xx * 2*np.pi)/dim[1] + (np.pi / 2)) *
              np.cos((yy * 2 * np.pi)/dim[0]))
    v = 15 * (np.sin((xx * 2*np.pi)/dim[1] + (np.pi / 2)) *
              np.sin((yy * 2 * np.pi)/dim[0]))

    # create synthetic images
    ima, imb = PIV.piv_image.create_synthetic_image_pair(dim, seed_dens=0.035,
                                                         u=u, v=v)

    # create and show images
    img = PIV.piv_image.PIVImage(ima, imb)
    img.plot_images_animation()

    if settings is None:
        settings = PIV.analysis.WidimSettings(init_WS=97,
                                              final_WS=33,
                                              WOR=0.5,
                                              n_iter_main=3,
                                              n_iter_ref=1,
                                              vec_val='NMT',
                                              interp='struc_cub')

    # analyse the image
    disp_field = PIV.analysis.widim(img, settings)

    # plot displacement field
    disp_field.plot_displacement_field(width=0.001, minshaft=2)
    plt.show()


if __name__ == "__main__":

    # configure analysis settings:
    settings = PIV.analysis.WidimSettings(init_WS=97,
                                          final_WS=33,
                                          WOR=0.5,
                                          n_iter_main=3,
                                          n_iter_ref=1,
                                          vec_val='NMT',
                                          interp='struc_cub')

    print("Experimental image example")
    experimental_example(im_number=1, settings=settings)

    print("Synthetic image example")
    synthetic_example(dim=(500, 500), settings=settings)
