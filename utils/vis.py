"""Utility functions for registration-related visualisation"""
import cv2
import imageio
import numpy as np
import os
from matplotlib import pyplot as plt

from utils.imageio import save_gif, save_png


def flow_to_hsv(opt_flow, max_mag=0.1, white_bg=False):
    """
    Encode optical flow to HSV.

    Args:
        opt_flow: 2D optical flow in (dx, dy) encoding, shape (H, W, 2)
        max_mag: flow magnitude will be normalised to [0, max_mag]

    Returns:
        hsv_flow_rgb: HSV encoded flow converted to RGB (for visualisation), same shape as input

    """
    # convert to polar coordinates
    mag, ang = cv2.cartToPolar(opt_flow[..., 0], opt_flow[..., 1])

    # hsv encoding
    hsv_flow = np.zeros((opt_flow.shape[0], opt_flow.shape[1], 3))
    hsv_flow[..., 0] = ang*180/np.pi/2  # hue = angle
    hsv_flow[..., 1] = 255.0  # saturation = 255
    hsv_flow[..., 2] = 255.0 * mag / max_mag
    # (wrong) hsv_flow[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # convert hsv encoding to rgb for visualisation
    # ([..., ::-1] converts from BGR to RGB)
    hsv_flow_rgb = cv2.cvtColor(hsv_flow.astype(np.uint8), cv2.COLOR_HSV2BGR)[..., ::-1]
    hsv_flow_rgb = hsv_flow_rgb.astype(np.uint8)

    if white_bg:
        hsv_flow_rgb = 255 - hsv_flow_rgb

    return hsv_flow_rgb


def blend_image_seq(images1, images2, alpha=0.7):
    """
    Blend two sequences of images.
    (used in this project to blend HSV-encoded flow with image)
    Repeat to fill RGB channels if needed.

    Args:
        images1: numpy array, shape (H, W, Ch, Frames) or (H, W, Frames)
        images2: numpy array, shape (H, W, Ch, Frames) or (H, W, Frames)
        alpha: mixing weighting, higher alpha increase image 2.  (1 - alpha) * images1 + alpha * images2

    Returns:
        blended_images: numpy array, shape (H, W, Ch, Frames)
    """
    if images1.ndim < images2.ndim:
        images1 = np.repeat(images1[:, :, np.newaxis, :], images2.shape[2], axis=2)
    elif images1.ndim > images2.ndim:
        images2 = np.repeat(images2[:, :, np.newaxis, :], images1.shape[2], axis=2)

    assert images1.shape == images2.shape, "Blending: images being blended have different shapes, {} vs {}".format(images1.shape, images2.shape)
    blended_images = (1 - alpha) * images1 + alpha * images2

    return blended_images.astype(np.uint8)


def save_flow_hsv(op_flow, background, save_result_dir, fps=20, max_mag=0.1):
    """
    Save HSV encoded optical flow overlayed on background image.
    GIF and PNG images

    Args:
        op_flow: numpy array of shape (N, H, W, 2)
        background: numpy array of shape (N, H, W)
        save_result_dir: path to save result dir
        fps: frames per second, for gif
        max_mag: maximum flow magnitude used in normalisation

    Returns:

    """

    # encode flow in hsv
    op_flow_hsv = []
    for fr in range(op_flow.shape[0]):
        op_flow_hsv += [flow_to_hsv(op_flow[fr, :, :, :], max_mag=max_mag)]  # a list of N items each shaped (H, W, ch)

    # save flow sequence into a gif file and a sequence of png files
    op_flow_hsv = np.array(op_flow_hsv).transpose(1, 2, 3, 0)  # (H, W, 3, N)

    # overlay on background images
    op_flow_hsv_blend = blend_image_seq(background.transpose(1, 2, 0), op_flow_hsv)

    # save gif and png
    save_result_dir = os.path.join(save_result_dir, 'hsv_flow')
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)
    save_gif(op_flow_hsv, os.path.join(save_result_dir, 'flow.gif'), fps=fps)
    save_gif(op_flow_hsv_blend, os.path.join(save_result_dir, 'flow_blend.gif'), fps=fps)
    save_png(op_flow_hsv_blend, save_result_dir)
    print("HSV flow saved to: {}".format(save_result_dir))


def save_flow_quiver(op_flow, background, save_result_dir, scale=1, interval=3, fps=20):
    """
    Plot quiver plot and save.

    Args:
        op_flow: numpy array of shape (N, H, W, 2)
        background: numpy array of shape (N, H, W)
        save_result_dir: path to save result dir

    Returns:
    """

    # set up saving directory
    save_result_dir = os.path.join(save_result_dir, 'quiver')
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)

    # create mesh grid of vector origins
    # note: numpy uses x-y order in generating mesh grid, i.e. (x, y) = (w, h)
    mesh_x, mesh_y = np.meshgrid(range(0, background.shape[1]-1, interval), range(0, background.shape[2]-1, interval))

    png_list = []
    for fr in range(background.shape[0]):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax = plt.imshow(background[fr, :, :], cmap='gray')
        ax = plt.quiver(mesh_x, mesh_y,
                        op_flow[fr, mesh_y, mesh_x, 1], op_flow[fr, mesh_y, mesh_x, 0],
                        angles='xy', scale_units='xy', scale=scale, color='g')
        save_path = os.path.join(save_result_dir, 'frame_{}.png'.format(fr))
        plt.axis('off')
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

        # read it back to make gif
        png_list += [imageio.imread(save_path)]

    # save gif
    imageio.mimwrite(os.path.join(save_result_dir, 'quiver.gif'), png_list, fps=fps)
    print("Flow quiver plots saved to: {}".format(save_result_dir))


def save_warp_n_error(warped_source, target, source, save_result_dir, fps=20):
    """
    Calculate warping and save results

    Args:
        warped_source: source images warped to target images, numpy array shaped (N, H, W)
        target: target image, numpy array shaped (N, H, W)
        source: numpy array shaped (N, H, W)
        save_result_dir: numpy array shaped (N, H, W)
        fps:

    Returns:

    """

    # transpose all to (H, W, N)
    warped_source = warped_source.transpose(1, 2, 0)
    target = target.transpose(1, 2, 0)
    source = source.transpose(1, 2, 0)

    # calculate error normalised to (0, 255)
    error = np.abs(warped_source - target)
    error_before = np.abs(source - target)

    save_gif(error, os.path.join(save_result_dir, 'error.gif'), fps=fps)
    save_gif(error_before, os.path.join(save_result_dir, 'error_before.gif'), fps=fps)
    save_gif(target, os.path.join(save_result_dir, 'target.gif'), fps=fps)
    save_gif(warped_source, os.path.join(save_result_dir, 'wapred_source.gif'), fps=fps)
    save_gif(source, os.path.join(save_result_dir, 'source.gif'), fps=fps)
    print("Warping and error saved to: {}".format(save_result_dir))


def show_warped_grid(ax, dvf, bg_img, interval=3, title="Grid", fontsize=20):
    """dvf shape (2, H, W)"""
    background = bg_img
    interval = interval
    id_grid_X, id_grid_Y = np.meshgrid(range(0, bg_img.shape[0]-1, interval),
                                       range(0, bg_img.shape[1]-1, interval))

    new_grid_X = id_grid_X + dvf[1, id_grid_Y, id_grid_X]
    new_grid_Y = id_grid_Y + dvf[0, id_grid_Y, id_grid_X]

    kwargs = {"linewidth": 1.5, "color": 'c'}
    for i in range(new_grid_X.shape[0]):
        ax.plot(new_grid_X[i,:], new_grid_Y[i,:], **kwargs)  # each draw a line
    for i in range(new_grid_X.shape[1]):
        ax.plot(new_grid_X[:,i], new_grid_Y[:,i], **kwargs)

    ax.set_title(title, fontsize=fontsize)
    ax.imshow(background, cmap='gray')
    ax.axis('off')