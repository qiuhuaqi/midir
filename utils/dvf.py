import torch
import torch.nn.functional as F

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
import imageio
from utils.imageio import save_gif, save_png, save_nifti
from model.submodules import spatial_transform


def flow_line_integral(op_flow):
    """
    Perform approximated line integral of frame-to-frame optical flow
    using Pytorch and GPU

    Args:
        op_flow: optical flow, Tensor, shape (N, 2, W, H)

    Returns:
        accum_flow: line integrated optical flow, same shape as input
    """
    # generate a standard grid
    h, w = op_flow.size()[-2:]
    std_grid_h, std_grid_w = torch.meshgrid([torch.linspace(-1, 1, h), torch.linspace(-1, 1, w)])
    std_grid = torch.stack([std_grid_h, std_grid_w])  # (2, H, W)
    std_grid = std_grid.cuda().float()

    # loop over frames
    accum_flow = []
    for fr in range(op_flow.size()[-1]):
        if fr == 0:
            new_grid = std_grid + op_flow[fr, ...]  # (2, H, W) + (2, H, W)
        else:
            # sample optical flow at current new_grid, then update new_grid by sampled offset
            # this line is unnecessarily complicated thanks to pytorch
            new_grid += F.grid_sample(op_flow[fr, ...].unsqueeze(0), new_grid.unsqueeze(0).permute(0, 2, 3, 1)).squeeze(0)  # (2, H, W)
        accum_flow += [new_grid - std_grid]

    accum_flow = torch.stack(accum_flow)  # (N, 2, H, W)
    return accum_flow



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



def dof_to_dvf(target_img, dofin, dvfout, output_dir):
    """
    Convert MIRTK format DOF file to dense Deformation Vector Field (DVF)
    by warping meshgrid using MIRTK transform-image

    Note: this function only handles 2D deformation a.t.m.

    Args:
        target_img: (str) full path of target image of the registration
        dofin: (str) full path to input DOF file
        dofdir: (str) full path of the dir of input DOF file
        dvfout: (str) output DVF file name under output_dir (without ".nii.gz")
        output_dir: (str) full path of output dir

    Returns:
        dvf: (ndarray of shape HxWx2) dense DVF, numpy array coordinate system
            this DVF is in number of pixels (not normalised)

    """

    # read a target image to get the affine transformation
    # this is to ensure the mirtk transform-image apply the DOF in the correct target space
    nim = nib.load(target_img)

    # generate the meshgrids
    mesh_x, mesh_y = np.meshgrid(range(0, nim.shape[0], 1), range(0, nim.shape[1], 1))
    mesh_x = mesh_x.astype('float')
    mesh_y = mesh_y.astype('float')

    # save both (x and y) into nifti files, using the target image header
    nim_meshx = nib.Nifti1Image(mesh_x, nim.affine)
    meshx_path = '{0}/{1}_meshx.nii.gz'.format(output_dir, dvfout)
    nib.save(nim_meshx, meshx_path)

    nim_meshy = nib.Nifti1Image(mesh_y, nim.affine)
    meshy_path = '{0}/{1}_meshy.nii.gz'.format(output_dir, dvfout)
    nib.save(nim_meshy, meshy_path)

    # use mirtk to transform it with DOF file as input
    warped_meshx_path = '{0}/{1}_meshx_warped.nii.gz'.format(output_dir, dvfout)
    warped_meshy_path = '{0}/{1}_meshy_warped.nii.gz'.format(output_dir, dvfout)
    os.system('mirtk transform-image {0} {1} -dofin {2} -target {3}'.format(meshx_path, warped_meshx_path, dofin, target_img))
    os.system('mirtk transform-image {0} {1} -dofin {2} -target {3}'.format(meshy_path, warped_meshy_path, dofin, target_img))

    # read in the generated mesh grid x and grid y
    warp_meshx = nib.load(warped_meshx_path).get_data()[:, :, 0]
    warp_meshy = nib.load(warped_meshy_path).get_data()[:, :, 0]

    # calculate the DVF by substracting the initial mesh grid from it
    dvf_x = warp_meshx - mesh_x
    dvf_y = warp_meshy - mesh_y
    dvf = np.array([dvf_y, dvf_x]).transpose(1, 2, 0)  # (H, W, 2), notice the x-y swap

    # save flow to nifti
    if dvfout is not None:
        ndvf = nib.Nifti1Image(dvf, nim.affine)
        nib.save(ndvf, '{0}/{1}.nii.gz'.format(output_dir, dvfout))

    # clean up: remove all the mesh files
    os.system('rm {0}/*mesh*'.format(output_dir))

    return dvf


def warp_numpy_cpu(source_img, dvf):
    """
    Warp numpy array image using Pytorch's resample based function on CPU

    Args:
        source_img: (ndarry shape: H,W,N) source image, assume square image
        dvf: (ndarray shape: H,W,N,1,2) dense displacement vector field, not normalised to [-1,1]

    Returns:
        warped_source_img: (ndarray shape: H,W,N) resample deformed source image
    """

    dvf_norm = 2 * dvf / source_img.shape[0]  # normalise to Pytorch coordinate system
    dvf_tensor = torch.from_numpy(dvf_norm[:, :, :, 0, :].transpose(2, 3, 0, 1)).float()  # tensor (N, 2, H, W)
    source_img_tensor = torch.from_numpy(source_img.transpose(2, 0, 1)).unsqueeze(1)  # tensor (N, 1, H, W)
    warped_source_img_tensor = spatial_transform(source_img_tensor, dvf_tensor)
    warped_source_img = warped_source_img_tensor.numpy().transpose(2, 3, 0, 1)[..., 0]  # (H, W, N)

    return warped_source_img


def show_warped_grid(ax, dvf, bg_img, interval=3, title="Grid", fontsize=20):
    """dvf shape (H, W, 2)"""
    background = bg_img
    interval = interval
    id_grid_X, id_grid_Y = np.meshgrid(range(0, bg_img.shape[0]-1, interval),
                                       range(0, bg_img.shape[1]-1, interval))

    new_grid_X = id_grid_X + dvf[id_grid_Y, id_grid_X, 1]
    new_grid_Y = id_grid_Y + dvf[id_grid_Y, id_grid_X, 0]

    kwargs = {"linewidth": 1.5, "color": 'c'}
    for i in range(new_grid_X.shape[0]):
        ax.plot(new_grid_X[i,:], new_grid_Y[i,:], **kwargs)  # each draw a line
    for i in range(new_grid_X.shape[1]):
        ax.plot(new_grid_X[:,i], new_grid_Y[:,i], **kwargs)

    ax.set_title(title, fontsize=fontsize)
    ax.imshow(background, cmap='gray')
    ax.axis('off')


