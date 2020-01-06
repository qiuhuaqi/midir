""" Run inference on full sequence of subjects """

import os
import argparse
import logging
import numpy as np
import h5py
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib
import imageio

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model.networks import BaseNet

from model.dataset_utils import CenterCrop, Normalise, ToTensor
from model.datasets import CardiacMR_2D_Eval_UKBB, CardiacMR_2D_Inference_UKBB
from model.submodules import resample_transform
from utils.metrics import contour_distances_stack, computeJacobianDeterminant2D
from utils import xutils, flow_utils

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default=None, help="Directory containing params.json")
parser.add_argument('--restore_file', default='best',
                    help="Name of the file in --model_dir containing weights to reload (w/o postfix)")
parser.add_argument('--data_path', default='data/inference',
                    help="Path to the dir containing inference data")

parser.add_argument('--no_cuda', action='store_true')
parser.add_argument('--num_workers', default=8, help='Number of processes used by dataloader, 0 means use main process')
parser.add_argument('--gpu', default=0, help='Choose GPU to run on')

parser.add_argument('--no_three_slices', action='store_true', help="Evaluate metrics on 3 slices.")
parser.add_argument('--metrics', action='store_true', help="Evaluating metrics")
parser.add_argument('--nifti', action='store_true', help="Save results in NIFTI files")

# parser.add_argument('--hsv_flow', action='store_true', help='Save hsv encoded flow (PNGs and GIF)')
# parser.add_argument('--quiver', action='store_true', help='Save quiver plot')



def plot_results(target, source, warped_source, op_flow, save_path=None, title_font_size=20, show_fig=False):
    """Plot all motion related results in a single figure,
    here we assume flow is normalised to [-1, 1] coordinate"""

    # convert flow into HSV flow with white background
    hsv_flow = flow_utils.flow_to_hsv(op_flow, max_mag=0.15, white_bg=True)

    ## set up the figure
    fig = plt.figure(figsize=(30, 18))
    title_pad = 10

    # source
    ax = plt.subplot(2, 4, 1)
    plt.imshow(source, cmap='gray')
    plt.axis('off')
    ax.set_title('Source', fontsize=title_font_size, pad=title_pad)

    # warped source
    ax = plt.subplot(2, 4, 2)
    plt.imshow(warped_source, cmap='gray')
    plt.axis('off')
    ax.set_title('Warped Source', fontsize=title_font_size, pad=title_pad)

    # calculate the error before and after reg
    error_before = target - source
    error_after = target - warped_source

    # error before
    ax = plt.subplot(2, 4, 3)
    plt.imshow(error_before, vmin=-255, vmax=255, cmap='gray')
    plt.axis('off')
    ax.set_title('Error before', fontsize=title_font_size, pad=title_pad)

    # error after
    ax = plt.subplot(2, 4, 4)
    plt.imshow(error_after, vmin=-255, vmax=255, cmap='gray')
    plt.axis('off')
    ax.set_title('Error after', fontsize=title_font_size, pad=title_pad)

    # target image
    ax = plt.subplot(2, 4, 5)
    plt.imshow(target, cmap='gray')
    plt.axis('off')
    ax.set_title('Target', fontsize=title_font_size, pad=title_pad)

    # hsv flow
    ax = plt.subplot(2, 4, 7)
    plt.imshow(hsv_flow)
    plt.axis('off')
    ax.set_title('HSV', fontsize=title_font_size, pad=title_pad)

    # quiver, or "Displacement Vector Field" (DVF)
    ax = plt.subplot(2, 4, 6)
    interval = 3  # interval between points on the grid
    background = source
    quiver_flow = np.zeros_like(op_flow)
    quiver_flow[:, :, 0] = op_flow[:, :, 0] * op_flow.shape[0] / 2
    quiver_flow[:, :, 1] = op_flow[:, :, 1] * op_flow.shape[1] / 2
    mesh_x, mesh_y = np.meshgrid(range(0, op_flow.shape[1] - 1, interval),
                                 range(0, op_flow.shape[0] - 1, interval))
    plt.imshow(background[:, :], cmap='gray')
    plt.quiver(mesh_x, mesh_y,
               quiver_flow[mesh_y, mesh_x, 1], quiver_flow[mesh_y, mesh_x, 0],
               angles='xy', scale_units='xy', scale=1, color='g')
    plt.axis('off')
    ax.set_title('DVF', fontsize=title_font_size, pad=title_pad)

    # det Jac
    ax = plt.subplot(2, 4, 8)
    jac_det, mean_grad_detJ, negative_detJ = computeJacobianDeterminant2D(op_flow)
    spec = [(0, (0.0, 0.0, 0.0)), (0.000000001, (0.0, 0.2, 0.2)),
            (0.12499999999, (0.0, 1.0, 1.0)), (0.125, (0.0, 0.0, 1.0)),
            (0.25, (1.0, 1.0, 1.0)), (0.375, (1.0, 0.0, 0.0)),
            (1, (0.94509803921568625, 0.41176470588235292, 0.07450980392156863))]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('detjac', spec)
    plt.imshow(jac_det, vmin=-1, vmax=7, cmap=cmap)
    plt.axis('off')
    ax.set_title('Jacobian (Grad: {0:.2f}, Neg: {1:.2f}%)'.format(mean_grad_detJ, negative_detJ * 100),
                 fontsize=int(title_font_size*0.9), pad=title_pad)
    # split and extend this axe for the colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(cax=cax1)
    cb.ax.tick_params(labelsize=20)

    # adjust subplot placements and spacing
    plt.subplots_adjust(left=0.0001, right=0.99, top=0.9, bottom=0.1, wspace=0.001, hspace=0.1)

    # saving
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=100)

    if show_fig:
        plt.show()
    plt.close()



def inference(model, subject_data_dir, eval_data, subject_output_dir, args, params):
    """
    Run inference on one subject.

    Args:
        model: (object) instantiated model
        subject_data_dir: (string) directory of the subject's data, absolute path
        eval_data: (dict) ED and ES images and labels to evaluate metrics
        subject_output_dir: (string) save results of the subject to this dir
        args
        params

    Returns:

    """
    # set model to evaluation mode
    model.eval()

    # send model to the right device
    model = model.to(device=args.device)  # (note: this may not send all parameters)

    # --- run inference on the whole sequence --- #
    # create a dataloader to load data of one subject
    inference_dataset = CardiacMR_2D_Inference_UKBB(subject_data_dir,
                                                    seq=params.seq,
                                                    transform=transforms.Compose([
                                                        CenterCrop(params.crop_size),
                                                        Normalise(),
                                                        ToTensor()])
                                                    )

    # loop over time frames
    logging.info("Running inference calculation...")
    op_flow_list = []
    target_list = []
    source_list = []
    warped_source_list = []
    for (target, source) in inference_dataset:
        # size (N, 1, H, W) to input model
        target = target.unsqueeze(1).to(device=args.device)
        source = source.unsqueeze(1).to(device=args.device)

        # run inference
        op_flow = model(target, source)
        warped_source = resample_transform(source, op_flow)

        # move to cpu and stack
        op_flow_list += [op_flow.data.cpu().numpy().transpose(0, 2, 3, 1)]  # (N, H, W, 2)
        target_list += [target.data.squeeze(1).cpu().numpy()[:, :, :] * 255]  # (N, H, W), here N = frames -1
        source_list += [source.data.squeeze(1).cpu().numpy()[:, :, :] * 255]  # (N, H, W), here N = frames -1
        warped_source_list += [warped_source.data.squeeze(1).cpu().numpy()[:, :, :] * 255]  # (N, H, W)
    logging.info("- Done.")

    # stack on time as dim 0, shape (T, N, H, W)
    op_flow_seq = np.stack(op_flow_list, axis=0)
    target_seq = np.stack(target_list, axis=0)
    source_seq = np.stack(source_list, axis=0)
    warped_source_seq = np.stack(warped_source_list, axis=0)

    # save the flow and target sequence to a HDF5 file for lateer
    h5py_file_path = os.path.join(subject_output_dir, 'save_data.hdf5')
    if os.path.exists(h5py_file_path): os.system("rm {}".format(h5py_file_path))
    with h5py.File(h5py_file_path, "w") as f:
        f.create_dataset('op_flow_seq', data=op_flow_seq)
        f.create_dataset('target_seq', data=target_seq)

    num_slices = op_flow_seq.shape[1]
    if args.three_slices:
        apical_idx = int(round((num_slices - 1) * 0.75))  # 75% from basal
        mid_ven_idx = int(round((num_slices - 1) * 0.5))  # 50% from basal
        basal_idx = int(round((num_slices - 1) * 0.25))  # 25% from basal
        slices_idx = [apical_idx, mid_ven_idx, basal_idx]
    else:
        slices_idx = np.arange(0, num_slices)

    # loop over slices
    for slice_num in slices_idx:
        logging.info("Saving results of slice no. {}".format(slice_num))
        # shape (T, H, W) or (T, H, W, 2)
        op_flow_slice_seq = op_flow_seq[:, slice_num, :, :]
        target_slice_seq = target_seq[:, slice_num, :, :]
        source_slice_seq = source_seq[:, slice_num, :, :]
        warped_source_slice_seq = warped_source_seq[:, slice_num, :, :]

        # set up saving directory
        output_dir_slice = os.path.join(subject_output_dir, 'slice_{}'.format(slice_num))
        if not os.path.exists(output_dir_slice):
            os.makedirs(output_dir_slice)

        # loop over time frame
        png_buffer = []
        for fr in range(op_flow_slice_seq.shape[0]):
            print('Frame: {}/{}'.format(fr, op_flow_slice_seq.shape[0]))
            op_flow_fr = op_flow_slice_seq[fr, :, :, :]
            target_fr = target_slice_seq[fr, :, :]
            source_fr = source_slice_seq[fr, :, :]
            warped_source_fr = warped_source_slice_seq[fr, :, :]

            fig_save_path = os.path.join(output_dir_slice, 'frame_{}.png'.format(fr))
            plot_results(target_fr, source_fr, warped_source_fr, op_flow_fr, save_path=fig_save_path)

            # read back the PNG to save a GIF animation
            png_buffer += [imageio.imread(fig_save_path)]
        imageio.mimwrite(os.path.join(output_dir_slice, 'results.gif'), png_buffer, fps=params.fps)

        # flow_utils.save_warp_n_error(warped_source_slice_seq, target_slice_seq, source_slice_seq, output_dir_slice, fps=params.fps)
        # if args.hsv_flow:
        #     flow_utils.save_flow_hsv(op_flow_slice_seq, target_slice_seq, output_dir_slice, fps=params.fps)
        # if args.quiver:
        #     flow_utils.save_flow_quiver(op_flow_slice_seq * (params.crop_size / 2), source_slice_seq, output_dir_slice, fps=params.fps)

    if args.metrics:
        # --- evaluate motion estimation accuracy metrics ---  #
        # unpack the ED ES data Tensor inputs, transpose from (1, N, H, W) to (N, 1, H, W)
        image_ed_batch = eval_data['image_ed_batch'].permute(1, 0, 2, 3).to(device=args.device)
        image_es_batch = eval_data['image_es_batch'].permute(1, 0, 2, 3).to(device=args.device)
        label_es_batch = eval_data['label_es_batch'].permute(1, 0, 2, 3).to(device=args.device)

        # compute optical flow and warped ed images using the trained model(source, target)
        op_flow = model(image_ed_batch, image_es_batch)

        # warp ED segmentation mask to ES using nearest neighbourhood interpolation
        with torch.no_grad():
            warped_label_es_batch = resample_transform(label_es_batch.float(), op_flow, interp='nearest')

        # move data to cpu to calculate metrics (also transpose into H, W, N)
        warped_label_es_batch = warped_label_es_batch.squeeze(1).cpu().numpy().transpose(1, 2, 0)
        label_es_batch = label_es_batch.squeeze(1).cpu().numpy().transpose(1, 2, 0)
        label_ed_batch = eval_data['label_ed_batch'].squeeze(0).numpy().transpose(1, 2, 0)

        # calculate contour distance metrics, metrics functions take inputs shaped in (H, W, N)
        mcd_lv, hd_lv = contour_distances_stack(warped_label_es_batch, label_ed_batch, label_class=1, dx=params.pixel_size)
        mcd_myo, hd_myo = contour_distances_stack(warped_label_es_batch, label_ed_batch, label_class=2, dx=params.pixel_size)
        mcd_rv, hd_rv = contour_distances_stack(warped_label_es_batch, label_ed_batch, label_class=3, dx=params.pixel_size)

        metrics = dict()
        metrics['mcd_lv'] = mcd_lv
        metrics['hd_lv'] = hd_lv
        metrics['mcd_myo'] = mcd_myo
        metrics['hd_myo'] = hd_myo
        metrics['mcd_rv'] = mcd_rv
        metrics['hd_rv'] = hd_rv

        # save the metrics to a JSON file
        metrics_save_path = os.path.join(subject_output_dir, 'metrics.json')
        xutils.save_dict_to_json(metrics, metrics_save_path)

        if args.nifti:
            # save wapred ES segmentations and original (but cropped) ED segmentation into niftis
            nim = nib.load(os.path.join(subject_data_dir, 'label_sa_ED.nii.gz'))
            nim_wapred_label_es = nib.Nifti1Image(warped_label_es_batch, nim.affine, nim.header)
            nib.save(nim_wapred_label_es, os.path.join(subject_output_dir, 'warped_label_ES.nii.gz'))
            nim_label_ed = nib.Nifti1Image(label_ed_batch, nim.affine, nim.header)
            nib.save(nim_label_ed, os.path.join(subject_output_dir, 'label_ED.nii.gz'))
            nim_label_es = nib.Nifti1Image(label_es_batch, nim.affine, nim.header)
            nib.save(nim_label_es, os.path.join(subject_output_dir, 'label_ES.nii.gz'))


if __name__ == '__main__':

    # parse arguments
    args = parser.parse_args()

    args.three_slices = not args.no_three_slices

    # set the GPU to use and device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    assert os.path.exists(args.model_dir), "No model dir found at {}".format(args.model_dir)

    # load params from model JSON file
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = xutils.Params(json_path)

    # set up save dir
    output_dir = os.path.join(args.model_dir, 'inference')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # set up logger
    xutils.set_logger(os.path.join(output_dir, 'inference.log'))
    logging.info("Inference model: {}".format(args.model_dir))

    # set up the model
    model = BaseNet()

    # reload network parameters from saved model file
    logging.info("Loading model from saved file: {}".format(os.path.join(args.model_dir, args.restore_file + '.pth.tar')))
    xutils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # set up eval dataloader to evaluate motion metrics
    eval_dataset = CardiacMR_2D_Eval_UKBB(args.data_path,
                                          seq=params.seq,
                                          augment=params.augment,
                                          transform=transforms.Compose([
                                              CenterCrop(params.crop_size),
                                              Normalise(),
                                              ToTensor()]),
                                          label_transform=transforms.Compose([
                                              CenterCrop(params.crop_size),
                                              ToTensor()])
                                          )

    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=params.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=args.cuda)

    # loop over subjects using evaluation dataloader
    logging.info("Starting inference...")
    for idx, (image_ed_batch, image_es_batch, label_ed_batch, label_es_batch) in enumerate(eval_dataloader):
        # pack the eval data into a dict
        eval_data = dict()
        eval_data['image_ed_batch'] = image_ed_batch
        eval_data['image_es_batch'] = image_es_batch
        eval_data['label_ed_batch'] = label_ed_batch
        eval_data['label_es_batch'] = label_es_batch

        # get the subject dir from dataset
        subject_id = eval_dataloader.dataset.dir_list[idx]
        logging.info("Subject: {}".format(subject_id))
        subject_data_dir = os.path.join(args.data_path, subject_id)
        assert os.path.exists(subject_data_dir), "Dir of inference data subject {} does not exist!".format(subject_id)
        subject_output_dir = os.path.join(output_dir, subject_id)
        if not os.path.exists(subject_output_dir): os.makedirs(subject_output_dir)

        # inference on the subject
        inference(model, subject_data_dir, eval_data, subject_output_dir, args, params)
    logging.info("Inference complete.")


