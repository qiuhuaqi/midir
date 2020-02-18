""" Run inference on full sequence of subjects """

import os
import argparse
import logging
import numpy as np
import h5py
import nibabel as nib
import imageio

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model.networks import BaseNet

from model.dataset_utils import CenterCrop, Normalise, ToTensor
from model.datasets import CardiacMR_2D_Eval_UKBB, CardiacMR_2D_Inference_UKBB
from model.submodules import spatial_transform
from utils.metrics import contour_distances_stack
from utils import xutils

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir',
                    default='experiments/base_model',
                    help="Directory containing params.json")

parser.add_argument('--restore_file',
                    default=None,
                    help="Name of the checkpoint file: 'best' for best model, or 'last' for the last saved checkpoint")

parser.add_argument('--data_path',
                    default='data/ukbb/cine_ukbb2964/small_set/sa/inference',
                    help="Path to the dir containing inference data")

parser.add_argument('--no_cuda',
                    action='store_true')

parser.add_argument('--num_workers',
                    default=8,
                    help='Number of processes used by dataloader, 0 means use main process')

parser.add_argument('--gpu',
                    default=0,
                    help='Choose GPU')

parser.add_argument('--no_three_slices',
                    action='store_true',
                    help="Evaluate metrics on 3 slices.")

parser.add_argument('--metrics',
                    action='store_true',
                    help="Evaluating metrics")

parser.add_argument('--nifti',
                    action='store_true',
                    help="Save results in NIFTI files")

# parser.add_argument('--hsv_flow', action='store_true', help='Save hsv encoded flow (PNGs and GIF)')
# parser.add_argument('--quiver', action='store_true', help='Save quiver plot')

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

        # linear transformation test for NMI: use (1-source) as source image
        if params.inverse:
            source = 1.0 - source

        # run inference
        op_flow, warped_source = model(target, source)
        # warped_source = spatial_transform(source, op_flow)

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

    if args.metrics:
        # --- evaluate motion estimation accuracy metrics ---  #
        # unpack the ED ES data Tensor inputs, transpose from (1, N, H, W) to (N, 1, H, W)
        image_ed_batch = eval_data['image_ed_batch'].permute(1, 0, 2, 3).to(device=args.device)
        image_es_batch = eval_data['image_es_batch'].permute(1, 0, 2, 3).to(device=args.device)
        label_es_batch = eval_data['label_es_batch'].permute(1, 0, 2, 3).to(device=args.device)

        # compute optical flow and warped ed images using the trained model(source, target)
        op_flow, warped_image_es_batch = model(image_ed_batch, image_es_batch)

        # warp ED segmentation mask to ES using nearest neighbourhood interpolation
        with torch.no_grad():
            warped_label_es_batch = spatial_transform(label_es_batch.float(), op_flow, interp='nearest')

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


