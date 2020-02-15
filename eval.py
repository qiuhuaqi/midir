"""Evaluates the model"""

from tqdm import tqdm
import os
import argparse
import logging
import numpy as np
import pandas as pd

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model.networks import BaseNet, SiameseFCN
from model.submodules import spatial_transform
from model.losses import loss_fn
from model.dataset_utils import CenterCrop, Normalise, ToTensor
from model.datasets import CardiacMR_2D_Eval_UKBB
from utils.metrics import categorical_dice_stack, contour_distances_stack, detJac_stack
from utils import xutils

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best',
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--no_three_slices', action='store_true', help="Evaluate metrics on all instead of 3 slices.")

parser.add_argument('--no_cuda', action='store_true')
parser.add_argument('--num_workers', default=8, help='Number of processes used by dataloader, 0 means use main process')
parser.add_argument('--gpu', default=0, help='Choose the GPU to run on, pass -1 to use CPU')


def evaluate(model, loss_fn, dataloader, params, args, val):
    """
    Evaluate the model on the test dataset
    Returns metrics as a dict and evaluation loss

    Args:
        model:
        loss_fn:
        dataloader:
        params:
        args:
        val: (boolean) indicates validation (True) or testing (False)

    Returns:

    """

    # set model to evaluation mode
    model.eval()

    # empty buffer lists
    val_loss_buffer = []

    dice_lv_buffer = []
    dice_myo_buffer = []
    dice_rv_buffer = []

    mcd_lv_buffer = []
    hd_lv_buffer = []
    mcd_myo_buffer = []
    hd_myo_buffer = []
    mcd_rv_buffer = []
    hd_rv_buffer = []

    mean_mag_grad_detJ_buffer = []
    negative_detJ_buffer = []


    with tqdm(total=len(dataloader)) as t:
        # iterate over validation subjects
        for idx, (image_ed_batch, image_es_batch, label_ed_batch, label_es_batch) in enumerate(dataloader):
            # (data all in shape of (c, N, H, W))

            # extend to (N, c, H, W)
            image_ed_batch = image_ed_batch.permute(1, 0, 2, 3).to(device=args.device)
            image_es_batch = image_es_batch.permute(1, 0, 2, 3).to(device=args.device)
            label_es_batch = label_es_batch.permute(1, 0, 2, 3).to(device=args.device)

            with torch.no_grad():

                # linear transformation test for NMI: use (1-source) as source image
                if params.inverse:
                    image_es_batch = 1.0 - image_es_batch

                # compute optical flow and warped ED images towards ES
                dvf, warped_image_es_batch = model(image_ed_batch, image_es_batch)

                # transform label mask of ES frame
                warped_label_es_batch = spatial_transform(label_es_batch.float(), dvf, interp='nearest')

            if args.cuda:
                # move data to cpu to calculate metrics
                # (the axis permutation is to comply with metric calculation code which takes input shape H, W, N)
                warped_label_es_batch = warped_label_es_batch.squeeze(1).cpu().numpy().transpose(1, 2, 0)
                label_ed_batch = label_ed_batch.squeeze(0).numpy().transpose(1, 2, 0)
                dvf = dvf.data.cpu().numpy().transpose(0, 2, 3, 1)  # (N, H, W, 2)
            else:
                # CPU version of the code
                warped_label_es_batch = warped_label_es_batch.squeeze(1).numpy().transpose(1, 2, 0)
                label_ed_batch = label_ed_batch.squeeze(0).numpy().transpose(1, 2, 0)
                dvf = dvf.data.numpy().transpose(0, 2, 3, 1)  # (N, H, W, 2)

            # calculate the metrics
            if args.three_slices:
                num_slices = label_ed_batch.shape[-1]
                apical_idx = int(round((num_slices - 1) * 0.75))  # 75% from basal
                mid_ven_idx = int(round((num_slices - 1) * 0.5))  # 50% from basal
                basal_idx = int(round((num_slices - 1) * 0.25))  # 25% from basal
                slices_idx = [apical_idx, mid_ven_idx, basal_idx]

                warped_label_es_batch = warped_label_es_batch[:, :, slices_idx]
                label_ed_batch = label_ed_batch[:, :, slices_idx]
                dvf = dvf[slices_idx, :, :, :]  # needed for detJac

            # dice
            dice_lv = categorical_dice_stack(warped_label_es_batch, label_ed_batch, label_class=1)
            dice_myo = categorical_dice_stack(warped_label_es_batch, label_ed_batch, label_class=2)
            dice_rv = categorical_dice_stack(warped_label_es_batch, label_ed_batch, label_class=3)

            dice_lv_buffer += [dice_lv]
            dice_myo_buffer += [dice_myo]
            dice_rv_buffer += [dice_rv]

            # contour distances
            mcd_lv, hd_lv = contour_distances_stack(warped_label_es_batch, label_ed_batch, label_class=1, dx=params.pixel_size)
            mcd_myo, hd_myo = contour_distances_stack(warped_label_es_batch, label_ed_batch, label_class=2, dx=params.pixel_size)
            mcd_rv, hd_rv = contour_distances_stack(warped_label_es_batch, label_ed_batch, label_class=3, dx=params.pixel_size)

            mcd_lv_buffer += [mcd_lv]
            hd_lv_buffer += [hd_lv]
            mcd_myo_buffer += [mcd_myo]
            hd_myo_buffer += [hd_myo]
            mcd_rv_buffer += [mcd_rv]
            hd_rv_buffer += [hd_rv]

            # determinant of Jacobian
            mean_grad_detJ, mean_negative_detJ = detJac_stack(dvf)
            mean_mag_grad_detJ_buffer += [mean_grad_detJ]
            negative_detJ_buffer += [mean_negative_detJ]

            t.update()

    if not val:
    # for testing only: save all metrics evaluated for all test subjects in pandas dataframe
        # save accuracy metrics
        subj_id_buffer = dataloader.dataset.dir_list
        df_buffer = []
        column_method = ['DL'] * len(subj_id_buffer)
        for struct in ['LV', 'MYO', 'RV']:
            if struct == 'LV':
                ls_dice = dice_lv_buffer
                ls_mcd = mcd_lv_buffer
                ls_hd = hd_lv_buffer
            elif struct == 'MYO':
                ls_dice = dice_myo_buffer
                ls_mcd = mcd_myo_buffer
                ls_hd = hd_myo_buffer
            elif struct == 'RV':
                ls_dice = dice_rv_buffer
                ls_mcd = mcd_rv_buffer
                ls_hd = hd_rv_buffer

            ls_struct = [struct] * len(subj_id_buffer)
            data = {'Method': column_method,
                    'ID': subj_id_buffer,
                    'Structure': ls_struct,
                    'Dice': ls_dice,
                    'MCD': ls_mcd,
                    'HD': ls_hd}
            df_buffer += [pd.DataFrame(data=data)]

        # concatenate df and save
        metrics_df = pd.concat(df_buffer, axis=0)
        metrics_df.to_pickle("{0}/network_all_subjects_accuracy_3slices_{1}.pkl".format(args.model_dir, args.three_slices))

        # save detJac metrics
        jac_data = {'Method': column_method,
                    'ID': subj_id_buffer,
                    'GradDetJac': mean_mag_grad_detJ_buffer,
                    'NegDetJac': negative_detJ_buffer}
        jac_df = pd.DataFrame(data=jac_data)
        jac_df.to_pickle("{0}/network_all_subjects_jac_3slices_{1}.pkl".format(args.model_dir, args.three_slices))



    # construct metrics dict
    metrics = {'dice_lv_mean': np.mean(dice_lv_buffer), 'dice_lv_std': np.std(dice_lv_buffer),
               'dice_myo_mean': np.mean(dice_myo_buffer), 'dice_myo_std': np.std(dice_myo_buffer),
               'dice_rv_mean': np.mean(dice_rv_buffer), 'dice_rv_std': np.std(dice_rv_buffer),

               'mcd_lv_mean': np.mean(mcd_lv_buffer), 'mcd_lv_std': np.std(mcd_lv_buffer),
               'mcd_myo_mean': np.mean(mcd_myo_buffer), 'mcd_myo_std': np.std(mcd_myo_buffer),
               'mcd_rv_mean': np.mean(mcd_rv_buffer), 'mcd_rv_std': np.std(mcd_rv_buffer),

               'hd_lv_mean': np.mean(hd_lv_buffer), 'hd_lv_std': np.std(hd_lv_buffer),
               'hd_myo_mean': np.mean(hd_myo_buffer), 'hd_myo_std': np.std(hd_myo_buffer),
               'hd_rv_mean': np.mean(hd_rv_buffer), 'hd_rv_std': np.std(hd_rv_buffer),

               'mean_mag_grad_detJ_mean': np.mean(mean_mag_grad_detJ_buffer),
               'mean_mag_grad_detJ_std': np.std(mean_mag_grad_detJ_buffer),

               'negative_detJ_mean': np.mean(negative_detJ_buffer),
               'negative_detJ_std': np.std(negative_detJ_buffer)
               }

    return metrics


if __name__ == '__main__':
    # parse runtime arguments
    args = parser.parse_args()

    # set the GPU to use and device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    # check whether the trained model exists
    assert os.path.exists(args.model_dir), "No model dir found at {}".format(args.model_dir)

    # set the three slices
    args.three_slices = not args.no_three_slices

    # set up a logger
    xutils.set_logger(os.path.join(args.model_dir, 'eval.log'))

    # load parameters from model JSON file
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = xutils.Params(json_path)

    # set dataset and DataLoader
    logging.info("Eval data path: {}".format(params.eval_data_path))
    eval_dataset = CardiacMR_2D_Eval_UKBB(params.eval_data_path,
                                          seq=params.seq,
                                          augment=params.augment,
                                          label_prefix=params.label_prefix,
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

    # instantiate model and move to device
    if params.network == "BaseNet":
        model = BaseNet()
    elif params.network == "SiameseFCN":
        model = SiameseFCN()
    else:
        raise ValueError("Unknown network!")
    model = model.to(device=args.device)

    # reload network parameters from saved model file
    logging.info("Loading model from saved file: {}".format(os.path.join(args.model_dir, args.restore_file + '.pth.tar')))
    xutils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # run the evaluation and calculate the metrics
    logging.info("Running evaluation...")
    eval_metrics = evaluate(model, loss_fn, eval_dataloader, params, args, val=False)

    # save the results in a JSON file
    save_path = os.path.join(args.model_dir, "test_results_{}_3slices_{}.json".format(args.restore_file, args.three_slices))
    xutils.save_dict_to_json(eval_metrics, save_path)
    logging.info("Evaluation complete. Metric results saved at: \n\t{}".format(save_path))
