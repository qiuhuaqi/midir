"""Evaluate metrics without registration"""
import os
import nibabel as nib
import argparse
import numpy as np
import logging
from tqdm import tqdm
import pandas as pd

from model.dataset_utils import CenterCrop
from utils.metrics import categorical_dice_stack, contour_distances_stack, detJac_stack
from utils import xutils

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/test', help="Directory containing test data")
parser.add_argument('--output_dir', default='experiments/demo', help="Directory to save output")
parser.add_argument('--pixel_size', default=1.8,
                    help='Dimension of in-plane pixels in canonical space, assume square pixel in-plane, unit mm')
parser.add_argument('--seq', default='sa', help='Imaging view, sa, la_2ch or la_4ch')
parser.add_argument('--no_three_slices', action='store_true', help="Evaluate metrics on 3 slices.")
parser.add_argument('--crop_size', default=160)
parser.add_argument('--label_prefix', default='label', help='Prefix of file names of segmentation masks')

# parse arguments
args = parser.parse_args()


data_dir = args.data_dir
label_prefix = args.label_prefix
output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

args.three_slices = not args.no_three_slices

# set up a logger
xutils.set_logger(os.path.join(output_dir, 'unreg_eval.log'))
logging.info('Starting Unreg evaluation...')

# unregistered metric buffers
unreg_dice_lv_buffer = []
unreg_dice_myo_buffer = []
unreg_dice_rv_buffer = []

unreg_mcd_lv_buffer = []
unreg_hd_lv_buffer = []
unreg_mcd_myo_buffer = []
unreg_hd_myo_buffer = []
unreg_mcd_rv_buffer = []
unreg_hd_rv_buffer = []

subj_id_buffer = []

logging.info('Looping over subjects...')

with tqdm(total=len(os.listdir(data_dir))) as t:
    # loop over subjects
    for subj_id in sorted(os.listdir(data_dir)):
        subj_dir = os.path.join(data_dir, subj_id)
        subj_id_buffer += [subj_id]

        # load in the ED and ES segmentation masks
        nseg_ED = nib.load(os.path.join(subj_dir, '{0}_sa_ED.nii.gz'.format(label_prefix)))
        nseg_ES = nib.load(os.path.join(subj_dir, '{0}_sa_ES.nii.gz'.format(label_prefix)))
        seg_ED = nseg_ED.get_data()
        seg_ES = nseg_ES.get_data()

        # cropping
        cropper = CenterCrop(output_size=args.crop_size)
        seg_ED_crop = cropper(seg_ED.transpose(2, 0, 1)).transpose(1, 2, 0)
        seg_ES_crop = cropper(seg_ES.transpose(2, 0, 1)).transpose(1, 2, 0)

        # three slices
        num_slices = seg_ED.shape[-1]
        slices_idx = np.arange(0, num_slices)
        if args.three_slices:
            apical_idx = int(round((num_slices - 1) * 0.75))  # 75% from basal
            mid_ven_idx = int(round((num_slices - 1) * 0.5))  # 50% from basal
            basal_idx = int(round((num_slices - 1) * 0.25))  # 25% from basal
            slices_idx = [apical_idx, mid_ven_idx, basal_idx]

            seg_ED_crop = seg_ED_crop[:, :, slices_idx]
            seg_ES_crop = seg_ES_crop[:, :, slices_idx]

        ## evaluate un-registered metrics

        # dice
        unreg_dice_lv = categorical_dice_stack(seg_ES_crop, seg_ED_crop, label_class=1)
        unreg_dice_myo = categorical_dice_stack(seg_ES_crop, seg_ED_crop, label_class=2)
        unreg_dice_rv = categorical_dice_stack(seg_ES_crop, seg_ED_crop, label_class=3)

        unreg_dice_lv_buffer += [unreg_dice_lv]
        unreg_dice_myo_buffer += [unreg_dice_myo]
        unreg_dice_rv_buffer += [unreg_dice_rv]

        # contour distances
        unreg_mcd_lv, unreg_hd_lv = contour_distances_stack(seg_ES_crop, seg_ED_crop, label_class=1, dx=args.pixel_size)
        unreg_mcd_myo, unreg_hd_myo = contour_distances_stack(seg_ES_crop, seg_ED_crop, label_class=2, dx=args.pixel_size)
        unreg_mcd_rv, unreg_hd_rv = contour_distances_stack(seg_ES_crop, seg_ED_crop, label_class=3, dx=args.pixel_size)

        unreg_mcd_lv_buffer += [unreg_mcd_lv]
        unreg_hd_lv_buffer += [unreg_hd_lv]
        unreg_mcd_myo_buffer += [unreg_mcd_myo]
        unreg_hd_myo_buffer += [unreg_hd_myo]
        unreg_mcd_rv_buffer += [unreg_mcd_rv]
        unreg_hd_rv_buffer += [unreg_hd_rv]

        t.update()


# save all metrics evaluated for al test subjects
# construct the pd.DataFrame
df_buffer = []
column_method = ['Unreg'] * len(subj_id_buffer)
for struct in ['LV', 'MYO', 'RV']:
    if struct == 'LV':
        ls_dice = unreg_dice_lv_buffer
        ls_mcd = unreg_mcd_lv_buffer
        ls_hd = unreg_hd_lv_buffer
    elif struct == 'MYO':
        ls_dice = unreg_dice_myo_buffer
        ls_mcd = unreg_mcd_myo_buffer
        ls_hd = unreg_hd_myo_buffer
    elif struct == 'RV':
        ls_dice = unreg_dice_rv_buffer
        ls_mcd = unreg_mcd_rv_buffer
        ls_hd = unreg_hd_rv_buffer

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
metrics_df.to_pickle("{0}/unreg_results.pkl".format(output_dir))


# the mean and std of metrics
unreg_metrics = {'dice_lv_mean': np.mean(unreg_dice_lv_buffer), 'dice_lv_std': np.std(unreg_dice_lv_buffer),
                 'dice_myo_mean': np.mean(unreg_dice_myo_buffer), 'dice_myo_std': np.std(unreg_dice_myo_buffer),
                 'dice_rv_mean': np.mean(unreg_dice_rv_buffer), 'dice_rv_std': np.std(unreg_dice_rv_buffer),
                 'mcd_lv_mean': np.mean(unreg_mcd_lv_buffer), 'mcd_lv_std': np.std(unreg_mcd_lv_buffer),
                 'mcd_myo_mean': np.mean(unreg_mcd_myo_buffer), 'mcd_myo_std': np.std(unreg_mcd_myo_buffer),
                 'mcd_rv_mean': np.mean(unreg_mcd_rv_buffer), 'mcd_rv_std': np.std(unreg_mcd_rv_buffer),
                 'hd_lv_mean': np.mean(unreg_hd_lv_buffer), 'hd_lv_std': np.std(unreg_hd_lv_buffer),
                 'hd_myo_mean': np.mean(unreg_hd_myo_buffer), 'hd_myo_std': np.std(unreg_hd_myo_buffer),
                 'hd_rv_mean': np.mean(unreg_hd_rv_buffer), 'hd_rv_std': np.std(unreg_hd_rv_buffer),
                 }

unreg_save_path = '{0}/unreg_results_3slices_{1}.json'.format(output_dir, args.three_slices)
xutils.save_dict_to_json(unreg_metrics, unreg_save_path)

logging.info("Evaluation of unregistered images complete. Metric results saved at: \n\t{}".format(output_dir))
