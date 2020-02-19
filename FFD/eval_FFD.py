"""Evaluate FFD on the task of cardiac motion estimation"""
import os
import nibabel as nib
import argparse
import numpy as np
import logging
from tqdm import tqdm
import pandas as pd

from model.dataset_utils import CenterCrop
from utils.data_utils import split_volume_idmat, split_volume
from utils.metrics import categorical_dice_stack, contour_distances_stack, detJac_stack
from utils import xutils
from utils.dvf_utils import dof_to_dvf

parser = argparse.ArgumentParser()
parser.add_argument('-CPS', default=8, help="B-spline FFD control point distance.")
parser.add_argument('-BE', default=1e-4, help="Bending Energy weighting.")
parser.add_argument('-sim', default='NMI', help="(Dis-)similarity measure of registration.")
parser.add_argument('-intensity', default=None, help="Intensity transformation applied to source images. "
                                                     "'inv' means inverse of source image")

parser.add_argument('--model_dir', default=None)
parser.add_argument('--data_dir', default='data/ukbb/cine_ukbb2964/small_set/sa/val_autoseg')
parser.add_argument('--label_prefix', default='seg', help='Prefix of file names of segmentation masks')
parser.add_argument('--pixel_size', default=1.8, help='Dimension of in-plane pixels in canonical space, assume ^2')
parser.add_argument('--seq', default='sa', help='Imaging view, sa, la_2ch or la_4ch')
parser.add_argument('--no_three_slices', action='store_true', help="Evaluate metrics on 3 slices.")
parser.add_argument('--crop_size', default=160)
parser.add_argument('--clean', action='store_true', help="Remove intermediate files.")


# parse arguments (this is probably lazy programming...)
args = parser.parse_args()
data_dir = args.data_dir
label_prefix = args.label_prefix
args.three_slices = not args.no_three_slices

model_dir = args.model_dir
assert os.path.exists(model_dir), "Model directory does not exist!"

# save parameters passed to MIRTK via command arguments to a par.conf file
parout = os.path.join(model_dir, "par.conf")

# set up a logger
xutils.set_logger(os.path.join(model_dir, 'ffd_eval.log'))
logging.info('Starting FFD evaluation...')

# metric buffers
dice_lv_buffer = []
dice_myo_buffer = []
dice_rv_buffer = []

mcd_lv_buffer = []
hd_lv_buffer = []
mcd_myo_buffer = []
hd_myo_buffer = []
mcd_rv_buffer = []
hd_rv_buffer = []

dice_buffer = []
mcd_buffer = []
hd_buffer = []

mean_mag_grad_detJ_buffer = []
negative_detJ_buffer = []

subj_id_buffer = []

logging.info('Looping over subjects...')

with tqdm(total=len(os.listdir(data_dir))) as t:
    # loop over subjects
    for subj_id in sorted(os.listdir(data_dir)):
        subj_dir = os.path.join(data_dir, subj_id)
        subj_output_dir = os.path.join(model_dir, "tmp", subj_id)
        if not os.path.exists(subj_output_dir):
            os.makedirs(subj_output_dir)

        # save a list of subject id for dataFrames
        subj_id_buffer += [subj_id]


        # load in the ED and ES images
        nim_ED = nib.load(os.path.join(subj_dir, 'sa_ED.nii.gz'))
        nim_ES = nib.load(os.path.join(subj_dir, 'sa_ES.nii.gz'))
        nseg_ED = nib.load(os.path.join(subj_dir, '{0}_sa_ED.nii.gz'.format(label_prefix)))
        nseg_ES = nib.load(os.path.join(subj_dir, '{0}_sa_ES.nii.gz'.format(label_prefix)))

        # shape (H, W, N)
        img_ED = nim_ED.get_data()
        img_ES = nim_ES.get_data()
        seg_ED = nseg_ED.get_data()
        seg_ES = nseg_ES.get_data()

        # cropping
        cropper = CenterCrop(output_size=args.crop_size)  # input shape should be (N, H, W)
        img_ED_crop = cropper(img_ED.transpose(2, 0, 1)).transpose(1, 2, 0)
        img_ES_crop = cropper(img_ES.transpose(2, 0, 1)).transpose(1, 2, 0)
        seg_ED_crop = cropper(seg_ED.transpose(2, 0, 1)).transpose(1, 2, 0)
        seg_ES_crop = cropper(seg_ES.transpose(2, 0, 1)).transpose(1, 2, 0)

        # optional intensity transformation
        if args.intensity == "inv":
            img_ES_crop = img_ES_crop.max() - img_ES_crop

        #--- using identity matrix as image2world matrix to ensure correct DOF to DVF conversion--- #
        # save the image and segmentation to NIFTI files with identity image-to-world transformation matrix
        nim_ED_crop = nib.Nifti1Image(img_ED_crop, np.eye(4))
        nib.save(nim_ED_crop, '{0}/sa_ED_crop.nii.gz'.format(subj_output_dir))
        nim_ES_crop = nib.Nifti1Image(img_ES_crop, np.eye(4))
        nib.save(nim_ES_crop, '{0}/sa_ES_crop.nii.gz'.format(subj_output_dir))

        nseg_ED_crop = nib.Nifti1Image(seg_ED_crop, np.eye(4))
        nib.save(nseg_ED_crop, '{0}/{1}_sa_ED_crop.nii.gz'.format(subj_output_dir, label_prefix))
        nseg_ES_crop = nib.Nifti1Image(seg_ES_crop, np.eye(4))
        nib.save(nseg_ES_crop, '{0}/{1}_sa_ES_crop.nii.gz'.format(subj_output_dir, label_prefix))


        # split volume into slices and change to identity i2w matrix
        # images
        split_volume_idmat('{0}/sa_ED_crop.nii.gz'.format(subj_output_dir), '{0}/sa_ED_crop_z'.format(subj_output_dir))
        split_volume_idmat('{0}/sa_ES_crop.nii.gz'.format(subj_output_dir), '{0}/sa_ES_crop_z'.format(subj_output_dir))

        # segmentation
        split_volume_idmat('{0}/{1}_sa_ED_crop.nii.gz'.format(subj_output_dir, label_prefix),
                           '{0}/{1}_sa_ED_crop_z'.format(subj_output_dir, label_prefix))
        split_volume_idmat('{0}/{1}_sa_ES_crop.nii.gz'.format(subj_output_dir, label_prefix),
                           '{0}/{1}_sa_ES_crop_z'.format(subj_output_dir, label_prefix))



        # three slices
        num_slices = img_ED.shape[-1]
        slices_idx = np.arange(0, num_slices)
        if args.three_slices:
            apical_idx = int(round((num_slices - 1) * 0.75))  # 75% from basal
            mid_ven_idx = int(round((num_slices - 1) * 0.5))  # 50% from basal
            basal_idx = int(round((num_slices - 1) * 0.25))  # 25% from basal
            slices_idx = [apical_idx, mid_ven_idx, basal_idx]

            # slice the ground truth accordingly
            seg_ED_crop = seg_ED_crop[:, :, slices_idx]
            seg_ES_crop = seg_ES_crop[:, :, slices_idx]

        # buffer for warped segmentation of ES frame and dense DDF
        warped_seg_sa_ES = []
        dvf = []


        # loop over the slices
        for z in slices_idx:

            # forward registration
            target_img_path = '{0}/sa_ED_crop_z{1:02d}.nii.gz'.format(subj_output_dir, z)
            source_img_path = '{0}/sa_ES_crop_z{1:02d}.nii.gz'.format(subj_output_dir, z)
            dof = '{0}/ffd_z{1:02d}_ED_to_ES.dof.gz'.format(subj_output_dir, z)
            os.system(f'mirtk register {target_img_path} {source_img_path}  '
                      f'-sim {args.sim} -ds {args.CPS} -be {args.BE} '
                      f'-model FFD -levels 3 -padding -1 -bins 64 '
                      f'-parout {parout} -dofout {dof} -verbose 0')

            # use MIRTK to warp ES frame segmentation mask to ED frame
            target_seg_path = '{0}/{2}_sa_ED_crop_z{1:02d}.nii.gz'.format(subj_output_dir, z, label_prefix)
            source_seg_path = '{0}/{2}_sa_ES_crop_z{1:02d}.nii.gz'.format(subj_output_dir, z, label_prefix)
            warped_seg_path = '{0}/warp_ffd_{2}_sa_ES_z{1:02d}.nii.gz'.format(subj_output_dir, z, label_prefix)
            os.system('mirtk transform-image '
                      '{0} {1} -dofin {2} -target {3} -interpolation NN'.format(source_seg_path, warped_seg_path, dof, target_seg_path))

            # read warped segmentation back in and form numpy arrays
            warped_seg_sa_ES += [nib.load(warped_seg_path).get_data()[:, :, 0]]

            # convert the DOF file to dense DDF
            dvf_name ='dvf_ffd_ED_to_ES_z{0:02d}'.format(z)
            dvf += [dof_to_dvf(target_seg_path, dof, dvf_name, subj_output_dir)]


        ## evaluate metrics
        # reshape warped segmentation to numpy array of shape (H, W, N)
        warped_seg_sa_ES = np.array(warped_seg_sa_ES).transpose(1, 2, 0)
        # sanity check on shapes
        assert warped_seg_sa_ES.shape == seg_ED_crop.shape, 'Shape mismatch between ED label and warped ES label'

        # dice
        dice_lv = categorical_dice_stack(warped_seg_sa_ES, seg_ED_crop, label_class=1)
        dice_myo = categorical_dice_stack(warped_seg_sa_ES, seg_ED_crop, label_class=2)
        dice_rv = categorical_dice_stack(warped_seg_sa_ES, seg_ED_crop, label_class=3)

        dice_lv_buffer += [dice_lv]
        dice_myo_buffer += [dice_myo]
        dice_rv_buffer += [dice_rv]

        # contour distances
        mcd_lv, hd_lv = contour_distances_stack(warped_seg_sa_ES, seg_ED_crop, label_class=1, dx=args.pixel_size)
        mcd_myo, hd_myo = contour_distances_stack(warped_seg_sa_ES, seg_ED_crop, label_class=2, dx=args.pixel_size)
        mcd_rv, hd_rv = contour_distances_stack(warped_seg_sa_ES, seg_ED_crop, label_class=3, dx=args.pixel_size)

        mcd_lv_buffer += [mcd_lv]
        hd_lv_buffer += [hd_lv]
        mcd_myo_buffer += [mcd_myo]
        hd_myo_buffer += [hd_myo]
        mcd_rv_buffer += [mcd_rv]
        hd_rv_buffer += [hd_rv]

        # Jacobian related metrics
        dvf = np.array(dvf)  # (N, H, W, 2)
        mean_mag_grad_detJ_mean, negative_detJ_mean = detJac_stack(dvf, rescaleFlow=False)
        mean_mag_grad_detJ_buffer += [mean_mag_grad_detJ_mean]
        negative_detJ_buffer += [negative_detJ_mean]

        t.update()


# save all metrics evaluated for all test subjects, used for boxplot visualisation of the results
# Dice & contour distances
df_buffer = []
column_method = ['FFD'] * len(subj_id_buffer)
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
    else:
        raise ValueError("Structure not recognised.")

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
metrics_df.to_pickle("{0}/FFD_all_subjects_accuracy_3slices_{1}.pkl".format(model_dir, args.three_slices))

# Jacobian-related metrics
jac_data = {'Method': column_method,
            'ID': subj_id_buffer,
            'GradDetJac': mean_mag_grad_detJ_buffer,
            'NegDetJac': negative_detJ_buffer}
jac_df = pd.DataFrame(data=jac_data)
jac_df.to_pickle("{0}/FFD_all_subjects_jac_3slices_{1}.pkl".format(model_dir, args.three_slices))


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


# save the results in a JSON file
save_path = '{0}/ffd_results_3slices_{1}.json'.format(model_dir, args.three_slices)
xutils.save_dict_to_json(metrics, save_path)


# clean up intermediate data
if args.clean:
    tmp_path = os.path.join(model_dir, "tmp")
    os.system(f"rm -rf {tmp_path}")

logging.info("Evaluation complete. Metric results saved at: \n\t{}".format(model_dir))
