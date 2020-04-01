"""Evaluate FFD on the task of cardiac motion estimation"""
import os
import nibabel as nib
import argparse
import numpy as np
import logging
from tqdm import tqdm
import pandas as pd

from data.utils import Normalise
from utils.split_nifti import split_volume_idmat
from utils.metrics import categorical_dice_stack, contour_distances_stack, detJac_stack, aee, rmse, rmse_dvf
from utils import misc
from utils.transform import dof_to_dvf
from utils.image import bbox_from_mask

parser = argparse.ArgumentParser()
## MIRTK FFD parameters
parser.add_argument('-sim', default='NMI', help="(Dis-)similarity measure of registration.")
parser.add_argument('-CPS', default=8, help="B-spline FFD control point distance.")
parser.add_argument('-BE', default=1e-4, help="Bending Energy weighting.")

parser.add_argument('--run_dir', default=None, help="Base directory to run FFD in")
parser.add_argument('--data_dir', default='data/brats17/miccai2020/test_crop192_sigma8_cps10_dispRange1.5-2.5_sliceRange70-90')
parser.add_argument('--test', action='store_true', help="True if testing.")
parser.add_argument('--clean', action='store_true', help="Remove intermediate files.")
parser.add_argument('--verbose', default=1, help="Verbose level of MIRTK FFD")

# parse arguments (this is just lazy programming...)
args = parser.parse_args()
data_dir = args.data_dir

# FFD model dir
model_dir = os.path.join(args.run_dir, f"ffd_sim_{args.sim}_CPS_{args.CPS}_BE_{args.BE}")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# save parameters passed to MIRTK via command arguments to a par.conf file
parout = os.path.join(model_dir, "par.conf")

# set up a logger
misc.set_logger(os.path.join(model_dir, 'ffd_eval.log'))
logging.info('Starting FFD evaluation...')



# metric result buffers
AEE_buffer = []
RMSE_DVF_buffer = []
RMSE_buffer = []

mean_mag_grad_detJ_buffer = []
negative_detJ_buffer = []
subj_id_buffer = []



# image intensity normaliser
normaliser_minmax = Normalise(mode="minmax")

logging.info('Looping over subjects...')
with tqdm(total=len(os.listdir(data_dir))) as t:
    # loop over subjects
    for subj_id in sorted(os.listdir(data_dir)):
        subj_data_dir = os.path.join(data_dir, subj_id)
        subj_output_dir = os.path.join(model_dir, "tmp", subj_id)  # temporary output dir per subject
        if not os.path.exists(subj_output_dir):
            os.makedirs(subj_output_dir)

        # save a list of subject id for dataFrames
        subj_id_buffer += [subj_id]

        # load in the ED and ES images
        nim_target = nib.load(os.path.join(subj_data_dir, f'{subj_id}_t1_deformed.nii.gz'))
        nim_source = nib.load(os.path.join(subj_data_dir, f'{subj_id}_t2.nii.gz'))
        nim_target_original = nib.load(os.path.join(subj_data_dir, f'{subj_id}_t1.nii.gz'))

        # shape (H, W, N)
        target, source, target_original = [x.get_data() for x in [nim_target, nim_source, nim_target_original]]

        # split volume into 2D slices
        split_volume_idmat(f'{subj_data_dir}/{subj_id}_t1_deformed.nii.gz', f'{subj_output_dir}/target_z')  # target, T1-syn
        split_volume_idmat(f'{subj_data_dir}/{subj_id}_t2.nii.gz', f'{subj_output_dir}/source_z')  # source, T2
        split_volume_idmat(f'{subj_data_dir}/{subj_id}_t1.nii.gz', f'{subj_output_dir}/target_original_z')  # target_original, T1

        # segmentation


        dvf_buffer = []
        warped_target_original_buffer = []

        # loop over the slices
        for z in range(target.shape[-1]):

            # forward registration
            target_img_path = '{0}/target_z{1:02d}.nii.gz'.format(subj_output_dir, z)
            source_img_path = '{0}/source_z{1:02d}.nii.gz'.format(subj_output_dir, z)
            dof = '{0}/ffd_z{1:02d}.dof.gz'.format(subj_output_dir, z)
            os.system(f'mirtk register {target_img_path} {source_img_path}  '
                      f'-sim {args.sim} -ds {args.CPS} -be {args.BE} '
                      f'-model FFD -levels 3 -padding -1 -bins 64 '
                      f'-parout {parout} -dofout {dof} -verbose {args.verbose}')

            # use MIRTK to apply transformation to target original image
            target_original_path = f'{subj_output_dir}/target_original_z{z:02d}.nii.gz'  # input path
            warped_target_original_path = f'{subj_output_dir}/warped_target_original_z{z:02d}.nii.gz'  # output path
            os.system(f'mirtk transform-image '
                      f'{target_original_path} {warped_target_original_path} '
                      f'-dofin {dof} -target {target_img_path}')

            # read warped target original image back in and form numpy arrays
            warped_target_original_buffer += [nib.load(warped_target_original_path).get_data()[:, :, 0]]  # (H, W) each

            # convert the DOF file to dense DDF
            dvf_name ='dvf_z{0:02d}'.format(z)
            dvf_buffer += [dof_to_dvf(target_img_path, dof, dvf_name, subj_output_dir)]

        """Evaluate metrics"""
        # find brian mask bbox mask
        brain_mask = nib.load(os.path.join(subj_data_dir, f'{subj_id}_brainmask.nii.gz')).get_data()
        mask_bbox, mask_bbox_mask = bbox_from_mask(brain_mask)

        ## AEE
        # todo: unify these shapes!!!
        # to numpy tensor
        dvf = np.array(dvf_buffer)  # (N, H, W, 2)
        dvf_gt = nib.load(os.path.join(subj_data_dir, f'{subj_id}_dvf_t2_to_t1_deformed.nii.gz')).get_data()  # (H, W, N, 2)
        dvf_gt = dvf_gt.transpose(2, 0, 1, 3)  # (N, H, W, 2) - metric functions work with this dimension (accuracy & Jacobian)
        # mask DVF with mask bbox (actually slicing)

        ## experimental: mask both prediction and ground truth with brain mask
        brain_mask = brain_mask.transpose(2, 0, 1)  # (N, H, W)
        dvf_brainmasked = dvf * brain_mask[..., np.newaxis]
        dvf_gt_brain_masked = dvf_gt * brain_mask[..., np.newaxis]
        ##

        # crop by brain mask bbox to reduce background
        dvf_bbox_cropped = dvf_brainmasked[:,
                     mask_bbox[0][0]:mask_bbox[0][1],
                     mask_bbox[1][0]:mask_bbox[1][1],
                     :]  # (N, H', W', 2)
        dvf_gt_bbox_cropped = dvf_gt_brain_masked[:,
                        mask_bbox[0][0]:mask_bbox[0][1],
                        mask_bbox[1][0]:mask_bbox[1][1],
                        :]  # (N, H', W', 2)
        AEE = aee(dvf_bbox_cropped, dvf_gt_bbox_cropped)
        AEE_buffer += [AEE]

        RMSE_DVF = rmse_dvf(dvf_bbox_cropped,dvf_gt_bbox_cropped)
        RMSE_DVF_buffer += [RMSE_DVF]

        ## RMSE(image)
        #   between the target (T1-moved by ground truth DVF) and the warped_target_original (T1-moved by predicted DVF)
        # to numpy tensor
        target = target.transpose(2, 0, 1)  # (N, H, W)
        warped_target_original = np.array(warped_target_original_buffer)  # (N, H, W)

        # [run49 fix] minmax normalise to [0,1] for metric evaluation (to be comparable Qin Chen's numbers)
        target = normaliser_minmax(target)  # T1 warped by ground truth DVF
        warped_target_original = normaliser_minmax(warped_target_original)  # T1 warped warped by predicted DVF
        ##

        target_masked = target[:,
                        mask_bbox[0][0]:mask_bbox[0][1],
                        mask_bbox[1][0]:mask_bbox[1][1],
                        ]  # (N, H', W')
        warped_target_original_masked = warped_target_original[:,
                                        mask_bbox[0][0]:mask_bbox[0][1],
                                        mask_bbox[1][0]:mask_bbox[1][1]]  # (N, H', W')

        RMSE = rmse(target_masked, warped_target_original_masked)
        RMSE_buffer += [RMSE]

        # Jacobian related metrics
        mean_mag_grad_detJ_mean, negative_detJ_mean = detJac_stack(dvf, rescaleFlow=False)
        mean_mag_grad_detJ_buffer += [mean_mag_grad_detJ_mean]
        negative_detJ_buffer += [negative_detJ_mean]


        """ Save DVF and warped target original at test time """
        if args.test:
            subj_test_output_dir = f"{model_dir}/test_output/{subj_id}"
            if not os.path.exists(subj_test_output_dir):
                os.makedirs(subj_test_output_dir)

            dvf = dvf.transpose(1, 2, 0, 3)  # (H, W, N, 2) - shape used in NIFTI
            nim_dvf = nib.Nifti1Image(dvf, nim_target.affine, nim_target.header)
            nib.save(nim_dvf, f"{subj_test_output_dir}/dvf.nii.gz")
            warped_target_original = warped_target_original.transpose(1, 2, 0)
            nim_warped_target_original = nib.Nifti1Image(warped_target_original, nim_target.affine, nim_target.header)
            nib.save(nim_warped_target_original, f"{subj_test_output_dir}/warped_target_original.nii.gz")

        t.update()
# end loop over subjects


""" Save mean & std metrics """
# construct metrics dict
results = {}

# RMSE (dvf) and RMSE (image)
rmse_criteria = ["AEE", "RMSE_DVF", "RMSE"]
for cr in rmse_criteria:
    result_name = cr
    the_buffer = locals()[f'{result_name}_buffer']
    results[f'{result_name}_mean'] = np.mean(the_buffer)
    results[f'{result_name}_std'] = np.std(the_buffer)

# regularity
reg_criteria = ['mean_mag_grad_detJ', 'negative_detJ']
for cr in reg_criteria:
    result_name = cr
    the_buffer = locals()[f'{result_name}_buffer']
    results[f'{result_name}_mean'] = np.mean(the_buffer)
    results[f'{result_name}_std'] = np.std(the_buffer)

# sanity check: proportion of negative Jacobian points should be lower than 1
assert results['negative_detJ_mean'] <= 1, "Invalid det Jac: Ratio of folding points > 1"

# save
save_path = os.path.join(model_dir, "test_results.json")
misc.save_dict_to_json(results, save_path)



""" Save metric numbers for individual subjects for testing """
# save evaluated metrics for individual test subjects in pandas dataframe for boxplots
df_buffer = []
column_method = ['FFD'] * len(subj_id_buffer)
# save RMSE and AEE
rmse_data = {'Method': column_method,
             'ID': subj_id_buffer,
             'RMSE': RMSE_buffer,
             'RMSE_DVF': RMSE_DVF_buffer,
             'AEE': AEE_buffer}
rmse_df = pd.DataFrame(data=rmse_data)
rmse_df.to_pickle(f"{model_dir}/FFD_test_results_rmse.pkl")

# save detJac metrics
jac_data = {'Method': column_method,
            'ID': subj_id_buffer,
            'GradDetJac': mean_mag_grad_detJ_buffer,
            'NegDetJac': negative_detJ_buffer}
jac_df = pd.DataFrame(data=jac_data)
jac_df.to_pickle(f"{model_dir}/FFD_test_results_Jacobian.pkl")



# clean up intermediate data
if args.clean:
    tmp_path = os.path.join(model_dir, "tmp")
    os.system(f"rm -rf {tmp_path}")

logging.info("Evaluation complete. Metric results saved at: \n\t{}".format(model_dir))
