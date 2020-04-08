"""Evaluates the model"""

from tqdm import tqdm
import os
from os import path

import argparse
import logging
import numpy as np
import pandas as pd
import nibabel as nib
import torch

from data.datasets import Data
from model.models import RegDVF
from model.submodules import spatial_transform
from model.losses import loss_fn
from utils.metrics import detJac_stack, rmse, rmse_dvf, aee
from utils.image import bbox_from_mask, normalise_intensity
from utils.transform import normalise_dvf, denormalise_dvf
from utils.misc import save_val_visual_results
from utils import misc

def evaluate(model, loss_fn, dataloader, params, args, epoch=0, val=False, save=False):
    """
    Evaluate the model and returns metrics as a dict and evaluation loss

    Args:
        model:
        loss_fn:
        dataloader: validation / testing dataloader
        params:
        args:
        val: (boolean) indicates validation (True) or testing (False)
        save: (boolen) save inference results (True)

    Returns:

    """
    model.eval()

    # empty lists buffer
    val_loss_buffer = []

    AEE_buffer = []
    RMSE_dvf_buffer = []
    RMSE_buffer = []

    mean_mag_grad_detJ_buffer = []
    negative_detJ_buffer = []

    # set up a test result directory
    if not val:
        test_result_dir = path.join(args.model_dir, "test_results")
        if not path.exists(test_result_dir):
            os.makedirs(test_result_dir)

    with tqdm(total=len(dataloader)) as t:
        # iterate over validation subjects
        for idx, data_point in enumerate(dataloader):

            # input data now is completely not normalised
            target, source, target_original, brain_mask, dvf_gt = data_point  # (1, N, (2,) H, W)

            # reshaping
            target = target.transpose(0, 1)  # (N, 1, H, W)
            source = source.transpose(0, 1)  # (N, 1, H, W)
            target_original = target_original.transpose(0, 1)  # (N, 1, H, W)
            brain_mask = brain_mask.transpose(0, 1)  # (N, 1, H, W)
            dvf_gt = dvf_gt[0, ...]  # (N, 2, H, W)

            # mono-modal case
            if params.modality == "mono":
                source = target_original

            ### (deprecated)
            # meanstd intensity normalisation (val & test data is only min-max normalised)
            target_input = target.numpy().copy()  # because normalisation op is in-place
            # target_input = normalise_intensity(target_input[:, 0, ...], mode="meanstd")[:, np.newaxis, ...]
            target_input = torch.from_numpy(target_input).to(device=args.device)  # (Nx1xHxW)

            source_input = source.numpy().copy()
            # source_input = normalise_intensity(source_input[:, 0, ...], mode="meanstd")[:, np.newaxis, ...]
            source_input = torch.from_numpy(source_input).to(device=args.device)  # (Nx1xHxW)
            ###

            with torch.no_grad():
                # compute DVF and warped source image towards target
                dvf_pred, warped_source_input = model(target_input, source_input)
                val_loss, _ = loss_fn(dvf_pred, target_input, warped_source_input, params)
                val_loss_buffer += [val_loss]

                # warp original target image using the predicted dvf
                target_pred = spatial_transform(target_original.to(device=args.device), dvf_pred).cpu()

                # warp minmax normalised source image using predicted dvf for visualisation
                warped_source = spatial_transform(source.to(device=args.device), dvf_pred).cpu()

                # reverse-normalise DVF to number of pixels
                dvf_pred = dvf_pred.cpu()
                dvf_pred = denormalise_dvf(dvf_pred)

                # set predicted DVF to zeros when measuring identity transformation (Id) baseline
                # if args.Id_baseline:
                #     dvf_pred *= 0.
                #     ## dev
                #     print(dvf_pred.mean(), dvf_pred.max())

            # cast images (N, 1, H, W) and dvf (N, 2, H, W) to numpy tensor
            # for metrics and visualisation
            target = target.numpy()
            target_original = target_original.numpy()
            target_pred = target_pred.numpy()
            source = source.numpy()
            warped_source = warped_source.numpy()
            brain_mask = brain_mask.numpy()

            dvf_pred = dvf_pred.numpy()  # (N, 2, H, W)
            dvf_gt = dvf_gt.numpy()  # (N, 2, H, W)

            """
            Calculating metrics
            """
            ## DVF accuracy vs. ground truth
            # find brian mask bbox mask
            mask_bbox, mask_bbox_mask = bbox_from_mask(brain_mask[:, 0, ...])

            # mask both prediction and ground truth with brain mask
            dvf_pred_brain_masked = dvf_pred * brain_mask   # (N, 2, H, W) * (N, 1, H, W) = (N, 2, H, W)
            dvf_gt_brain_masked = dvf_gt * brain_mask   # (N, 2, H, W) * (N, 1, H, W) = (N, 2, H, W)

            # crop out DVF within the brain mask bbox
            dvf_pred_brain_bbox_cropped = dvf_pred_brain_masked[:, :,
                                     mask_bbox[0][0]:mask_bbox[0][1],
                                     mask_bbox[1][0]:mask_bbox[1][1]]  # (N, 2, H', W')
            dvf_gt_brain_bbox_cropped = dvf_gt_brain_masked[:, :,
                                        mask_bbox[0][0]:mask_bbox[0][1],
                                        mask_bbox[1][0]:mask_bbox[1][1]]  # (N, 2, H', W')

            # measure both Averaged End-point Error (AEE) and RMSE of DVF (in Qin et al. IPMI 2019)
            AEE_buffer += [aee(dvf_pred_brain_bbox_cropped, dvf_gt_brain_bbox_cropped)]
            RMSE_dvf_buffer += [rmse_dvf(dvf_pred_brain_bbox_cropped, dvf_gt_brain_bbox_cropped)]


            ## RMSE(image) between the target and the target_pred
            # crop out the images within the brain mask bbox to avoid background
            target_brain_bbox_cropped = target[:, :,
                                        mask_bbox[0][0]:mask_bbox[0][1],
                                        mask_bbox[1][0]:mask_bbox[1][1]]  # (N, 1, H', W')
            target_pred_brain_bbox_cropped = target_pred[:, :,
                                             mask_bbox[0][0]:mask_bbox[0][1],
                                             mask_bbox[1][0]:mask_bbox[1][1]]  # (N, 1, H', W')

            RMSE_buffer += [rmse(target_brain_bbox_cropped, target_pred_brain_bbox_cropped)]


            ## Jacobian metrics
            mean_grad_detJ, mean_negative_detJ = detJac_stack(dvf_pred_brain_bbox_cropped.transpose(0, 2, 3, 1),  # (N, H, W, 2)
                                                              rescaleFlow=False)
            mean_mag_grad_detJ_buffer += [mean_grad_detJ]
            negative_detJ_buffer += [mean_negative_detJ]
            """"""

            """ Save predicted DVF and warped images """
            if args.save:
                test_output_dir = path.join(test_result_dir, "output")
                if not path.exists(test_output_dir):
                    os.makedirs(test_output_dir)

                subj_id = dataloader.dataset.subject_list[idx]
                subject_output_dir = path.join(test_output_dir, subj_id)
                if not path.exists(subject_output_dir):
                    os.makedirs(subject_output_dir)

                nib.save(nib.Nifti1Image(target.transpose(2,3,0,1), np.eye(4)),
                         f"{subject_output_dir}/target.nii.gz")

                nib.save(nib.Nifti1Image(source.transpose(2,3,0,1), np.eye(4)),
                         f"{subject_output_dir}/source.nii.gz")

                nib.save(nib.Nifti1Image(target_original.transpose(2,3,0,1), np.eye(4)),
                         f"{subject_output_dir}/target_original.nii.gz")

                nib.save(nib.Nifti1Image(target_pred.transpose(2,3,0,1), np.eye(4)),
                         f"{subject_output_dir}/target_pred.nii.gz")

                nib.save(nib.Nifti1Image(dvf_gt.transpose(2,3,0,1), np.eye(4)),
                         f"{subject_output_dir}/dvf_gt.nii.gz")

                nib.save(nib.Nifti1Image(dvf_pred.transpose(2,3,0,1), np.eye(4)),
                         f"{subject_output_dir}/dvf_pred.nii.gz")

            t.update()

    # construct metrics dict
    results_dict = {}

    # RMSE (dvf) and RMSE (image)
    rmse_criteria = ["AEE", "RMSE_dvf", "RMSE"]
    for cr in rmse_criteria:
        result_name = cr
        the_buffer = locals()[f'{result_name}_buffer']  # get buffer object from local namespace
        results_dict[f'{result_name}_mean'] = np.mean(the_buffer)
        results_dict[f'{result_name}_std'] = np.std(the_buffer)

    # regularity
    reg_criteria = ['mean_mag_grad_detJ', 'negative_detJ']
    for cr in reg_criteria:
        result_name = cr
        the_buffer = locals()[f'{result_name}_buffer']
        results_dict[f'{result_name}_mean'] = np.mean(the_buffer)
        results_dict[f'{result_name}_std'] = np.std(the_buffer)

    # sanity check: proportion of negative Jacobian points should be lower than 1
    assert results_dict['negative_detJ_mean'] <= 1, "Invalid det Jac: Ratio of folding points > 1"

    """
    Validation only:
    """
    if val:
        # determine if this is the best model so far
        current_one_metric = np.mean([results_dict['RMSE_mean'], results_dict['AEE_mean']])  # error metrics, lower the better
        if epoch + 1 == params.val_epochs:
            # initialise for the first validation
            params.is_best = False
            params.best_one_metric = current_one_metric
        elif current_one_metric < params.best_one_metric:
            params.is_best = True
            params.best_one_metric = current_one_metric

        ## Saving results JSON files
        # save results of this validation
        save_path = path.join(args.model_dir,
                                 f"val_results_last.json")
        misc.save_dict_to_json(results_dict, save_path)

        # save the results for the best model
        if params.is_best:
            save_path = path.join(args.model_dir,
                                     f"val_results_best.json")
            misc.save_dict_to_json(results_dict, save_path)

        ## validation visual results
        val_results_dir = args.model_dir + "/val_visual_results"
        if not path.exists(val_results_dir):
            os.makedirs(val_results_dir)

        save_val_visual_results(target,
                                target_original,
                                source,
                                warped_source,
                                target_pred,
                                dvf_pred,
                                dvf_gt,
                                val_results_dir, epoch, dpi=50)

    """
    Testing only: 
    """
    if not val:
        # save the overall test results
        save_path = path.join(test_result_dir, "test_results.json")
        misc.save_dict_to_json(results_dict, save_path)

        # save evaluated metrics for individual test subjects in pandas dataframe for boxplots
        subj_id_buffer = dataloader.dataset.subject_list  # this list should be in the same order as subjects are evaluated
        df_buffer = []
        column_method = ['DL'] * len(subj_id_buffer)

        # save RMSE and AEE
        rmse_data = {'Method': column_method,
                     'ID': subj_id_buffer,
                     'AEE': AEE_buffer,
                     'RMSE_dvf': RMSE_dvf_buffer,
                     'RMSE': RMSE_buffer
                     }
        rmse_df = pd.DataFrame(data=rmse_data)
        rmse_df.to_pickle(f"{test_result_dir}/DL_test_subjects_rmse.pkl")

        # save detJac metrics
        jac_data = {'Method': column_method,
                    'ID': subj_id_buffer,
                    'GradDetJac': mean_mag_grad_detJ_buffer,
                    'NegDetJac': negative_detJ_buffer}
        jac_df = pd.DataFrame(data=jac_data)
        jac_df.to_pickle(f"{test_result_dir}/DL_test_subjects_jacDet.pkl")

        # todo: option to save all outputs to be analysed later

    return results_dict, torch.stack(val_loss_buffer).mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',
                        default='experiments/base_model',
                        help="Directory containing params.json")

    parser.add_argument('--Id_baseline',
                        action='store_true',
                        help="Test identity transformation baseline")

    parser.add_argument('--restore_file',
                        default='best',
                        help="Prefix of the checkpoint file:"
                             " 'best' for best model, or 'last' for the last saved checkpoint")

    parser.add_argument('--save',
                        action='store_true',
                        help="Save deformed images and predicted DVF etc. if True.")

    parser.add_argument('--cpu',
                        action='store_true',
                        help='Use CPU if given')

    parser.add_argument('--num_workers',
                        default=8,
                        help='Number of processes used by dataloader, 0 means use main process')

    parser.add_argument('--gpu',
                        default=0,
                        help='Choose the GPU to run on, pass -1 to use CPU')


    args = parser.parse_args()

    # set the GPU to use and device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.cuda = not args.cpu and torch.cuda.is_available()
    if args.cuda:
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    # check whether the model directory exists
    assert path.exists(args.model_dir), "No model dir found at {}".format(args.model_dir)

    # set up a logger
    misc.set_logger(path.join(args.model_dir, 'eval.log'))
    logging.info("What a beautiful day to save lives!")
    logging.info("Model: {}".format(args.model_dir))

    # load parameters from model JSON file
    json_path = path.join(args.model_dir, 'params.json')
    assert path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = misc.Params(json_path)

    """Data"""
    logging.info("Setting up data loaders...")
    data = Data(args, params)
    data.use_brain()
    logging.info("- Done.")
    """"""

    """Model and parameter loading"""
    model = RegDVF(params.network, transform_model=params.transform_model)
    model = model.to(device=args.device)

    logging.info(f"Loading model parameters from: "
                 f"{path.join(args.model_dir, args.restore_file + '.pth.tar')}")
    misc.load_checkpoint(path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    """Run evaluation"""
    logging.info("Running evaluation...")
    evaluate(model, loss_fn, data.test_dataloader, params, args, val=False)
    logging.info("Evaluation complete. Model path {}".format(args.model_dir))
