"""Evaluates the model"""

from tqdm import tqdm
import os
import argparse
import logging
import numpy as np
import pandas as pd
import torch

from data.datasets import Data
from model.models import RegDVF
from model.submodules import spatial_transform
from model.losses import loss_fn
from utils.metrics import categorical_dice_stack, contour_distances_stack, detJac_stack, rmse, rmse_dvf, aee
from utils import xutils

from utils.image_utils import bbox_from_mask
from utils.xutils import save_val_visual_results

from data.dataset_utils import Normalise

def evaluate(model, loss_fn, data, params, args, epoch=0, val=False, save=False):
    """
    Evaluate the model and returns metrics as a dict and evaluation loss

    Args:
        model:
        loss_fn:
        data: (instance of Data object)
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

    with tqdm(total=len(data.val_dataloader)) as t:
        # iterate over validation subjects
        for idx, data_point in enumerate(data.val_dataloader):

            # input data now is completely not normalised
            target, source, target_original, brain_mask, dvf_gt = data_point  # (1, N, (2,) H, W)

            if params.modality == "mono":
                source = target_original

            # intensity normalisation (val & test data are saved w/o normalisation)
            normaliser_meanstd = Normalise(mode="meanstd")
            normaliser_minmax = Normalise(mode="minmax")
            target_input = normaliser_minmax(normaliser_meanstd(target.numpy())).transpose(1, 0, 2, 3)
            target_input = torch.from_numpy(target_input)
            source_input = normaliser_minmax(normaliser_meanstd(source.numpy())).transpose(1, 0, 2, 3)
            source_input = torch.from_numpy(source_input)  # (N, 1, H, W)
            ##

            target_input = target_input.to(device=args.device)  # (Nx1xHxW)
            source_input = source_input.to(device=args.device)  # (Nx1xHxW)

            print(target_input.size())
            print(source_input.size())

            with torch.no_grad():
                # compute DVF and warped source image towards target
                dvf, warped_source = model(target_input, source_input)
                val_loss, _ = loss_fn(dvf, target_input, warped_source, params)
                val_loss_buffer += [val_loss]

                # warp original target image using the same dvf
                target_original = target_original.to(device=args.device).permute(1, 0, 2, 3)  # (Nx1xHxW)
                warped_target_original = spatial_transform(target_original, dvf)

                # reverse-normalise DVF to number of pixels
                dvf *= params.crop_size / 2

            """
            Calculating metrics
            (metrics calculation works with shape image: NxHxW, DVF: NxHxWx2)
            """
            ## DVF error
            # to numpy tensor
            dvf = dvf.data.cpu().numpy().transpose(0, 2, 3, 1)  # (N, H, W, 2)
            dvf_gt = dvf_gt.numpy().squeeze().transpose(0, 2, 3, 1)  # (N, H, W, 2)

            # mask both prediction and ground truth with brain mask
            brain_mask = brain_mask.cpu().numpy()[0,...]  # (N, H, W)
            dvf_brainmasked = dvf * brain_mask[..., np.newaxis]  # (N, H, W, 2) * (N, H, W, 1)
            dvf_gt_brain_masked = dvf_gt * brain_mask[..., np.newaxis]

            # find brian mask bbox mask
            mask_bbox, mask_bbox_mask = bbox_from_mask(brain_mask)

            # mask DVF with mask bbox (actually slicing)
            dvf_brainmasked_bbox_cropped = dvf_brainmasked[:,
                         mask_bbox[0][0]:mask_bbox[0][1],
                         mask_bbox[1][0]:mask_bbox[1][1],
                         :]  # (N, H', W', 2)
            dvf_gt_brain_masked_bbox_cropped = dvf_gt_brain_masked[:,
                            mask_bbox[0][0]:mask_bbox[0][1],
                            mask_bbox[1][0]:mask_bbox[1][1],
                            :]  # (N, H', W', 2)

            # measure both Averaged End-point Error (AEE) and RMSE of DVF (in Qin et al. IPMI 2019)
            AEE = aee(dvf_brainmasked_bbox_cropped, dvf_gt_brain_masked_bbox_cropped)
            AEE_buffer += [AEE]
            RMSE_dvf = rmse_dvf(dvf_brainmasked_bbox_cropped, dvf_gt_brain_masked_bbox_cropped)
            RMSE_dvf_buffer += [RMSE_dvf]

            ## RMSE(image)
            #   between the target (T1-moved by ground truth DVF) and the warped_target_original (T1-moved by predicted DVF)
            # [run49 fix] minmax normalise to [0,1] for metric evaluation (to be comparable Qin Chen's numbers)
            # todo: to measure RMSE the target and the warped target original really should be normalised by the same ratio
            target = target.cpu().numpy().squeeze()  # (N, H, W)
            target = normaliser_minmax(target)  # T1 warped by ground truth DVF
            warped_target_original = warped_target_original.cpu().numpy().squeeze()  # (N, H, W)
            warped_target_original = normaliser_minmax(warped_target_original)  # T1 warped warped by predicted DVF
            ##

            # measure only the brain bounding box area to avoid background errors
            target_masked = target[:,
                            mask_bbox[0][0]:mask_bbox[0][1],
                            mask_bbox[1][0]:mask_bbox[1][1],
                            ]  # (N, H', W')
            warped_target_original_masked = warped_target_original[:,
                                            mask_bbox[0][0]:mask_bbox[0][1],
                                            mask_bbox[1][0]:mask_bbox[1][1]]  # (N, H', W')

            RMSE = rmse(target_masked, warped_target_original_masked)
            RMSE_buffer += [RMSE]

            # determinant of Jacobian
            mean_grad_detJ, mean_negative_detJ = detJac_stack(dvf_brainmasked_bbox_cropped, rescaleFlow=False)
            mean_mag_grad_detJ_buffer += [mean_grad_detJ]
            negative_detJ_buffer += [mean_negative_detJ]
            """"""

            t.update()

    # construct metrics dict
    results = {}

    # RMSE (dvf) and RMSE (image)
    rmse_criteria = ["AEE", "RMSE_dvf", "RMSE"]
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

    """
    Validation:
    """
    if val:
        # determine if this is the best model yet
        current_one_metric = np.mean([results['RMSE_mean'], results['AEE_mean']])  # error metrics, lower the better
        if epoch + 1 == params.val_epochs:
            # initialise for the first validation
            params.is_best = False
            params.best_one_metric = current_one_metric
        elif current_one_metric < params.best_one_metric:
            params.is_best = True
            params.best_one_metric = current_one_metric

        # save the most recent results JSON
        save_path = os.path.join(args.model_dir,
                                 f"val_results_last.json")
        xutils.save_dict_to_json(results, save_path)

        # save the validation results for the best model JSON
        if params.is_best:
            save_path = os.path.join(args.model_dir,
                                     f"val_results_best.json")
            xutils.save_dict_to_json(results, save_path)

        # save validation visual results
        # (outputs of the last iteration should still be in scope)
        val_results_dir = args.model_dir + "/val_visual_results"
        if not os.path.exists(val_results_dir):
            os.makedirs(val_results_dir)

        # visualisation
        target_original = target_original.cpu().numpy().squeeze()  # (N, H, W)
        source = source.numpy().squeeze()  # (N, H, W)
        warped_source = warped_source.cpu().numpy().squeeze()  # (N, H, W), normalised

        # normalise for correct visualisation of error
        # todo: again the warped source is not normalised the same way as source target, etc. This is why the network model should only output DVF
        target_original = normaliser_minmax(target_original)
        source = normaliser_minmax(source)

        save_val_visual_results(target, target_original, source, warped_source, warped_target_original,
                                dvf, dvf_gt,
                                val_results_dir, epoch, dpi=50)

    """
    Testing: 
    """
    if not val:
        # save the overall test results
        save_path = os.path.join(args.model_dir,
                                 "test_results.json".format(args.restore_file, args.three_slices))
        xutils.save_dict_to_json(results, save_path)


        # save evaluated metrics for individual test subjects in pandas dataframe for boxplots
        subj_id_buffer = data.val_dataloader.dataset.subject_list  # this list should be in the same order as subjects are evaluated
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
        rmse_df.to_pickle(f"{args.model_dir}/DL_test_subjects_rmse.pkl")

        # save detJac metrics
        jac_data = {'Method': column_method,
                    'ID': subj_id_buffer,
                    'GradDetJac': mean_mag_grad_detJ_buffer,
                    'NegDetJac': negative_detJ_buffer}
        jac_df = pd.DataFrame(data=jac_data)
        jac_df.to_pickle(f"{args.model_dir}/DL_test_subjects_jacDet.pkl")

        # todo: option to save all outputs to be analysed later

    return results, torch.stack(val_loss_buffer).mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',
                        default='experiments/base_model',
                        help="Directory containing params.json")
    parser.add_argument('--restore_file',
                        default='best',
                        help="Prefix of the checkpoint file:"
                             " 'best' for best model, or 'last' for the last saved checkpoint")

    parser.add_argument('--no_three_slices',
                        action='store_true',
                        help="Evaluate metrics on all instead of 3 slices.")

    parser.add_argument('--no_cuda',
                        action='store_true')

    parser.add_argument('--num_workers',
                        default=8,
                        help='Number of processes used by dataloader, 0 means use main process')

    parser.add_argument('--gpu',
                        default=0,
                        help='Choose the GPU to run on, pass -1 to use CPU')
    args = parser.parse_args()

    # set the GPU to use and device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    # check whether the model directory exists
    assert os.path.exists(args.model_dir), "No model dir found at {}".format(args.model_dir)

    # set the three slices
    args.three_slices = not args.no_three_slices

    # set up a logger
    xutils.set_logger(os.path.join(args.model_dir, 'eval.log'))

    # load parameters from model JSON file
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = xutils.Params(json_path)

    # set up data
    """Data"""
    logging.info("Setting up data loaders...")
    data = Data(args, params)
    data.use_brain()
    logging.info("- Done.")
    """"""

    """Model & Optimiser"""
    model = RegDVF(params.network, transform_model=params.transform_model)
    model = model.to(device=args.device)

    # reload network parameters from saved model file
    logging.info(
        "Loading model from saved file: {}".format(os.path.join(args.model_dir, args.restore_file + '.pth.tar')))
    xutils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # run the evaluation and calculate the metrics
    logging.info("Running evaluation...")
    evaluate(model, loss_fn, data, params, args, val=False)
    logging.info("Evaluation complete. Model path {}".format(args.model_dir))
