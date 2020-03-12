"""Evaluates the model"""

from tqdm import tqdm
import os
import argparse
import logging
import numpy as np
import pandas as pd
import torch

from data.datasets import Data
from model.networks import BaseNet, SiameseFCN
from model.submodules import spatial_transform
from model.losses import loss_fn
from utils.metrics import categorical_dice_stack, contour_distances_stack, detJac_stack, rmse, aee
from utils import xutils

from utils.image_utils import bbox_from_mask
from utils.xutils import save_val_visual_results


# def evaluate_cardiac(model, loss_fn, data, params, args, epoch=0, val=False):
#     """
#     Evaluate the model and returns metrics as a dict and evaluation loss
#
#     Args:
#         model:
#         loss_fn:
#         data: (instance of Data object)
#         params:
#         args:
#         val: (boolean) indicates validation (True) or testing (False)
#
#     Returns:
#
#     """
#     model.eval()
#
#     # empty lists buffer
#     val_loss_buffer = []
#
#     dice_lv_buffer = []
#     dice_myo_buffer = []
#     dice_rv_buffer = []
#
#     mcd_lv_buffer = []
#     hd_lv_buffer = []
#     mcd_myo_buffer = []
#     hd_myo_buffer = []
#     mcd_rv_buffer = []
#     hd_rv_buffer = []
#
#     mean_mag_grad_detJ_buffer = []
#     negative_detJ_buffer = []
#
#
#     with tqdm(total=len(data.val_dataloader)) as t:
#         # iterate over validation subjects
#         for idx, data_point in enumerate(data.val_dataloader):
#             # (c, N, H, W) to (N, c, H, W)
#             image_ed_batch = image_ed_batch.permute(1, 0, 2, 3).to(device=args.device)
#             image_es_batch = image_es_batch.permute(1, 0, 2, 3).to(device=args.device)
#             label_es_batch = label_es_batch.permute(1, 0, 2, 3).to(device=args.device)
#
#             with torch.no_grad():
#                 # # linear transformation test for NMI: use (1-source) as source image
#                 # if params.inverse:
#                 #     image_es_batch = 1.0 - image_es_batch
#
#                 # compute DVF and warped source image towards target
#                 dvf, warped_source = model(image_ed_batch, image_es_batch)
#
#                 # transform label mask of ES frame
#                 warped_label_es_batch = spatial_transform(label_es_batch.float(), dvf, mode="nearest")
#
#             if args.cuda:
#                 # move data to cpu to calculate metrics
#                 # (the axis permutation is to comply with metric calculation code which takes input shape H, W, N)
#                 warped_label_es_batch = warped_label_es_batch.squeeze(1).cpu().numpy().transpose(1, 2, 0)
#                 label_ed_batch = label_ed_batch.squeeze(0).numpy().transpose(1, 2, 0)
#                 dvf = dvf.data.cpu().numpy().transpose(0, 2, 3, 1)  # (N, H, W, 2)
#             else:
#                 # CPU version of the code
#                 warped_label_es_batch = warped_label_es_batch.squeeze(1).numpy().transpose(1, 2, 0)
#                 label_ed_batch = label_ed_batch.squeeze(0).numpy().transpose(1, 2, 0)
#                 dvf = dvf.data.numpy().transpose(0, 2, 3, 1)  # (N, H, W, 2)
#
#             # calculate the metrics
#             if args.three_slices:
#                 num_slices = label_ed_batch.shape[-1]
#                 apical_idx = int(round((num_slices - 1) * 0.75))  # 75% from basal
#                 mid_ven_idx = int(round((num_slices - 1) * 0.5))  # 50% from basal
#                 basal_idx = int(round((num_slices - 1) * 0.25))  # 25% from basal
#                 slices_idx = [apical_idx, mid_ven_idx, basal_idx]
#
#                 warped_label_es_batch = warped_label_es_batch[:, :, slices_idx]
#                 label_ed_batch = label_ed_batch[:, :, slices_idx]
#                 dvf = dvf[slices_idx, :, :, :]  # needed for detJac
#
#             # dice
#             dice_lv = categorical_dice_stack(warped_label_es_batch, label_ed_batch, label_class=1)
#             dice_myo = categorical_dice_stack(warped_label_es_batch, label_ed_batch, label_class=2)
#             dice_rv = categorical_dice_stack(warped_label_es_batch, label_ed_batch, label_class=3)
#
#             dice_lv_buffer += [dice_lv]
#             dice_myo_buffer += [dice_myo]
#             dice_rv_buffer += [dice_rv]
#
#             # contour distances
#             mcd_lv, hd_lv = contour_distances_stack(warped_label_es_batch, label_ed_batch,
#                                                     label_class=1, dx=params.pixel_size)
#             mcd_myo, hd_myo = contour_distances_stack(warped_label_es_batch, label_ed_batch,
#                                                       label_class=2, dx=params.pixel_size)
#             mcd_rv, hd_rv = contour_distances_stack(warped_label_es_batch, label_ed_batch,
#                                                     label_class=3, dx=params.pixel_size)
#
#             mcd_lv_buffer += [mcd_lv]
#             hd_lv_buffer += [hd_lv]
#             mcd_myo_buffer += [mcd_myo]
#             hd_myo_buffer += [hd_myo]
#             mcd_rv_buffer += [mcd_rv]
#             hd_rv_buffer += [hd_rv]
#
#             # determinant of Jacobian
#             mean_grad_detJ, mean_negative_detJ = detJac_stack(dvf)
#             mean_mag_grad_detJ_buffer += [mean_grad_detJ]
#             negative_detJ_buffer += [mean_negative_detJ]
#
#             t.update()
#
#     # construct metrics dict
#     results = {}
#     # accuracy
#     structures = ['lv', 'myo', 'rv']
#     acc_criteria = ['dice', 'mcd', 'hd']
#     for st in structures:
#         for cr in acc_criteria:
#             result_name = f'{cr}_{st}'
#             the_buffer = locals()[f'{result_name}_buffer']
#             results[f'{result_name}_mean'] = np.mean(the_buffer)
#             results[f'{result_name}_std'] = np.std(the_buffer)
#     # regularity
#     reg_criteria = ['mean_mag_grad_detJ', 'negative_detJ']
#     for cr in reg_criteria:
#         result_name = cr
#         the_buffer = locals()[f'{result_name}_buffer']
#         results[f'{result_name}_mean'] = np.mean(the_buffer)
#         results[f'{result_name}_std'] = np.std(the_buffer)
#
#     # sanity check: proportion of negative Jacobian points should be lower than 1
#     assert results['negative_detJ_mean'] <= 1, "Invalid det Jac: Ratio of folding points > 1"
#
#     """
#     Validation:
#     """
#     if val:
#         # use one metric to set is_best flag
#         current_one_metric = np.mean([results['dice_lv_mean'],
#                                       results['dice_myo_mean'],
#                                       results['dice_rv_mean']])
#         if epoch + 1 == params.val_epochs:
#             # initialise for the first validation
#             params.is_best = False
#             params.best_one_metric = current_one_metric
#         elif current_one_metric >= params.best_one_metric:
#             params.is_best = True
#             params.best_one_metric = current_one_metric
#
#         # save the most recent results
#         save_path = os.path.join(args.model_dir,
#                                  f"val_results_last_3slices_{args.three_slices}.json")
#         xutils.save_dict_to_json(results, save_path)
#
#         # save the validation results for the best model separately
#         if params.is_best:
#             save_path = os.path.join(args.model_dir,
#                                      f"val_results_best_3slices_{args.three_slices}.json")
#             xutils.save_dict_to_json(results, save_path)
#
#     """
#     Testing:
#     """
#     if not val:
#         # save the overall test results
#         save_path = os.path.join(args.model_dir,
#                                  "test_results_{}_3slices_{}.json".format(args.restore_file, args.three_slices))
#         xutils.save_dict_to_json(results, save_path)
#
#         # save evaluated metrics for individual test subjects in pandas dataframe
#         # which can be used to create boxplots
#         subj_id_buffer = data.val_dataloader.dataset.dir_list
#         df_buffer = []
#         column_method = ['DL'] * len(subj_id_buffer)
#         for struct in ['LV', 'MYO', 'RV']:
#             if struct == 'LV':
#                 ls_dice = dice_lv_buffer
#                 ls_mcd = mcd_lv_buffer
#                 ls_hd = hd_lv_buffer
#             elif struct == 'MYO':
#                 ls_dice = dice_myo_buffer
#                 ls_mcd = mcd_myo_buffer
#                 ls_hd = hd_myo_buffer
#             elif struct == 'RV':
#                 ls_dice = dice_rv_buffer
#                 ls_mcd = mcd_rv_buffer
#                 ls_hd = hd_rv_buffer
#
#             ls_struct = [struct] * len(subj_id_buffer)
#             data = {'Method': column_method,
#                     'ID': subj_id_buffer,
#                     'Structure': ls_struct,
#                     'Dice': ls_dice,
#                     'MCD': ls_mcd,
#                     'HD': ls_hd}
#             df_buffer += [pd.DataFrame(data=data)]
#
#         # concatenate df and save
#         metrics_df = pd.concat(df_buffer, axis=0)
#         metrics_df.to_pickle("{0}/DL_all_subjects_accuracy_3slices_{1}.pkl".format(args.model_dir, args.three_slices))
#
#         # save detJac metrics
#         jac_data = {'Method': column_method,
#                     'ID': subj_id_buffer,
#                     'GradDetJac': mean_mag_grad_detJ_buffer,
#                     'NegDetJac': negative_detJ_buffer}
#         jac_df = pd.DataFrame(data=jac_data)
#         jac_df.to_pickle("{0}/DL_all_subjects_jac_3slices_{1}.pkl".format(args.model_dir, args.three_slices))
#
#     return results


def evaluate_brain(model, loss_fn, data, params, args, epoch=0, val=False, save=False):
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
    RMSE_buffer = []

    mean_mag_grad_detJ_buffer = []
    negative_detJ_buffer = []


    with tqdm(total=len(data.val_dataloader)) as t:
        # iterate over validation subjects
        for idx, data_point in enumerate(data.val_dataloader):

            target, source, target_original, brain_mask, dvf_gt = data_point
            target = target.to(device=args.device).permute(1, 0, 2, 3)  # (Nx1xHxW), slices in a brain as one batch
            source = source.to(device=args.device).permute(1, 0, 2, 3)  # (Nx1xHxW)

            with torch.no_grad():
                # compute DVF and warped source image towards target
                dvf, warped_source = model(target, source)
                val_loss, _ = loss_fn(dvf, target, warped_source, params)
                val_loss_buffer += [val_loss]

                # warp original target image using the same dvf
                target_original = target_original.to(device=args.device).permute(1, 0, 2, 3)  # (Nx1xHxW)
                warped_target_original = spatial_transform(target_original, dvf)

                # todo: warping labels for dice


            """
            Calculating metrics
            (metrics calculation works with shape image: NxHxW, DVF: NxHxWx2)
            """

            # find brian mask bbox mask
            mask_bbox, mask_bbox_mask = bbox_from_mask(brain_mask)

            ## AEE
            # to numpy tensor
            dvf = dvf.data.cpu().numpy().transpose(0, 2, 3, 1)  # (N, H, W, 2)
            dvf *= params.crop_size / 2
            dvf_gt = dvf_gt.numpy().squeeze().transpose(0, 2, 3, 1)  # (N, H, W, 2)
            # mask DVF with mask bbox (actually slicing)
            dvf_masked = dvf[:,
                         mask_bbox[0][0]:mask_bbox[0][1],
                         mask_bbox[1][0]:mask_bbox[1][1],
                         :]  # (N, H', W', 2)
            dvf_gt_masked = dvf_gt[:,
                            mask_bbox[0][0]:mask_bbox[0][1],
                            mask_bbox[1][0]:mask_bbox[1][1],
                            :]  # (N, H', W', 2)
            AEE = aee(dvf_masked, dvf_gt_masked)
            AEE_buffer += [AEE]

            ## RMSE(image)
            #   between the target (T1-moved by ground truth DVF) and the warped_target_original (T1-moved by predicted DVF)
            # to numpy tensor
            target = target.cpu().numpy().squeeze()  # (N, H, W)
            warped_target_original = warped_target_original.cpu().numpy().squeeze()  # (N, H, W)

            target_masked = target[:,
                            mask_bbox[0][0]:mask_bbox[0][1],
                            mask_bbox[1][0]:mask_bbox[1][1],
                            ]  # (N, H', W')
            warped_target_original_masked = warped_target_original[:,
                                            mask_bbox[0][0]:mask_bbox[0][1],
                                            mask_bbox[1][0]:mask_bbox[1][1]]  # (N, H', W')

            RMSE = rmse(target_masked, warped_target_original_masked)
            RMSE_buffer += [RMSE]

            # todo: dice metrics

            # determinant of Jacobian
            mean_grad_detJ, mean_negative_detJ = detJac_stack(dvf_masked)
            mean_mag_grad_detJ_buffer += [mean_grad_detJ]
            negative_detJ_buffer += [mean_negative_detJ]
            """"""

            t.update()

    # construct metrics dict
    results = {}

    # RMSE (dvf) and RMSE (image)
    rmse_criteria = ["AEE", "RMSE"]
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

        target_original = target_original.cpu().numpy().squeeze()  # (N, H, W)
        source = source.cpu().numpy().squeeze()  # (N, H, W)
        warped_source = warped_source.cpu().numpy().squeeze()  # (N, H, W)

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
        subj_id_buffer = data.val_dataloader.dataset.dir_list  # this list should be in the same order as subjects are evaluated
        df_buffer = []
        column_method = ['DL'] * len(subj_id_buffer)

        # save RMSE and AEE
        rmse_data = {'Method': column_method,
                     'ID': subj_id_buffer,
                     'RMSE': RMSE_buffer,
                     'AEE': AEE_buffer}
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
    logging.info("Eval data path: {}".format(params.eval_data_path))
    data = Data(args, params)
    data.use_ukbb_cardiac()

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
    evaluate_brain(model, loss_fn, data, params, args, val=False)
    logging.info("Evaluation complete. Model path {}".format(args.model_path))
