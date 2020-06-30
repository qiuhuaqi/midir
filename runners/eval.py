from os import path
from tqdm import tqdm

import numpy as np

import torch
from model.transformations import spatial_transform
from runners.inference import process_batch
from runners.helpers import LossReporter, MetricReporter

import utils.metric as metrics_utils
import utils.misc as misc_utils
import utils.image_io as imageio_utils
import utils.visualise as vis_utils


def evaluate(model, loss_fn, dataloader, args, val=False, tb_writer=None):
    """
    Evaluate the model and returns metrics as a dict and evaluation loss

    Args:
        model:
        loss_fn:
        dataloader: validation / testing dataloader
        args:
        val: (boolean) indicates validation (True) or testing (False)
        tb_writer: (TensorboardX writer)

    Returns:

    """
    # evaluation mode
    model.eval()

    # set up reporters
    loss_reporter = LossReporter()
    metrics_reporter = MetricReporter()
    metrics_reporter.id_list = dataloader.dataset.subject_list

    # set up output dir
    if val:
        result_dir = misc_utils.setup_dir(args.model_dir + "/val_results")
    else:
        result_dir = misc_utils.setup_dir(args.model_dir + "/test_results")


    with tqdm(total=len(dataloader)) as t:
        for idx, data_dict in enumerate(dataloader):

            # reshaping (ch, N, *(dims)) to (N, ch, *(dims))
            for name in ["target", "source", "target_original", "roi_mask"]:
                data_dict[name] = data_dict[name].transpose(0, 1)  # (N, 1, *(dims))
            data_dict["dvf_gt"] = data_dict["dvf_gt"][0, ...]  # (N, dim, *(dims))

            """ Inference & loss """
            with torch.no_grad():
                eval_losses = process_batch(model, data_dict, loss_fn, args)
                loss_reporter.collect_value(eval_losses)

                # warp original target image using the predicted dvf
                # (comment this out if images are not synthesised)
                data_dict["target_pred"] = spatial_transform(data_dict["target_original"].to(device=args.device),
                                                             data_dict["dvf_pred"])

            # cast to numpy array
            for name, one_data in data_dict.items():
                data_dict[name] = one_data.cpu().numpy()
            """"""

            """
            Calculate metrics
            """
            metric_results = metrics_utils.metrics_fn(data_dict, model.params.metric_groups)
            metrics_reporter.collect_value(metric_results)
            """"""

            """ 
            Save predicted DVF and warped images 
            """
            if args.save:
                output_dir = misc_utils.setup_dir(result_dir + "/output")
                subj_id = dataloader.dataset.subject_list[idx]
                subj_output_dir = misc_utils.setup_dir(path.join(output_dir, subj_id))

                for name, save_data in data_dict.items():
                    # data_dict data shape (N, (1/dim), *size)
                    # save in shape: (*sizes, N, (dim)) for 2D, or (*sizes, (dim)) for 3D
                    imageio_utils.save_nifti(np.moveaxis(save_data, [0, 1], [-2, -1]).squeeze(),
                                             f"{subj_output_dir}/{name}.nii.gz")
            """"""

            t.update()

    # generate summarised reports
    loss_reporter.summarise()
    metrics_reporter.summarise()

    if val:  # Validation
        # determine if best_model
        model.update_best_model(metric_results)

        # save metric results to JSON files
        metrics_reporter.save_mean_std(result_dir + "/val_metrics_results_last.json")
        if model.is_best:
            metrics_reporter.save_mean_std(result_dir + "/val_metrics_results_best.json")

        # log loss and metrics to Tensorboard
        loss_reporter.log_to_tensorboard(tb_writer, step=model.iter_num)
        metrics_reporter.log_to_tensorboard(tb_writer, step=model.iter_num)

        # save validation visual results for training
        val_vis_dir = misc_utils.setup_dir(result_dir + "/val_visual_results")
        vis_utils.save_val_visual_results(data_dict, val_vis_dir, model.epoch_num, dpi=50, axis=0)
        # v0.3-dev: visualise all planes of the 3D volume
        if model.params.dim == 3:
            vis_utils.save_val_visual_results(data_dict, val_vis_dir, model.epoch_num, dpi=50, axis=1)
            vis_utils.save_val_visual_results(data_dict, val_vis_dir, model.epoch_num, dpi=50, axis=2)

    else:  # Testing
        # save mean-std and dataframe of metric results
        metrics_reporter.save_mean_std(result_dir + "/test_metrics_results.json")
        metrics_reporter.save_df(result_dir + "/test_metrics_results.pkl")

