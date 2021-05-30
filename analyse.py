""" Calculate metric results from the model predictions/outputs"""
import os
import argparse
from tqdm import tqdm
import numpy as np

from utils.image_io import load_nifti
from utils.metric import measure_metrics, MetricReporter


def analyse_output(inference_output_dir, save_dir, metric_groups):
    print("Running output analysis:")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    metric_reporter = MetricReporter(id_list=os.listdir(inference_output_dir),
                                     save_dir=save_dir)

    for d in tqdm(os.listdir(inference_output_dir)):
        subj_output_dir = inference_output_dir + f'/{d}'

        # load saved output data from inference
        file_names = os.listdir(subj_output_dir)
        data_dict = dict()
        for fn in file_names:
            # file names as dict keys
            k = fn.split('.')[0]
            data_dict[k] = load_nifti(subj_output_dir + f'/{fn}')

        ndim = data_dict['disp_pred'].shape[-1]
        for k, x in data_dict.items():
            # reshape from saved for analysis:
            if ndim == 2:
                # 2D: img (H, W, N) -> (N=num_slice, 1, H, W)
                #     disp (H, W, N, 2) -> (N=num_slice, 2, H, W)
                if k == 'disp_gt' or k == 'disp_pred':
                    data_dict[k] = x.transpose(2, 3, 0, 1)
                else:
                    data_dict[k] = x.transpose(2, 0, 1)[:, np.newaxis, ...]

            if ndim == 3:
                # 3D: img (H, W, D) -> (N=1, 1, H, W, D)
                #     disp (H, W, D, 3) -> (N=1, 3, H, W, D)
                if k == 'disp_gt' or k == 'disp_pred':
                    data_dict[k] = x.transpose(3, 0, 1, 2)[np.newaxis, ...]
                else:
                    data_dict[k] = x[np.newaxis, np.newaxis, ...]

        # calculate metric for one validation batch
        metric_result_step = measure_metrics(data_dict, metric_groups)
        metric_reporter.collect(metric_result_step)

    # save the metric results
    metric_reporter.summarise()
    metric_reporter.save_mean_std()
    metric_reporter.save_df()


if __name__ == '__main__':
    # main single run to analyse outputs of one model
    import sys
    sys.path.append('../')

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--model_dir')
    parser.add_argument('-o', '--inference_output_dir')
    parser.add_argument('-s', '--save_dir')
    parser.add_argument('-m', '--metric_groups',
                        nargs='*',
                        type=str,
                        default=["disp_metrics", "image_metrics", "seg_metrics"])
    args = parser.parse_args()

    # default inference output directory
    if args.inference_output_dir is None:
        args.inference_output_dir = args.model_dir + '/outputs'

    # default save directory
    if args.save_dir is None:
        args.save_dir = args.model_dir + '/analysis'

    # pretty print args
    for k, i in args.__dict__.items():
        print(f'{k}: {i}')

    # run analysis
    delattr(args, 'model_dir')
    analyse_output(**args.__dict__)
