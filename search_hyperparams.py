"""Peform hyperparemeters search"""

import argparse
import os
from subprocess import check_call
import sys

from utils import xutils


PYTHON = sys.executable  # note: need to activate virtualenv before
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/learning_rate',
                    help='Directory containing params.json')
parser.add_argument('--gpu', default=0, help='Choose GPU to run on')

# pass the range of hyperparameters that are being searched
parser.add_argument('--alphas', nargs='*', default=None, help='weight of similarity loss. Arg format: --alphas a b c d')
parser.add_argument('--h_alphas', nargs='*', default=None, help='spatial smoothness. Arg format: --h_alphas a b c d')


def launch_training_job(parent_dir, job_name, params, gpu):
    """Launch one training of the model with a set of hyperparameters in parent_dir/job_name
    Args:
        parent_dir: (string) root directory of this search
        job_name: (string) a unique name of this training job
        params: (dict) containing hyperparameters of this training job
        gpu: (int)
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters of current config in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} src/train.py --model_dir={model_dir} --gpu {gpu}".format(python=PYTHON,
                                                                         model_dir=model_dir,
                                                                         gpu=gpu)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = xutils.Params(json_path)

    # Perform hypersearch of parameters
    for huber_spatial in args.huber_spatials:
        for huber_temporal in args.huber_temporal:
            # Modify the relevant parameter in params
            params.huber_spatial = float(huber_spatial)
            params.huber_temporal = float(huber_temporal)

            # Launch job (name has to be unique)
            job_name = f"spt_{params.huber_spatial}_temp_{params.huber_temporal}"
            launch_training_job(args.parent_dir, job_name, params, args.gpu)
