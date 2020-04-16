"""Peform hyperparemeters search"""

import argparse
import os
import subprocess
import sys

from utils import misc

PYTHON = sys.executable  # note: need to activate virtualenv before
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir',
                    default=None,
                    help='Parent directory of the models')

parser.add_argument('--model_parent_name',
                    default=None,
                    help="Parent name of the model")

# pass the range of hyperparameters that are being searched
parser.add_argument('--reg_weights',
                    nargs='*',
                    default=None,
                    help='Regularisation weights to try')

parser.add_argument('--gpu',
                    default=0,
                    help='Choose GPU to run on')

parser.add_argument('--local',
                    action='store_true',
                    help="True if not running on cluster to skip taskset")

args = parser.parse_args()


# Load the base parameters from json file in parent_dir
json_path = os.path.join(args.parent_dir, 'params.json')
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
params = misc.Params(json_path)

# RUn hyper parameter search
for reg_weight in args.reg_weights:
    # Modify the tuned parameter in params
    params.reg_weight = float(reg_weight)

    # Create a new folder in parent_dir with unique_name "job_name"
    model_name = f"{args.model_parent_name}_diff{reg_weight}"
    model_dir = os.path.join(args.parent_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters of current config in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training
    cmd = f"python src/run.py " \
                          f"--run train " \
                          f"--model_dir {model_dir} " \
                          f"--gpu {args.gpu}"
    if args.local:
        cmd += " --local"

    print(f"Launching job: \n\t{cmd}")
    subprocess.run(cmd, check=True, shell=True)
