""" Hyperparemeters search / tuning """

import argparse
import os
import subprocess
import sys
import pandas as pd

from utils import misc

# note: need to activate virtualenv before

PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir',
                    default=None,
                    help='Parent directory of models')

parser.add_argument('--params_list_csv',
                    default=None,
                    help='Path to the CSV file containing the list of parameters to search')

parser.add_argument('--gpu_num',
                    default=0,
                    help='Choose GPU to run on')

parser.add_argument('--num_workers',
                    default=0,
                    type=int,
                    help='Number of processes used by dataloader in parallel, set to 0 to use main process only')

parser.add_argument('--local',
                    action='store_true',
                    help="True if not running on cluster to skip taskset")

args = parser.parse_args()


# load the base parameters from the JSON file in parent_dir
json_path = os.path.join(args.parent_dir, 'params.json')
params = misc.Params(json_path)

# read the parameter settings from csv file
params_dict = pd.read_csv(args.params_list_csv).to_dict('list')
num_settings = len(params_dict[list(params_dict.keys())[0]])

for _, p in params_dict.items():
    assert len(p) == num_settings, "Parameters to search have different number of settings"


# loop through the parameter settings
for i in range(num_settings):
    model_name = "tune"

    # modify the params
    for param_name in params_dict.keys():
        param_value = params_dict[param_name][i]
        params.dict[param_name] = param_value

        # update model dir name
        model_name += f"_{param_name}_{param_value}"

    # create a new model_dir in parent_dir
    model_dir = os.path.join(args.parent_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # write the parameters of current config in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # training command
    cmd = f"python src/run.py " \
          f"--run train " \
          f"--model_dir {model_dir} " \
          f"--gpu_num {args.gpu_num} " \
          f"--num_workers {args.num_workers}"

    if args.local:
        cmd += " --local"

    # run
    print(f"Launching job: \n\t{cmd}")
    subprocess.run(cmd, check=True, shell=True)

