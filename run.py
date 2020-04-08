"""Run experiments"""
import argparse
import os
import subprocess
import sys

PYTHON = sys.executable  # note: need to activate virtualenv before

parser = argparse.ArgumentParser()
parser.add_argument('--run',
                    default='train',
                    help='"train" or "eval"')

parser.add_argument('--model_dir',
                    default='experiments/base_model',
                    help='Directory containing params.json')

parser.add_argument('--restore_file',
                    default='best',
                    help="Optional, name of the file in --model_dir containing weights to reload before training w/o postfix")

parser.add_argument('--gpu',
                    default=0,
                    help='Choose GPU to run on')

parser.add_argument('--no_cuda',
                    action='store_true')

parser.add_argument('--local',
                    action='store_true',
                    help="True if not running on cluster to skip taskset")


args = parser.parse_args()
json_path = os.path.join(args.model_dir, 'params.json')
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)

# main run shell command
cmd = f"python src/{args.run}.py " \
                      f"--model_dir {args.model_dir} " \
                      f"--gpu {args.gpu}"

# add $taskset for running on monal servers
if not args.local:
    cpu_core_base = int(args.gpu) * 4
    cmd = f"taskset -c {cpu_core_base},{cpu_core_base+1},{cpu_core_base+2},{cpu_core_base+3} " + cmd

# run command
print(cmd)
subprocess.run(cmd, check=True, shell=True)


# collect results if evaluating/testing
if args.run == 'eval':
    cmd_results = f"python src/synthesize_results.py " \
                                  f"--parent_dir {args.model_dir}/test_results " \
                                  f"--result_file test_results.json " \
                                  f"--save_file test_results.csv"
    print("Collecting test results: ")
    print(cmd_results)
    subprocess.run(cmd, check=True, shell=True)
