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

parser.add_argument('--ckpt_file',
                    default=None,
                    help="Name of the checkpoint file in model_dir:"
                         " 'best.pth.tar' for best model, or 'last.pth.tar' for the last saved checkpoint")

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

# main run command
cmd = f"python src/main.py " \
      f"--run {args.run} " \
      f"--model_dir {args.model_dir} " \
      f"--gpu_num {args.gpu_num} " \
      f"--num_workers {args.num_workers} " \
      f"--ckpt_file {args.ckpt_file}"

# add taskset for shared servers
if not args.local:
    cpu_num = int(args.gpu_num) * 4
    cmd = f"taskset -c {cpu_num},{cpu_num+1},{cpu_num+2},{cpu_num+3} " + cmd

# print the final command and run
print(cmd)
subprocess.run(cmd, check=True, shell=True)

# # collect results if evaluating/testing
# if args.run == 'eval':
#     cmd_results = f"python src/synthesize_results.py " \
#                                   f"--parent_dir {args.model_dir}/test_results " \
#                                   f"--result_file test_results.json " \
#                                   f"--save_file test_results.csv"
#     print("Collecting test results: ")
#     print(cmd_results)
#     subprocess.run(cmd, check=True, shell=True)
