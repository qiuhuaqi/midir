"""Run experiments"""
import argparse
import os
from subprocess import check_call
import sys
from utils import xutils

PYTHON = sys.executable  # note: need to activate virtualenv before

parser = argparse.ArgumentParser()
parser.add_argument('--run', default='train')
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before training w/o postfix")
parser.add_argument('--gpu', default=0, help='Choose GPU to run on')
parser.add_argument('--no_cuda', action='store_true')

parser.add_argument('--local', action='store_true')



args = parser.parse_args()
json_path = os.path.join(args.model_dir, 'params.json')
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)

cmd = f"python src/train.py --model_dir {args.model_dir} --gpu {args.gpu}"

if not args.local:
    cpu_core_base = int(args.gpu) * 4
    cmd = f"taskset -c {cpu_core_base},{cpu_core_base+1},{cpu_core_base+2},{cpu_core_base+3} " + cmd

print(cmd)
check_call(cmd, shell=True)
