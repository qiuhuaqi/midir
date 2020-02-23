import os, subprocess, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/ukbb/cine_ukbb2964/small_set/sa/val_autoseg')
parser.add_argument('--run_dir', default='experiments/ffd/mutual_info', help="Root directory of the tuning experiments")

parser.add_argument('-sim', default='NMI', help="(Dis-)similarity measure of registration.")
parser.add_argument('-intensity', default=None, help="Intensity transformation applied to source images. "
                                                     "'inv' means inverse of source image")

args = parser.parse_args()

params_tuning = {
    "CPS": [2, 3, 4, 5, 6, 7, 8],
    "BE": [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
}


for CPS in params_tuning["CPS"]:
    for BE in params_tuning["BE"]:

        model_dir = os.path.join(args.run_dir, f"CPS{CPS}_BE{BE}")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        cmd = f"python src/eval_FFD.py " \
              f"-sim {args.sim} -CPS {CPS} -BE {BE} " \
              f"-intensity {args.intensity} " \
              f"--data_dir {args.data_dir} " \
              f"--model_dir {model_dir} " \
              f"--clean"

        """ -dofin id"""
        print(f"Running command: {cmd}")
        subprocess.run(cmd, check=True, shell=True)
