import os, subprocess, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/brats17/miccai2020/val_crop192_sigma8_cps10_dispRange1.5-2.5_sliceRange70-90')
parser.add_argument('--run_dir', default='experiments/2_MICCAI2020_mutual_info/FFD/Tuning',
                    help="Root directory of the tuning experiments")

parser.add_argument('-sim', default='NMI', help="(Dis-)similarity measure of registration.")
parser.add_argument('-CPSs', nargs='*', help="Control point spacing options to try.")
parser.add_argument('-BEs', nargs='*', help="Bending Energy options to try.")

args = parser.parse_args()


#
# params_tuning = {
#     "CPS": [6],
#     "BE": [1e-3, 1e-4, 1e-5]
# }



for CPS in args.CPSs:
    for BE in args.BEs:

        # model_dir = os.path.join(args.run_dir, f"CPS{CPS}_BE{BE}")
        # if not os.path.exists(model_dir):
        #     os.makedirs(model_dir)

        cmd = f"python src/eval_FFD.py " \
              f"-sim {args.sim} -CPS {CPS} -BE {BE} " \
              f"--data_dir {args.data_dir} " \
              f"--run_dir {args.run_dir} " \
              f"--clean --verbose 0"

        """ -dofin id"""
        print(f"Running command: {cmd}")
        subprocess.run(cmd, check=True, shell=True)
