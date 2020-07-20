import os, subprocess, argparse

parser = argparse.ArgumentParser()
parser.add_argument('-modality',
                    default='multi',
                    help="mono-modal or multi-modal.")

parser.add_argument('--data_dir',
                    default='/vol/biomedic2/hq615/PROJECTS/2_mutual_info/data/brats17/miccai2020/val_crop192_sigma8_cps10_dispRange1.5-2.5_sliceRange70-90')

parser.add_argument('--run_dir',
                    default=None,
                    help="Root directory of the tuning experiments")

parser.add_argument('-sim',
                    default='NMI',
                    help="(Dis-)similarity measure of registration.")

parser.add_argument('-CPSs',
                    nargs='*',
                    help="Control point spacing options to try.")

parser.add_argument('-BEs',
                    nargs='*',
                    help="Bending Energy options to try.")

parser.add_argument('-verbose',
                    default=0,
                    help="Verbose level of MIRTK FFD.")

args = parser.parse_args()

for CPS in args.CPSs:
    for BE in args.BEs:

        cmd = f"python src/eval_FFD.py " \
                              f"-modality {args.modality} " \
                              f"-sim {args.sim} " \
                              f"-CPS {CPS} " \
                              f"-BE {BE} " \
                              f"-verbose {args.verbose} " \
                              f"--data_dir {args.data_dir} " \
                              f"--run_dir {args.run_dir}"

        print(f"Running command: \n{cmd}")
        subprocess.run(cmd, check=True, shell=True)
