"""Script for brain extraction and subcortical structural segmentation using FSL"""
import os
from os import path
import subprocess
from tqdm import tqdm
from glob import glob

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",
                    default="/vol/vipdata/ixi/ixi-db/images")

parser.add_argument("--output_dir",
                    default=None)

parser.add_argument("--process_bet",
                    action="store_true",
                    help="Run brain extraction using FSL BET")

parser.add_argument("--process_first",
                    action="store_true",
                    help="Run segmentation using FSL BET")

args = parser.parse_args()


if __name__ == '__main__':

    assert os.path.exists(args.data_dir), "Source data dir does not exist."

    t1_list = glob(args.data_dir + "/*T1*")
    t2_list = glob(args.data_dir + "/*T2*")

    # # Sanity check if all subjects has both T1 and T2
    # for t1, t2 in zip(sorted(t1_list), sorted(t2_list)):
    #     t1_subj = "".join(t1.split("-")[:-1])
    #     t2_subj = "".join(t2.split("-")[:-1])
    #     print(t1_subj, t2_subj, t1_subj==t2_subj)
    #     assert t1_subj == t2_subj

    with tqdm(total=len(t1_list)) as t:
        for t1 in sorted(t1_list):

            subject_path_prefix = "-".join(t1.split("-")[:-1])  # $PATH_TO_DATA/$subject_prefix
            subject_prefix = subject_path_prefix.split("/")[-1]

            print(f"Processing: {subject_prefix}")

            # setup subject output dir
            subject_output_dir = path.join(args.output_dir, subject_prefix)
            if not path.exists(subject_output_dir):
                os.makedirs(subject_output_dir)

            # set original image paths
            t1_img_path = subject_path_prefix + "-T1.nii.gz"
            t2_img_path = subject_path_prefix + "-T2.nii.gz"
            assert path.exists(t1_img_path), f"T1 image does not exist for subject: {subject_prefix}"
            assert path.exists(t2_img_path), f"T2 image does not exist for subject: {subject_prefix}"

            # set output image paths
            t1_brain_path = subject_output_dir + "/T1-brain.nii.gz"
            t2_brain_path = subject_output_dir + "/T2-brain.nii.gz"


            """
            Run brain extraction / skull stripping
            """
            if args.process_bet:

                # create symbolic links to original images
                os.system(f"ln -s {t1_img_path} {subject_output_dir + '/T1.nii.gz'}")
                os.system(f"ln -s {t2_img_path} {subject_output_dir + '/T2.nii.gz'}")

                # run FSL bet to extract the brain and get the mask
                cmd_bet = f"bet {t1_img_path} {t1_brain_path} -m -R"
                subprocess.run(cmd_bet, check=True, shell=True)

                # check if t1 mask and extracted brain has been generated
                t1_mask_path = subject_output_dir + "/T1-brain_mask.nii.gz"  # _mask added by fsl bet
                assert path.exists(t1_mask_path), f"T1 mask has not been generated for subject {subject_prefix}"
                assert path.exists(t1_brain_path), f"T1 brain has not been extracted for subject {subject_prefix}"


                # register T2 image to T1 image
                t2_img_registered_path = subject_output_dir + "/T2-registered.nii.gz"
                cmd_register_t2 = f"mirtk transform-image " \
                                  f"{t2_img_path} {t2_img_registered_path} -target {t1_img_path}"
                subprocess.run(cmd_register_t2, check=True, shell=True)

                # apply T1 mask to registered T2 image to extract T2 brain
                cmd_extract_t2 = f"mirtk calculate " \
                                 f"{t2_img_registered_path} -mask {t1_mask_path} -pad 0 " \
                                 f"-output {t2_brain_path}"
                subprocess.run(cmd_extract_t2, check=True, shell=True)


            """
            Run segmentation
            """
            if args.process_first:
                t1_brain_resampled_label_path = subject_output_dir + "/T1-brain-resampled_subcor_label.nii.gz"
                assert path.exists(t1_brain_path), \
                    "T1 brain resampled does not exist, check brain extraction?"

                # skip the subject is segmentation already exists
                if path.exists(subject_output_dir + "/T1-brain-resampled_subcor_label_all_fast_firstseg.nii.gz"):
                    continue

                bad_datapoint = ["IXI028-Guys-1038"]
                if subject_prefix in bad_datapoint:
                    print(f"WARNING: skipping subject {subject_prefix}")
                    continue

                # run segmentation using FSL FIRST
                cmd_segment_t1_resampled = f"run_first_all -b -m auto " \
                                           f"-i {t1_brain_path} -o {t1_brain_resampled_label_path} "
                subprocess.run(cmd_segment_t1_resampled, check=True, shell=True)

                # # clean up the additional FSL FIRST output
                # subject_first_output_dir = subject_output_dir + "/first_output"
                # if not path.exists(subject_first_output_dir):
                #     os.makedirs(subject_first_output_dir)
                # first_output_list = glob(subject_output_dir + "/*.vtk") \
                #                     + glob(subject_output_dir + "/*bvars") \
                #                     + glob(subject_output_dir + "/*.logs") \
                #                     + glob(subject_output_dir + "/*.com*") \
                #                     + glob(subject_output_dir + "/*firstseg.nii.gz") \
                #                     + glob(subject_output_dir + "/*_std_sub*")
                # for x in first_output_list:
                #     os.system(f"mv {x} {subject_output_dir}")

            t.update()

