import os
import argparse
import logging
import torch

import model.utils
import utils.experiment
import utils.experiment.experiment
import utils.misc
import utils.experiment.model
from datasets.datasets import BrainData
from archive.legacy_model.models import DLRegModel, IdBaselineModel, MIRTKBaselineModel
from model.loss import loss_fn

from archive.runners.train import train_and_validate
from archive.runners.eval import evaluate

torch.autograd.set_detect_anomaly(True)

# set random seed for workers generating random deformation
import random
random.seed(12)


parser = argparse.ArgumentParser()
parser.add_argument('--run',
                    default='train',
                    help="'train' or 'eval'")

parser.add_argument('--model_dir',
                    default='experiments/base_model',
                    help="Directory containing params.json")

parser.add_argument('--ckpt_file',
                    default=None,
                    help="Name of the checkpoint file in model_dir:"
                         " 'best.pth.tar' for best model, or 'last.pth.tar' for the last saved checkpoint")

parser.add_argument('--save',
                    action='store_true',
                    help="Save deformed images and predicted DVFs in evaluation if True")

parser.add_argument('--cpu',
                    action='store_true',
                    help='Use CPU')

parser.add_argument('--gpu_num',
                    default=0,
                    help='Choose GPU')

parser.add_argument('--num_workers',
                    default=2,
                    type=int,
                    help='Number of processes used by dataloader, 0 means use main process')


args = parser.parse_args()

# set up device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)  # select GPU
args.cuda = not args.cpu and torch.cuda.is_available()
if args.cuda:
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

assert os.path.exists(args.model_dir), f"Model directory does not exist at \n\t {args.model_dir}."

# set up the logger
utils.experiment.set_logger(f"{args.model_dir}/{args.run}.log")
logging.info("What a beautiful day to save lives!")
logging.info(f"Model: \n\t{args.model_dir}")

# load config parameters from the JSON file
json_path = os.path.join(args.model_dir, 'params.json')
logging.info(f"Loading JSON configuration...")
assert os.path.isfile(json_path), "No JSON configuration file found at {}".format(json_path)
params = utils.misc.Params(json_path)
logging.info("- Done.")

"""Data"""
logging.info("Setting up Dataloader...")
brain_data = BrainData(args, params)
logging.info("- Done.")
""""""

"""Model"""
logging.info("Setting up Model...")
if params.model_name == "DL":
    reg_model = DLRegModel(params)
    reg_model = reg_model.to(device=args.device)

elif params.model_name == "IdBaseline":
    reg_model = IdBaselineModel(params)

elif params.model_name == "MIRTK":
    # work dir set up in model dir
    work_dir = args.model_dir + "/work_dir"
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    reg_model = MIRTKBaselineModel(params, work_dir)

else:
    raise ValueError("Registration model not recognised.")
logging.info("- Done.")
""""""


""" Run """
if args.run == "train":
    logging.info(f"Running training and validation for {params.num_epochs} epochs...")
    train_and_validate(reg_model, loss_fn, brain_data, args)
    logging.info("Training and validation complete.")

elif args.run == "test":
    if params.model_name == "DL":
        model_ckpt_path = f"{args.model_dir}/{args.ckpt_file}"
        assert os.path.exists(model_ckpt_path), "Model checkpoint does not exist."
        logging.info(f"Loading model parameters from: {model_ckpt_path}")
        model.utils.load_checkpoint(model_ckpt_path, reg_model)

    logging.info("Running testing...")
    evaluate(reg_model, loss_fn, brain_data.test_dataloader, args, val=False)
    logging.info("Testing complete.")

else:
    raise ValueError("Run mode not recognised.")
""""""
