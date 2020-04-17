import os
import argparse
import logging
import torch

from data.datasets import Brain2dData
from model.models import RegModel
from model.losses import loss_fn
import utils.misc as misc_utils

from runners.train import train_and_validate
from runners.eval import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--run',
                    default=None,
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
                    help="Save deformed images and predicted DVFs in evaluation if True.")

parser.add_argument('--cpu',
                    action='store_true',
                    help='Use CPU if given')

parser.add_argument('--gpu',
                    default=0,
                    help='Choose GPU to run on')

parser.add_argument('--num_workers',
                    default=8,
                    help='Number of processes used by dataloader, 0 means use main process')

args = parser.parse_args()

assert os.path.exists(args.model_dir), f"Model directory does not exist at \n\t {args.model_dir}."

# set up device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # select GPU
args.cuda = not args.cpu and torch.cuda.is_available()
if args.cuda:
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

# set up the logger
misc_utils.set_logger(f"{args.model_dir}/{args.run}.log")
logging.info("What a beautiful day to save lives!")
logging.info(f"Model: \n\t{args.model_dir}")

# load config parameters from the JSON file
json_path = os.path.join(args.model_dir, 'params.json')
logging.info(f"Loading JSON configuration...")
assert os.path.isfile(json_path), "No JSON configuration file found at {}".format(json_path)
params = misc_utils.Params(json_path)
logging.info("- Done.")

"""Data"""
logging.info("Setting up Dataloader...")
brain2d_data = Brain2dData(args, params)
logging.info("- Done.")
""""""

"""Model"""
logging.info("Setting up Model...")
reg_model = RegModel(params)
reg_model = reg_model.to(device=args.device)
logging.info("- Done.")
""""""

""" Run """
if args.run == "train":
    logging.info(f"Running training and validation for {params.num_epochs} epochs...")
    train_and_validate(reg_model, loss_fn, brain2d_data, args)
    logging.info("Training and validation complete.")

elif args.run == "test":
    model_ckpt_path = f"{args.model_dir}/{args.ckpt_file}"
    assert os.path.exists(model_ckpt_path), "Model checkpoint does not exist."
    logging.info(f"Loading model parameters from: {model_ckpt_path}")
    misc_utils.load_checkpoint(model_ckpt_path, reg_model)

    logging.info("Running testing...")
    evaluate(reg_model, loss_fn, brain2d_data.test_dataloader, args, val=False)
    logging.info("Testing complete.")

else:
    raise ValueError("Run mode not recognised.")
""""""
