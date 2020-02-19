""" Main training function"""

from tqdm import tqdm
import os
import argparse
import logging
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model.networks import BaseNet, SiameseFCN
from model.losses import loss_fn
from model.dataset_utils import CenterCrop, Normalise, ToTensor
from model.datasets import CardiacMR_2D_UKBB, CardiacMR_2D_Eval_UKBB

from eval import evaluate
from utils import xutils


def train_and_validate(model, optimizer, loss_fn, dataloaders, params):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        dataloaders: (dict) train and val dataloaders
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        params: (instance of Params) configuration parameters
    """
    # reload weights from a specified file to resume training
    if args.restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        xutils.load_checkpoint(restore_path, model, optimizer)

    # set up TensorboardX summary writers
    train_summary_writer = xutils.set_summary_writer(args.model_dir, 'train')
    val_summary_writer = xutils.set_summary_writer(args.model_dir, 'val')

    # unpack dataloaders
    train_dataloader = dataloaders['train']
    val_dataloader = dataloaders['val']

    """Training loop"""
    for epoch in range(params.num_epochs):
        """Train for 1 epoch"""
        logging.info('Epoch number {}/{}'.format(epoch + 1, params.num_epochs))

        model.train()

        with tqdm(total=len(train_dataloader)) as t:
            for it, (target, source) in enumerate(train_dataloader):
                # target shape (1, 1, H, W), source shape (1, seq_length, H, W)
                # send input data and the model to device
                # expand target and source images to a view of (seq_length, 1, H, W)
                target = target.to(device=args.device).expand(source.size()[1], -1, -1, -1)
                source = source.to(device=args.device).permute(1, 0, 2, 3)

                # linear transformation test for NMI: use (1-source) as source image
                if params.inverse:
                    source = 1.0 - source

                # network inference & loss
                dvf, warped_source = model(target, source)  # dvf (N, 2, H, W), warped_source (N, 1, H, W)
                loss, losses = loss_fn(dvf, target, warped_source, params)

                # update parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # write losses to Tensorboard
                if it % params.save_summary_steps == 0:
                    train_summary_writer.add_scalar('loss',
                                                    loss.data,
                                                    global_step=epoch * len(train_dataloader) + it)
                    for loss_name, loss_value in losses.items():
                        train_summary_writer.add_scalar('losses/{}'.format(loss_name),
                                                        loss_value.data,
                                                        global_step=epoch * len(train_dataloader) + it)

                # update tqdm & show the loss value after the progress bar
                t.set_postfix(loss='{:05.3f}'.format(loss.data))
                t.update()
        """"""

        """Validation"""
        if (epoch + 1) % params.val_epochs == 0 or (epoch + 1) == params.num_epochs:
            logging.info("Validating at epoch: {} ...".format(epoch + 1))
            val_metrics = evaluate(model, loss_fn, val_dataloader, params, args, epoch=epoch, val=True)
            # save model checkpoint
            xutils.save_checkpoint({'epoch': epoch + 1,
                                    'state_dict': model.state_dict(),
                                    'optim_dict': optimizer.state_dict()},
                                   is_best=params.is_best,
                                   checkpoint=args.model_dir)
            # write validation metric results
            for key, value in val_metrics.items():
                val_summary_writer.add_scalar('metrics/{}'.format(key),
                                              value,
                                              global_step=epoch * len(train_dataloader))

            # save training results
            logging.info("Saving training results...")
            save_result_dir = os.path.join(args.model_dir, "train_results")
            if not os.path.exists(save_result_dir):
                os.makedirs(save_result_dir)
            # GPU tensor to CPU numpy array & transpose
            dvf_np = dvf.data.cpu().numpy().transpose(0, 2, 3, 1)  # (N, H, W, 2)
            warped_source_np = warped_source.data.cpu().numpy()[:, 0, :, :] * 255  # (N, H, W)
            target_np = target.data.cpu().numpy()[:, 0, :, :] * 255  # (N, H, W)
            source_np = source.data.cpu().numpy()[:, 0, :, :] * 255  # (N, H, W), here N = frames -1
            xutils.save_train_result(target_np, source_np, warped_source_np, dvf_np,
                                     save_result_dir, epoch=epoch + 1, fps=params.fps)
            logging.info("Done.")
        """"""

    # flush Tensorboard summaries
    train_summary_writer.close()
    val_summary_writer.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',
                        default='experiments/base_model',
                        help="Directory containing params.json")

    parser.add_argument('--restore_file',
                        default=None,
                        help="Name of the checkpoint file: 'best' for best model, or 'last' for the last saved checkpoint")

    parser.add_argument('--no_three_slices',
                        action='store_true',
                        help="Evaluate metrics on all instead of 3 slices.")

    parser.add_argument('--no_cuda',
                        action='store_true')

    parser.add_argument('--gpu',
                        default=0,
                        help='Choose GPU to run on')

    parser.add_argument('--num_workers',
                        default=8,
                        help='Number of processes used by dataloader, 0 means use main process')

    args = parser.parse_args()

    # set up device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # select GPU
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cpu')  # CPU by default
    if args.cuda:
        args.device = torch.device('cuda')

    # set up model dir
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # set up the logger
    xutils.set_logger(os.path.join(args.model_dir, 'train.log'))
    logging.info("What a beautiful day to save lives!")
    logging.info("Model: {}".format(args.model_dir))

    # load config parameters from the JSON file
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No JSON configuration file found at {}".format(json_path)
    params = xutils.Params(json_path)

    # set the three slices
    args.three_slices = not args.no_three_slices

    """Data"""
    logging.info("Setting up data loaders...")
    dataloaders = {}

    # training dataset
    train_dataset = CardiacMR_2D_UKBB(params.train_data_path,
                                      seq=params.seq,
                                      seq_length=params.seq_length,
                                      augment=params.augment,
                                      transform=transforms.Compose([
                                          CenterCrop(params.crop_size),
                                          Normalise(),
                                          ToTensor()
                                      ]))
    # training dataloader
    dataloaders['train'] = DataLoader(train_dataset,
                                      batch_size=params.batch_size, shuffle=False,
                                      num_workers=args.num_workers, pin_memory=args.cuda)


    # validation dataset
    val_dataset = CardiacMR_2D_Eval_UKBB(params.val_data_path,
                                         seq=params.seq,
                                         augment=params.augment,
                                         label_prefix=params.label_prefix,

                                         transform=transforms.Compose([
                                             CenterCrop(params.crop_size),
                                             Normalise(),
                                             ToTensor()]),
                                         label_transform=transforms.Compose([
                                             CenterCrop(params.crop_size),
                                             ToTensor()])
                                         )

    dataloaders['val'] = DataLoader(val_dataset,
                                    batch_size=params.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=args.cuda)
    logging.info("- Done.")
    """"""

    """Model & Optimiser"""
    if params.network == "BaseNet":
        model = BaseNet()
    elif params.network == "SiameseFCN":
        model = SiameseFCN()
    else:
        raise ValueError("Unknown network!")
    model = model.to(device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    """"""

    """Train and validate """
    logging.info("Starting training and validation for {} epochs.".format(params.num_epochs))
    train_and_validate(model, optimizer, loss_fn, dataloaders, params)
    logging.info("Training and validation complete.")
    """"""
