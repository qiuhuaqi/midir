""" Main training function"""

from tqdm import tqdm
import os
import argparse
import logging
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model.networks import BaseNet

from model.losses import huber_loss_fn, nmi_loss_fn
from model.dataset_utils import CenterCrop, Normalise, ToTensor
from model.datasets import CardiacMR_2D_UKBB, CardiacMR_2D_Eval_UKBB
from model.submodules import resample_transform

from eval import evaluate
from utils import xutils, flow_utils

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before training w/o postfix")
parser.add_argument('--no_three_slices', action='store_true', help="Evaluate metrics on all instead of 3 slices.")

parser.add_argument('--no_cuda', action='store_true')
parser.add_argument('--gpu', default=0, help='Choose GPU to run on')
parser.add_argument('--num_workers', default=8, help='Number of processes used by dataloader, 0 means use main process')


def train(model, optimizer, loss_fn, dataloader, params, epoch, summary_writer):
    """
    Train the model for one epoch

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        params: (instance of Params) configuration parameters
        epoch: (int) number of epoch this is training (for the summary writer)
        summary_writer: (Class instance) TensorBoardX SummaryWriter()
    """

    # set model in training mode
    model.train()

    with tqdm(total=len(dataloader)) as t:
        for it, (target, source) in enumerate(dataloader):
            # target shape (1, 1, H, W), source shape (1, seq_length, H, W)
            # send input data and the model to device
            # expand target and source images to a view of (seq_length, 1, H, W)
            target = target.to(device=args.device).expand(source.size()[1], -1, -1, -1)
            source = source.to(device=args.device).permute(1, 0, 2, 3)


            # compute outputs and loss
            flow = model(target, source)  # (N, 2, H, W)
            loss, losses = loss_fn(flow, target, source, params)


            # clear previous gradients, calculate gradient and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # save summary of loss every some steps
            if it % params.save_summary_steps == 0:
                summary_writer.add_scalar('loss', loss.data, global_step=epoch * len(dataloader) + it)

                for loss_name, loss_value in losses.items():
                    summary_writer.add_scalar('losses/{}'.format(loss_name), loss_value.data, global_step=epoch * len(dataloader) + it)

            # update tqdm, show the loss value after the progress bar
            t.set_postfix(loss='{:05.3f}'.format(loss.data))
            t.update()



            # save visualisation of training results
            if (epoch + 1) % params.save_result_epochs == 0 or (epoch + 1) == params.num_epochs:
                if it == len(dataloader) - 1:

                    # warp source image with full resolution flow
                    warped_source = resample_transform(source, flow)

                    # [flow and warped source] -> cpu -> numpy array
                    op_flow = flow.data.cpu().numpy().transpose(0, 2, 3, 1)  # (N, H, W, 2)
                    warped_source = warped_source.data.cpu().numpy()[:, 0, :, :] * 255  # (N, H, W)

                    # [input images] -> cpu -> numpy array -> [0, 255]
                    target = target.data.cpu().numpy()[:, 0, :, :] * 255  # (N, H, W)
                    source = source.data.cpu().numpy()[:, 0, :, :] * 255  # (N, H, W), here N = frames -1

                    # set up the result dir for this epoch
                    save_result_dir = os.path.join(args.model_dir, "train_results", "epoch_{}".format(epoch + 1))
                    if not os.path.exists(save_result_dir):
                        os.makedirs(save_result_dir)

                    # NOTE: the following code saves all N frames in a batch
                    # save flow (hsv + quiver), target, source, warped source and error
                    # flow_utils.save_flow_hsv(op_flow, target, save_result_dir, fps=params.fps)
                    flow_utils.save_warp_n_error(warped_source, target, source, save_result_dir, fps=params.fps)
                    flow_utils.save_flow_quiver(op_flow * (target.shape[-1] / 2), source, save_result_dir,
                                                fps=params.fps)


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

    # ------------------------------------------- #
    # Training loop
    # ------------------------------------------- #
    for epoch in range(params.num_epochs):
        logging.info('Epoch number {}/{}'.format(epoch + 1, params.num_epochs))


        # train the model for one epoch
        logging.info("Training...")
        train(model, optimizer, loss_fn, train_dataloader, params, epoch, train_summary_writer)

        # validation
        if (epoch + 1) % params.val_epochs == 0 or (epoch + 1) == params.num_epochs:
            logging.info("Validating at epoch: {} ...".format(epoch + 1))
            val_metrics = evaluate(model, loss_fn, val_dataloader, params, args, val=True)

            # save the most recent results in a JSON file
            save_path = os.path.join(args.model_dir, "val_results_last_3slices_{}.json".format(args.three_slices))
            xutils.save_dict_to_json(val_metrics, save_path)

            # determine whether this is the best model based on mean dice score
            if params.seq == 'sa':
                mean_val_dice = np.mean([val_metrics['dice_lv_mean'], val_metrics['dice_myo_mean'], val_metrics['dice_rv_mean']])
                mean_val_mcd = np.mean([val_metrics['mcd_lv_mean'], val_metrics['mcd_myo_mean'], val_metrics['mcd_rv_mean']])
                mean_val_hd = np.mean([val_metrics['hd_lv_mean'], val_metrics['hd_myo_mean'], val_metrics['hd_rv_mean']])
            else:
                mean_val_dice = val_metrics['dice_mean']
                mean_val_mcd = val_metrics['mcd_mean']
                mean_val_hd = val_metrics['hd_mean']

            # add mean metrics to the write metrics dict
            val_metrics['mean_val_dice'] = mean_val_dice
            val_metrics['mean_val_mcd'] = mean_val_mcd
            val_metrics['mean_val_hd'] = mean_val_hd

            logging.info("Mean val dice: {:05.3f}".format(mean_val_dice))
            logging.info("Mean val mcd: {:05.3f}".format(mean_val_mcd))
            logging.info("Mean val hd: {:05.3f}".format(mean_val_hd))
            logging.info("Mean val negative detJ: {:05.3f}".format(val_metrics['negative_detJ_mean']))
            logging.info("Mean val mag grad detJ: {:05.3f}".format(val_metrics['mean_mag_grad_detJ_mean']))
            assert val_metrics['negative_detJ_mean'] <= 1, "Invalid det Jac: Ratio of folding points > 1"  # sanity check

            # determine if the best model
            is_best = False
            best_one_metric = 0.0
            current_one_metric = mean_val_dice  # use val dice to choose best model
            # current_one_metric = np.mean([mean_val_mcd , mean_val_hd])  # use contour distance to choose the best model
            if epoch + 1 == params.val_epochs:  # first validation
                best_one_metric = current_one_metric
            if current_one_metric >= best_one_metric:
                is_best = True
                best_one_metric = current_one_metric

            # save model checkpoint
            xutils.save_checkpoint({'epoch': epoch + 1,
                                    'state_dict': model.state_dict(),
                                    'optim_dict': optimizer.state_dict()},
                                   is_best=is_best,
                                   checkpoint=args.model_dir)

            for key, value in val_metrics.items():
                val_summary_writer.add_scalar('metrics/{}'.format(key), value, global_step=epoch * len(train_dataloader))

            # save the validation results for the best model separately
            if is_best:
                save_path = os.path.join(args.model_dir, "val_results_best_3slices_{}.json".format(args.three_slices))
                xutils.save_dict_to_json(val_metrics, save_path)

    # close TensorBoard summary writers
    train_summary_writer.close()
    val_summary_writer.close()



if __name__ == '__main__':
    # parse arguments and read parameters from JSON file
    args = parser.parse_args()

    # set the GPU to use and device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')


    # set up model dir
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # set up the logger
    xutils.set_logger(os.path.join(args.model_dir, 'train.log'))
    logging.info("What a beautiful day to save lives!")
    logging.info("Model: {}".format(args.model_dir))


    # load setting parameters from a JSON file
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No JSON configuration file found at {}".format(json_path)
    params = xutils.Params(json_path)


    # set the three slices
    args.three_slices = not args.no_three_slices

    # set up dataset and DataLoader
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



    # instantiate model and move to device
    model = BaseNet()
    model = model.to(device=args.device)

    # set up the loss function and optimiser
    if params.loss_fn == "unsupervised":
        loss_fn = huber_loss_fn
    elif params.loss_fn == "unsupervised_nmi":
        loss_fn = nmi_loss_fn

    # set up optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)


    ## train and validate
    logging.info("Starting training and validation for {} epochs.".format(params.num_epochs))
    train_and_validate(model, optimizer, loss_fn, dataloaders, params)
    logging.info("Training and validation complete.")
