""" Main training function"""

from tqdm import tqdm
import os
import argparse
import logging
import torch

from data.datasets import Data
from model.models import RegDVF
from model.losses import loss_fn
from eval import evaluate
from utils import misc

# set random seed for workers generating random deformation
import random
random.seed(12)


def train_and_validate(model, optimizer, loss_fn, data, params):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        data: (instance of class Data) data object
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        params: (instance of Params) configuration parameters
    """
    # reload weights from a specified file to resume training
    if args.restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        misc.load_checkpoint(restore_path, model, optimizer)

    # set up TensorboardX summary writers
    train_summary_writer = misc.set_summary_writer(args.model_dir, 'train')
    val_summary_writer = misc.set_summary_writer(args.model_dir, 'val')

    """Training loop"""
    for epoch in range(params.num_epochs):
        """Train for 1 epoch"""
        logging.info('Epoch number {}/{}'.format(epoch + 1, params.num_epochs))

        model.train()

        with tqdm(total=len(data.train_dataloader)) as t:
            for it, data_point in enumerate(data.train_dataloader):

                """brain data"""
                target, source, target_original, brain_mask, dvf_gt = data_point
                if params.modality == "mono":
                    source = target_original

                target = target.to(device=args.device)  # (Nx1xHxW), N=batch_size
                source = source.to(device=args.device)  # (Nx1xHxW)
                """"""

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
                                                    global_step=epoch * len(data.train_dataloader) + it)

                    for loss_name, loss_value in losses.items():
                        train_summary_writer.add_scalar('losses/{}'.format(loss_name),
                                                        loss_value.data,
                                                        global_step=epoch * len(data.train_dataloader) + it)

                # update tqdm & show the loss value after the progress bar
                t.set_postfix(loss='{:05.3f}'.format(loss.data))
                t.update()
        """"""

        """Validation"""
        if (epoch + 1) % params.val_epochs == 0 or (epoch + 1) == params.num_epochs:
            logging.info("Validating at epoch: {} ...".format(epoch + 1))
            val_metrics, val_loss = evaluate(model, loss_fn, data, params, args, epoch=epoch, val=True)

            if params.is_best:
                logging.info("Best model found at epoch {} ...".format(epoch+1))

            # save model checkpoint
            misc.save_checkpoint({'epoch': epoch + 1,
                                    'state_dict': model.state_dict(),
                                    'optim_dict': optimizer.state_dict()},
                                 is_best=params.is_best,
                                 checkpoint=args.model_dir)

            # write validation metric results to Tensorboard
            for key, value in val_metrics.items():
                val_summary_writer.add_scalar('metrics/{}'.format(key),
                                              value,
                                              global_step=epoch * len(data.train_dataloader))
            # write validation loss to Tensorborad
            val_summary_writer.add_scalar('val_loss', val_loss,
                                          global_step=epoch * len(data.train_dataloader))
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
                        help="Prefix of the checkpoint file:"
                             " 'best' for best model, or 'last' for the last saved checkpoint")

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
    misc.set_logger(os.path.join(args.model_dir, 'train.log'))
    logging.info("What a beautiful day to save lives!")
    logging.info("Model: {}".format(args.model_dir))

    # load config parameters from the JSON file
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No JSON configuration file found at {}".format(json_path)
    params = misc.Params(json_path)

    """Data"""
    logging.info("Setting up data loaders...")
    data = Data(args, params)
    data.use_brain()
    logging.info("- Done.")
    """"""

    """Model & Optimiser"""
    model = RegDVF(params.network, transform_model=params.transform_model)
    model = model.to(device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    """"""

    """Train and validate """
    logging.info("Starting training and validation for {} epochs.".format(params.num_epochs))
    train_and_validate(model, optimizer, loss_fn, data, params)
    logging.info("Training and validation complete.")
    """"""
