import torch
from tqdm import tqdm
import logging

from archive.runners.inference import process_batch
from archive.runners.eval import evaluate
import utils.misc as misc_utils


def train_and_validate(model, loss_fn, data, args):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: (function callback) loss function
        data: (instance of class Data) data object
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=model.params.learning_rate)

    # set up TensorboardX summary writers
    train_tb_writer = misc_utils.set_summary_writer(args.model_dir, 'train')
    val_tb_writer = misc_utils.set_summary_writer(args.model_dir, 'val')

    for epoch in range(model.params.num_epochs):
        """Train for one epoch"""
        logging.info('Epoch number {}/{}'.format(epoch + 1, model.params.num_epochs))
        model.epoch_num = epoch

        # put model in training mode
        model.train()

        # training loop for one epoch
        with tqdm(total=len(data.train_dataloader)) as t:
            for it, data_dict in enumerate(data.train_dataloader):
                model.iter_num = epoch * len(data.train_dataloader) + it

                """ Inference, loss and train step """
                losses = process_batch(model, data_dict, loss_fn, args)

                optimizer.zero_grad()
                losses["loss"].backward()
                optimizer.step()
                """"""

                # write losses to Tensorboard
                if model.iter_num % model.params.save_summary_steps == 0:
                    for loss_name, loss_value in losses.items():
                        train_tb_writer.add_scalar(f'losses/{loss_name}', loss_value.data, global_step=model.iter_num)

                # update tqdm & show the loss value
                t.set_postfix(loss=f'{losses["loss"].data:05.3f}')
                t.update()
        """"""

        """Validation"""
        if (epoch + 1) % model.params.val_epochs == 0 or (epoch + 1) == model.params.num_epochs:
            logging.info("Validating at epoch: {} ...".format(epoch + 1))

            evaluate(model, loss_fn, data.val_dataloader, args, tb_writer=val_tb_writer, val=True)

            # save model
            misc_utils.save_checkpoint({'epoch': epoch + 1,
                                        'state_dict': model.state_dict(),
                                        'optim_dict': optimizer.state_dict()},
                                       is_best=model.is_best,
                                       checkpoint=args.model_dir)

            if model.is_best:
                logging.info("Best model found at epoch {} ...".format(epoch+1))
        """"""

    # flush Tensorboard summaries
    train_tb_writer.close()
    val_tb_writer.close()
