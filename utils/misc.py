import os
import json
import logging
import shutil

import torch
from tensorboardX import SummaryWriter


class Params(object):
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        if os.path.exists(log_path):
            print("Logger already exists. Overwritting.")
            os.system("mv -f {} {}.backup".format(log_path, log_path))

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(stream_handler)


def set_summary_writer(model_dir, run_name):
    """
    Returns a Tensorboard summary writer
    which writes to [model_dir]/tb_summary/[run_name]/

    Args:
        model_dir: directory of the model
        run_name: sub name of the summary writer (usually 'train' or 'val')

    Returns:
        summary writer

    """
    summary_dir = os.path.join(model_dir, 'tb_summary', run_name)
    if not os.path.exists(summary_dir):
        print("TensorboardX summary directory does not exist...\n Making directory {}".format(summary_dir))
        os.makedirs(summary_dir)
    else:
        print("TensorboardX summary directory already exist at {}...\nOverwritting!".format(summary_dir))
        os.system("rm -rf {}".format(summary_dir))
        os.makedirs(summary_dir)
    return SummaryWriter(summary_dir)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        if state['epoch'] == 1:
            print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # needs to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def setup_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def param_dim_setup(param, dim):
    """
    Check dimensions of paramters and extend dimension if needed.

    Args:
        param: (int/float, tuple or list) check dimension match if tuple or list is given,
                expand to `dim` by repeating if a single integer/float number is given.
        dim: (int) data/model dimension

    Returns:
        param: (tuple)
    """

    if isinstance(param, int) or isinstance(param, float):
        param = (param,) * dim
    elif isinstance(param, tuple):
        assert len(param) == dim, \
            f"Dimension mismatch with data ({dim})"
    elif isinstance(param, list):
        assert len(param) == dim, \
            f"Dimension mismatch with data ({dim})"
        param = tuple(param)
    else:
        raise TypeError("Parameter type not int, tuple or list")
    return param

