"""Run model inference and save outputs for analysis"""
import os
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from datasets.brain import BrainMRInterSubj3D
from datasets.cardiac import CardiacMR2D

from model.lightning import LightningDLReg
from model.baselines import Identity, MIRTK
from core_modules.transform.utils import warp
from utils.image_io import save_nifti
from utils.misc import setup_dir
from analyse import analyse_output

import random
random.seed(7)


def get_inference_dataloader(cfg, pin_memory=False):
    if cfg.data.name == 'brain_camcan':
        dataset = BrainMRInterSubj3D(**cfg.data.dataset)
    elif cfg.data.name == 'cardiac_ukbb':
        dataset = CardiacMR2D(**cfg.data.dataset)
    else:
        raise ValueError(f'Dataset config ({cfg.data.dataset}) not recognised.')
    return DataLoader(dataset,
                      shuffle=False,
                      pin_memory=pin_memory,
                      **cfg.data.dataloader)


def get_inference_model(cfg, device=torch.device('cpu')):
    if cfg.model.name == 'id':
        model = Identity()

    elif cfg.model.name == 'mirtk':
        model = MIRTK(**cfg.model.mirtk_params)

    elif cfg.model.name == 'dl':
        assert os.path.exists(cfg.inference_model.ckpt_path)
        model = LightningDLReg.load_from_checkpoint(cfg.inference_model.ckpt_path)
        model = model.to(device=device)
        model.evaluate()

    else:
        raise ValueError(f"Unknown inference model type: {cfg.inference_model.type}")
    return model


def inference(model, dataloader, output_dir, device=torch.device('cpu')):
    for idx, batch in enumerate(tqdm(dataloader)):
        # -- Note on data shape and reshape -- #
        # 2d: (N=1, num_slices, H, W) -> (num_slices, N=1, H, W)
        # 3d: (N=1, 1, H, W, D) -> (1, N=1, H, W, D)
        # -------------------------------------#
        # reshape data for inference
        for k, x in batch.items():
            batch[k] = x.transpose(0, 1).to(device=device)

        # model inference
        out = model(batch['target'], batch['source'])

        if isinstance(out, tuple):
            # (flows, disps), multi-level list
            batch['disp_pred'] = out[1][-1]
        elif isinstance(out, list):
            # disps only, multi-level list
            batch['disp_pred'] = out[-1]
        else:
            batch['disp_pred'] = out

        # warp images and segmentation using predicted disp
        batch['warped_source'] = warp(batch['source'], batch['disp_pred'])
        if 'source_seg' in batch.keys():
            batch['warped_source_seg'] = warp(batch['source_seg'], batch['disp_pred'],
                                              interp_mode='nearest')
        if 'target_original' in batch.keys():
            batch['target_pred'] = warp(batch['target_original'], batch['disp_pred'])

        # save the outputs
        subj_id = dataloader.dataset.subject_list[idx]
        output_id_dir = setup_dir(output_dir + f'/{subj_id}')
        for k, x in batch.items():
            x = x.detach().cpu().numpy()
            # reshape for saving:
            # 2D: img (N=num_slice, 1, H, W) -> (H, W, N);
            #     disp (N=num_slice, 2, H, W) -> (H, W, N, 2)
            # 3D: img (N=1, 1, H, W, D) -> (H, W, D);
            #     disp (N=1, 3, H, W, D) -> (H, W, D, 3)
            x = np.moveaxis(x, [0, 1], [-2, -1]).squeeze()
            save_nifti(x, path=output_id_dir + f'/{k}.nii.gz')


@hydra.main(config_path="conf_inference", config_name="config")
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())

    # configure GPU
    gpu = cfg.gpu
    if gpu is not None and isinstance(gpu, int):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # configure dataset & model
    dataloader = get_inference_dataloader(cfg, pin_memory=(device is torch.device('cuda')))
    model = get_inference_model(cfg, device=device)

    # run inference
    output_dir = setup_dir(os.getcwd() + '/outputs')  # cwd = hydra.run.dir
    inference(model, dataloader, output_dir, device=device)

    # (optional) run analysis on the current inference outputs
    if cfg.analyse:
        analyse_output(output_dir, os.getcwd() + '/analysis', cfg.metric_groups)


if __name__ == '__main__':
    main()
