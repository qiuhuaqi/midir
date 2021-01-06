"""Run model inference and save outputs for analysis"""
import os
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import numpy as np
import torch

from datasets.brain import BrainInterSubject3DEval
from model.lightning import LightningDLReg
from model.baselines import Identity, MirtkFFD
from core_modules.transform.utils import warp
from utils.image_io import save_nifti
from utils.misc import setup_dir
from analyse import analyse_output

import random
random.seed(7)


def get_inference_model(cfg, device=torch.device('cpu')):
    if cfg.inference_model.type == 'id':
        model = Identity(cfg.ndim)

    elif cfg.inference_model.type == 'mirtk':
        model = MirtkFFD(hparams=cfg.inference_model.mirtk_params)

    elif cfg.inference_model.type == 'dl':
        assert os.path.exists(cfg.inference_model.ckpt_path)
        model = LightningDLReg.load_from_checkpoint(cfg.inference_model.ckpt_path).to(device=device)
        model.eval()

    else:
        raise ValueError(f"Unknown inference model type: {cfg.inference_model.type}")
    return model


def inference(model, inference_dataset, output_dir, device=torch.device('cpu')):
    for idx, batch in enumerate(tqdm(inference_dataset)):
        # reshape data for inference
        for k, x in batch.items():
            batch[k] = x.unsqueeze(1).to(device=device)

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

        batch['warped_source'] = warp(batch['source'], batch['disp_pred'])
        if 'source_seg' in batch.keys():
            batch['warped_source_seg'] = warp(batch['source_seg'], batch['disp_pred'],
                                              interp_mode='nearest')
        if 'target_original' in batch.keys():
            batch['target_pred'] = warp(batch['target_original'], batch['disp_pred'])

        # save the outputs
        subj_id = inference_dataset.subject_list[idx]
        output_id_dir = setup_dir(output_dir + f'/{subj_id}')
        for k, x in batch.items():
            x = x.detach().cpu().numpy()
            # TODO: doc this output shape
            x = np.moveaxis(x, [0, 1], [-2, -1]).squeeze()
            save_nifti(x, path=output_id_dir + f'/{k}.nii.gz')


@hydra.main(config_path="conf", config_name="config_inference")
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
    inference_dataset = BrainInterSubject3DEval(**cfg.data)
    model = get_inference_model(cfg, device=device)

    # run inference
    output_dir = setup_dir(os.getcwd() + '/outputs')  # cwd = hydra.run.dir
    inference(model, inference_dataset, output_dir, device=device)

    # (optional) run analysis on the current inference outputs
    if cfg.analyse:
        analyse_output(output_dir, os.getcwd() + '/analysis', cfg.metric_groups)


if __name__ == '__main__':
    main()
