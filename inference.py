import os
import hydra
from omegaconf import DictConfig

from tqdm import tqdm
import numpy as np
import torch

from data.datasets import BrainLoadingDataset
from model.baselines import Identity
from model.transformations import spatial_transform
from utils.image_io import save_nifti

from analyse import analyse_output

# fix random seeding for eval dataset for fair comparison
import random
random.seed(7)


def get_model(cfg):
    model = Identity(cfg.data.dim)
    # TODO: DL model 1) load model checkpoint; 2) eval mode; 3)put on GPU
    return model


def inference(model, inference_dataset, output_dir, device=torch.device('cpu')):
    print("Running inference:")
    for idx, batch in enumerate(tqdm(inference_dataset)):
        # reshape data for inference.yaml
        for k, x in batch.items():
            # images 2d: (N, 1, H, W), 3d: (1, 1, H, W, D)
            # dvf 2d: (N, 2, H, W), 3d: (1, 3, H, W, D)
            if k != 'dvf_gt':
                batch[k] = x.unsqueeze(1).to(device=device)

        # model inference
        batch['dvf_pred'] = model(batch['target'].to(device=device), batch['source'].to(device=device))

        # deformed images with predicted DVF
        batch['target_pred'] = spatial_transform(batch['target_original'].to(device=device),
                                                 batch['dvf_pred'].to(device=device))
        batch['warped_source'] = spatial_transform(batch['source'].to(device=device),
                                                   batch['dvf_pred'].to(device=device))

        # deformed segmentation with predicted DVF
        batch['target_cor_seg_pred'] = spatial_transform(batch['source_cor_seg'].to(device=device),
                                                         batch['dvf_pred'].to(device=device),
                                                         interp_mode='nearest')
        batch['target_subcor_seg_pred'] = spatial_transform(batch['source_subcor_seg'].to(device=device),
                                                            batch['dvf_pred'].to(device=device),
                                                            interp_mode='nearest')

        # save the outputs
        id = inference_dataset.subject_list[idx]
        output_id_dir = output_dir + f'/{id}'
        if not os.path.exists(output_id_dir):
            os.makedirs(output_id_dir)

        for k, x in batch.items():
            x = x.detach().cpu().numpy()
            x = np.moveaxis(x, [0, 1], [-2, -1]).squeeze()

            save_nifti(x, path=output_id_dir + f'/{k}.nii.gz')


@hydra.main(config_path="conf", config_name="inference")
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
    inference_dataset = BrainLoadingDataset(**cfg.data)
    model = get_model(cfg)

    # run inference
    output_dir = os.getcwd() + '/outputs'
    inference(model, inference_dataset, output_dir, device=device)

    # (optional) run analysis on the current inference outputs
    if cfg.analyse:
        analyse_output(output_dir, os.getcwd() + '/analysis', cfg.metric_groups)


if __name__ == '__main__':
    main()
