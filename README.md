# Learning Diffeomorphic and Modality-invariant Registration using B-splines
Welcome!

This repository contains code for the Modality-Invariant Diffeomorphic Deep Learning Image Registration (MIDIR) framework 
presented in the paper:

```
@inproceedings{
qiu2021learning,
title={Learning Diffeomorphic and Modality-invariant Registration using B-splines},
author={Huaqi Qiu and Chen Qin and Andreas Schuh and Kerstin Hammernik and Daniel Rueckert},
booktitle={Medical Imaging with Deep Learning},
year={2021},
url={https://openreview.net/forum?id=eSI9Qh2DJhN}
}
```
Please consider citing the paper if you use the code in this repository.

<img src="example_image.png" width="600">

## Installation
1. Clone this repository
2. In a fresh Python 3.7.x virtual environment, install dependencies via:
    ```
    pip install -r <path_to_cloned_repository>/requirements.txt
    ```

### Using GPU
If you want to run the code on GPU (which is recommended), you should check your CUDA and CuDNN installations. 
This code has been tested on CUDA 10.1 and CuDNN 7.6.5. Later versions should be backward-competible. 

To install the exact CUDA and CuDNN versions with the corresponding Pytorch build:
```
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```


## File structure
- `conf`: Hydra configuration for training deep learning models
- `conf_inference`: Hydra configuration for inference
- `data`:  data loading
- `model`:
    - `lightning.py`: the LightningModule which puts everything together
    - `loss.py`: image similarity loss and transformation regularity loss
    - `network.py`: dense and b-spline model networks
    - `transformation.py`: dense and b-spline parameterised SVF transformation modules
    - `utils.py`: main configuration parsing is in here
- `utils`: some handy utility functions


## Configurations
We use [Hydra](https://hydra.cc/docs/intro) for structured configurations. 
Each directory in `conf/` is a *config group* which contains alternative configurations for that group. 
The final configuration is the *composition* of all *config groups*.
The default options for the groups are set in `conf/config.yaml`. 
To use a different configuration for a group, for example the loss function:
```
python loss=<lncc/mse/nmi> ...
```

Any configuration in this structure can be conveniently over-written in CLI at runtime. For example, to change the regularisation weight at runtime:
```
python loss.reg_weight=<your_reg_weight> ...
```

See [Hydra documentation](https://hydra.cc/docs/intro) for more details.



## Training your own model
### Configure data
- Create your own data configuration file `conf/data/your_data.yaml`
- Specifying the paths to your training and validation data as well as other configurations. 
Your data should be organised as one subject per directory. 
- The configurations are parsed to construct `Dataset` in [`model/utils.py`](https://github.com/qiuhuaqi/midir/blob/4fc8b458cd24778c12ecdf9becafb127e19dcf99/model/utils.py#L70) 
and `DataLoader` in [`model/lightning.py`](https://github.com/qiuhuaqi/midir/blob/4fc8b458cd24778c12ecdf9becafb127e19dcf99/model/lightning.py#L32)
- For 3D brain images inter-subject registration, see `conf/data/brain_camcan.yaml` for reference. 
- For 2D cardiac image intra-subject (frame-to-frame) registration, see `conf/data/cardiac_ukbb.yaml` for reference.

### Configure model and training
The code contained in this repository has some generic building blocks for deep learning registration, 
so you can build your own registration model by playing with the configurations.
- `conf/loss`: Image similarity loss function configurations
- `conf/network`: CNN-based network configurations
- `conf/transformation`: Transformation model configurations
- `conf/training`: Training configurations
- If you don't have anatomical segmentation for validation, you can switch off the related metric evaluation by 
removing "image_metrics" from `meta.metric_groups` and "mean_dice" from `meta.hparam_metrics` in `config.yaml`


### Run a single training
To train the default model on your own data:
```
python train.py hydra.run.dir=<model_dir> \
    data=<your_data> \
    meta.gpu=<gpu_num>
```
Training logs and outputs will be saved in `model_dir`. 
On default settings, a checkpoint of the model will be saved at `model_dir/checkpoints/last.ckpt`
A copy of the configurations will be saved to `model_dir/.hydra` automatically.
As mentioned above, you can overwrite any configuration in CLI at runtime, or chagne the default values in `conf/`


### Hyper-parameter tuning
To tune any hyper-parameters such as regularisation loss weight, b-spline control point spacing or network channels,
we can simply use the sweeping run feature of Hydra. For example, to tune the regularisation weight:
```
python train.py \
    -m hydra.sweep.dir=<sweep_parent_dir> \
    hydra.sweep.subdir=\${loss.reg_loss}_\${loss.reg_weight} \
    loss.reg_weight=<weight1,weight2,weight3,...> \
    meta.gpu=<gpu_num>
```


## Inference/Testing
To run inference of a trained model, 
```
python inference.py hydra.run.dir=<output_dir> \
    model=dl \
    data=<inference_data>
    model.ckpt_path=<path_to_model_checkpoint> \
    gpu=<gpu_num>
```
A different set of configuration files are used for inference (see `conf_inference/`)



## Data
We cannot share the data we use in the paper directly. But you can apply to download at:
- [CamCAN](https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/)
- [UK Biobank](https://www.ukbiobank.ac.uk/enable-your-research) 


## Contact Us
If you have any question or need any help running the code, feel free to open an issue or email us at:
[huaqi.qiu15@imperial.ac.uk](mailto:huaqi.qiu15@imperial.ac.uk)


## Others
- [Hydra configuration](https://hydra.cc/docs/intro)
- [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/1.1.0/)
