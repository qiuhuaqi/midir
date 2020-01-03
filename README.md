# Deep Learning Registration + Cardiac Motion Estimation
Image based cardiac motion analysis using deep learning registration.
Implemented in Pytorch-1.0.0.


# Change Log
- `Added` for new features.
- `Changed` for changes in existing functionality.
- `Deprecated` for soon-to-be removed features.
- `Removed` for now removed features.
- `Fixed` for any bug fixes.


## 2019-06-10
`Added`
- Started Change Log
- Add a base `params.json`

## 2019-06-13
`Changed`
- changed `args.three_slices` to negative logic `args.no_three_slices` in `train.py`
- changed legacy `alpha`, `beta` to `h_alpha`, `h_beta` in `search_hyperparams.py`

## 2019-07-25
`Added`
- script to generate supervised data using FFD
- supervised dataset
- supervised training loss
- mixed training loss

`Removed`
- validation loss
- inactive code for LAX metrics

`Changed`
- hyperparamter search alpha/beta for supervision mixture, h_alpha/h_beta for smoothness

## 2019-08-09
Repo migration to github.com
