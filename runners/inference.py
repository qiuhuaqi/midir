""" Run inference on full sequence of subjects """
from model.transformations import spatial_transform


def process_batch(model, data_dict, loss_fn, args):
    # for mono-modal case
    if model.params.modality == "mono":
        data_dict["source"] = data_dict["target_original"]

    # send relevant data to device
    for name in ["target", "source"]:
        data_dict[name] = data_dict[name].to(device=args.device)

    # network inference & compute loss
    data_dict["dvf_pred"] = model(data_dict["target"], data_dict["source"])
    data_dict["warped_source"] = spatial_transform(data_dict["source"], data_dict["dvf_pred"])
    losses = loss_fn(data_dict, model.params)
    return losses