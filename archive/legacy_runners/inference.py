""" Run inference.yaml on full sequence of subjects """
from model.transform.transform_fn import spatial_transform


def process_batch(model, data_dict, loss_fn, args):

    # define source image based on modality setting
    if model.params.modality == "mono":
        data_dict["source"] = data_dict["target_original"]
    elif model.params.modality == "pseudo":
        data_dict["source"] = 1 - data_dict["target_original"]
    elif model.params.modality == "multi":
        pass
    else:
        raise ValueError("Data modality setting not recognised")

    # send relevant data to device
    for name in ["target", "source"]:
        data_dict[name] = data_dict[name].to(device=args.device)

    # network inference.yaml & compute loss
    data_dict["dvf_pred"] = model(data_dict["target"], data_dict["source"])
    data_dict["warped_source"] = spatial_transform(data_dict["source"], data_dict["dvf_pred"])
    losses = loss_fn(data_dict, model.params)
    return losses