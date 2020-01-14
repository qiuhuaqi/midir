import torch
import numpy as np


def nmi_from_joint_entropy_numpy(joint_hist, eps=1e-12):
    """Compute Normalised Mutual Information (NMI) using joint histogram"""
    # normalise joint histogram to acquire joint pdf
    joint_pdf = joint_hist / np.sum(joint_hist)

    # avoid log(0)
    joint_pdf += eps

    # marginalise the joint distribution to get marginal distributions
    # reference bins in dim0, target bins in dim1
    pdf_marginal_ref = np.sum(joint_pdf, axis=1)
    pdf_marginal_tar = np.sum(joint_pdf, axis=0)

    # marginal entropy
    entropy_ref = - np.sum(pdf_marginal_ref * np.log(pdf_marginal_ref))
    entropy_tar = - np.sum(pdf_marginal_tar * np.log(pdf_marginal_tar))

    # joint entropy
    entropy_joint = - np.sum(joint_pdf * np.log(joint_pdf))

    NMI = (entropy_ref + entropy_tar) / entropy_joint
    return NMI, entropy_joint, entropy_ref, entropy_tar


def nmi_from_joint_entropy_pytorch(joint_hist, eps=1e-12):
    """Compute Normalised Mutual Information (NMI) using joint histogram in Pytorch"""

    # normalise joint histogram to acquire joint pdf
    joint_pdf = joint_hist / torch.sum(joint_hist)

    # avoid log(0)
    joint_pdf += eps

    # marginalise the joint distribution to get marginal distributions
    # batch size in dim0, reference bins in dim1, target bins in dim2
    pdf_marginal_ref = torch.sum(joint_pdf, dim=2)
    pdf_marginal_tar = torch.sum(joint_pdf, dim=1)

    # marginal entropy (N, 1)
    entropy_ref = - torch.sum(pdf_marginal_ref * torch.log(pdf_marginal_ref), dim=1)
    entropy_tar = - torch.sum(pdf_marginal_tar * torch.log(pdf_marginal_tar), dim=1)
    
    # joint entropy (N, 1)
    entropy_joint = - torch.sum(joint_pdf * torch.log(joint_pdf), dim=(1,2))

    nmi = torch.mean((entropy_ref + entropy_tar) / entropy_joint)
    return nmi
