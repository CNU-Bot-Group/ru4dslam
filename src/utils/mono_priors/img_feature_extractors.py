from typing import Dict, List, Tuple, Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

"""
From FiT3D, here we subclass the model instead of overriding the "get_intermediate_layers" method
as it will cause errors in multipprocessing setup of the SLAM system
"""


class Fit3DModels(torch.nn.Module):
    def __init__(self, extractor_model, device):
        super().__init__()
        self.model = torch.hub.load("ywyue/FiT3D", extractor_model).to(device).eval()

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n=1,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ):
        outputs = self.model._intermediate_layers(x, n)
        if norm:
            outputs = [self.model.norm(out) for out in outputs]
        if return_class_token:
            prefix_tokens = [out[:, 0] for out in outputs]
        else:
            prefix_tokens = [
                out[:, 0 : self.model.num_prefix_tokens] for out in outputs
            ]
        outputs = [out[:, self.model.num_prefix_tokens :] for out in outputs]

        if reshape:
            B, C, H, W = x.shape
            grid_size = (
                (H - self.model.patch_embed.patch_size[0])
                // self.model.patch_embed.proj.stride[0]
                + 1,
                (W - self.model.patch_embed.patch_size[1])
                // self.model.patch_embed.proj.stride[1]
                + 1,
            )
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]

        if return_prefix_tokens or return_class_token:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)


"""
Done with overwriting get_intermediate_layers of FiT3D model
"""


def get_feature_extractor(cfg: Dict) -> nn.Module:
    """
    Get the feature extractor model based on the configuration.
    """
    device = cfg["device"]
    extractor_model = cfg["mono_prior"]["feature_extractor"]

    if extractor_model in ["dinov2_reg_small_fine", "dinov2_small_fine"]:
        return Fit3DModels(extractor_model, device)
    elif extractor_model in ["dinov2_vits14", "dinov2_vits14_reg"]:
        return (
            torch.hub.load("facebookresearch/dinov2", extractor_model).to(device).eval()
        )
    else:
        # If use other feature extractor as prior, add code here
        raise NotImplementedError("Unsupported feature extractor")


@torch.no_grad()
def predict_img_features(
    model: nn.Module,
    idx: int,
    input_tensor: torch.Tensor,
    cfg: Dict,
    device: str,
    save_feat: bool = True,
    suffix: str = "",
) -> torch.Tensor:
    """
    Predict image features using the given model.

    Args:
        model (nn.Module): The feature extractor model.
        idx (int): Image index.
        input_tensor (torch.Tensor): Input image tensor of shape (1, 3, H, W).
        cfg (Dict): Configuration dictionary.
        device (str): Device to run the model on.
        save_feat (bool): Whether to save the features.
        suffix (str): Suffix for the output file name.

    Returns:
        torch.Tensor: Extracted features.
    """
    extractor_model = cfg["mono_prior"]["feature_extractor"]
    stride = 14
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    normalize = transforms.Normalize(mean=mean, std=std)
    image_resized = process_image(input_tensor, stride, normalize, device)

    if extractor_model in ["dinov2_reg_small_fine", "dinov2_small_fine"]:
        features = model.get_intermediate_layers(
            image_resized,
            n=[8, 9, 10, 11],
            reshape=True,
            return_prefix_tokens=False,
            return_class_token=False,
            norm=True,
        )
        features = features[-1].squeeze().permute((1, 2, 0))
    elif extractor_model in ["dinov2_vits14", "dinov2_vits14_reg"]:
        features_dict = model.forward_features(image_resized)
        features = features_dict["x_norm_patchtokens"].view(
            image_resized.shape[2] // 14, image_resized.shape[3] // 14, -1
        )
    else:
        # If use other feature extractor as prior, add code here
        raise NotImplementedError("Unsupported feature extractor")

    if save_feat:
        _save_features(features, cfg, idx, suffix)

    return features


def process_image(
    image: torch.Tensor, stride: int, transforms: nn.Module, device: str = "cuda"
) -> torch.Tensor:
    """
    Process the input image for feature extraction.

    Args:
        image (torch.Tensor): Input image tensor.
        stride (int): Stride for resizing.
        transforms (nn.Module): Normalization transforms.
        device (str): Device to run the processing on.

    Returns:
        torch.Tensor: Processed image tensor.
    """
    image_tensor = transforms(image).float().to(device)
    h, w = image_tensor.shape[2:]
    height_int = (h // stride) * stride
    width_int = (w // stride) * stride
    return F.interpolate(image_tensor, size=(height_int, width_int), mode="bilinear")


def _save_features(features: torch.Tensor, cfg: Dict, idx: int, suffix: str) -> None:
    """
    Save the extracted features to a file.

    Args:
        features (torch.Tensor): Extracted features.
        cfg (Dict): Configuration dictionary.
        idx (int): Image index.
        suffix (str): Suffix for the output file name.
    """
    output_dir = f"{cfg['data']['output']}/{cfg['scene']}"
    output_path = f"{output_dir}/mono_priors/features/{idx:05d}{suffix}.npy"
    final_feat = features.detach().cpu().float().numpy()
    np.save(output_path, final_feat)
