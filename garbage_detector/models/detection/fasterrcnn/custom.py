import torch.nn as nn
from garbage_detector.models.classification.cnn import (CNNModel,
                                                        CNNModelGenerator)
from garbage_detector.models.util import load_states_


def resnet50_backbone(out_features):
    """
    Creates a new Resnet50 model that can be used as backbone in
    a Faster RCNN Model.

    The newly created model updates its weights by loading a checkpoint
    before it being returned.

    Parameters
    ----------
    out_features: int
        The number of classes that can be classified by backbone.

    Returns
    -------
    torchvision.models.ResNet
    """
    model, _, _ = CNNModelGenerator.get_pretrained(
        CNNModel.RESNET_50, 1.0, 0.7, 1, out_features)
    load_states_(CNNModel.RESNET_50, model)
    layers = list(model.children())[: -1]
    backbone = nn.Sequential(*layers)
    backbone.out_channels = 2048
    return backbone


def mobile_net_backbone(out_features):
    """
    Creates a new MobileNet model that can be used as backbone in
    a Faster RCNN Model.

    The newly created model updates its weights by loading a checkpoint
    before it being returned.

    Parameters
    ----------
    model: torchvision.models.detection.FasterRCNN
        The model.

    num_classes: int
        The number of classes that need to be detected.

    Parameters
    ----------
    out_features: int
        The number of classes that can be classified by backbone.

    Returns
    -------
    torchvision.models.MobileNetV3
    """
    model, _, _ = CNNModelGenerator.get_pretrained(
        CNNModel.MOBILE_NET, 1.0, 0.7, 1, out_features)
    load_states_(CNNModel.MOBILE_NET, model)
    backbone = model.features
    backbone.out_channels = 960
    return backbone
