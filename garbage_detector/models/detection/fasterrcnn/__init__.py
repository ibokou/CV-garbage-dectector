from enum import Enum

import torch.optim
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator


class FasterRCNNModel(str, Enum):
    """
    An enum class that represents the different Faster RCNN models that are used in the jupyter notebook.
    """
    RESNET_50 = 'fasterrcnn_resnet50'
    MOBILE_NET = 'fasterrcnn_mobilenet'
    CUSTOM_RESNET_50 = 'fasterrcnn_custom_resnet50'
    CUSTOM_MOBILE_NET = 'fasterrcnn_custom_mobilenet'


# Urls from which the checkpoint of the models can be download for transfer learning
# Download of checkpoint always occurs before running demo application, but can be
# turned off via a flag (see jupyter notebook)
model_state_urls = {
    FasterRCNNModel.RESNET_50: 'https://drive.google.com/uc?id=193ZLbUbpa8jfg03Nhqoz8h95Uz3SJrOh&export=download',
    FasterRCNNModel.MOBILE_NET: 'https://drive.google.com/uc?id=1QxQ_x9vDEEDG3vfYQudLf_-NgorGDibm&export=download',
    FasterRCNNModel.CUSTOM_RESNET_50: 'https://drive.google.com/uc?id=1rbJ4qUP4zxGNHLwLPjH2dhpqWcPrtuHL&export=download',
    FasterRCNNModel.CUSTOM_MOBILE_NET: 'https://drive.google.com/uc?id=1sAL_up7nFllZpLQT53Y2C0iEAok029dr&export=download'
}


class FasterRCNNModelGenerator():
    """
    A class that is responsible for generating CNN models
    """

    def __init__(self):
        pass

    def __set_box_predictor_(self, model, num_classes):
        """
        Set the box predictor of the model.

        Parameters
        ----------
        model: torchvision.models.detection.FasterRCNN
            The model.

        num_classes: int
            The number of classes that need to be detected.

        Returns
        -------
        None
        """
        None
        if model is not None:
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, num_classes)

    def __generate_and_optimizer_scheduler(self, model, lr, gamma, step_size):
        """
        Creates the optimizer and scheduler for a model.

        Parameters
        ----------
        model: torchvision.models.detection.FasterRCNN
            The model.

        lr: float
            Learning rate for the optimizer of the model.

        step_size: int
            Step size for the scheduler.

        gamma: float
            Gamma value for the scheduler.

        Returns
        -------
        tuple
            containing optimizer and scheduler
        """
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=0.9, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=step_size,
                                                    gamma=gamma)

        return optimizer, scheduler

    def create_model(self, model_type, lr, gamma, step_size, num_classes):
        """
        Creates a pretrained model from PyTorch with necessary adjustments, such
        as setting the box predictor, so that it can be used for different types
        of datasets with varying number of classes.

        The pretrained models contain weights obtained by the training with the
        COCO dataset, which contains 1.5 millions object instances.

        See https://cocodataset.org/#home

        Parameters
        ----------
        model: torchvision.models.detection.FasterRCNN
            The model.

        lr: float
            Learning rate for the optimizer of the model

        step_size: int
            Step size for the scheduler

        gamma: float
            Gamma value for the scheduler

        num_classes: int
            The number of classes that need to be detected.

        Returns
        -------
        tuple
            containing optimizer and scheduler
        """
        model = None

        if model_type == FasterRCNNModel.RESNET_50:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
                                                                            weights_backbone=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)

        if model_type == FasterRCNNModel.MOBILE_NET:
            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
                weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1,
                weights_backbone=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
            )

        self.__set_box_predictor_(model, num_classes)

        optimizer, scheduler = self.__generate_and_optimizer_scheduler(
            model, lr, gamma, step_size)

        return model, optimizer, scheduler

    def create_model_with_custom_backbone(self, backbone, lr, gamma, step_size, num_classes):
        """
        Creates a model with a self trained backbone model for the classifcation part of
        the FasterRCNN.

        See also https://hasty.ai/docs/mp-wiki/model-architectures/faster-r-cnn

        See https://cocodataset.org/#home

        Parameters
        ----------
        backbone: Function -> [..., torchvision.models.ResNet]
            Function that returns the model which is used for classifcation in
            the Faster RCNN

        lr: float
            Learning rate for the optimizer of the model

        step_size: int
            The step size for the scheduler of the Faster RCNN

        gamma: float
            The gamma value for the scheduler of the Faster RCNN

        num_classes: int
            The number of classes that need to be detected.

        Returns
        -------
        tuple
            containing Faster RCNN model, optimizer and scheduler
        """

        backbone = backbone(num_classes)

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )

        model = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )

        optimizer, scheduler = self.__generate_and_optimizer_scheduler(model,
                                                                       lr, gamma, step_size)

        return model, optimizer, scheduler
