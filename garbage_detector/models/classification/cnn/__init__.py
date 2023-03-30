
from enum import Enum

import torch.optim as optim
from garbage_detector.models import (adjust_classification_layer_,
                                     freeze_model_params_)
from torch.optim.lr_scheduler import StepLR
from torchvision import models


class CNNModel(str, Enum):
    """
    An enum class that represents the different CNN models that are used in the jupyter notebook.
    """
    RESNET_50 = 'cnn_resnet50'
    MOBILE_NET = 'cnn_mobilenet'
    EFFICIENT_NET = 'cnn_efficientnet'
    GOOGLE_NET = 'cnn_googlenet'


# Urls from which the checkpoint of the models (after performing the benchmark) can be download for transfer learning
model_state_urls = {
    CNNModel.RESNET_50: 'https://drive.google.com/uc?id=1YaMSJiHNfGvs5y4phpLFWuJvF8hYg-vC&export=download',
    CNNModel.MOBILE_NET: 'https://drive.google.com/uc?id=1aHbB7U_KTp6SprRy_tW1TnxCfLT7urZ7&export=download',
    CNNModel.EFFICIENT_NET: 'https://drive.google.com/uc?id=1-3-SQECiUa6ZZXcn2VjjAN0rDnPF_ogR&export=download',
    CNNModel.GOOGLE_NET: 'https://drive.google.com/uc?id=1gKVIAuw0bmNlAOvAG50bh4hqO4Q5uOCw&export=download'
}


class CNNModelGenerator:
    """
    A class that is responsible for generating CNN models.
    """

    @classmethod
    def get_pretrained(self, model_name, lr, gamma, step_size, out_features):
        """Returns model with pretrained weights provided by PyTorch.

        Parameters
        ----------
        model_name: str
            Name of the model to be generated.

        lr: float
            Learning rate for the optimizer of the model.

        step_size: int
            Step size for the scheduler.

        gamma: float
            Gamma value for the scheduler.

        out_features: int
            The number of classes that can be classified.

        Returns
        -------
        tuple
            containing model, optimizer and scheduler.
        """

        model = None
        if (model_name == CNNModel.RESNET_50):
            model = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2)
        if (model_name == CNNModel.MOBILE_NET):
            model = models.mobilenet_v3_large(
                weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        if (model_name == CNNModel.EFFICIENT_NET):
            model = models.efficientnet_v2_m(
                weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        if (model_name == CNNModel.GOOGLE_NET):
            model = models.googlenet(
                weights=models.GoogLeNet_Weights.IMAGENET1K_V1)

        # Parameter for all layers are frozen and the last layer is replaced
        # with a new linear layer (adjust_classification_layer_). The weights
        # of only this layers are updated, which speeds up the trainig.

        # Under different circumstances, the weights of the frozen layers are
        # still updated by the optimizer.

        # See following discussion:
        # https://discuss.pytorch.org/t/parameters-with-requires-grad-false-are-updated-during-training/90096
        freeze_model_params_(model)
        adjust_classification_layer_(model, out_features)

        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size, gamma=gamma)

        return (model, optimizer, scheduler)
