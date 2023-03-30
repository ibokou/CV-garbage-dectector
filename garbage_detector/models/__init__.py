import torch
import torch.nn as nn
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def is_faster_rcnn(model):
    """Checks whether the passed model is of type FasterRCNN.

    Parameters
    ----------
    model: Optional[..., torchvision.models.detection.FasterRCNN]

    Returns
    -------
    bool
    """
    return 'FasterRCNN' in model.__class__.__name__


def freeze_model_params_(model):
    """Freezes parameters of all layers, so that weights are 
    not updated during training.

    Parameters
    ----------
    model: Optional[..., torchvision.models.ResNet]

    Returns
    -------
    None
    """
    for param in model.parameters():
        param.requires_grad = False


def adjust_classification_layer_(model, out_features):
    """Replaces last layer for classification with custom linear layer
    Different types of CNN models have different object signatures and
    need to be handled accordingly

    Parameters
    ----------
    model: Optional[..., torchvision.models.ResNet]

    out_features: int
        Number of outputs of the linear layer

    Returns
    -------
    None
    """
    if model.__class__.__name__ == 'EfficientNet':
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Linear(num_ftrs, out_features)
        return

    if model.__class__.__name__ == 'MobileNetV3':
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(
            num_ftrs, out_features=out_features)
        return

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, out_features)


def train(model, device, train_loader, optimizer, loss_criterion=nn.MSELoss()):
    """Trains model for one epoch. Data needs to be transformed first before
    passing it to Faster RCNN models.

    Parameters
    ----------
    model: Optional[..., torchvision.models.detection.FasterRCNN]

    device: str
        The device (or hardware) on which the model is executed.

    train_loader: torch.utils.data.DataLoader
        The data loader, which contains the data to be trained with.

    optimizer: Optional[, ... torch.optim.Adadelta] 

    loss_criterion Optional[, ... torch.nn.MSELoss] 
        The default is set to nn.MSELoss.
        The loss_criterion is not used when training Faster RCNN models.

    Returns
    -------
    None
    """
    model.train()
    model.to(device)
    for (data, targets) in train_loader:
        optimizer.zero_grad()
        losses = None

        if is_faster_rcnn(model):
            images = list(img.to(device) for img in data)
            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        else:
            data, targets = data.to(device), targets.to(device)
            out = model(data)
            losses = loss_criterion(out, targets)

        losses.backward()
        optimizer.step()


def validate(model, device, test_loader):
    """Validates the model against the data inside the test_loader
    For the CNN models, only the label with the highest probability
    is considered for comparison to the target label. The accuracy
    results from ratio of correct predictions to overall predictions.

    For Faster RCNN models, the Mean-Average-Precision (mAP) is used as 
    the performance indicator.

    See https://torchmetrics.readthedocs.io/en/stable/detection/mean_average_precision.html


    Parameters
    ----------
    model: Optional[..., torchvision.models.detection.FasterRCNN]

    device: str
        The device (or hardware) on which the model is executed.

    test_loader: torch.utils.data.DataLoader
        The data loader, which contains the data to be validated against.

    Returns
    -------
    Optional[dict, int]
    """
    model.eval()
    model.to(device)

    rcnn_pred_dicts = list()
    rcnn_truth_dicts = list()
    rcnn_metric = MeanAveragePrecision()

    cnn_preds_correct = 0

    with torch.no_grad():
        for (data, targets) in test_loader:

            if is_faster_rcnn(model):
                images = list(img.to(device) for img in data)
                targets = [{k: v.to(device) for k, v in t.items()}
                           for t in targets]
                loss_dict = model(images)
                rcnn_pred_dicts.extend([{'boxes': loss_dict[i]['boxes'], 'labels':loss_dict[i]['labels'], 'scores':loss_dict[i]['scores']}
                                        for i in range(len(images))])
                rcnn_truth_dicts.extend([{'boxes': targets[i]['boxes'],
                                          'labels':targets[i]['labels']} for i in range(len(images))])

            else:
                data, targets = data.to(device), targets.to(device)
                out = model(data)
                cnn_preds_correct += (torch.argmax(out, 1)
                                      == targets).float().sum()

    rcnn_metric.update(rcnn_pred_dicts, rcnn_truth_dicts)

    metric = (rcnn_metric.compute()
              if is_faster_rcnn(model) else round(100 * (cnn_preds_correct.item()/len(test_loader.dataset)), 2))

    return metric
