import os

import garbage_detector.models.classification.cnn as cnn
import garbage_detector.models.detection.fasterrcnn as frrcnn
import gdown
import torch
from garbage_detector.models import is_faster_rcnn
from garbage_detector.util import io


def save_states(dir, model, optimizer):
    """Saves the state_dict of model and optionally an optimizer
    for later use in transfer learning.


    Parameters
    ----------
    model: Optional[..., torchvision.models.detection.FasterRCNN]

    optimizer: Optional[, ... torch.optim.Adadelta]

    Returns
    -------
    None
    """
    category = 'detection' if is_faster_rcnn(model) else 'classification'
    root = os.path.join(io.get_project_root_dir(),
                        'model_states', category)
    path = os.path.join(root, dir)
    if not os.path.exists(path):
        os.makedirs(path)

    checkpoint_file_path = os.path.join(path, 'checkpoint.pth')
    torch.save({'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict()}, checkpoint_file_path)


def load_states_(dir, model, optimizer=None, force_download=False):
    """Loads the checkpoint, containing state_dict of model and optimizer.
    When the checkpoint file is not found locally, the checkpoint is downloaded from
    Google Drive.


    Parameters
    ----------
    model: Optional[..., torchvision.models.detection.FasterRCNN]

    force_download: bool
        If true, then checkpoint is always downloaded from Google Drive.
        Default is set to false

    Returns
    -------
    None
    """
    category = 'detection' if is_faster_rcnn(model) else 'classification'
    root = os.path.join(io.get_project_root_dir(),
                        'model_states', category)
    path = os.path.join(root, dir, 'checkpoint.pth')

    if force_download or not os.path.exists(path):

        if category == 'detection':
            gdown.download(
                frrcnn.model_state_urls[dir], path)
        else:
            gdown.download(
                cnn.model_state_urls[dir], path)

    map_location = "cuda:0" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location)

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
