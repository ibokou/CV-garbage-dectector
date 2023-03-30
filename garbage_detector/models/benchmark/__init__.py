from enum import Enum

import numpy as np
import torch
from torch.utils.data import default_collate


class DatasetSize(str, Enum):
    """
    An enum class that represents the different size categories for datasets used in the benchmark
    of the classification and detection models
    """
    SMALL = 'small'
    MEDIUM = 'medium'
    LARGE = 'large'


class DatasetGenerator():
    """
    A class that creates the different train loaders with data subsets of different sizes.
    It also creates the test loader which contains, by default, the largest data subset for 
    evaluation purposes.

    Attributes
    --------
    train_loaders: dict[torch.utils.data.DataLoader]
        The dict containing data loaders at different dataset sizes.

    test:loader: torch.utils.data.DataLoader
        The data loader containing, the largest data subset.
    """

    def __init__(self, dataset, kwargs, benchmark_set_sizes, batch_size=64, test_batch_size=100, collate_fn=default_collate):
        """
        Parameters
        ----------
        dataset : Optional[torchvision.datasets, garbage_detector.models.detection.fasterrcnn.data.GarbageDetectionDataset]
            The dataset containing all images.
        model : Optional[..., torchvision.models.detection.FasterRCNN]
            The model that runs the classification or detection
            of the detected objects
        batch_size : int
            The batch size for train_loaders. Default is 64
        test_batch_size: int
            The batch size for test_loader
        collate_fn: Function
            Function that performs batching.
            See https://pytorch.org/docs/stable/data.html#loading-batched-and-non-batched-data

        """
        data_subsets = {x: torch.utils.data.Subset(
            dataset, np.random.choice(len(dataset), benchmark_set_sizes[x], replace=False)) for x in benchmark_set_sizes.keys()}

        self.train_loaders = {x: torch.utils.data.DataLoader(data_subsets[x], batch_size,
                                                             shuffle=True, **kwargs, collate_fn=collate_fn)
                              for x in benchmark_set_sizes.keys()}

        self.test_loader = torch.utils.data.DataLoader(dataset, test_batch_size,
                                                       shuffle=True, **kwargs, collate_fn=collate_fn)
