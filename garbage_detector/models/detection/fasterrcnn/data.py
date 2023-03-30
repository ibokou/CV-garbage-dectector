import json
import os

import torch
from garbage_detector.util.io import get_project_root_dir
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.ops import box_convert


def collate_fn(batch):
    """Function that performs batching.
    See https://pytorch.org/docs/stable/data.html#loading-batched-and-non-batched-data
    Batches are returned as a tuple.

    Parameters
    ----------
    batch: torch.Tensor

    Returns
    -------
    tuple
        containing image and targets (containing bounding boxes) as tensors
    """
    return tuple(zip(*batch))


class GarbageDetectionDataset(Dataset):
    """
        A data data loader that has similiar behavior to torchvision.datasets.ImageFolder
        As described in the documentation of ImageFolder, the images are arranged in this way by default: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        This class inherits from Dataset, so that torch.utils.data.DataLoader can use the same interface
        to access necessary data.

        Attributes
        -------

        transform:
            The composition of all transformations that need to be applied to the image
            before returning it to the data loader.

        classes:
            The different classes of all images. Each subdirectory is interpreted as class.

        images: list 
            The list of all images with their absolute path that the dataset finds in the subdirectories
    """

    def __init__(self, dir, transform=None):
        """
        The subdirectory starting from the dir are traversed and 
        the image pathes are added to self.images.

        It is expected that each subdirectory contains a JSON file
        with the same name. If it is not found, the images of this 
        directory are not added to self.images.

        The same happens if no bounding box regions are defined in the 
        JSON file.

        Parameters
        ----------
        dir : str
            The relative path to the folder containing the images

        transform: torchvision.transforms.Compose
            The composition of transformations. Default is None.

        """
        self.transform = transform

        root = get_project_root_dir()
        path = os.path.join(root, dir)

        self.classes = [d for d in os.listdir(
            path) if os.path.isdir(os.path.join(path, d))]

        self.imgs = []
        for label in range(len(self.classes)):
            d = os.path.join(path, self.classes[label])

            annotations_file_path = os.path.join(
                d, f'{self.classes[label]}.json')

            if not os.path.isfile(annotations_file_path):
                return

            annotations_file = open(annotations_file_path)
            annotations = json.load(annotations_file)
            fnames = [os.path.join(d, f)
                      for f in os.listdir(d) if f.endswith('.jpg') or f.endswith('.png')]

            for f in fnames:
                file_name = os.path.basename(f)
                boxes = []
                labels = []
                regions = annotations[file_name]['regions']

                if len(regions) == 0:
                    continue

                for region in regions:
                    box_shape = region['shape_attributes']
                    boxes.append([box_shape['x'], box_shape['y'],
                                  box_shape['width'], box_shape['height']])
                    labels.append(label+1)
                self.imgs.append((f, labels, boxes))

            annotations_file.close()

    def __len__(self):
        """Returns number of images, the dataset contains.

        Returns
        -------
        int
        """
        return len(self.imgs)

    def __getitem__(self, idx):
        """Returns the image, bounding boxes, and labels of the bounding boxes (class)
        as tensors.
        This function is called by the data loader, to which objects of this class type are passed

        Parameters
        ----------
        idx: int
            ID of image

        Returns
        -------
        tuple
            containing image and target (bounding boxes, labels, and image_id) as tensors
        """

        img_path, labels, boxes = self.imgs[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        boxes = box_convert(torch.as_tensor(
            boxes, dtype=torch.float32), in_fmt='xywh', out_fmt='xyxy')

        assert len(boxes) == len(labels)

        target = {}
        target['boxes'] = boxes
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
        target['image_id'] = torch.tensor([idx])

        return (img, target)

    def get_classes(self):
        """Returns list of all classes

        Returns
        -------
        list[str]
        """
        return self.classes
