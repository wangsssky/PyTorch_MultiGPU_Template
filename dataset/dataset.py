from __future__ import print_function

import numpy as np
import random
import torch
from torchvision import datasets
from PIL import Image


class ImageFolderInstance(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 two_crop=False, use_label=False):
        super(ImageFolderInstance, self).__init__(root, transform, target_transform)
        self.two_crop = two_crop
        self.num = self.__len__()

    def __getitem__(self, index):
        """
        Args:
            index (int): index
        Returns:
            tuple: (image, index, ...)
        """
        path, target = self.imgs[index]
        image = self.loader(path)

        # # image
        if self.transform is not None:
            img = self.transform(image)
            if self.two_crop:
                img2 = self.transform(image)
                img = torch.cat([img, img2], dim=0)
        else:
            img = image

        return img, target


