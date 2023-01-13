"""
copycat.py contains utility functions
    that mimicks certain torch functions
"""
import os, sys, argparse, time, requests, random
from typing import Tuple, List, Dict
from enum import Enum
from pathlib import Path
from PIL import Image


import torch
from torch import nn
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset



sys.path.append(os.path.dirname(__file__))
from .utils import *



class ImageFolderClone(Dataset):
    """
    Mimick version of torch.utils.data.Dataset.ImageFolder
    """
    def __init__(self, imgDir: str, imgExtension: str=".jpg", transform :TransformsCompose = None):
        self.paths = list(Path(imgDir).glob(f"*/*{imgExtension}"))            # Get all img files
        self.transform = transform                                                  # Setup transform
        self.classes, self.class_to_idx = ImageFolderClone.find_classes(imgDir)    # Create classes and class_to_idx attributes

    def load_image(self, index: int) -> Image.Image:
        """
        Opens an image via a path and returns it.
        """
        image_path = self.paths[index]
        return Image.open(image_path)

    # 5. Overwrite __len__()
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)

    # 6. Overwrite __getitem__() method to return a particular sample
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name = self.paths[index].parent.name # expects path in format: data_folder/class_name/image.jpg
        class_idx = self.class_to_idx[class_name]

        # return data, label (X, y)   or    untransformed image and label
        return self.transform(img), class_idx if self.transform else\
                img, class_idx


    @staticmethod
    def find_classes(dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Get class names based on directory name
            Dog/  <- `this is the class name`
                |__ maya.png

        torch version: `datasets.ImageFolder().classes || .class_to_idx`

        ### Args
            - `dir`: path to the directory containing the images
        
        ### Return
            - A tuple (classes_name, a dict of class_name and index)\n
                in sorted order
        """
        classes = sorted(entry.name for entry in os.scandir(dir) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class in {dir}... check dir structure!")

        class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
        return classes, class_to_idx
