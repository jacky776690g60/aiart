import os, sys, argparse, time, requests, random
from typing import NewType, Literal, Union
from enum import Enum
from pathlib import Path
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import torch
from torch import nn
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from torch.utils.data import Dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================
# Types
# =============================
TorchDataset = NewType("torch.utils.data.Dataset", Dataset)
TransformsCompose = NewType("torchvision.transforms.Compose", transforms.Compose)

Optimizer = Union[torch.optim.Adam, torch.optim.SparseAdam,
                  torch.optim.SGD, torch.optim.Adadelta, 
                  torch.optim.Optimizer, any]
"torch.optim.{any optimizer}"

LossFunction = Union[nn.CrossEntropyLoss, nn.MSELoss,
                     nn.L1Loss, nn.GaussianNLLLoss,
                     any]
"nn.{any loss function}"





# =============================
# Utility Functions
# =============================
def second_to_standard(sec: float) -> str:
    """
    Convert xxxx.xxxx seconds to hh:mm:ss.ssss

    ### return
        - (string) hh:mm:ss.ssss
    """
    mm = sec // 60

    hh = str(int(mm // 60)).zfill(2)
    mm = str(int(mm % 60)).zfill(2)
    sec = round(sec % 60, 4)
    return f"{hh}:{mm}:{sec}"