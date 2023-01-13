"""
contains utility functions related 
    to plotting/processing images
"""
import os, sys, random
from typing import List, Dict, Tuple, Union
from enum import Enum, unique
from pathlib import Path
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


import torch
from torch import nn
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TorchTFFunc

sys.path.append(os.path.dirname(__file__))
from .utils import *




class DaVinci():
    """
    This class contains methods that
        are more for plotting results. 
        (More realistic plotting functions)
    """
    @staticmethod
    def plot_single_img(img: torch.Tensor, title: str="", figsize: tuple=(4, 3)):
        """
        plot one single picture

        - `img`: provide a tensor of image, and it will 
                automatically get permuted
        """
        fig = plt.figure(figsize= figsize)
        plt.imshow(img.permute(1, 2, 0))
        plt.title(f"{title}\nshape: {img.shape}")
        plt.axis(False)
        return fig
        

    @staticmethod
    def plot_transformed_imgs(image_paths: list[str], transform: TransformsCompose, 
                              n: int=3, seed: int=None, figsize=(3, 3)):
        """
        Selects random images from paths and loads & transforms
            them, then plot original against transformed version.

        param
        ---------
            - `image_paths`: a list of image paths
            - `transform`: a torch transform function (Compose or Sequential)
            - `n`: counter (how many images)
            - `seed`: seed for random indexes
            - `figsize`: size for the figure
        """
        if seed: random.seed(seed)
        random_paths = random.sample(image_paths, k=n)
        for p in random_paths:
            with Image.open(p) as f:
                # plt.figure(figsize = (10,10))
                fig, ax = plt.subplots(nrows=1, ncols=2, figsize = figsize)
                # ◼︎ original
                ax[0].imshow(f)
                ax[0].set_title(f"Original\nSize: {f.size}", fontdict={"size": 10})
                ax[0].axis(False)

                # ◼︎ transformed
                transformed_image = transform(f).permute(1, 2, 0)  # change shape for matplotlib (C, H, W) -> (H, W, C)
                ax[1].imshow(transformed_image)
                ax[1].set_title(f"Transformed\nShape: {transformed_image.shape}", fontdict={"size": 10})
                ax[1].axis("off")

                fig.suptitle(f"Class: {p.parent.stem}", fontsize=12)
                

    @staticmethod
    def plot_random_imgs(dataset: TorchDataset, classes: List[str] = None,
                            n: int = 10, seed: int = None, figsize = (16, 8)):
        """
        A function for displaying random images 
            from a torch dataset using matplotlib
        """
        fontsize = 11
        if n > 7:
            print(f"For display purposes, max n is 7. Setting n = 7 ...")
            n = 7
            fontsize = 7

        if seed: random.seed(seed)

        print(len(dataset))
        rand_idxs = random.sample(range(len(dataset)), k=n)

        plt.figure(figsize= figsize)

        for i, img_idx in enumerate(rand_idxs):
            img, label = dataset[img_idx][0], dataset[img_idx][1]

            tmpImg = img.permute(1, 2, 0) # [H, W, C]

            plt.subplot(1, n, i+1)
            plt.imshow(tmpImg)
            plt.axis("off")
            if classes:
                title = f"Class: {classes[label]}"
                title = title + f"\nshape: {tmpImg.shape}"
            plt.title(title, fontdict={"size": fontsize})


    @staticmethod
    def plot_pred_img(model: torch.nn.Module, img_path: str, class_names: List[str] = None, 
                      transform: TransformsCompose=None, device=DEVICE):
        """
        Make a prediction on a target image with 
            a trained model and plot the image and prediction.
        """
        # Load image
        target_image = torchvision.io.read_image(str(img_path)).type(torch.float32)
        # Divide pixel values by 255 to get them between [0, 1]
        target_image = target_image / 255.0

        # Transform if necessary
        if transform: target_image = transform(target_image)

        # Make sure the model is on the target device
        model.to(device)

        model.eval()
        with torch.inference_mode():
            # Add an extra dimension to the image (this is the batch dimension, e.g. our model will predict on batches of 1x image)
            target_image = target_image.unsqueeze(0)
            # Make a prediction
            z = model(target_image.to(device)) # logits

        # logits -> probabilities
        probs = torch.softmax(z, dim=1)
        # probabilities -> labels
        y_hat = torch.argmax(probs, dim=1)

        plt.imshow(target_image.squeeze().permute(1, 2, 0))  # remove batch dimension and rearrange shape to be HWC
        if class_names:
            title = f"Pred: {class_names[y_hat.cpu()]} | Prob: {probs.max().cpu():.3f}"
        else:
            title = f"Pred: {y_hat} | Prob: {probs.max().cpu():.3f}"
        plt.title(title)
        plt.axis(False)


    @staticmethod
    def plot_patchified_img(img_permuted: torch.Tensor, patch_h: int, patch_w: int,
                            title: str="",  figsize=(8,6)):
        """
        plot a patchified version of the image.
            patch size must be able to divide the
            entire image; otherwise, raise error
        """
        imgH, imgW = len(img_permuted[0]), len(img_permuted[1])
        assert imgH % patch_h == 0, "patch height cannot divide image height"
        assert imgW % patch_w == 0, "patch width cannot divide image width"
        
        p_perCol, p_perRow = int(imgH // patch_h), int(imgW // patch_w)
        p_cnt = int(p_perCol * p_perRow)

        print(f"patches per row: {imgW // patch_w} | patches per column: {imgH // patch_h}")
        print(f"Total patches: {p_cnt}")
        print(f"Patch (h*w): {patch_h} x {patch_w}")

        # ◼︎ Create a series of subplots
        fig, axs = plt.subplots(
                                nrows=int(imgH // patch_h),
                                ncols=int(imgW // patch_w), 
                                figsize=figsize,
                                sharex=True,
                                sharey=True
                                )

        # ◼︎ Loop through height and width
        for i, startH in enumerate(range(0, imgH, patch_h)): # thru height
            for j, startW in enumerate(range(0, imgW, patch_w)): # thru width
                # ◼︎ (Height, Width, Color Channels)
                axs[i, j].imshow(img_permuted[startH:startH + patch_h, startW:startW + patch_w, :])
                # ◼︎ Set up label
                axs[i, j].set_ylabel(i+1, 
                                        rotation="horizontal", 
                                        horizontalalignment="right", 
                                        verticalalignment="center") 
                axs[i, j].set_xlabel(j+1) 
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
                axs[i, j].label_outer()
        fig.suptitle(f"Patchified\n{title}\npatch_size: {patch_h} * {patch_w}", fontsize=10)
        return fig






class Picasso():
    """
    This classs contains plotting methods that
        are more functional, like plotting curves.
        (Images that are more abstract or charts)
    """
    @staticmethod
    def plot_loss_curves(results: Dict[str, List[float]]):
        """
        Plots training curves of a results dictionary.

        ### args
            - `results`: a dictionary that must contain keys\n
                - train_loss
                - train_loss
                - train_acc
                - test_acc
        """
        loss            = results["train_loss"]
        accuracy        = results["train_acc"]
        test_loss       = results["test_loss"]
        test_accuracy   = results["test_acc"]

        if not (len(loss) == len(accuracy) == len(test_loss) == len(test_accuracy)) or\
            any(len(x) ==0 for x in [loss, accuracy, test_loss, test_accuracy]):
            raise ValueError("Dictionary has incorrect formats or no record.")

        # total epochs
        epochs = range(len(results["train_loss"]))

        # Setup a plot
        plt.figure(figsize=(10, 5))

        # Plot the loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, label="train_loss")
        plt.plot(epochs, test_loss, label="test_loss")
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.legend()

        # Plot the accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracy, label="train_accuracy")
        plt.plot(epochs, test_accuracy, label="test_accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.legend()

    @staticmethod
    def plot_random_feature_maps(tensor_from_nnLayer: torch.Tensor, length: int, n: int=1,
                                 cmap="Blues" , verbal=False):
        """
        plot random feature maps from the output of an NN layer
            provide a tensor from that output
        
        ### args
            - `tensor_from_nnLayer`: a tensor from the output of a NN layer
            - `length`: the embedding size; e.g.:\n
                        [32, 768, 14, 14]    [N * C * H * W]\n
                        `provide 768`
        """
        randn_idxes = random.sample(range(0, length), k= n) # pick n numbers between 0 and the embedding size
        if verbal: 
            print(f"Showing random convolutional feature maps from indexes: {randn_idxes}")

        # Create plot
        fig, axs = plt.subplots(nrows=1, ncols= n, figsize=(12, 12))
        
        # Plot random image feature maps
        for i, idx in enumerate(randn_idxes):
            image_conv_feature_map = tensor_from_nnLayer[:, idx, :, :]
            axs[i].imshow(image_conv_feature_map.squeeze().detach().numpy(), cmap= cmap)
            axs[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        return fig


@unique
class PILResampling(Enum):
    def __repr__(self):
        return self.value
    """
    A selection of resampling methods.
        Only methods that generate 3 color channels
        are selected for tensor processing

    This is not the state of the art for scaling image
        anymore. For best resampling, upscaler performs
        a better job (but slower).
    """
    Bilinear = (Image.Resampling.BILINEAR)
    "computationally cheapest, but not best representive. creates artifacts"
    # ◼︎ Differnece in Lanczos and Bicubic are sometimes negligible
    Bicubic  = (Image.Resampling.BICUBIC)
    "quality second to lanczos, but faster"
    Lanczos  = (Image.Resampling.LANCZOS)
    "expensive but best quality; involves a sinc filter"
    Nearest  = (Image.Resampling.NEAREST)
    "aka proximal interpolation; can be good for pixel art"

    def getPILResample(name: str="lanczos"):
        """
        Get a PIL resampling method based on its name 
            from a collection, input will be 
            auto-converted to lowercase
        """
        name = name.lower()
        if name ==   "bilinear": return PILResampling.Bilinear
        elif name == "bicubic":  return PILResampling.Bicubic
        elif name == "lanczos":  return PILResampling.Lanczos
        elif name == "nearest":  return PILResampling.Nearest
        else:
            raise ValueError("Incorrect resampling method of tensor, check your input")

PILResamplingType = Union[PILResampling, str]

class Assistant():
    """
    This class contains methods for preparing images, etc.,
        for tensor processing in PyTorch
    """
    @staticmethod
    def imageToTensor(PIL_img: Image.Image, size: Tuple[int, int]=None,
                      resampling: PILResamplingType=PILResampling.Lanczos) -> torch.Tensor:
        """
        Convert input image to `RGB` color space (get
            3 channels), and resize accordingly, then
            transform the image into a torch tensor
        
        ### args
            - PIL_img: an image opened by PIL.Image
            - size: new size for the image
        
        ### return
            - torch.Tensor: [3, H, W]
        """
        if isinstance(resampling, str): 
            resampling = PILResampling.getPILResample(resampling)

        w, h = PIL_img.size if not size else size
        img = PIL_img.convert('RGB').resize((w, h), resampling.value)
        return TorchTFFunc.to_tensor(img)

    