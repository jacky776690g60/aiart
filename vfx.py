"""
Contains Torch model to perform vfx
"""
import os, sys, math
from typing import NewType, Literal, Union, List, Tuple
from enum import Enum
from PIL import ImageFile, Image

import numpy as np
from numpy.random import randint

import torch
import kornia.augmentation as Korn
from torch import nn


class KorniaAug(Enum):
    Jt  = lambda : Korn.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.7)
    Jg  = lambda : Korn.ColorJiggle(contrast=.2, saturation=1.1)
    Rsh = lambda : Korn.RandomSharpness(sharpness=0.3, p=0.5)
    Rse = lambda : Korn.RandomPerspective(distortion_scale=0.5, p=0.7)
    Rso = lambda : Korn.RandomRotation(degrees=15, p=0.7)
    Ra  = lambda : Korn.RandomAffine(degrees=15, translate=0.1, shear=5, p=0.7, padding_mode='zeros', keepdim=True)
    Rst = lambda : Korn.RandomElasticTransform(p=0.7)
    Rss = lambda : Korn.RandomThinPlateSpline(scale=0.8, same_on_batch=True, p=0.7)
    Rsr = lambda : Korn.RandomErasing(scale=(.1, .4), ratio=(.3, 1/.3), same_on_batch=True, p=0.7)
    Rsn = lambda : Korn.RandomGaussianNoise(mean=0.0, std=1., p=0.5)
    Rgb = lambda : Korn.RandomGaussianBlur(keepdim=(5, 5), sigma=(), border_type="reflect", p=0.2)
    Rbb = lambda : Korn.RandomBoxBlur(border_type="reflect", p=0.2)
    Rc  = lambda cutSize=1: Korn.RandomCrop(size=(cutSize, cutSize), pad_if_needed=True, padding_mode='reflect', p=0.5)
    Rrc = lambda cutSize=1: Korn.RandomResizedCrop(size=(cutSize, cutSize), scale=(0.1,1),  ratio=(0.75,1.333), cropping_mode='resample', p=0.5)


class AugmentationSTK(nn.Module):
    """
    A pooling model which uses Kornia to apply
    visual effects based on user input
    """
    def __init__(self, kornia_aug, cut_size, cutn, padding: int=0):
        super().__init__()
        self.cut_size = cut_size + padding
        self.cut_cnt = cutn
        # ◼︎ cutout layers for augmenting
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size= self.cut_size)
        self.max_pool = nn.AdaptiveMaxPool2d(output_size= self.cut_size)
        # ◼︎ select augmentations & their order
        modules = [self._visual_aug_selector(s) for s in kornia_aug]
        self.augs = nn.Sequential(*modules)
        self.noise_fac = 0.1

    def forward(self, input):
        cutouts = []

        for _ in range(self.cut_cnt):
            cutouts.append((self.avg_pool(input) + self.max_pool(input)) / 2)

        batch = self.augs(torch.cat(cutouts, dim=0))

        if self.noise_fac:
            facs = batch.new_empty([self.cut_cnt, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch

    
    def _visual_aug_selector(self, s: str) -> KorniaAug:
        if s == "Jt":    return KorniaAug.Jt()
        elif s == "Jg":  return KorniaAug.Jg()
        elif s == "Rsh": return KorniaAug.Rsh()    
        elif s == "Rse": return KorniaAug.Rse()    
        elif s == "Rso": return KorniaAug.Rso()    
        elif s == "Ra":  return KorniaAug.Ra()    
        elif s == "Rst": return KorniaAug.Rst()    
        elif s == "Rss": return KorniaAug.Rss()    
        elif s == "Rsr": return KorniaAug.Rsr()    
        elif s == "Rsn": return KorniaAug.Rsn()    
        elif s == "Rgb": return KorniaAug.Rgb()    
        elif s == "Rbb": return KorniaAug.Rbb()    
        elif s == "Rc":  return KorniaAug.Rc(self.cut_size)
        elif s == "Rrc": return KorniaAug.Rrc(self.cut_size)
        else:
            raise ValueError("[ERROR] unknown augmentation (check input)")


class Shader():
    """
    This class provides various shaders.
    """
    @staticmethod
    def create_noise_img(w: int, h: int) -> Image.Image:
        """
        Generate an image of random noises
            with width and height with 3 color channels
        """
        img = Image.fromarray(
                            randint(0, 255, size=(w,h,3), dtype=np.dtype('uint8'))
                            )
        return img

    @staticmethod
    def ramp2d(w: int, h: int, start: int, end: int, 
                    horizontal: bool) -> np.ndarray:
        """
        directional

        ### args
            - `step` control the smoothness of the gradient
        """
        def helper(step, rows) -> np.ndarray:
            return  np.tile(
                            np.linspace(start, end, step),
                            (rows, 1)
                        )
        return helper(step = w, rows = h) if horizontal else\
            helper(step = w, rows = h).T

    @staticmethod
    def ramp3d(w: int, h: int, starts: List[int], ends: List[int], 
                    hors: List[bool]) -> np.ndarray:
        """
        directional

        ### return
            an numpy array which has w * h * 3
                (not image ready because of floating points)
        """
        res = np.zeros(shape=(h, w, len(starts)), dtype=float)

        for i, (start, stop, is_horizontal) in enumerate(zip(starts, ends, hors)):
            edge_idx = randint(0, 2)
            if i == edge_idx:
                res[:, :, i] = Shader.ramp2d(w, h, stop, start, is_horizontal)
            else:
                res[:, :, i] = Shader.ramp2d(w, h, start, stop, is_horizontal)
        return res


    class RampStyle(Enum):
        """
        Style for creating a gradient image
        """
        diverse  = lambda : ([randint(0, 255), randint(0,255)], [randint(0, 255), randint(0,255)], [randint(0, 255), randint(0,255)])
        redish   = lambda : ([randint(160, 255), randint(160,255)], [randint(0, 127), randint(0,127)], [randint(0, 127), randint(0,127)])
        greenish = lambda : ([randint(0, 127), randint(0,127)], [randint(160, 255), randint(160,255)], [randint(0, 127), randint(0,127)])
        bluish   = lambda : ([randint(0, 127), randint(0,127)], [randint(0, 127), randint(0,127)], [randint(160, 255), randint(160,255)])

    @staticmethod
    def getRampStyle(s: str) -> RampStyle:
        """
        Get a ramp style based on string
        """
        if s == "diverse":    return Shader.RampStyle.diverse 
        elif s == "redish":   return Shader.RampStyle.redish
        elif s == "greenish": return Shader.RampStyle.greenish
        elif s == "bluish":   return Shader.RampStyle.bluish
        else:
            raise ValueError("Incorrect string for selecting ramp style.")

    @staticmethod
    def create_gradient_img(w: int, h: int, style: RampStyle=RampStyle.diverse) -> Image.Image:
        """
        generate a random gradient image with
            width and height and 3 color channels
        
        ### args
            - `w`: image width
            - `h`: image height
            - `style`: pick a style from `RampStyle`
        """
        def rangeSwap(ls: List) -> List:
            n1, n2 = ls[0], ls[1]
            ls[0] = min(n1, n2)
            ls[1] = max(n1, n2)
            return ls

        ranges = np.array([rangeSwap(ls) for ls in style()])
        rL, rR, gL, gR, bL, bR = ranges.ravel()
        rR += 1
        gR += 1
        bR += 1

        arr = Shader.ramp3d(w, h, 
                            starts=(rL, gL, bL), 
                            ends=(randint(rL,rR), randint(gL,gR), randint(bL,bR)), 
                            hors=(True, False, False)
                           )
        img = Image.fromarray(np.uint8(arr))
        return img