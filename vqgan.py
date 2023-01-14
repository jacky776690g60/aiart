"""
VQGAN+CLIP AI ART GENERATOR

(credits to these developers)
A combinations of:
https://github.com/nerdyrodent/VQGAN-CLIP
https://github.com/openai/CLIP
"""

import sys, os, argparse, warnings, random
from typing import List, Optional, Tuple, Dict
from enum import Enum
from PIL import ImageFile, Image, PngImagePlugin
import requests
from pathlib import Path
sys.path.append(os.path.join(os.getcwd(), 'taming-transformers'))
# suppress deprecated warnings
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from CLIP import clip

from omegaconf import OmegaConf
from taming.models import cond_transformer, vqgan # taming transformers

import torch
from torch import nn
from torch.nn import functional as TorchNNFunc
from torchvision import transforms
from torchvision.transforms import functional as TorchTfFunc
from torch_optimizer import DiffGrad, AdamP, RAdam
from torch.optim import Adam, AdamW, Adagrad, Adamax, RMSprop

from utils.progressbar import ProgressBar, TermArtist
from vfx import AugmentationSTK, Shader


import torchtools
from torchtools import DaVinci, Picasso, Assistant


import ssl
ssl._create_default_https_context = ssl._create_unverified_context
  

if not torch.cuda.is_available():
    DEFAULT_IMG_RES = 128
elif torch.cuda.get_device_properties(0).total_memory <= 2 ** 32: # vram for first device
    DEFAULT_IMG_RES = 256
else:
    DEFAULT_IMG_RES = 512
# =============================
# Add arguments
# =============================
parser = argparse.ArgumentParser(description='Image generation using VQGAN+CLIP')

parser.add_argument("-tp",  "--text_prompts", type=str, nargs='+',
                    help="Text prompts")
parser.add_argument("-ip",  "--image_prompts", type=str, nargs="+", default=[],
                    help="Paths to images to be used as prompts")
parser.add_argument("-i",   "--iterations", type=int, default=500, 
                    help="Total number of iterations")
parser.add_argument("-se",  "--save_every", type=int, default=50,
                    help="Save image after every n iterations (use this to see how image evolves)")
parser.add_argument("-s",   "--save", type=str, default="output.png",
                    help="provide a file name for saving (it will be put in output/)")
parser.add_argument("-sig", "--save_itergrp", action='store_true',
                    help="Save each iteration group as different image")
parser.add_argument("-rs",  "--resolution", type=int, nargs=2, default=[DEFAULT_IMG_RES, DEFAULT_IMG_RES],
                    help="Output resolution (default: %(default)s)")
parser.add_argument("-iw", "--init_weight", type=float, default=0.,
                     help="Initial weight")
parser.add_argument("-m", "--clip_model", type=str, default='ViT-B/32',                          
                    help="CLIP model (e.g. ViT-B/32, ViT-B/16)")
parser.add_argument("-conf", "--vqgan_config", type=str, default=f'checkpoints/vqgan_imagenet_f16_16384.yaml',
                    help="VQGAN config")
parser.add_argument("-ckpt", "--vqgan_checkpoint", type=str, default=f'checkpoints/vqgan_imagenet_f16_16384.ckpt', 
                    help="VQGAN checkpoint")
parser.add_argument("-ns", "--noise_prompt_seeds", nargs="+", type=int, default=[],
                    help="Noise prompt seeds")
parser.add_argument("-nsw", "--noise_prompt_weights", nargs="+", type=float, default=[],
                    help="Noise prompt weights")
parser.add_argument("-cuts", "--num_cuts", type=int, default=32, 
                    help="Number of cuts")
parser.add_argument("-cutp", "--cut_padding", type=int, default=0, 
                    help="Cut padding")
parser.add_argument("-sd", "--torch_seed", type=int, default=None,
                    help="Define a torch seed for similar result")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.1, 
                    help="Specify a learning rate (0 ~ 1)")
parser.add_argument("-opt", "--optimizer", type=str, choices=['Adam','AdamW','Adagrad','Adamax','DiffGrad','AdamP','RAdam','RMSprop'], default='Adam', 
                    help="Choose an optimizer")
parser.add_argument("-cpe", "--iter_group", type=int, default=0,
                    help="Change prompt after how many iterations")
parser.add_argument("-d", "--cudnn_determinism", action='store_true',
                    help="(CUDA) Allow result to be reproducible")
parser.add_argument("-aug", "--augments", type=str, nargs='+', choices=["Jt","Jg","Rsh","Rse","Rso","Ra","Rst","Rss","Rsr","Rsn","Rgb","Rbb","Rc","Rrc"], 
                    default=['Rsh', 'Jg', "Ra"],
                    help="Use kornia augmentations (pick from choices)")
parser.add_argument("-cpu", "--cpu", action='store_true',
                    help="Use CPU")


init_grp = parser.add_mutually_exclusive_group()
init_grp.add_argument("-ii", "--init_image", type=str, default="",
                      help="Set the initial image (provide a file path or URL directly to image)")
init_grp.add_argument("-in",  "--init_noise", action="store_true", default=False, 
                      help="Use noises for inital image")
init_grp.add_argument("-acc", "--accent", type=str, choices=['diverse','redish','bluish','greenish'], default=None, 
                       help="Put emphasize on a range of colors on initial prompt")



args = parser.parse_args()

# =============================
# Setting up
# =============================
torch.backends.cudnn.benchmark = False
ImageFile.LOAD_TRUNCATED_IMAGES = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if args.text_prompts == args.image_prompts == None:
    raise ValueError("Please provide prompt for generating image")

if args.cudnn_determinism:
   torch.backends.cudnn.deterministic = True

# ◼︎ Convert input to phrases
if args.text_prompts:
    # We need to extract phrases from all sentences
    # phrases is not a complete sentence. 
    # e.g., In this afternonn, on the table, ...
    sentences = [[p.strip() for p in s.strip().split("|")] for s in args.text_prompts]
    cur_sentence = sentences[0]

# ◼︎ process save path
args.save = os.path.join("output", args.save)
save_dir = os.path.dirname(args.save)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)





class CustomGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)

custom_grad = CustomGrad.apply


class ClampWithGrad(torch.autograd.Function):
    """
    clamping vqgan output between min, max
    """
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None

clamp_with_grad = ClampWithGrad.apply



def compose(z) -> torch.Tensor:
    """
    Compose the tensor

    ### return
        a tensor with shape [1, 3, 256, 256]
    """
    def vector_quantize(x, codebook):
        d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
        indices = d.argmin(-1)
        x_q = TorchNNFunc.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
        return custom_grad(x_q, x)

    z_q = vector_quantize(z.movedim(1, 3),
                          VQGAN_MDL.quantize.embed.weight if gumbel else\
                          VQGAN_MDL.quantize.embedding.weight
                        ).movedim(3, 1)
    
    out_vq = VQGAN_MDL.decode(z_q).add(1).div(2)
    return clamp_with_grad(out_vq, 0, 1)






# =============================
# Models
# =============================
def load_vqgan_model(config_path, checkpoint_path):
    global gumbel
    gumbel = False
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.vqgan.GumbelVQ':
        model = vqgan.GumbelVQ(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
        gumbel = True
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model


class PromptMDL(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        """
        Registering buffers that should not be conisdered as
        model paramters, but they are part of the persistent
        state of the model.
        """
        super().__init__()
        # ◼︎ parameters
        self.register_buffer('embed',   embed)
        self.register_buffer('weight',  torch.as_tensor(weight))
        self.register_buffer('stop',    torch.as_tensor(stop))

    def forward(self, input):
        input_normed = TorchNNFunc.normalize(input.unsqueeze(1), dim=2)
        embed_normed = TorchNNFunc.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * custom_grad(dists, torch.maximum(dists, self.stop)).mean()


VQGAN_MDL = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(DEVICE)
PERCEPTOR = clip.load(args.clip_model, jit=False)[0]\
                .eval()\
                .requires_grad_(False)\
                .to(DEVICE)


clip_sz = PERCEPTOR.visual.input_resolution
fffffff = 2**(VQGAN_MDL.decoder.num_resolutions - 1)

CUTER_MDL = AugmentationSTK(args.augments, clip_sz, args.num_cuts, padding=args.cut_padding)





def get_weight(prompt):
    """
    Split the string based on semi-colon
    to get the weight for this prompt.

    ## return
        `txt` the prompt\n
        `weight` weight for this prompt\n
        `stop`
    """
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):] # replace default based on length
    txt, weight, stop = vals[0], float(vals[1]), float(vals[2])
    return txt, weight, stop


 


# ◼︎ Gumble distribution
if gumbel:
    e_dim = 256
    n_toks = VQGAN_MDL.quantize.n_embed
    z_min = VQGAN_MDL.quantize.embed.weight.min(dim=0).values[None, :, None, None]
    z_max = VQGAN_MDL.quantize.embed.weight.max(dim=0).values[None, :, None, None]
else:
    e_dim = VQGAN_MDL.quantize.e_dim
    n_toks = VQGAN_MDL.quantize.n_e
    z_min = VQGAN_MDL.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = VQGAN_MDL.quantize.embedding.weight.max(dim=0).values[None, :, None, None]


# =============================
# First frame
# =============================
# ◼︎ currently not supporting overlapping

toksX, toksY = args.resolution[0] // fffffff, args.resolution[1] // fffffff
sideX, sideY = toksX * fffffff, toksY * fffffff


if args.init_image or args.init_noise or args.accent:
    if "http" in args.init_image:
        req = requests.get(args.init_image, stream=True)
        init_img = Image.open(req.raw)
    elif args.init_image:
        init_img = Image.open(args.init_image)
    elif args.init_noise:
        init_img = Shader.create_noise_img(args.resolution[0], args.resolution[1])
    elif args.accent:
        init_img = Shader.create_gradient_img(args.resolution[0], args.resolution[1], style=Shader.getRampStyle(args.accent))

    pil_tensor = Assistant.imageToTensor(init_img, (sideX, sideY))
    z, *_ = VQGAN_MDL.encode(pil_tensor.to(DEVICE).unsqueeze(0) * 2 - 1)
        
else:
    one_hot = TorchNNFunc.one_hot(torch.randint(n_toks, [toksY * toksX], device=DEVICE), n_toks).float()

    if gumbel:
        z = one_hot @ VQGAN_MDL.quantize.embed.weight
    else:
        z = one_hot @ VQGAN_MDL.quantize.embedding.weight

    z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)


def get_optimizer(name: str, learning_rate: float=args.learning_rate):
    """
    Get the optimzer based on user inputs

    ## param
        - `name` the name of the optimizer
        - `learning_rate` learning rate for this optimizer (0 ~ 1)
    """
    if name == "Adam":       opt = Adam([z], lr=learning_rate)
    elif name == "AdamW":    opt = AdamW([z], lr=learning_rate)
    elif name == "Adagrad":  opt = Adagrad([z], lr=learning_rate)
    elif name == "Adamax":   opt = Adamax([z], lr=learning_rate)
    elif name == "DiffGrad": opt = DiffGrad([z], lr=learning_rate, eps=1e-9, weight_decay=1e-9)
    elif name == "AdamP":    opt = AdamP([z], lr=learning_rate)
    elif name == "RAdam":    opt = RAdam([z], lr=learning_rate)
    elif name == "RMSprop":  opt = RMSprop([z], lr=learning_rate)
    else:
        raise ValueError("Incorrect value for optimizer. Please check the inputs.")
    return opt

optimizer = get_optimizer(args.optimizer, args.learning_rate)


seed = torch.seed() if not args.torch_seed else args.torch_seed
torch.manual_seed(seed)

print(TermArtist.ORANGE)
print('Device:', DEVICE)
print('Optimizer:', args.optimizer)
print('Seed:', seed)
print(TermArtist.WHITE)

print(f"{TermArtist.UNDERLINE}Prompt Collector{TermArtist.WHITE}")
if args.text_prompts:
    print("[INFO] Detected text prompts")
if args.image_prompts:
    print("[INFO] Detected image prompts")
if args.init_image:
    print("[INFO] Detected initial image")
if args.noise_prompt_weights:
    print("[INFO] Detected noise prompt weights")



z_orig = z.clone()
z.requires_grad_(True)


normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]
                                )


prompt_models = []

# ◼︎ feeder
def feedText(phrase: str):
    txt, weight, stop = get_weight(phrase)
    embed = PERCEPTOR.encode_text(clip.tokenize(txt).to(DEVICE)).float()
    prompt_models.append(PromptMDL(embed, weight, stop).to(DEVICE))
    return txt, weight, stop


def feedImg(img_path: str):
    path, weight, stop = get_weight(img_path)
    PIL_img = Image.open(path).convert('RGB')

    resized_img = PIL_img.resize((sideX, sideY), Image.Resampling.LANCZOS)

    batch = CUTER_MDL(TorchTfFunc.to_tensor(resized_img).unsqueeze(0).to(DEVICE))
    embed = PERCEPTOR.encode_image(normalize(batch)).float()
    prompt_models.append(PromptMDL(embed, weight, stop).to(DEVICE))

    return path, weight, stop


def feedNoises():
    weightTmp = [1.] * len(args.noise_prompt_seeds)
    weightTmp[:len(args.noise_prompt_weights)] = args.noise_prompt_weights[:]
    
    for seed, weight in zip(args.noise_prompt_seeds, weightTmp):
        gen = torch.Generator().manual_seed(seed)
        embed = torch.empty([1, PERCEPTOR.visual.output_dim]).normal_(generator=gen)
        prompt_models.append(PromptMDL(embed, weight).to(DEVICE))
    return args.noise_prompt_seeds, weightTmp




if args.text_prompts == args.image_prompts == None:
    raise RuntimeError("No prompt provided. At least provide text or image prompt.")





def train_step(i):
    losses = []

    img_tensor = compose(z)

    clip_logits = PERCEPTOR.encode_image(normalize(CUTER_MDL(img_tensor))).float()

    if args.init_weight:
        losses.append(TorchNNFunc.mse_loss(z, torch.zeros_like(z_orig)) * 
                    ((1/torch.tensor(i*2 + 1)) * args.init_weight) / 2)

    for mdl in prompt_models:
        # print(mdl)
        losses.append(mdl(clip_logits))

    losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
    loss = sum(losses)

    print(f'\ti: {i}, total loss: {loss.item():g}, each: {losses_str}', end="")
    
    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    optimizer.step()

    if i % args.save_every == 0:
        output = compose(z)
        metadata = PngImagePlugin.PngInfo()
        info = ""
        if args.text_prompts:
            info += f"{cur_sentence} "
        if args.image_prompts:
            info += f"{cur_imgPath} "
        metadata.add_text('Sentence', f'{info}')

        if args.save_itergrp:
            save_path = os.path.splitext(Path(args.save))
            save_path = save_path[0] + f"_{i}" + save_path[1]
        else: 
            save_path = Path(args.save)

        TorchTfFunc.to_pil_image(output[0].cpu()).save(save_path, pnginfo=metadata)

    with torch.inference_mode():
        z.copy_(z.maximum(z_min).minimum(z_max))



phr_prm_id = img_prm_idx = noi_prmt_idx = 0 
progress_bar = ProgressBar(args.iterations, bar_len=50, style=2)
try:
    for i in range(args.iterations):
        # ◼︎ Change Iteration Group
        if args.iter_group and i % args.iter_group == 0:
            # ◼︎ Text prompts
            prompt_models = []
            if args.text_prompts:
                if phr_prm_id >= len(sentences): phr_prm_id = 0
                
                cur_sentence = sentences[phr_prm_id]
                print("\n", "-" * 50)
                print("Using Sentence: ", cur_sentence)

                minLen = max(max(len(el[0]) for el in cur_sentence), 30)

                
                for phr in cur_sentence:
                    txt, weight, stop = feedText(phr)
                    
                    print(f"Text: {txt}{' ' * (minLen - len(txt)-1)}||Weight: {weight}{' ' * (minLen - len(str(weight)) - 1)}||Stop: {stop}")

                phr_prm_id += 1

            # ◼︎ Image Prompts
            if args.image_prompts:
                if img_prm_idx >= len(args.image_prompts): img_prm_idx = 0
                cur_imgPath = args.image_prompts[img_prm_idx]
                path, weight, stop = feedImg(cur_imgPath)
                print(f"\nUse Image: {cur_imgPath} ||Weight: {weight} ||Stop: {stop}")
                img_prm_idx += 1

            # ◼︎ noise prompts
            if args.noise_prompt_seeds:
                seeds, weights = feedNoises()
                print(f"\nnoise seed: {seeds} ||Weight: {weights}")


        # ◼︎ Starts Training
        progress_bar.draw(i)
        train_step(i)
    print(f"{TermArtist.GREEN}\n[INFO] Image Generated")
except KeyboardInterrupt:
    print(f"{TermArtist.FAIL}\n[WARN] Process Stopped by user")
    sys.exit()