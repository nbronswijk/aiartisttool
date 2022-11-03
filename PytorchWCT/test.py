from tqdm import tqdm
from PytorchWCT.util_wct import *
import torch.nn.functional as F
import os
from torchvision import transforms
from PIL import Image
import math

# import sys
# sys.path.append("../..")
from tools import unpadding, preprocess
from thumb_instance_norm import init_thumbnail_instance_norm


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def styleTransfer(wct, content, sFs, alpha, device, wct_mode):
    sF5 = sFs[0]
    if wct_mode == 'gpu':
        cF5 = wct.args['e5'](content).squeeze(0)
        csF5 = wct.transform_v2(cF5, sF5, alpha, index=1)
    else:
        cF5 = wct.args['e5'](content).data.cpu().squeeze(0)
        csF5 = wct.transform_v2(cF5, sF5, alpha, index=1).to(device)
    Im5 = wct.args['d5'](csF5)
    del csF5, cF5; torch.cuda.empty_cache()

    sF4 = sFs[1]
    if wct_mode == 'gpu':
        cF4 = wct.args['e4'](Im5).squeeze(0)
        csF4 = wct.transform_v2(cF4, sF4, alpha, index=2)
    else:
        cF4 = wct.args['e4'](Im5).data.cpu().squeeze(0)
        csF4 = wct.transform_v2(cF4, sF4, alpha, index=2).to(device)
    Im4 = wct.args['d4'](csF4)
    del csF4, cF4; torch.cuda.empty_cache()

    sF3 = sFs[2]
    if wct_mode == 'gpu':
        cF3 = wct.args['e3'](Im4).squeeze(0)
        csF3 = wct.transform_v2(cF3, sF3, alpha, index=3)
    else:
        cF3 = wct.args['e3'](Im4).data.cpu().squeeze(0)
        csF3 = wct.transform_v2(cF3, sF3, alpha, index=3).to(device)
    Im3 = wct.args['d3'](csF3)
    del csF3, cF3; torch.cuda.empty_cache()

    sF2 = sFs[3]
    if wct_mode == 'gpu':
        cF2 = wct.args['e2'](Im3).squeeze(0)
        csF2 = wct.transform_v2(cF2, sF2, alpha, index=4)
    else:
        cF2 = wct.args['e2'](Im3).data.cpu().squeeze(0)
        csF2 = wct.transform_v2(cF2, sF2, alpha, index=4).to(device)
    Im2 = wct.args['d2'](csF2)
    del csF2, cF2; torch.cuda.empty_cache()

    sF1 = sFs[4]
    if wct_mode == 'gpu':
        cF1 = wct.args['e1'](Im2).squeeze(0)
        csF1 = wct.transform_v2(cF1, sF1, alpha, index=5)
    else:
        cF1 = wct.args['e1'](Im2).data.cpu().squeeze(0)
        csF1 = wct.transform_v2(cF1, sF1, alpha, index=5).to(device)
    Im1 = wct.args['d1'](csF1)
    del csF1, cF1; torch.cuda.empty_cache()

    return Im1


def save_image(image, save_path):
    image = image.mul_(255.0).add_(0.5).clamp_(0, 255)
    image = image.squeeze(0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
    image = Image.fromarray(image)
    image.save(save_path)
    
    
def style_transfer_thumbnail(wct, thumb, sFs, alph, device, save_path, save=True, wct_mode="cpu"):
    init_thumbnail_instance_norm(wct, collection=True)
    stylized = styleTransfer(wct, thumb, sFs, alph, device, wct_mode)
    if save:
        save_image(stylized, save_path)
        image = stylized.mul_(255.0).add_(0.5).clamp_(0, 255)
        image = image.squeeze(0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
        return Image.fromarray(image)


def style_transfer_high_resolution(wct, patches, sFs, padding, alph, device, collection, save_path, save=True, wct_mode="cpu"):
    stylized_patches = []
    init_thumbnail_instance_norm(wct, collection=collection)
    for patch in tqdm(patches):
        patch = patch.unsqueeze(0).to(device)
        stylized_patch = styleTransfer(wct, patch, sFs, alph, device, wct_mode)
        stylized_patch = F.interpolate(stylized_patch, patch.shape[2:], mode='bilinear', align_corners=True)
        stylized_patch = unpadding(stylized_patch, padding=padding)
        stylized_patches.append(stylized_patch.cpu())

    stylized_patches = torch.cat(stylized_patches, dim=0)
    
    b, c, h, w = stylized_patches.shape
    stylized_patches = stylized_patches.unsqueeze(dim=0)
    stylized_patches = stylized_patches.view(1, b, c * h * w).permute(0, 2, 1).contiguous()
    output_size = (int(math.sqrt(b) * h), int(math.sqrt(b) * w))
    stylized_image = F.fold(stylized_patches, output_size=output_size,
                            kernel_size=(h, w), stride=(h, w))
    if save:
        save_image(stylized_image, save_path)


def generate_sFs(wct, style, mode="cpu"):
    if mode == 'cpu':
        sF5 = wct.args['e5'](style).data.cpu().squeeze(0)
        sF4 = wct.args['e4'](style).data.cpu().squeeze(0)
        sF3 = wct.args['e3'](style).data.cpu().squeeze(0)
        sF2 = wct.args['e2'](style).data.cpu().squeeze(0)
        sF1 = wct.args['e1'](style).data.cpu().squeeze(0)
    else:
        sF5 = wct.args['e5'](style).squeeze(0)
        sF4 = wct.args['e4'](style).squeeze(0)
        sF3 = wct.args['e3'](style).squeeze(0)
        sF2 = wct.args['e2'](style).squeeze(0)
        sF1 = wct.args['e1'](style).squeeze(0)
    return [sF5, sF4, sF3, sF2, sF1]

