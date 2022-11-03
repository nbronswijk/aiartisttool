from tqdm import tqdm
import torch.nn.functional as F
import time
import os
import argparse
from torchvision import transforms
from PIL import Image
import numpy as np
import math
import base64
from io import BytesIO
import logging
import azure.functions as func

from PytorchWCT.util_wct import *
from PytorchWCT.test import generate_sFs, style_transfer_thumbnail, style_transfer_high_resolution
from tools import unpadding, preprocess
from thumb_instance_norm import init_thumbnail_instance_norm

class NeuralStyleTransferV2():
    def __init__(self, input_picture, style_choice):
        """ Initialize parameters of the program + corresponding paths """
        self.URST = True
        self.alpha = 1
        self.patch_size = 1000
        self.thumb_size = 256
        self.style_size = 1024
        self.padding = 32
        self.test_speed = False

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.repeat = 15 if self.test_speed else 1
        self.time_list = []

        #self.content_path = './1. Content' 
        self.content_image = Image.open(BytesIO(base64.b64decode(input_picture))) 
        #self.content_options = {i: styletype for i, styletype in enumerate(os.listdir(self.content_path))}

        self.style_path = './2. Style'
        self.style_options = {i: styletype for i, styletype in enumerate(os.listdir(self.style_path))}
        self.style_chosen = style_choice
        
        self.save_prev_path = './3. Preview'
        self.save_path = './4. Output'
        self.WCT = self.full_transformer()

        self._RUN()
    

    def full_transformer(self):
        fake_args = {'mode': '16x',
                     'e5': './trained_models/wct_se_16x_new/5SE.pth',
                     'e4': './trained_models/wct_se_16x_new/4SE.pth',
                     'e3': './trained_models/wct_se_16x_new/3SE.pth',
                     'e2': './trained_models/wct_se_16x_new/2SE.pth',
                     'e1': './trained_models/wct_se_16x_new/1SE.pth',
                     'd5': './trained_models/wct_se_16x_new_sd/5SD.pth',
                     'd4': './trained_models/wct_se_16x_new_sd/4SD.pth',
                     'd3': './trained_models/wct_se_16x_new_sd/3SD.pth',
                     'd2': './trained_models/wct_se_16x_new_sd/2SD.pth',
                     'd1': './trained_models/wct_se_16x_new_sd/1SD.pth'}

        return WCT(fake_args).to(self.device)


    def test_transform(self, size, crop):
        transform_list = []
        if size != 0:
            transform_list.append(transforms.Resize(size))
        if crop:
            transform_list.append(transforms.CenterCrop(size))
        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)
        return transform


    def watermark(self, ax, fig, size_divided=6):
        """ Create watermark for thumbnail images """
        img = Image.open('Logo_Datacation.png')
        width, height = ax.figure.get_size_inches() * fig.dpi
        wm_width = int(width/size_divided) # make the watermark 1/4 of the figure size
        scaling = (wm_width / float(img.size[0]))
        wm_height = int(float(img.size[1])*float(scaling))
        img = img.resize((wm_width, wm_height), Image.Resampling.LANCZOS)

        imagebox = OffsetImage(img, zoom=1, alpha=1.0)
        imagebox.image.axes = ax

        ao = AnchoredOffsetbox(4, pad=0.01, borderpad=0, child=imagebox)
        ao.patch.set_alpha(0)
        ax.add_artist(ao)


    def preview_all_styles(self, style_tf, thumbnail, content_name = 'inputimage.png'):
        """ Transform content in all available style types """  
        options_style = list(self.style_options.values())
        chosen_style = options_style[self.style_chosen]

        # Setup of the corresponding style
        style = Image.open(f"{self.style_path}/{chosen_style}")
        style = style_tf(style).unsqueeze(0).to(self.device)

        # Stylize image
        with torch.no_grad():
            sFs = generate_sFs(self.WCT, style, mode="gpu")
            style_transfer_thumbnail(self.WCT, thumbnail, sFs, self.alpha, self.device, 
                                        save=True, wct_mode="gpu",
                                        save_path=os.path.join(self.save_prev_path, f"{content_name.split('.')[0]}-{self.thumb_size}.{content_name.split('.')[-1]}"))

            # Create and plot image
            self.artwork = Image.open(f"{self.save_prev_path}/{content_name.split('.')[0]}-{self.thumb_size}.{content_name.split('.')[-1]}")

                
    def _RUN(self):
        PATCH_SIZE = self.patch_size
        PADDING = self.padding
        content_tf = self.test_transform(0, False)
        style_tf = self.test_transform(self.style_size, True)

        # If image is added, retrieve image name, transform and show
        image = self.content_image

        IMAGE_WIDTH, IMAGE_HEIGHT = image.size
        aspect_ratio = IMAGE_WIDTH / IMAGE_HEIGHT

        # Start by resizing the image for preview images
        thumbnail = image.resize((int(aspect_ratio * self.thumb_size), self.thumb_size))
        thumbnail = content_tf(thumbnail).unsqueeze(0).to(self.device)
        #print("thumbnail:", thumbnail.shape)

        # Preview all different styles and choose one
        self.preview_all_styles(style_tf=style_tf, thumbnail=thumbnail)

        
            