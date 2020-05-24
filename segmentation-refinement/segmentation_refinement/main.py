import os

import numpy as np
import torch
from torchvision import transforms

from segmentation_refinement.models.psp.pspnet import RefinementModule
from segmentation_refinement.eval_helper import process_high_res_im, process_im_single_pass
from segmentation_refinement.download import download_file_from_google_drive


class Refiner:
    def __init__(self, device='cpu', model_folder=None):
        """
        Initialize the segmentation refinement model.
        device can be 'cpu' or 'cuda'
        model_folder specifies the folder in which the model will be downloaded and stored. Defaulted in ~/.segmentation-refinement.
        """
        self.model = RefinementModule()
        self.device = device
        if model_folder is None:
            model_folder = os.path.expanduser("~/.segmentation-refinement")

        if not os.path.exists(model_folder):
            os.makedirs(model_folder, exist_ok=True)

        model_path = os.path.join(model_folder, 'model')
        if not os.path.exists(model_path):
            print('Downloading the model file into: %s...' % model_path)
            download_file_from_google_drive('103nLN1JQCs2yASkna0HqfioYZO7MA_J9', model_path)

        model_dict = torch.load(model_path)
        new_dict = {}
        for k, v in model_dict.items():
            name = k[7:] # Remove module. from dataparallel
            new_dict[name] = v
        self.model.load_state_dict(new_dict)
        self.model.eval().to(device)

        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.seg_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            ),
        ])

    def refine(self, image, mask, fast=False, L=900):
        with torch.no_grad():
            """
            Refines an input segmentation mask of the image.

            image should be of size [H, W, 3]. Range 0~255.
            Mask should be of size [H, W] or [H, W, 1]. Range 0~255. We will make the mask binary by thresholding at 127.
            Fast mode - Use the global step only. Default: False. The speedup is more significant for high resolution images.
            L - Hyperparameter. Setting a lower value reduces memory usage. In fast mode, a lower L will make it runs faster as well.
            """
            image = self.im_transform(image).unsqueeze(0).to(self.device)
            mask = self.seg_transform((mask>127).astype(np.uint8)*255).unsqueeze(0).to(self.device)
            if len(mask.shape) < 4:
                mask = mask.unsqueeze(0)

            if fast:
                output = process_im_single_pass(self.model, image, mask, L)
            else:
                output = process_high_res_im(self.model, image, mask, L)

            return (output[0,0].cpu().numpy()*255).astype('uint8')
