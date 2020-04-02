import os
from os import path
from torch.utils.data.dataset import Dataset
from torchvision import transforms, utils
from torchvision.transforms import functional
from PIL import Image
import numpy as np
import progressbar

from dataset.make_bb_trans import *

class OfflineDataset(Dataset):
    def __init__(self, root, in_memory=False, need_name=False, resize=False, do_crop=False):
        self.root = root
        self.need_name = need_name
        self.resize = resize
        self.do_crop = do_crop
        self.in_memory = in_memory

        imgs = os.listdir(root)
        imgs = sorted(imgs)

        """
        There are three kinds of files: _im.png, _seg.png, _gt.png
        """
        im_list = [im for im in imgs if 'im' in im[-7:].lower()]

        self.im_list = [path.join(root, im) for im in im_list]

        print('%d images found' % len(self.im_list))

        # Make up some transforms
        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.gt_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.seg_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            ),
        ])

        if self.resize:
            self.resize_bi = lambda x: x.resize((224, 224), Image.BILINEAR)
            self.resize_nr = lambda x: x.resize((224, 224), Image.NEAREST)
        else:
            self.resize_bi = lambda x: x
            self.resize_nr = lambda x: x

        if self.in_memory:
            print('Loading things into memory')
            self.images = []
            self.gts = []
            self.segs = []
            for im in progressbar.progressbar(self.im_list):
                image, seg, gt = self.load_tuple(im)

                self.images.append(image)
                self.segs.append(seg)
                self.gts.append(gt)
        
    def load_tuple(self, im):
        seg = Image.open(im[:-7]+'_seg.png').convert('L')
        crop_lambda = self.get_crop_lambda(seg)

        image = self.resize_bi(crop_lambda(Image.open(im).convert('RGB')))
        gt = self.resize_bi(crop_lambda(Image.open(im[:-7]+'_gt.png').convert('L')))
        seg = self.resize_bi(crop_lambda(Image.open(im[:-7]+'_seg.png').convert('L')))

        return image, seg, gt

    def get_crop_lambda(self, seg):
        if self.do_crop:
            seg = np.array(seg)
            h, w = seg.shape
            try:
                bb = get_bb_position(seg)
                rmin, rmax, cmin, cmax = scale_bb_by(*bb, h, w, 0.15, 0.15)
                return lambda x: functional.crop(x, rmin, cmin, rmax-rmin, cmax-cmin)
            except:
                return lambda x: x
        else:
            return lambda x: x

    def __getitem__(self, idx):
        if self.in_memory:
            im = self.images[idx]
            gt = self.gts[idx]
            seg = self.segs[idx]
        else:
            im, seg, gt = self.load_tuple(self.im_list[idx])

        im = self.im_transform(im)
        gt = self.gt_transform(gt)
        seg = self.seg_transform(seg)

        if self.need_name:
            return im, seg, gt, os.path.basename(self.im_list[idx][:-7])
        else:
            return im, seg, gt

    def __len__(self):
        return len(self.im_list)
        
if __name__ == '__main__':
    o = OfflineDataset('data/val_static')
