import torch

from models.psp.pspnet import PSPNet

import sys

# Construct model
model = PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50').cuda()

L = int(sys.argv[1])
batch_size = 1

def safe_forward(model, im, seg):

    b, _, ph, pw = seg.shape
    if (ph % 8 != 0) or (pw % 8 != 0):
        newH = ((ph//8+1)*8)
        newW = ((pw//8+1)*8)
        p_im = torch.zeros(b, 3, newH, newW).cuda()
        p_seg = torch.zeros(b, 1, newH, newW).cuda() - 1

        p_im[:,:,0:ph,0:pw] = im
        p_seg[:,:,0:ph,0:pw] = seg
        im = p_im
        seg = p_seg

    images = model(im, seg)

    return images

with torch.no_grad():
    for _ in range(10):
        im = torch.zeros((1, 3, L, L)).cuda()
        seg = torch.zeros((1, 1, L, L)).cuda()
        images = safe_forward(model, im, seg)

        print(torch.cuda.max_memory_allocated()/1024/1024/1024)

        del im
        del seg
        del images
