import numpy as np
from PIL import Image
import progressbar

from util.compute_boundary_acc import compute_boundary_acc_multi_class
from util.file_buffer import FileBuffer
from dataset.make_bb_trans import get_bb_position, scale_bb_by

from argparse import ArgumentParser
import glob
import os
import re
from pathlib import Path
from shutil import copyfile

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

parser = ArgumentParser()

parser.add_argument('--mask_dir', help='Directory with all the _mask.png outputs',
        default=os.path.join('./output/ade_output'))

parser.add_argument('--gt_dir', help='Directory with original size GT images (in P mode)',
        default=os.path.join('./data/ADE/annotations'))

parser.add_argument('--seg_dir', help='Directory with original size input segmentation images (in L mode)',
        default=os.path.join('./data/ADE/inputs'))

parser.add_argument('--split_dir', help='Directory with the processed split dataset',
        default=os.path.join('./data/ADE/split_ss'))

# Optional
parser.add_argument('--im_dir', help='Directory with original size input images (in RGB mode)',
        default=os.path.join('.', './data/ADE/images'))

parser.add_argument('--output', help='Output of temp results',
        default=None)

args = parser.parse_args()

def get_iu(seg, gt):
    intersection = np.count_nonzero(seg & gt)
    union = np.count_nonzero(seg | gt)
    
    return intersection, union 

total_old_correct_pixels = 0
total_new_correct_pixels = 0
total_num_pixels = 0

total_seg_mba = 0
total_mask_mba = 0
total_num_images = 0

small_objects = 0

num_classes = 150

new_class_i = [0] * num_classes
new_class_u = [0] * num_classes
old_class_i = [0] * num_classes
old_class_u = [0] * num_classes
edge_class_pixel = [0] * num_classes
old_gd_class_pixel = [0] * num_classes
new_gd_class_pixel = [0] * num_classes

all_gts = os.listdir(args.seg_dir)
mask_path = Path(args.mask_dir)

if args.output is not None:
    os.makedirs(args.output, exist_ok=True)
    file_buffer = FileBuffer(os.path.join(args.output, 'results_post.txt'))

for gt_name in progressbar.progressbar(all_gts):
    
    gt = np.array(Image.open(os.path.join(args.gt_dir, gt_name)
                                ).convert('P'))

    seg = np.array(Image.open(os.path.join(args.seg_dir, gt_name)
                                ).convert('L'))

    # We pick the highest confidence class label for overlapping region
    mask = seg.copy()
    confidence = np.zeros_like(gt) + 0.5
    keep = False
    for mask_name in mask_path.glob(gt_name[:-4] + '*mask*'):
        class_mask_prob = np.array(Image.open(mask_name).convert('L')).astype('float') / 255
        class_string = re.search(r'\d+.\d+', mask_name.name[::-1]).group()[::-1]
        this_class = int(class_string.split('.')[0])
        class_seg = np.array(
                Image.open(
                    os.path.join(args.split_dir, mask_name.name.replace('mask', 'seg'))
                ).convert('L')
            ).astype('float') / 255

        try:
            rmin, rmax, cmin, cmax = get_bb_position(class_seg)
            rmin, rmax, cmin, cmax = scale_bb_by(rmin, rmax, cmin, cmax, seg.shape[0], seg.shape[1], 0.25, 0.25)
        except:
            # Sometimes we cannot get a proper bounding box
            rmin = cmin = 0
            rmax, cmax = seg.shape

        if (cmax==cmin) or (rmax==rmin):
            print(gt_name, this_class)
            continue
        class_mask_prob = np.array(Image.fromarray(class_mask_prob).resize((cmax-cmin, rmax-rmin), Image.BILINEAR))

        background_classes = [1,2,3,4,6,7,10,12,14,17,22,26,27,29,30,47,49,52,53,55,60,61,62,69,80,85,92,95,97,102,106,110,114,129,141]
        if this_class in background_classes:
            class_mask_prob = class_mask_prob * 0.51

        # Record the current higher confidence level for each pixel
        mask[rmin:rmax, cmin:cmax] = np.where(class_mask_prob>confidence[rmin:rmax, cmin:cmax], 
                                                this_class, mask[rmin:rmax, cmin:cmax])
        confidence[rmin:rmax, cmin:cmax] = np.maximum(confidence[rmin:rmax, cmin:cmax], class_mask_prob)

    total_classes = np.union1d(np.unique(gt), np.unique(seg))
    seg[gt==0] = 0
    mask[gt==0] = 0
    total_classes = total_classes[1:] # Remove background class
    # Shift background class to -1
    total_classes -= 1

    for c in total_classes:
        gt_class = (gt == (c+1))
        seg_class = (seg == (c+1))
        mask_class = (mask == (c+1))

        old_i, old_u = get_iu(gt_class, seg_class)
        new_i, new_u = get_iu(gt_class, mask_class)

        total_old_correct_pixels += old_i
        total_new_correct_pixels += new_i
        total_num_pixels += gt_class.sum()

        new_class_i[c] += new_i
        new_class_u[c] += new_u
        old_class_i[c] += old_i
        old_class_u[c] += old_u

    seg_acc, mask_acc = compute_boundary_acc_multi_class(gt, seg, mask)
    total_seg_mba += seg_acc
    total_mask_mba += mask_acc
    total_num_images += 1

    if args.output is not None and keep:
        gt = Image.fromarray(gt,mode='P')
        seg = Image.fromarray(seg,mode='P')
        mask = Image.fromarray(mask,mode='P')
        gt.putpalette(color_map())
        seg.putpalette(color_map())
        mask.putpalette(color_map())

        gt.save(os.path.join(args.output, gt_name.replace('.png', '_gt.png')))
        seg.save(os.path.join(args.output, gt_name.replace('.png', '_seg.png')))
        mask.save(os.path.join(args.output, gt_name.replace('.png', '_mask.png')))

        if args.im_dir is not None:
            copyfile(os.path.join(args.im_dir, gt_name.replace('.png','.jpg')), 
            os.path.join(args.output, gt_name.replace('.png','.jpg')))

file_buffer.write('New pixel accuracy: ', total_new_correct_pixels / total_num_pixels)
file_buffer.write('Old pixel accuracy: ', total_old_correct_pixels / total_num_pixels)

file_buffer.write('Number of small objects: ', small_objects)

file_buffer.write('Now giving class information')

new_class_iou = [0] * num_classes
old_class_iou = [0] * num_classes
new_class_boundary = [0] * num_classes
old_class_boundary = [0] * num_classes

print('\nNew IOUs: ')
for i in range(num_classes):
    new_class_iou[i] = new_class_i[i] / (new_class_u[i] + 1e-6)
    print('%.3f' % (new_class_iou[i]), end=' ')

print('\nOld IOUs: ')
for i in range(num_classes):
    old_class_iou[i] = old_class_i[i] / (old_class_u[i] + 1e-6)
    print('%.3f' % (old_class_iou[i]), end=' ')

file_buffer.write()
file_buffer.write('Average over classes')

old_miou = np.array(old_class_iou).mean()
new_miou = np.array(new_class_iou).mean()
old_mba = total_seg_mba/total_num_images
new_mba = total_mask_mba/total_num_images

file_buffer.write('Old mIoU    : ', old_miou)
file_buffer.write('New mIoU    : ', new_miou)
file_buffer.write('mIoU Delta  : ', new_miou - old_miou)
file_buffer.write('Old mBA     : ', old_mba)
file_buffer.write('New mBA     : ', new_mba)
file_buffer.write('mBA Delta   : ', new_mba - old_mba)
