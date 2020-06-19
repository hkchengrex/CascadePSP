import cv2
import time
import matplotlib.pyplot as plt
import segmentation_refinement as refine


image = cv2.imread('test/aeroplane.jpg')
mask = cv2.imread('test/aeroplane.png', cv2.IMREAD_GRAYSCALE)

# model_path can also be specified here
# This step takes some time to load the model
refiner = refine.Refiner(device='cuda:0') # device can also be 'cpu'

# Fast - Global step only.
# Smaller L -> Less memory usage; faster in fast mode.
output = refiner.refine(image, mask, fast=False, L=900) 

plt.imshow(output)
plt.show()
