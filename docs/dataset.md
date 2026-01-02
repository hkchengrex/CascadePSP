# Dataset

Here we provide our annotated dataset for evaluation, as well as segmentation results from other models. We do not hold the license for the RGB images. 

## BIG
BIG is a high-resolution segmentation dataset that has been hand-annotated by us. Images are collected from Flickr. Please do not use for commercial purposes.
BIG contains 50 validation objects, and 100 test objects with resolution ranges from 2048\*1600 to 5000\*3600. 

| Sample image  | Mask overlay |
| ------------- | ------------- |
| ![big_image](images/big_sample_image.jpg)  |  ![big_mask](images/big_sample_mask.jpg) |

(Downsampled to reduce file size here)

- [Google Drive](https://drive.google.com/open?id=1cLQvy1giJTSrHV4FGzXgadBgI0zNtIxN)
- [Google Drive (Backup)](https://drive.google.com/file/d/1yXP-PDyBD1BIvr7wJ6aLSRMO9UQqo8V5/view?usp=sharing)

## Relabeled PASCAL VOC 2012
We have relabeled 500 images from PASCAL VOC 2012 to have more accurate boundaries. 
Below shows an example of our relabeled segmentation.

![relabeled_pascal](images/relabeled_pascal.png)

- [Google Drive](https://drive.google.com/open?id=1vtkR05TTSQYu6XPrNr88sh3m7UxDazn2)
- [Google Drive (Backup)](https://drive.google.com/file/d/1iiOPhjp-YBfc-lUpTbbF2ApsxPNYnqqf/view?usp=sharing)

## Segmentation Results

For convenience, we provide segmentation results from other models for evaluation. 
We tried our best to match the performance in their original papers, and use official code whenever available. 
<!-- These are NOT an official result from the authors of the paper.  -->
<!-- We recommend you to get the segmentation results manually from the original author's code release to test our model.  -->
<!-- We also include multi-scale evaluation  -->

| Segmentation |             | Refined  | Segmentation input  |    Source    |
|--------------|-------------|:---:|:-------------------------:|:-------------------------------:|
| BIG (Test)   | DeeplabV3+  | [Download](https://drive.google.com/drive/folders/1tqkNan0cwHMFJWfbRbFabvAuTEKphc-8?usp=sharing)  | [Download](https://drive.google.com/drive/folders/1_71QIWf0rwkKlVEtEDH3gXrfB8lEozKA?usp=sharing) | [Link](https://github.com/tensorflow/models/tree/master/research/deeplab) |
|              | RefineNet   | [Download](https://drive.google.com/drive/folders/1tqkNan0cwHMFJWfbRbFabvAuTEKphc-8?usp=sharing)  | [Download](https://drive.google.com/drive/folders/1_71QIWf0rwkKlVEtEDH3gXrfB8lEozKA?usp=sharing) | [Link](https://github.com/guosheng/refinenet) |
|              | PSPNet      | [Download](https://drive.google.com/drive/folders/1tqkNan0cwHMFJWfbRbFabvAuTEKphc-8?usp=sharing)  | [Download](https://drive.google.com/drive/folders/1_71QIWf0rwkKlVEtEDH3gXrfB8lEozKA?usp=sharing) | [Link](https://github.com/hszhao/PSPNet) |
|              | FCN-8s      | [Download](https://drive.google.com/drive/folders/1tqkNan0cwHMFJWfbRbFabvAuTEKphc-8?usp=sharing)  | [Download](https://drive.google.com/drive/folders/1_71QIWf0rwkKlVEtEDH3gXrfB8lEozKA?usp=sharing) | [Link](https://github.com/developmentseed/caffe-fcn/tree/master/fcn-8s) |
| PASCAL       | DeeplabV3+  | [Download](https://drive.google.com/drive/folders/17mxQIV4AMAfwi57MozKljOYR11DKFPej?usp=sharing)  | [Download](https://drive.google.com/drive/folders/1XNupcNFtyelF_OEkZNsjv18hhg74XBlk?usp=sharing) | [Link](https://github.com/tensorflow/models/tree/master/research/deeplab) |
|              | RefineNet   | [Download](https://drive.google.com/drive/folders/17mxQIV4AMAfwi57MozKljOYR11DKFPej?usp=sharing) | [Download](https://drive.google.com/drive/folders/1XNupcNFtyelF_OEkZNsjv18hhg74XBlk?usp=sharing) | [Link](https://github.com/guosheng/refinenet) |
|              | PSPNet      | [Download](https://drive.google.com/drive/folders/17mxQIV4AMAfwi57MozKljOYR11DKFPej?usp=sharing) | [Download](https://drive.google.com/drive/folders/1XNupcNFtyelF_OEkZNsjv18hhg74XBlk?usp=sharing) | [Link](https://github.com/hszhao/PSPNet) |
|              | FCN-8s      | [Download](https://drive.google.com/drive/folders/17mxQIV4AMAfwi57MozKljOYR11DKFPej?usp=sharing)  | [Download](https://drive.google.com/drive/folders/1XNupcNFtyelF_OEkZNsjv18hhg74XBlk?usp=sharing) | [Link](https://github.com/developmentseed/caffe-fcn/tree/master/fcn-8s) |

| Scene Parsing |    |  Refined | Pre-processed 'split' input(*) | Segmentation input   |      Source      |
|---------------|-----------|:---:|:---:|:-------------------------:|:-------------------------------:|
| ADE20K        | RefineNet | [Download](https://drive.google.com/drive/folders/1YF2ly1c2HiZs0-4eEGHPDwKTW7Rylk3l?usp=sharing) | [Download](https://drive.google.com/drive/folders/1DkJeU2yC3K-V0x58Gk2oDew2JjuzYEw9?usp=sharing)| [Download](https://drive.google.com/drive/folders/1fwicCJMokr1E97JSzH7SnHs490f6Rzcx?usp=sharing) | [Link](https://github.com/guosheng/refinenet) |
|               | EncNet    | [Download](https://drive.google.com/drive/folders/1YF2ly1c2HiZs0-4eEGHPDwKTW7Rylk3l?usp=sharing) | [Download](https://drive.google.com/drive/folders/1DkJeU2yC3K-V0x58Gk2oDew2JjuzYEw9?usp=sharing)| [Download](https://drive.google.com/drive/folders/1fwicCJMokr1E97JSzH7SnHs490f6Rzcx?usp=sharing) | [Link](https://github.com/zhanghang1989/PyTorch-Encoding) | 
|               | PSPNet    | [Download](https://drive.google.com/drive/folders/1YF2ly1c2HiZs0-4eEGHPDwKTW7Rylk3l?usp=sharing) | [Download](https://drive.google.com/drive/folders/1DkJeU2yC3K-V0x58Gk2oDew2JjuzYEw9?usp=sharing)| [Download](https://drive.google.com/drive/folders/1fwicCJMokr1E97JSzH7SnHs490f6Rzcx?usp=sharing) | [Link](https://github.com/hszhao/semseg) | 

(*) Generated from segmentation input for our evaluation. 
