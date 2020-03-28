# Dataset

Here we provide our annotated dataset for evaluation, as well as segmentation results from other models.

## BIG
BIG is a high-resolution segmentation dataset that has been hand-annotated by us. 
BIG contains 50 validation objects, and 100 test objects with resolution ranges from 2048\*1600 to 5000\*3600. 

- [One Drive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jchungaa_connect_ust_hk/EeTPE6gisqBBndX2ABIy2QEBTZR_OxPrpaCdKhuP8Q95QA?e=6rCUSQ)

## Relabeled PASCAL VOC 2012
We have relabeled 500 images from PASCAL VOC 2012 to have more accurate boundaries. 
Below shows an example of our relabeled segmentation.

![](images/relabeled_pascal.png)

- [One Drive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jchungaa_connect_ust_hk/EbtbHa40zNJDpNlD3UbDadQB4eG_dNfFI7YDit3OYOXAkw?e=Gmuaym)

## Segmentation Results

For convenience, we provide segmentation results from other models for evaluation. 
We tried our best to match the performance in their original papers, and use official code whenever available. 
<!-- These are NOT an official result from the authors of the paper.  -->
<!-- We recommend you to get the segmentation results manually from the original author's code release to test our model.  -->
<!-- We also include multi-scale evaluation  -->

| Segmentation |             |   |                         |                               |
|--------------|-------------|---|-------------------------|-------------------------------|
| BIG (Test)   | DeeplabV3+  |   | [Download](https://hkustconnect-my.sharepoint.com/:f:/g/personal/jchungaa_connect_ust_hk/Em8xxjDNRVNFpZaWwJV49NkBXxQwXd_AAIahQniAnq5IkQ?e=OwheVV) | [Source](https://github.com/tensorflow/models/tree/master/research/deeplab) |
|              | RefineNet   |   | [Download](https://hkustconnect-my.sharepoint.com/:f:/g/personal/jchungaa_connect_ust_hk/Em8xxjDNRVNFpZaWwJV49NkBXxQwXd_AAIahQniAnq5IkQ?e=OwheVV) | [Source](https://github.com/guosheng/refinenet) |
|              | PSPNet      |   | [Download](https://hkustconnect-my.sharepoint.com/:f:/g/personal/jchungaa_connect_ust_hk/Em8xxjDNRVNFpZaWwJV49NkBXxQwXd_AAIahQniAnq5IkQ?e=OwheVV) | [Source](https://github.com/hszhao/PSPNet) |
|              | FCN-8s      |   | [Download](https://hkustconnect-my.sharepoint.com/:f:/g/personal/jchungaa_connect_ust_hk/Em8xxjDNRVNFpZaWwJV49NkBXxQwXd_AAIahQniAnq5IkQ?e=OwheVV) | [Source](https://github.com/developmentseed/caffe-fcn/tree/master/fcn-8s) |
| PASCAL       | DeeplabV3+  |   | [Download](https://hkustconnect-my.sharepoint.com/:f:/g/personal/jchungaa_connect_ust_hk/EhTt-3DzfdZHoRsjQEC8_xABjjQEHbK9rKgXE78btCfE0g?e=EvsRGH) | [Source](https://github.com/tensorflow/models/tree/master/research/deeplab) |
|              | RefineNet   |   | [Download](https://hkustconnect-my.sharepoint.com/:f:/g/personal/jchungaa_connect_ust_hk/EhTt-3DzfdZHoRsjQEC8_xABjjQEHbK9rKgXE78btCfE0g?e=EvsRGH) | [Source](https://github.com/guosheng/refinenet) |
|              | PSPNet      |   | [Download](https://hkustconnect-my.sharepoint.com/:f:/g/personal/jchungaa_connect_ust_hk/EhTt-3DzfdZHoRsjQEC8_xABjjQEHbK9rKgXE78btCfE0g?e=EvsRGH) | [Source](https://github.com/hszhao/PSPNet) |
|              | FCN-8s      |   | [Download](https://hkustconnect-my.sharepoint.com/:f:/g/personal/jchungaa_connect_ust_hk/EhTt-3DzfdZHoRsjQEC8_xABjjQEHbK9rKgXE78btCfE0g?e=EvsRGH) | [Source](https://github.com/developmentseed/caffe-fcn/tree/master/fcn-8s) |

| Scene Parsing |           |   |                         |                               |
|---------------|-----------|---|-------------------------|-------------------------------|
| ADE20K        | RefineNet |   | [Download](https://hkustconnect-my.sharepoint.com/:f:/g/personal/jchungaa_connect_ust_hk/EvIgfKbjdNdJkjchYL5GBgcBzNX5n4DoLWoLx2dJjFBWgA?e=wGGxNt) | [Source](https://github.com/guosheng/refinenet) |
|               | EncNet    |   | [Download](https://hkustconnect-my.sharepoint.com/:f:/g/personal/jchungaa_connect_ust_hk/EvIgfKbjdNdJkjchYL5GBgcBzNX5n4DoLWoLx2dJjFBWgA?e=wGGxNt) | [Source](https://github.com/zhanghang1989/PyTorch-Encoding) |
|               | PSPNet    |   | [Download](https://hkustconnect-my.sharepoint.com/:f:/g/personal/jchungaa_connect_ust_hk/EvIgfKbjdNdJkjchYL5GBgcBzNX5n4DoLWoLx2dJjFBWgA?e=wGGxNt) | [Source](https://github.com/hszhao/PSPNet) |