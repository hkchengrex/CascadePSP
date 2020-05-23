# Testing on Semantic Segmentation

Pretrained models can be downloaded [here](models.md). 
For convenience, we offer pre-processed segmentation inputs from other segmentation models [here](dataset.md). Pre-computed results from our method can also be found [here](dataset.md)

## Test set Structure

Our test script expects the following structure:

```
+ testset_directory
  - imagename_gt.png
  - imagename_seg.png
  - imagename_im.jpg
```

Where `_gt`, `_seg`, and `_im` denote the input segmentation, ground-truth segmentation, and RGB image respectively. Segmentations should be in binary format (i.e. only one object at a time).

## Testing

To refine on high-resolution segmentations using both the Global and Local step (i.e. for the BIG dataset), use the following:
``` bash
# From CascadePSP/
python eval.py \
    --dir testset_directory \
    --model model_name \
    --output output_directory
```

To refine on low-resolution segmentations, we can skip the Local step (though using both will not deteriorate the result) by appending a `--global_only` flag, i.e.: 

``` bash
# From CascadePSP/
python eval.py \
    --dir testset_directory \
    --model model_name \
    --output output_directory \
    --global_only
```

You can obtain the accurate metrics (i.e. IoU and mBA) by running a separate script -- this allows you to test your own results easily:

``` bash
# From CascadePSP/
python eval_post.py \
    --dir output_directory
```
