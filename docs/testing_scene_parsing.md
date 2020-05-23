# Testing on Scene Parsing

Pretrained models can be downloaded [here](models.md). 
For convenience, we offer pre-processed scene parsing inputs from other segmentation models [here](dataset.md). Pre-computed results from our method can also be found [here](dataset.md)

## Test set Structure
Evaluation on scene parsing dataset is more complicated. Read this [document](testing_segmentation.md) about testing on segmentation first for starters. 

We need to perform the following steps:

1. Obtain initial segmentations from other models.
2. Break down the scene parse into individual components using the method described in the paper.
3. Process each component separately using CascadePSP.
4. Combine the processed components to a final scene parse.

You can skip step 1 and 2 by downloading our pre-processed dataset. 

## Testing

To run step 3, append an extra flag `--ade` to `eval.py`.
``` bash
# From CascadePSP/
python eval.py \
    --dir testset_directory \
    --model model_name \
    --output output_directory \
    --ade
```

And to run step 4, 

``` 
python eval_post_ade.py \
    --mask_dir [Output directory in step3] \
    --seg_dir [Directory with the original initial segmentations] \
    --gt_dir [Directory with the ground truth segmentations] \
    --split_dir [Directory with the broken-down initial segmentations] \
    --output output_directory
```
