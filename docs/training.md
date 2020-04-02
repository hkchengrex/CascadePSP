# Training

### Download the dataset 

We have prepared a script for downloading the training dataset. 
The script below downloads and merges the following datasets: MSRA-10K, DUT-OMRON, ECSSD, and FSS-1000.
 
```
# From CascadePSP/scripts/
python download_training_dataset.py
```

Note that the following script will create a dataset folder as follows:
```
+ CascadePSP/data/
  + DUTS-TE/
     - image_name_01.jpg
     - image_name_01.png
     - ...
  + DUTS-TR/
     - image_name_01.jpg
     - image_name_01.png
     - ...
  + ecssd/
     - image_name_01.jpg
     - image_name_01.png
     - ...
  + fss/
    + class_name/
         - image_name_01.jpg
         - image_name_01.png
     - ...
  + MSRA_10K/
     - image_name_01.jpg
     - image_name_01.png
     - ...
```

### Running the training script

*NOTE*: Hyperparameters have been adjusted, and code are restructured. The new code yields slightly better performance with faster training time. Both the model used in the paper and the model trained with new code can be downloaded [here](models.md).

Training can be done with following command with some distinguishable id:

```
# From CascadePSP/
python train.py some_unique_id
```

Note that you can change the hyperparameter by specifying arguments, e.g. to change batch size.

```
# From CascadePSP/
python train.py -b 10 some_unique_id
```
Please check [hyper_para.py](../util/hyper_para.py) for more options.

### After training

Tensorboard log file will be stored in `CascadePSP/log/some_unique_id_timestamp`

Model will be saved in `CascadePSP/weights/some_unique_id_timestamp`