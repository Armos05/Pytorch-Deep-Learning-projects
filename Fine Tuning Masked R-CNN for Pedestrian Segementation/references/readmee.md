# Fast R-CNN Object Detection with PennFudanDataset

This is a PyTorch implementation of object detection using Fast R-CNN on the PennFudanDataset.

## Dataset

The PennFudanDataset is a dataset for pedestrian detection and segmentation. It consists of 170 images with 345 instances of pedestrians, annotated with ground-truth bounding boxes and masks. The dataset can be downloaded from [here](https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip).

The dataset has Pedestrain Images and their correspondin Masks.
The typical imgage is RBG (3, 559, 556): 

The following is an example:

## Requirements

- PyTorch 1.7+
- torchvision
- numpy
- matplotlib
- Pillow
- tqdm

## Training

To train the Fast R-CNN model on the PennFudanDataset clone the repository and run ```python model.py```.

This will train the model on the dataset for 10 epochs with a learning rate of 0.001 and a batch size of 4. You can modify the hyperparameters according to your needs.

# Concepts:
 