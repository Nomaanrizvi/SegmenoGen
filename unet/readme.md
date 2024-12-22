# UNet Image Segmentation Training

This repository contains code for training a UNet model for image segmentation tasks. The UNet architecture is particularly effective for biomedical image segmentation and other similar tasks where precise boundary detection is important.

## U-Net architecture
![UNet Architecture](/unet/u-net-architecture.png)


## Directory Structure

```
.
├── dataset/
│   ├── train_images/
│   ├── train_masks/
│   ├── val_images/
│   └── val_masks/
├── runs/
│   └── train/
└── train.py
└── model.py
└── dataset.py
└── utils.py
```



## Setup Instructions

1. Create the required directories:
- make the directories for the train_images, train_mask, val_images, val_masks

2. Place your training and validation data in the respective directories:
   - Training images: `dataset/train_images/`
   - Training masks: `dataset/train_masks/`
   - Validation images: `dataset/val_images/`
   - Validation masks: `dataset/val_masks/`

## Configuration

Update the following paths in train.py:

```python
TRAIN_IMG_DIR = "path-to-your-train_images/"
TRAIN_MASK_DIR = "path-to-your-train_masks/"
VAL_IMG_DIR = "path-to-your-val_images/"
VAL_MASK_DIR = "path-to-your-val_masks/"
```

## Training Parameters

The model can be trained with the following recommended hyperparameters:
- Learning rate: 3.16e-04
- Batch size: any based on your system specs (8 recommended)
- Epochs: 100
- Optimizer: AdamW
- Loss function: Binary Cross Entropy with Logits

## Training Process

1. The model will save checkpoints in the `runs/train` directory
2. Training progress will be logged including:
   - Training loss
   - Validation loss

## Data Requirements

- Images and masks should be in the following formats i.e. PNG (for masks) or JPEG (for images) or both in PNG
- Masks should be binary (0 for background, 1 for segmentation)

## Monitoring Training

Training progress can be monitored through:
- Terminal output showing loss values and metrics
- logs in the runs/train directory
- Saved model checkpoints

## Expected Outputs

The training process will generate:
1. Model checkpoints
2. Training logs
3. Validation metrics
4. Best and last model weights

## Troubleshooting

Common issues and solutions:
1. Out of memory errors: Reduce batch size
2. Slow training: Check data loading pipeline
3. Poor convergence: Adjust learning rate or increase epochs
4. Overfitting: Implement additional augmentation or adjust dropout
5. deprecation of the dependencies: check online or contact me

For any additional issues, please check the documentation or raise an issue in the repository or mail me at nomanrizvi007@gmail.com.