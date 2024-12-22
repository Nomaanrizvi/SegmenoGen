import psutil
import GPUtil
import time
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns 
import torch
import albumentations as A 
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import get_loaders, save_checkpoint, load_checkpoint, save_predictions_as_imgs
warnings.simplefilter("ignore")

# Hyperparameters

LEARNING_RATE = 3.16e-04
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 100
NUM_WORKERS = 8
IMAGE_HEIGHT = 250  # 250 originally
IMAGE_WIDTH = 250  # 250 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "dataset/train_images/"
TRAIN_MASK_DIR = "dataset/train_masks/"
VAL_IMG_DIR = "dataset/val_images/"
VAL_MASK_DIR = "dataset/val_masks/"

def get_gpu_memory():
    # Get the memory usage of all available GPUs
    gpu_memory = [gpu.memoryFree for gpu in GPUtil.getGPUs()]
    return sum(gpu_memory)



start_time = time.time()
start_cpu_usage = psutil.cpu_percent()
start_memory_usage = psutil.virtual_memory().percent
start_gpu_memory = get_gpu_memory()


train_losses = []
def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
    
    return loss.item()


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1, dropout_prob=0.4).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    scaler = torch.cuda.amp.GradScaler()

    best_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        print(f'epoch: {epoch+1}/{NUM_EPOCHS}')
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        train_losses.append(train_loss)
        
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        
        if train_loss < best_loss:
            best_loss = train_loss
            save_checkpoint(checkpoint)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="runs/train/", device=DEVICE
        )

# save model
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer":optimizer.state_dict(),
    }
    save_checkpoint(checkpoint, filename="last.pth")


    # Plotting the loss
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.savefig('runs/train/seg_losses.png')

if __name__=='__main__':
    main()

    end_time = time.time()
    end_cpu_usage = psutil.cpu_percent()
    end_memory_usage = psutil.virtual_memory().percent
    end_gpu_memory = get_gpu_memory()

    elapsed_time = end_time - start_time
    cpu_time = end_cpu_usage - start_cpu_usage
    memory_usage = end_memory_usage - start_memory_usage
    gpu_memory_usage = start_gpu_memory - end_gpu_memory

    print(f"Elapsed time: {elapsed_time} seconds")
    print(f"CPU usage: {cpu_time}%")
    print(f"Memory usage: {memory_usage}%")
    print(f"GPU memory usage: {gpu_memory_usage} bytes")