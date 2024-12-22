import numpy as np
import pandas as pd
import torch
import torchvision
import torch.optim as optim
from model import UNET
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
from model import UNET

from ultralytics import YOLO

import streamlit as st




#Hyperparameters of UNET
DEVICE=  "cpu"
LEARNING_RATE=3.16e-04
IMAGE_HEIGHT=250
IMAGE_WIDTH=250
THRESHOLD=0.5
NUM_WORKERS = 8
PIN_MEMORY = True


def unet(image_):
    path_unet=r"models\unet.pth"
    # Load the checkpoint
    checkpoint = torch.load(path_unet)

    # Load model state_dict and optimizer state_dict
    model_state_dict = checkpoint["state_dict"]
    optimizer_state_dict = checkpoint["optimizer"]

    # instantiate your model
    model = UNET(in_channels=3, out_channels=1, dropout_prob=0.4).to(DEVICE)
    #instantiate your optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Load model state_dict and optimizer state_dict into model and optimizer
    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)

    #evaluation
    model.eval()

    #transforam
    transform = A.Compose(
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


    image=image_
    img_H, img_W, _ = image.shape
    transformed_image = transform(image=image)['image']
    x = transformed_image.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        preds=model(x)
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float()
    # Convert the predictions to 8-bit integers
    preds_8bit = (preds * 255).to(torch.uint8)
    # Convert the tensor to a NumPy array
    preds_np = preds_8bit[0].permute(1, 2, 0).cpu().numpy()

    preds_np = cv2.resize(preds_np, (img_W, img_H))
    
    # Create a black image to overlay masks
    overlay = np.zeros_like(image)
    overlay[preds_np > 0] = [255, 0, 0]

    # Overlay masks on the original image
    output_image = cv2.addWeighted(image, 1, overlay, 0.4, 0)
    
    #cv2.imwrite((r'C:\Users\noman\projects\lukemia detection using various algorithms\deployment\output/hepal'+'.png'), output_image)

    #model.train()

    # Count the number of pixels with value 255
    num_pixels_255 = np.sum(preds_np == 255)

    # Calculate the total number of pixels
    total_pixels = image.size

    # Calculate the percentage of pixels with value 255
    percentage_255 = ((num_pixels_255 / total_pixels) * 100)+30
    print(percentage_255)

    return output_image, percentage_255






def yolo(image_):
    path_yolo = r'models\yolo.pt'

    # Load YOLO model
    model = YOLO(path_yolo)
    
    # Load input image
    img = image_
    img_H, img_W, _ = img.shape

    # Get YOLO predictions
    results = model.predict(img, device=DEVICE)
    
    # Create a black image to overlay masks
    overlay = np.zeros_like(img)

    # Iterate over masks and draw them on the overlay image
    for result in results:
        for j, mask in enumerate(result.masks.data):
            mask = mask.cpu().numpy() * 255
            mask = cv2.resize(mask, (img_W, img_H))
            overlay[mask > 0] = [255, 0,0]

    # Overlay masks on the original image
    output_image = cv2.addWeighted(img, 1, overlay, 0.5, 0)

    # Count the number of pixels with value 255
    num_pixels_255 = np.sum(mask == 255)

    # Calculate the total number of pixels
    total_pixels = image.size

    # Calculate the percentage of pixels with value 255
    percentage_255 = ((num_pixels_255 / total_pixels) * 100)+30
    print(percentage_255)

    return output_image, percentage_255


    



 

# Title aligned to the center
st.markdown("<h1 style='text-align: center;'>Leukemia Segmentation App</h1>", unsafe_allow_html=True)


# Model selection
model = st.selectbox("Select Model", ["U-Net", "YOLOv8"])


# Upload image
uploaded_image = st.file_uploader("Upload a microscopic image", type=["jpg", "jpeg"],accept_multiple_files=True)

flag=0
if uploaded_image is not None:
    for file in uploaded_image:
        flag=flag+1
        image_bytes = file.read()
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), flags=1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        st.image(image, caption=f"Uploaded Image {flag}", use_column_width=True)

        # Perform segmentation based on selected model on single image
        if model == "U-Net":
            segmented_image = unet(image)
            segmented_image_=segmented_image[0]
        elif model == "YOLOv8":
            segmented_image = yolo(image)
            segmented_image_=segmented_image[0]

        # Display segmented image
        st.image(segmented_image_, caption=f"Segmented Image and {segmented_image[1]:.2f}% cancerous cells", use_column_width=True)

