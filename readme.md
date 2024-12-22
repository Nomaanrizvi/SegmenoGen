# SegmenoGen - a web based leukemia segmentation app

## Overview
This is a medical image analysis application built using Streamlit that performs automated segmentation of leukemia cells in microscopic images. The app allows users to choose between two different deep learning approaches:
- YOLOv8 for object detection-based segmentation
- U-Net for semantic segmentation

Users can select their preferred model based on their specific needs and image characteristics.

## Features
- Upload and process microscopic blood smear images
- Choose between YOLOv8 or U-Net for cell segmentation
- Interactive visualization of segmentation results
- Model selection interface

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- CUDA-capable GPU (recommended for optimal performance)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/Nomaanrizvi/SegmenoGen.git
cd SegmenoGen
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage
1. Navigate to the project directory:
```bash
cd SegmenoGen
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your web browser and go to `http://localhost:8501`

4. Use the app:
   - Upload microscopic images through the interface
   - Select your preferred model (YOLOv8 or U-Net)
   - View the results


## Dependencies
- streamlit>=1.24.0
- ultralytics>=8.0.0  # For YOLOv8
- torch>=2.0.0
- opencv-python>=4.7.0
- numpy>=1.23.5
- pandas>=1.5.3
- scikit-image>=0.20.0

## Model Information

### Model Selection
- Users must choose one model at a time for processing
- Each model has its own strengths and use cases
- Models cannot be used simultaneously

### YOLOv8
- Object detection-based approach
- Best for clearly separated cells
- Detection confidence threshold: 0.5 (configurable)

### U-Net
- Semantic segmentation approach
- Better for densely packed or overlapping cells

## Troubleshooting

Common issues and solutions:
1. Out of memory errors: Reduce batch size
2. deprecation of the dependencies: check online or contact me

## Contributing
1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
Nomaan Rizvi - [nomanrizvi007@gmail.com](mailto:your.email@example.com)