# Grad-CAM Discoverer
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

[![Watch the video](https://img.youtube.com/vi/1U5GHGVs9-s/0.jpg)](https://youtu.be/1U5GHGVs9-s?si=6ZYWnAMzUpxZHQp7)  
*Click the image above to watch a system demonstration on YouTube.*

Grad-CAM Discoverer is a Python application for visualizing 3D medical imaging data (e.g., CT scans) using Grad-CAM (Gradient-weighted Class Activation Mapping) with a PyQt6-based GUI and VTK for volume rendering. It allows users to load NIfTI files, process them with a pre-trained model, visualize the results with customizable transfer functions, and interact with the visualization through rotation controls and feature selection.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Notes](#notes)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features

- **Load and Process NIfTI Files**: Load medical imaging data in NIfTI format (`.nii` or `.nii.gz`) and process it using a pre-trained UNETCNX model.
- **Grad-CAM Visualization**: Generate and display Grad-CAM heatmaps overlaid on 3D olume data.
- **Interactive GUI**: Built with PyQt6, featuring:
  - File selection for loading NIfTI files.
  - Transfer function editor for customizing color and opacity.
  - Layer and feature selection for controlling Grad-CAM outputs.
  - Rotation speed control for 3D visualization.
  - Screenshot and video recording capabilities.
- **VTK Rendering**: High-quality 3D volume rendering with axes indicators and smooth rotation.
- **Console Output**: Real-time feedback on processing steps and errors.

## Prerequisites

- A compatible GPU with CUDA support is recommended for faster processing.
- NIfTI files (`.nii` or `.nii.gz`) for input data.
- Pre-trained model checkpoint file (`unetcnx.pth`).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/JoyceHsu0/GCD.git
   cd GCD
   ```
2. Create a `dat` directory in the project root and place the required files:
   - **Pre-trained model weights**: Place the `unetcnx.pth` file in the `dat` directory.
   - **CT scan data**: Place your NIfTI files (e.g., `demo.1.nii.gz`) in the `dat` directory.

   Example directory structure:
   ```
   GCD/
   ├── dat/
   │ ├── unetcnx.pth
   │ └── demo.1.nii.gz
   ├── gcd_core.py
   ├── gcd.py
   ├── gcd_render.py
   ├── requirements.txt
   └── README.md
   ```
3. Create and activate a Conda virtual environment with Python 3.10, then install the required packages:
   ```bash
   conda create -n GCD python=3.10
   conda activate GCD
   ```
4. Install PyTorch with CUDA 11.3 support:
   ```
   pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
   ```

5. Install the remaining dependencies from requirements.txt:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python gcd.py
   ```

2. The GUI will open with the following controls:
   - **Open File**: Select a NIfTI file from the `dat` directory or elsewhere.
   - **Transfer Function Editor**: Click and drag to adjust control points for color and opacity. Double-click to change colors.
   - **Layer Selection**: Choose a layer from the model to compute Grad-CAM.
   - **Feature Selection**: Adjust the feature range using input fields or shift buttons (`<` and `>`).
   - **Rotation Controls**: Use the slider to adjust rotation speed or start/stop rotation.
   - **Overlay/Heatmap**: Toggle between overlay mode (heatmap + CT) and heatmap-only mode.
   - **Save Screenshot**: Save the current view as a PNG file.
   - **Record Video**: Record a video of the visualization as an MP4 file.

## Files

- `gcd.py`: Main application script with the PyQt6 GUI.
- `gcd_core.py`: Core logic for loading, processing, and computing Grad-CAM.
- `gcd_render.py`: VTK-based rendering for 3D visualization.
- `dat/unetcnx.pth`: Pre-trained model weights (must be provided by the user).
- `dat/*.nii.gz`: Input NIfTI files (e.g., CT scans).

## Notes

- Ensure the `dat` directory contains the `unetcnx.pth` file and your NIfTI files before running the application.
- The application assumes a model input size of 128x128x128 and specific spacing (`(0.7, 0.7, 1.0)`). Adjust these in `gcd_core.py` if needed.
- Video recording requires sufficient disk space and may take time depending on the rotation speed and number of frames.
- Public datasets are available to test run. For example, https://www.kaggle.com/datasets/rajendrakpandey/mm-whs-2017-dataset-5-62-gb-158-files-ct-and-mr

## Troubleshooting

- **Missing `unetcnx.pth`**: Ensure the pre-trained model weights are placed in the `dat` directory.
- **NIfTI file errors**: Verify that input files are valid NIfTI files and not corrupted.
- **Performance issues**: Use a CUDA-enabled GPU for faster processing. Check console output for errors.
- **GUI rendering issues**: Ensure VTK and PyQt6 are correctly installed and compatible with your Python version.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

