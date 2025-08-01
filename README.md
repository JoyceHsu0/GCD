# Grad-CAM Discoverer

Grad-CAM Discoverer is a Python application for visualizing 3D medical imaging data (e.g., CT scans) using Grad-CAM (Gradient-weighted Class Activation Mapping) with a PyQt6-based GUI and VTK for volume rendering. It allows users to load NIfTI files, process them with a pre-trained model, visualize the results with customizable transfer functions, and interact with the visualization through rotation controls and feature selection.

## Features

- **Load and Process NIfTI Files**: Load medical imaging data in NIfTI format (`.nii` or `.nii.gz`) and process it using a pre-trained UNETCNX model.
- **Grad-CAM Visualization**: Generate and display Grad-CAM heatmaps overlaid on 3D volume data.
- **Interactive GUI**: Built with PyQt6, featuring:
  - File selection for loading NIfTI files.
  - Transfer function editor for customizing color and opacity.
  - Layer and feature selection for controlling Grad-CAM outputs.
  - Rotation speed control for 3D visualization.
  - Screenshot and video recording capabilities.
- **VTK Rendering**: High-quality 3D volume rendering with axes indicators and smooth rotation.
- **Console Output**: Real-time feedback on processing steps and errors.
