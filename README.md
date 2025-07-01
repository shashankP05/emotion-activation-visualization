# Emotion Detection & Activation Map Visualization

## Project Overview
This project demonstrates how to visualize activation maps from a convolutional neural network (CNN) trained for facial emotion detection. The goal is to understand which regions of an image activate specific CNN filters, providing insight into the model's decision-making process.

## Features
- Visualizes activation maps from any pre-trained Keras/TensorFlow model for emotion detection.
- Uses an image dataset organized by emotion categories (see the `train/` and `test/` folders).
- (Optional) Includes a GUI for emotion prediction, but the main focus is on visualization, not the GUI.

## Getting Started

### 1. Download the Pre-trained Model
Download the Keras model file (`emotion2_model.keras`) from your provided Google Drive link:

**[Download Model from Google Drive](YOUR_GOOGLE_DRIVE_LINK_HERE)**

Place the downloaded file in the project root directory.

### 2. Dataset
The dataset should be organized as image folders by emotion, as already present in the `train/` and `test/` directories.

### 3. Install Requirements
Install all dependencies using:

```bash
pip install -r requirements.txt
```

### 4. Visualize Activation Maps
Run the `Visualization_layer.ipynb` notebook to visualize activation maps for any input image. This notebook loads the pre-trained model, processes an image, and displays the feature maps from a convolutional layer.

## Notes
- The GUI (`gui.py`) is optional and not required for activation map visualization.
- The Haar Cascade XML file is included for possible face detection but is not used in the current notebooks.

## Main Intention
**Visualize Activation Maps:**
- Description: Visualize activation maps to understand which image regions activate CNN filters for emotion detection.
- Guidelines: You can use any of your pre-trained models for this task. GUI is not necessary.
