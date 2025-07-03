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

**[Download Model from Google Drive](https://drive.google.com/drive/folders/1XmQIvwEC-0PHBtssFhfmi9NZiuqWIlWw?usp=drive_link)**

Place the downloaded file in the project root directory.

### 2. Dataset
The dataset should be organized as image folders by emotion, as already present in the `train/` and `test/` directories.

### 3. Install Requirements
Install all dependencies using:

```bash
pip install -r requirements.txt
```

### 4. Visualize Activation Maps
Run the Visualization_layer.ipynb notebook located inside the anaconda_projects/ folder to visualize activation maps for any input image. This notebook loads the pre-trained model, processes an image, and displays the feature maps from a convolutional layer.

## About 'anaconda_projects/Visualization_layer.ipynb`

The Visualization_layer.ipynb notebook (found inside the anaconda_projects/ folder) is dedicated to interpreting and understanding the inner workings of the emotion detection model by visualizing what the neural network "sees" and focuses on during prediction. Here’s what is accomplished in this notebook:

1. **Model and Image Preparation**
   - Loads the pre-trained Keras model (`emotion2_model.keras`).
   - Loads and preprocesses a sample image from the dataset, resizing and normalizing it to fit the model’s input requirements.

2. **Activation Map Visualization**
   - Extracts and visualizes the activation maps (feature maps) from the first convolutional layer. This shows which low-level features (like edges or textures) are detected by the model at the initial stage.
   - Further, it visualizes activation maps from a deeper convolutional layer, revealing how the model captures more complex and abstract features as the data moves deeper into the network.

3. **Grad-CAM (Gradient-weighted Class Activation Mapping)**
   - Implements Grad-CAM to generate a heatmap that highlights the regions of the input image that most strongly influence the model’s prediction for a particular emotion class.
   - The notebook overlays this heatmap on the original image, making it easy to see which areas the model considers most important for its decision.

4. **Interpretability and Insights**
   - By visualizing both early and deep layer activations, as well as the Grad-CAM heatmap, the notebook provides a comprehensive look at how the model processes facial images and which regions are most relevant for emotion classification.

This notebook is a valuable tool for gaining insights into model behavior, debugging, and building trust in the emotion detection system by making its decision process more transparent and interpretable.

## Notes
- The GUI (`gui.py`) is optional and not required for activation map visualization.

## Main Intention
**Visualize Activation Maps:**
- Description: Visualize activation maps to understand which image regions activate CNN filters for emotion detection.

