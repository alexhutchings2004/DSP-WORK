# DeepFake Detection System

A comprehensive deepfake detection platform that combines advanced deep learning techniques with a user-friendly interface to identify AI-generated and manipulated facial images.

## Project Overview

This project provides an end-to-end solution for detecting deepfake images through multiple components:

1. **AI Model Training Framework** - EfficientNet-based deep learning models trained on deepfake datasets
2. **Backend API Server** - Flask-based server that processes images and provides detection results
3. **React Web Application** - Modern web interface for image upload and analysis
4. **Chrome Extension** - Browser extension for analysing images directly from social media

## Repository Structure

- `/backend` - Flask API server with model inference capabilities
  - `/models` - Trained deepfake detection models
  - `/static` - Storage for processed images and visualisations
- `/DeepdakeDetectionApril` - Model training framework and dataset management
  - `/deepfake_detection` - Core Python package with model architectures and training utilities
  - `/results` - Saved model checkpoints and performance metrics
  - `/train`, `/valid`, `/test` - Dataset directories (not included in git due to folder sizes)
- `/deepfake-detection` - React frontend application
  - `/src` - React components and application logic
  - `/public` - Static assets
- `/chrome-extension` - Browser extension for social media integration
  - Manifest and browser integration files

## Key Features

### AI Detection Models

- **Multi-modal Analysis**: Processes images through multiple representations (RGB, noise pattern, frequency domain)
- **Advanced Feature Extraction**: Uses EfficientNet architecture with specialised layers for deepfake artifacts
- **Face Detection**: Automatically locates and analyses faces in images
- **Explainable AI**: Provides heatmaps and visual explanations of detection results

### User Interface

- **Drag-and-drop Interface**: Easy image upload through the React web application
- **Real-time Analysis**: Quick processing and visualisation of results
- **Confidence Metrics**: Detailed probability scores and confidence levels
- **Visual Explanations**: GradCAM heatmaps showing which parts of the image influenced the decision

### Chrome Extension

- **Social Media Integration**: Directly analyse images from Twitter/X
- **One-click Analysis**: Simple interface for quick checks
- **Seamless Handoff**: Transfer images to the main application for detailed analysis

## Technology Stack

- **Deep Learning**: PyTorch, TorchVision, EfficientNet, ResNet
- **Backend**: Flask, OpenCV, Numpy
- **Frontend**: React, Material-UI, TailwindCSS
- **Browser Extension**: JavaScript, Chrome Extensions API

## Dataset

The models were trained on a curated dataset of real and fake facial images from:
- https://universe.roboflow.com/deep-facke-detection/deep-fake-detection-xxa8f
- License: CC BY 4.0

## Getting Started

### Backend Setup

1. Navigate to the backend directory:
   ```
   cd backend
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Start the Flask server:
   ```
   python app.py
   ```

### Frontend Setup

1. Navigate to the React app directory:
   ```
   cd deepfake-detection
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Start the development server:
   ```
   npm start
   ```

### Chrome Extension

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked" and select the `chrome-extension` directory

   ```

## License

This project uses datasets under the CC BY 4.0 license. The code is available for educational and research purposes.

## Acknowledgments

- Dataset provided by Roboflow users
- EfficientNet implementation from PyTorch