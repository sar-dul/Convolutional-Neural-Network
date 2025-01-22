# CNN Transfer Learning Repository

This repository contains implementations and experiments with Convolutional Neural Networks (CNNs), focusing on transfer learning for image classification tasks. Below is a detailed overview of the contents and methodology.

## Table of Contents

1. [Overview](#overview)
2. [Implemented Models](#implemented-models)
3. [Datasets](#datasets)
4. [Transfer Learning](#transfer-learning)
5. [Project Structure](#project-structure)
6. [Inference](#inference)
7. [How to Use](#how-to-use)
8. [Results](#results)
9. [Dependencies](#dependencies)

## Overview

This project demonstrates the following workflows:

- Implementation of VGG-11 and AlexNet from scratch.
- Training these models on a subset of the ImageNet dataset with 10 classes.
- Transfer learning for binary classification (cats vs. dogs).
- Transfer learning using pre-trained ResNet and ConvNeXtV2 models from Hugging Face.
- Model inference using Gradio for the frontend and FastAPI for the backend.

## Implemented Models

1. **VGG-11**

   - Built and trained from scratch using PyTorch.
   - Optimized for classification on a 10-class ImageNet subset.

2. **AlexNet**

   - Implemented and trained similarly to VGG-11.

## Datasets

- **ImageNet Subset (10 Classes):** Used for initial training of VGG-11 and AlexNet.
- **Cats vs. Dogs Dataset:** Used for transfer learning experiments.

## Transfer Learning

1. **Custom Models (VGG-11 and AlexNet):**

   - The classification head was modified to handle binary classification (cats vs. dogs).
   - Pre-trained weights from the 10-class ImageNet training were utilized.

2. **Pre-trained ResNet and ConvNeXtV2:**

   - Leveraged models available on Hugging Face.
   - Fine-tuned on the cats vs. dogs dataset.

## Project Structure

- **Root Folder:**

  - `CNN_AlexNet.ipynb`: Implementation of AlexNet from scratch.
  - `CNN_manual_backpropagation.ipynb`: Manual backpropagation implementation.
  - `CNN_VGG.ipynb`: Implementation of VGG-11 from scratch.
  - `Transfer_learning_on_own_AlexNet_model.ipynb`: Transfer learning experiments with custom AlexNet.
  - `Transfer_learning_on_pretrained_models.ipynb`: Transfer learning experiments with pre-trained models (ResNet, ConvNeXtV2).

- **App Folder:**

  - `test_images/`: Sample images for testing.
  - `gradio_UI.py`: Gradio interface for model inference.
  - `inference.py`: Backend logic for inference.
  - `main.py`: FastAPI app entry point.
  - `ResNet_model.pth`: Pre-trained ResNet model file.

## Inference

- **Frontend:**
  - Built using [Gradio](https://gradio.app) for user-friendly interaction.
- **Backend:**
  - Implemented using [FastAPI](https://fastapi.tiangolo.com) for fast and scalable API endpoints.

## How to Use

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Run Inference Server:**
   ```bash
   uvicorn app:app --reload
   ```
3. **Launch Gradio Interface:**
   ```bash
   python gradio_UI.py
   ```

## Results

| Model           | Dataset             | Accuracy |
| --------------- | ------------------- | -------- |
| VGG-11          | ImageNet (10-class) | 62.90%      |
| AlexNet         | ImageNet (10-class) | 74.62%      |
| AlexNet (TL)    | Cats vs. Dogs       | 66.88%      |
| ResNet (TL)     | Cats vs. Dogs       | 97.75%      |
| ConvNeXtV2 (TL) | Cats vs. Dogs       | 85.12%      |

## Dependencies

- PyTorch
- Gradio
- FastAPI
- Hugging Face Transformers

## Contributing

Feel free to fork this repository, submit issues, or create pull requests for improvements and bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

