# Image Enhancement Resnet

This repository contains the implementation and evaluation of various convolutional neural network (CNN) architectures for different image enhancement tasks, including denoising, colorization, and super-resolution. Below is an overview of the contents, along with sample results and descriptions of the architectures used.

## Architectures Overview

We use different CNN-based models, mainly ResNet and VGG architectures, to enhance image quality. Below are the architectures implemented:

### ResNetAE and ResNetAE_skip

These models use ResNet as the backbone for autoencoder structures, with _ResNetAE_skip_ incorporating skip connections between the input layer and the output layer for better feature retention.

<div align="center">
    <img src="https://github.com/JaimeGlez22/Image_Enhacement_Resnet/blob/main/auxiliar/EN_resnetAE_background.png" alt="ResNetAE Architecture" width="90%">
    <p><em>ResNetAE Architecture</em></p>
</div>

<div align="center">
    <img src="https://github.com/JaimeGlez22/Image_Enhacement_Resnet/blob/main/auxiliar/EN_resnetAE_skip_bg.png" alt="ResNetAE_skip Architecture" width="50%">
    <p><em>ResNetAE_skip Architecture</em></p>
</div>


ResNetAE Architecture: The model enhances images by learning compressed representations and then reconstructing them through a decoding process. The skip connections in _ResNetAE_skip_ improve detail preservation, particularly in high-frequency regions of the image.

### VGGAE
A modified VGG architecture is used to enhance images, taking advantage of its depth for detailed feature extraction.

<div align="center">
    <img src="https://github.com/JaimeGlez22/Image_Enhacement_Resnet/blob/main/auxiliar/EN_vggAE_background.png" alt="VGGAE Architecture" width="90%">
    <p><em>VGGAE Architecture</em></p>
</div>

_VGGAE Architecture_: Designed for image enhancement tasks by leveraging the deep feature extraction capabilities of VGG.

____

## Image Enhancement Tasks

### 1. Image Denoising (RGB and MNIST)

We employ the implemented models to reduce noise in images, imporving clarity while preserving important details. The following examples demostrate the denosing process apllied to both color and grayscale images.

#### MNIST Denoising

_Before and After Denoising on MNIST_: Enhancing the clarity of grayscale images while preserving the shape of digits. Using the MNIST dataset. More info [here](https://github.com/JaimeGlez22/Image_Enhacement_Resnet/tree/main/denoising_mnist).

<div align="center">
    <img src="https://github.com/JaimeGlez22/Image_Enhacement_Resnet/blob/main/denoising_mnist/figures/comparation.png" alt="mnist denoising" width="90%">
    <p><em>Example of color image denoising</em></p>
</div>

#### RGB Denoising 

_Before and After Denoising on RGB Images_: The model effectively removes noise while maintaining color and detail integrity. Using the Thumbanils 128x128 dataset. More info [here](https://github.com/JaimeGlez22/Image_Enhacement_Resnet/tree/main/denoising_color).

<div align="center">
    <img src="https://github.com/JaimeGlez22/Image_Enhacement_Resnet/blob/main/denoising_color/figures/comparation.png" alt="rgb denoising" width="90%">
    <p><em>Example of color image denoising</em></p>
</div>

---

### 2. Image Colorization

Using the deep learning models, we restore colors in grayscale images predicting the color values for each pixel. Using the Landscape Image dataset. More info [here](https://github.com/JaimeGlez22/Image_Enhacement_Resnet/tree/main/image%20colorization).

_Before and After Colorization_: The model predicts accurate color representations, giving life to grayscale images.

<div align="center">
    <img src="https://github.com/JaimeGlez22/Image_Enhacement_Resnet/blob/main/image%20colorization/figures/comparation_AE.png" alt="image colorization" width="90%">
    <p><em>Example of image colorization using ResNetAE</em></p>
</div>

<div align="center">
    <img src="https://github.com/JaimeGlez22/Image_Enhacement_Resnet/blob/main/image%20colorization/figures/comparation_mod.png" alt="Skip image colorization" width="90%">
    <p><em>Example of image colorization using ResNetAE_skip</em></p>
</div>

---

### 3. Super-Resolution

Our super-resolution models generate high-resolution versions of low-resolution images by learning to upscale while preserving fine details. Using the Labeled Faces in the Wild dataset. More info [here](https://github.com/JaimeGlez22/Image_Enhacement_Resnet/tree/main/superresolution)

_Before and After Super-Resolution_: The model sharpens and upscales images, enhancing resolution without significant quality loss.

<div align="center">
    <img src="https://github.com/JaimeGlez22/Image_Enhacement_Resnet/blob/main/superresolution/figures/comparation.png" alt="superresolution" width="90%">
    <p><em>Example of Super-Resolution</em></p>
</div>

---

## Folder Structure

- [AE_ResNet](https://github.com/JaimeGlez22/Image_Enhacement_Resnet/tree/main/AE_RESNET): contains the Python implementation of the two proposed ResNetAE models (resnetAE.py and resnetAE_skip.py).

- [AE_VGG](https://github.com/JaimeGlez22/Image_Enhacement_Resnet/tree/main/AE_VGG): contains the Python implementation of VGGAE.

- [denoising_color](https://github.com/JaimeGlez22/Image_Enhacement_Resnet/tree/main/denoising_color): Jupyter notebooks for RGB image denoising.

- [denoising_mnsit](https://github.com/JaimeGlez22/Image_Enhacement_Resnet/tree/main/denoising_mnist): Jupyter notebooks for MNIST image denoising.

- [image colorization](https://github.com/JaimeGlez22/Image_Enhacement_Resnet/tree/main/image%20colorization): Jupyter notebooks for image colorization.

- [superresolution](https://github.com/JaimeGlez22/Image_Enhacement_Resnet/tree/main/superresolution): Jupyter notebooks for super-resolution tasks.
