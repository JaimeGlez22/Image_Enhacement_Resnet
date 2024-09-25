# Image Enhancement Resnet

This repository contains the implementation and evaluation of various convolutional neural network (CNN) architectures for different image enhancement tasks, including denoising, colorization, and super-resolution. Below is an overview of the contents, along with sample results and descriptions of the architectures used.

## Architectures Overview

We use different CNN-based models, mainly ResNet and VGG architectures, to enhance image quality. Below are the architectures implemented:

### ResNetAE and ResNetAE_skip

These models use ResNet as the backbone for autoencoder structures, with `ResNetAE_skip` incorporating skip connections between the input layer and the output layer for better feature retention.

<div align="center">
    <img src="https://github.com/JaimeGlez22/Image_Enhacement_Resnet/blob/main/auxiliar/EN_resnetAE_background.png" alt="ResNetAE Architecture" width="90%">
    <p><em>ResNetAE Architecture</em></p>
</div>

<div align="center">
    <img src="https://github.com/JaimeGlez22/Image_Enhacement_Resnet/blob/main/auxiliar/EN_resnetAE_skip_bg.png" alt="ResNetAE_skip Architecture" width="50%">
    <p><em>ResNetAE_skip Architecture</em></p>
</div>


ResNetAE Architecture: The model enhances images by learning compressed representations and then reconstructing them through a decoding process. The skip connections in `ResNetAE_skip` improve detail preservation, particularly in high-frequency regions of the image.

### VGGAE
A modified VGG architecture is used to enhance images, taking advantage of its depth for detailed feature extraction.

<div align="center">
    <img src="https://github.com/JaimeGlez22/Image_Enhacement_Resnet/blob/main/auxiliar/EN_vggAE_background.png" alt="VGGAE Architecture" width="90%">
    <p><em>VGGAE Architecture</em></p>
</div>

`VGGAE Architecture`: Designed for image enhancement tasks by leveraging the deep feature extraction capabilities of VGG.
____
## Image Enhancement Tasks

### 1. Image Denoising (RGB and MNIST)

We employ the implemented models to reduce noise in images, imporving clarity while preserving important details. The following examples demostrate the denosing process apllied to both color and grayscale images.

#### MNIST Denoising

`Before and After Denoising on MNIST`: Enhancing the clarity of grayscale images while preserving the shape of digits. Using the MNIST dataset. More info [`here`]("https://github.com/JaimeGlez22/Image_Enhacement_Resnet/tree/main/denoising_mnist")

<div align="center">
    <img src="https://github.com/JaimeGlez22/Image_Enhacement_Resnet/blob/main/denoising_mnist/figures/comparation.png" alt="VGGAE Architecture" width="90%">
    <p><em>Example of color image denoising</em></p>
</div>

#### RGB Denoising 

`Before and After Denoising on RGB Images`: The model effectively removes noise while maintaining color and detail integrity. Using the Thumbanils 128x128 dataset. More info [`here`]("https://github.com/JaimeGlez22/Image_Enhacement_Resnet/tree/main/denoising_color")

<div align="center">
    <img src="https://github.com/JaimeGlez22/Image_Enhacement_Resnet/blob/main/denoising_color/figures/comparation.png" alt="VGGAE Architecture" width="90%">
    <p><em>Example of color image denoising</em></p>
</div>


- [`AE_ResNet`](https://github.com/JaimeGlez22/Image_Enhacement_Resnet/tree/main/AE_RESNET): contains the python implementation of the two ResNetAE proposed. The file resnetAE.py contains the python class corresponding to the ResNetAE, meanwhile the python file resnetAE_mod.py contains the implementation of the ResNetAE_skip.

/AE_VGG: contains the python implementation of the VGGAE.

/denoising_color: contains the jupyter notebooks for the training of the model for the denoising task in RGB images and the evaluation results.

/denoising_mnsit: contains the jupyter notebooks for the training of the model for the denoising task for the MNIST dateset and the evaluation results.

/image colorization: contains the jupyter notebooks for the training of the model for the colorization task and the evaluation results.

/superresolution: contains the jupyter notebooks for the training of the model for the super-resolution task  and the evaluation results.
