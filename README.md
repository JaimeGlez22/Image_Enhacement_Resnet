# Image Enhancement Resnet

This repository contains the implementation and evaluation of various convolutional neural network (CNN) architectures for different image enhancement tasks, including denoising, colorization, and super-resolution. Below is an overview of the contents, along with sample results and descriptions of the architectures used.

## Architectures Overview

We use different CNN-based models, mainly ResNet and VGG architectures, to enhance image quality. Below are the architectures implemented:

### ResNetAE and ResNetAE_skipç

These models use ResNet as the backbone for autoencoder structures, with `ResNetAE_skip` incorporating skip connections between the input layer and the output layer for better feature retention.

![ResNetAE Architecture](https://github.com/JaimeGlez22/Image_Enhacement_Resnet/blob/main/auxiliar/resnetAE.png)
![ResNetAE_skip Architecture](https://github.com/JaimeGlez22/Image_Enhacement_Resnet/blob/main/auxiliar/resnetAE_skip.png)

ResNetAE Architecture: The model enhances images by learning compressed representations and then reconstructing them through a decoding process. The skip connections in `ResNetAE_skip` improve detail preservation, particularly in high-frequency regions of the image.

- [`AE_ResNet`](https://github.com/JaimeGlez22/Image_Enhacement_Resnet/tree/main/AE_RESNET): contains the python implementation of the two ResNetAE proposed. The file resnetAE.py contains the python class corresponding to the ResNetAE, meanwhile the python file resnetAE_mod.py contains the implementation of the ResNetAE_skip.

/AE_VGG: contains the python implementation of the VGGAE.

/denoising_color: contains the jupyter notebooks for the training of the model for the denoising task in RGB images and the evaluation results.

/denoising_mnsit: contains the jupyter notebooks for the training of the model for the denoising task for the MNIST dateset and the evaluation results.

/image colorization: contains the jupyter notebooks for the training of the model for the colorization task and the evaluation results.

/superresolution: contains the jupyter notebooks for the training of the model for the super-resolution task  and the evaluation results.
