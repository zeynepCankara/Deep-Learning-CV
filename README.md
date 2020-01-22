# NTU Deep Learning for Computer Vision (Fall 2019)

Course website: http://vllab.ee.ntu.edu.tw/dlcv.html

The course introduces interesting deep learning approaches to various computer vision problems. This repository contains my solutions to the homework assignments together with the final project for the DLVC course that I took in National Taiwan University.


## Homework Assignments:

- `Assignment 1`: PCA method for dimension reduction, Bag of Words for image recognition, K-nearest Neighbors 
    - `Topics`: PCA, Bag of Words, K-nearest Neighbors, Bag of Words

- `Assignment 2`: Designing various CNN models using ResNet backbone for Image Segmentation problems.
    - `Topics`: Object recognition, CNNs in PyTorch, Image Segmentation, Fine-tuning the CNN models.

- `Assignment 3`: Implementing DANN, ACGAN, GAN for Image Generation, Feature Disentanglement and Domain Adaptation.
    - `Topics`: Advesarial Neural Networks, Feature Disentanglement, Domain Adaptation with Generative Advesarial Networks.

- `Assignment 4`: Trimmed action recognition and temporal action segmentation in full-length videos.
    - `Topics`: Recurrent Neural Networks architectures, video pre-processing.


## Final Project (Chosen as the Best Project):

- `Dunhuang Image Restoration`:
- The mural paintings from Dunhuang caves suffering from corrosion and aging. 
- We used image inpainting, a task of synthesizing contents in the missing regions to generate images which are as close as possible to the original image. 
- Ground truth images are encoded in Adobe RGB, while the input masked images are in sRGB. The encoding difference creates some ripples that decrease the validation accuracy. We propose a weighted average methodology to smooth out the final result.
