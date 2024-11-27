# Image Style Transfer Using Convolutional Neural Network 

##  Description
Style transfer is a technique designed to change the appearance of an image while preserving its original content. 
By using a source image and a style reference, It creates an output image that maintains the essential features of the input while incorporating the stylistic details from the selected style image. 
The implementation is based on the paper of Gatys et al. emphasizes the effective transfer of styles from the source image while ensuring that the semantic content of the target image is preserved.
[Image Style Transfer Using Convolutional Neural Networks, Gatys et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)  

## Table of contents
* [Inroduction](#Introducation)
* [Idea Behind](#idea-behind) 
* [Prerequisites](#prerequisites)
* [Installation and Setup](#Installationand_Setup)
* [Output](#OutputVideo)
* [Acknowledgement](#Acknowledgement)

## Introduction
The paper implementation is to explore the concept of combining CNN(Convolutional Neural Network) layers   
for two different images create a new images that blend the layers to one image 
with the aesthetics of well-known artworks.

## Idea Behind:

1- The input image and style images are taken and resized to equal shapes

2- A pre-trained CNN (VGG-16) is loaded.

3- It is known that layers responsible for style (such as basic shapes, colors, etc.) and those responsible for content (image-specific features) 
 can be distinguished, allowing for the separation of layers to work independently on content and style. 

4- The task is then set as an optimization problem where the following losses are minimized: 

- **Content loss**: The distance between the input and output images, ensuring the preservation of content.

- **Style loss**: the distance between the style and output images, ensuring a new style is applied.

- **Total variation loss**: a regularization term used for spatial smoothness to denoise the output image
	
## Prerequisites

- **Python:** Required programming language; ensure version 3.9 or later is installed.
- **Pytorch**: Required framework for DeepLearning models; ensure version 1.8.0 or later.
- **Conda Virtual Environment (optional):** To manage dependencies for running python packages.

## Installation and Setup

1. **Clone the Repository:**

	```sh
	git clone https://github.com/OmarEl-Khashab/Image-Style-Transfer-using-CNN.git
	cd Style_Transfer
	```

2.  **Add your image paths :**

	Add your images path in the Training.py:

	```
	base_path = "/Style_Transfer/"
	content_image = "/Style_Transfer/image/{content.png}"
	style_image = "/Style_Transfer/image/{style.png}"
	```
3. Start Training your Model by running:

 	```sh
	  python Training.py
	```

## Output Video is Uploaded on Linkedin 
Check the Results:
https://www.linkedin.com/feed/update/urn:li:activity:6760321850833690624/

## Acknowledgement

This is a big thanks for (https://github.com/AbdallaGomaa) for guidance in my is Machine learning Projects.

Feel free to contribute reach out for the project by opening issues or submitting pull requests. If you have any questions, contact me at omar_khashab11@hotmail.com

