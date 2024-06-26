# Art Generation with Neural Style Transfer

This project demonstrates the implementation of **Neural Style Transfer (NST)** to create stunning artworks by combining the content of one image with the style of another. Neural Style Transfer is a deep learning technique that uses convolutional neural networks to merge the style of a reference image with the content of another image.


<div align="center">
<img src="images\art.png" style="width:750px;"> <br>
</div>

![Deep Learning](https://img.shields.io/badge/Skill-Deep%20Learning-yellow)
![Convolutional Neural Networks](https://img.shields.io/badge/Skill-Convolutional%20Neural%20Networks-blueviolet)
![TensorFlow](https://img.shields.io/badge/Skill-TensorFlow-orange)
![Keras](https://img.shields.io/badge/Skill-Keras-yellow)
![Computer Vision](https://img.shields.io/badge/Skill-Computer%20Vision-brightblue)
![Image Processing](https://img.shields.io/badge/SkillImage%20Processing-brightblue)
![Python Programming](https://img.shields.io/badge/Skill-Python%20Programming-orange)

## Project Architecture

**Input Processing:** Load and preprocess content and style images.
**Model Architecture:** Use a pre-trained VGG19 network to extract features.
**Loss Calculation:** Compute content loss, style loss, and total variation loss.
**Optimization:** Minimize the total loss using an optimizer to generate the final image.

## Frameworks and Libraries
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-orange.svg?style=flat&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-3.3.3-red.svg?style=flat&logo=keras)
![NumPy](https://img.shields.io/badge/NumPy-1.26.4-blue.svg?style=flat&logo=numpy)
![SciPy](https://img.shields.io/badge/SciPy-1.10.1-yellow.svg?style=flat&logo=SciPy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.6.2-green.svg?style=flat&logo=matplotlib)

## Key Features
- Preprocessing of images for face detection and alignment
- Feature extraction using a pre-trained CNN.
- Face verification and identification

## Usage
**Clone the repository:**
```bash
git clone https://github.com/yourusername/Art_Generation_with_Neural_Style_Transfer.git
```
**Navigate to the project directory:**
```bash
cd Art_Generation_with_Neural_Style_Transfer
```
**Install the required dependencies:**
```bash
pip install -r requirements.txt
```
**Run the Jupyter Notebook to see the implementation:**
```bash
jupyter notebook Art_Generation_with_Neural_Style_Transfer.ipynb
```
## Implementation
The project implements Neural Style Transfer by leveraging a pre-trained VGG19 model. The VGG19 model, pre-trained on ImageNet, is used to extract features from the content and style images. The loss function is computed as a combination of content loss, style loss, and total variation loss. An optimizer is then used to iteratively update the generated image to minimize this loss.


## Results

As you can see the output of the project is a series of images where the style of the reference image is applied to the content of the original image, resulting in aesthetically pleasing artwork.


<div align="center">
<img src="images\art.png" style="width:750px;"> <br>

<center><img src="images/perspolis_vangogh.png" style="width:750px;height:300px;"></center>

<center><img src="images/pasargad_kashi.png" style="width:750px;height:300px;"></center>

<center><img src="images/circle_abstract.png" style="width:750px;height:300px;"></center>
</div>
