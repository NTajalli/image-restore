# Colorizing Black and White Images with Generative Adversarial Networks (GANs)

## Overview

This project implements a Generative Adversarial Network (GAN) to restore color to black and white images. The model is designed to take a grayscale image as input and produce a full-color image as output. The project includes scripts for data gathering, dataset creation, model building, loss computation, training, and visualization of results. The entire training process is designed to run on AWS EC2 instances, utilizing their computational resources effectively.

## Project Structure

- **COCO-getter.py**: Script to gather the dataset from the COCO dataset repository. It automates the downloading and preprocessing of images needed for training.
- **dataset.py**: Handles the creation of the dataset, including loading images, converting them to grayscale, and preparing them for training the GAN.
- **image-getter.py**: Contains logic for retrieving and processing images for use in the model.
- **models.py**: Defines the architecture of the GAN, including the generator and discriminator models.
- **loss.py**: Implements the custom loss functions used during training.
- **training_script.py**: Script to train the GAN on the dataset. Includes logic for periodically saving the model, displaying loss curves, and showing image outputs.
- **utils.py**: Contains utility functions used across the project, such as image visualization, data augmentation, and metric calculations.

## Requirements

- Python 3.8 or higher
- PyTorch (for deep learning model implementation)
- AWS EC2 instance with GPU support (for training)
- COCO dataset or any other suitable dataset of black and white images

## Setup

1. **Clone the repository**:

    `git clone https://github.com/NTajalli/image-restore.git`

    `cd image-restore`

2. **Install dependencies**:

    `pip install -r requirements.txt`

3. **Download and prepare the dataset**:

    Run the `COCO-getter.py` script to download and preprocess the COCO dataset, or modify the script to work with your dataset.

    `python COCO-getter.py`

4. **Create the dataset**:

    Run the `dataset.py` script to process the images and prepare them for training.

    `python dataset.py`

## Model Architecture

The GAN consists of two primary components:

- **Generator**: Takes a grayscale image as input and generates a color image. The architecture is designed to capture the spatial and semantic information from the grayscale image and translate it into realistic colors.
- **Discriminator**: Evaluates the authenticity of the generated color image by distinguishing between real and fake images. This adversarial setup ensures that the generator produces high-quality, realistic images.

## Loss Functions

Custom loss functions are implemented in `loss.py`, which includes a combination of adversarial loss (to ensure realism) and pixel-wise loss (to ensure the colorized image closely matches the ground truth). This combination helps the model learn both global structure and fine details.

## Training

The training process is handled by the `training_script.py`. Key features of the training script include:

- Periodic saving of the model to allow for resuming training or deployment.
- Visualization of loss curves to monitor training progress.
- Intermediate image outputs to visualize how the model's colorization improves over time.

To start the training, run:

`python training_script.py`

The script is designed to be run on the cloud with GPU support to accelerate the training process. It outputs the trained model, loss curves, and sample colorized images.

## Results

The GAN can effectively colorize grayscale images, producing outputs that are visually appealing and closely resemble the original color images. The results improve as the model trains, with the adversarial loss driving the generator to produce more realistic colors. It does have some trouble completing capturting complex features and it is only limited to a set size of image. 

## Conclusion

This project demonstrates the power of GANs in image restoration tasks, specifically in the context of colorizing black and white images. The modular nature of the codebase allows for easy experimentation with different datasets, model architectures, and loss functions, making it a flexible tool for various image processing tasks.

## Future Work

- **Fine-tuning**: Experiment with different architectures, loss functions, and hyperparameters to further improve the quality of colorized images. I tried adding SelfAttention, but so far have been unsuccessful. Hopefully I can experiment and add this optimally. 
- **Generalization**: Test the model on a wider variety of images to assess its ability to generalize to different styles and content.
- **Deployment**: Explore deploying the model as a web service or integrating it into an application.

---

Feel free to reach out for any questions or collaboration opportunities!
