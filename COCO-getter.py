import fiftyone as fo
import fiftyone.zoo as foz
import os
from PIL import Image
import os
import requests
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageChops
from io import BytesIO
import random

def apply_vintage_effects(image):
    effects = [apply_sepia, apply_contrast, apply_noise, apply_blur]
    image = apply_grayscale(image)
    effect = random.choice(effects)
    return effect(image)

def apply_sepia(image):
    sepia_filter = Image.new("RGB", image.size, (255, 240, 192))
    return Image.blend(image.convert("RGB"), sepia_filter, 0.2)

def apply_grayscale(image):
    return ImageOps.grayscale(image)

def apply_contrast(image):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(2)

def apply_noise(image):
    # Adding random noise
    noise = Image.effect_noise(image.size, 10)
    return ImageChops.add(image, noise, 2, 0)

def apply_blur(image):
    return image.filter(ImageFilter.GaussianBlur(radius=1))


original_dir = './downloaded_images'
vintage_dir = './vintage_images'
os.makedirs(original_dir, exist_ok=True)
os.makedirs(vintage_dir, exist_ok=True)

# Define the number of samples you want to download from each split
max_samples_per_split = 10000  # Adjust this number based on your requirement

# Download and process a subset of COCO dataset
splits = ["train", "validation", "test"]
for split in splits:
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split=split,
        max_samples=max_samples_per_split,
        shuffle=True
    )

    for sample in dataset.take(max_samples_per_split):
        # Load the image
        image_path = sample.filepath
        image = Image.open(image_path)

        # Save the original image
        original_image_path = os.path.join(original_dir, os.path.basename(image_path))
        image.save(original_image_path)

        # Apply the vintage filter and save
        vintage_image = apply_vintage_effects(image)
        vintage_image_path = os.path.join(vintage_dir, os.path.basename(image_path))
        vintage_image.save(vintage_image_path)

        print(f"Processed {image_path}")
