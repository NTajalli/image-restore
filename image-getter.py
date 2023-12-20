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

def download_and_process_images(search_term, subscription_key, total_images=150, output_folder='downloaded_images', vintage_folder='vintage_images'):
    # Ensure output folders exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(vintage_folder):
        os.makedirs(vintage_folder)

    # Bing Image Search API setup
    search_url = "https://api.bing.microsoft.com/v7.0/images/search"
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    downloaded_count = 0
    offset = 0
    downloaded_urls = set()  # Set to store downloaded URLs

    while downloaded_count < total_images:
        params = {"q": search_term, "count": 150, "offset": offset}
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()

        for result in search_results["value"]:
            if downloaded_count >= total_images:
                break

            image_url = result["contentUrl"]
            if image_url in downloaded_urls:
                continue  # Skip if the image has already been downloaded

            try:
                image_data = requests.get(image_url)
                image_data.raise_for_status()
                image = Image.open(BytesIO(image_data.content))
                image = crop_to_square(image).resize((256, 256), Image.Resampling.LANCZOS)

                color_image_filename = f"{search_term.replace(' ', '_')}_{downloaded_count + 1}.jpg"
                color_image_path = os.path.join(output_folder, color_image_filename)

                vintage_image_filename = f"{search_term.replace(' ', '_')}_{downloaded_count + 1}.jpg"
                vintage_image_path = os.path.join(vintage_folder, vintage_image_filename)

                if not os.path.exists(color_image_path) and not os.path.exists(vintage_image_path):
                    image.save(color_image_path, "JPEG")
                    vintage_image = apply_vintage_effects(image)
                    vintage_image.save(vintage_image_path, "JPEG")
                    downloaded_count += 1
                    downloaded_urls.add(image_url)
            except Exception as e:
                print(f"Error downloading or processing image: {e}")

        offset += 150

def crop_to_square(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

def main():
    SUBSCRIPTION_KEY = "78865a655c24410281f00b8ef08951ca"
    SEARCH_TERMS = [
        'modern political rally', 'contemporary inauguration ceremony', 'modern historical reenactment', 
        'current military parade', 'recent public speeches', 'contemporary historical celebration', 
        'modern sports event', 'recent Olympic games', 'contemporary athletes in action', 
        'modern football match', 'recent basketball game', 'contemporary tennis tournament',
        'landscape photography', 'wildlife in nature', 'modern national parks', 
        'contemporary nature reserve', 'recent nature exploration', 'beautiful natural scenery',
        'recent technological advancements', 'modern scientific discoveries', 'contemporary art exhibition', 
        'recent cultural festivals', 'modern space exploration', 'significant current events',
        'modern celebrity portraits', 'recent red carpet events', 'contemporary musicians in concert', 
        'modern actors and actresses', 'recent celebrity gatherings', 'contemporary public figures',
        'iconic photographs of the 21st century', 'famous modern artworks', 'notable contemporary photographs', 
        'renowned photographers modern work', 'famous photojournalism today', 'acclaimed modern photography'
    ]
    for term in SEARCH_TERMS:
        download_and_process_images(term, SUBSCRIPTION_KEY)

if __name__ == "__main__":
    main()

