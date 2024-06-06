import numpy as np
from PIL import Image

def load_image(image_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        return np.array(img)
    

def resize_image(image, size=(256, 256)):
    img_pil = Image.fromarray(image)
    img_resized = img_pil.resize(size, Image.LANCZOS)
    return np.array(img_resized)

def normalize_image(image):
    return (image / 255.0).astype(np.float32)

def process_image(image_path):
    image = load_image(image_path)
    image_resized = resize_image(image, size=(256, 256))
    image_normalized = normalize_image(image_resized)
    return image_normalized