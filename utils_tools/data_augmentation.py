from PIL import Image, ImageOps, ImageEnhance
import numpy as np

def rotate_image(image, angle):
    return image.rotate(angle)

def scale_image(image, scale_factor):
    width, height = image.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return image.resize((new_width, new_height))

def flip_image(image, direction='horizontal'):
    if direction == 'horizontal':
        return ImageOps.mirror(image)
    elif direction == 'vertical':
        return ImageOps.flip(image)

def sharpen_image(image, factor=2.0):
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)

def crop_image(image, crop_box):
    return image.crop(crop_box)

if __name__ == "__main__":
    # Load an example image
    image_path = 'path/to/your/image.jpg'
    image = Image.open(image_path)

    # Rotate the image by 45 degrees
    rotated_image = rotate_image(image, 45)
    rotated_image.show()

    # Scale the image by a factor of 0.5 (reduce size by half)
    scaled_image = scale_image(image, 0.5)
    scaled_image.show()

    # Flip the image horizontally
    flipped_image_h = flip_image(image, 'horizontal')
    flipped_image_h.show()

    # Flip the image vertically
    flipped_image_v = flip_image(image, 'vertical')
    flipped_image_v.show()

    # Sharpen the image
    sharpened_image = sharpen_image(image, 2.0)
    sharpened_image.show()

    # Crop the image (left, upper, right, lower)
    crop_box = (50, 50, 200, 200)
    cropped_image = crop_image(image, crop_box)
    cropped_image.show()
