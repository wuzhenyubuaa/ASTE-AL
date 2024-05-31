import random
import numpy as np
from PIL import Image


def divide_image(image, N):
    width, height = image.size
    patch_width = width // N
    patch_height = height // N

    patches = []
    for i in range(N):
        for j in range(N):
            left = j * patch_width
            upper = i * patch_height
            right = left + patch_width
            lower = upper + patch_height
            patch = image.crop((left, upper, right, lower))
            patches.append(patch)

    return patches, patch_width, patch_height


def merge_image(patches, N, patch_width, patch_height):
    width = N * patch_width
    height = N * patch_height
    merged_image = Image.new('RGB', (width, height))

    for i in range(N):
        for j in range(N):
            patch = patches[i * N + j]
            merged_image.paste(patch, (j * patch_width, i * patch_height))

    return merged_image


def randomize_patches(patches):
    patches_copy = patches[:]
    random.shuffle(patches_copy)
    return patches_copy


def process_images(natural_image_path, mask_image_path, N, randomize=False):
    natural_image = Image.open(natural_image_path)
    mask_image = Image.open(mask_image_path)

    natural_patches, patch_width, patch_height = divide_image(natural_image, N)
    mask_patches, _, _ = divide_image(mask_image, N)

    if randomize:
        natural_patches = randomize_patches(natural_patches)
        mask_patches = randomize_patches(mask_patches)

    merged_natural_image = merge_image(natural_patches, N, patch_width, patch_height)
    merged_mask_image = merge_image(mask_patches, N, patch_width, patch_height)

    return merged_natural_image, merged_mask_image


# Example usage
natural_image_path = 'path/to/your/natural_image.jpg'
mask_image_path = 'path/to/your/mask_image.jpg'
N = 4  # Number of patches per dimension
randomize = True  # Set to False to disable randomization

merged_natural_image, merged_mask_image = process_images(natural_image_path, mask_image_path, N, randomize)

# Display the results
merged_natural_image.show(title="Merged Natural Image")
merged_mask_image.show(title="Merged Mask Image")
