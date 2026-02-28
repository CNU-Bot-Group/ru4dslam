import os
from PIL import Image
import re


def create_gif_from_directory(directory_path, output_filename, duration=100, online=True):
    """
    Creates a GIF from all PNG images in a given directory.

    :param directory_path: Path to the directory containing PNG images.
    :param output_filename: Output filename for the GIF.
    :param duration: Duration of each frame in the GIF (in milliseconds).
    """
    # Function to extract the number from the filename
    def extract_number(filename):
        # Pattern to find a number followed by '.png'
        match = re.search(r'(\d+)\.png$', filename)
        if match:
            return int(match.group(1))
        else:
            return None


    if online:
        # Get all PNG files in the directory
        image_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.png')]

        # Sort the files based on the number in the filename
        image_files.sort(key=extract_number)
    else:
        # Get all PNG files in the directory
        image_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.png')]

        # Sort the files based on the number in the filename
        image_files.sort()

    # Load images
    images = [Image.open(file) for file in image_files]

    # Convert images to the same mode and size for consistency
    images = [img.convert('RGBA') for img in images]
    base_size = images[0].size
    resized_images = [img.resize(base_size, Image.LANCZOS) for img in images]

    # Save as GIF
    resized_images[0].save(output_filename, save_all=True, append_images=resized_images[1:], optimize=False, duration=duration, loop=0)