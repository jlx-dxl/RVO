import os
from PIL import Image

def create_gif_from_folder(input_folder, output_gif_path, duration=100):
    """
    Create a GIF from all images in a folder.

    Args:
        input_folder (str): Path to the folder containing images.
        output_gif_path (str): Path to save the resulting GIF.
        duration (int): Duration between frames in milliseconds.
    """
    # Get list of image files, sorted by filename
    image_files = sorted([
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not image_files:
        raise ValueError("No images found in the specified folder.")

    # Load all images
    images = [Image.open(f) for f in image_files]

    # Save as GIF
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

    print(f"GIF created successfully at {output_gif_path}")

# Example usage
if __name__ == "__main__":
    input_folder = f"./frames_circle_with_obstacles_20"
    output_gif_path = f"circle_with_obstacles_20.gif"
    # Create GIF from the specified folder
    print(f"Creating GIF from {input_folder}...")
    create_gif_from_folder(input_folder, output_gif_path, duration=10)