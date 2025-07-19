# -*- encoding: utf-8 -*-
"""
Script to count segmentation images that are more than 95% black
"""

import os
import cv2
import numpy as np

def count_blackout_images(directory_path, threshold=1):
    """Count images with more than threshold percentage of black pixels

    Args:
        directory_path (str): Path to directory containing segmentation images
        threshold (float): Threshold for black pixel percentage (default: 0.95)

    Returns:
        tuple: (count of blackout images, total count of images, list of blackout image names)
    """
    blackout_count = 0
    total_count = 0
    blackout_images = []

    # Get all segmentation images
    for filename in os.listdir(directory_path):
        if filename.endswith('_segmentation.jpg'):
            total_count += 1
            image_path = os.path.join(directory_path, filename)
            
            # Read image in grayscale
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                # Calculate percentage of black pixels (assuming black is 0)
                total_pixels = img.size
                black_pixels = np.sum(img == 0)
                black_percentage = black_pixels / total_pixels

                if black_percentage >= threshold:
                    blackout_count += 1
                    blackout_images.append(filename)

    return blackout_count, total_count, blackout_images

def main():
    # Path to the inference/images directory
    images_dir = os.path.join(os.path.dirname(__file__), 'inference', 'images')
    
    # Count blackout images
    blackout_count, total_count, images = count_blackout_images(images_dir)
    
    print(f"{blackout_count} out of {total_count} images are blank (more than 95% black)")
    if blackout_count > 0:
        print("\nBlackout images:")
        for img in images:
            print(f"- {img}")

if __name__ == '__main__':
    main()