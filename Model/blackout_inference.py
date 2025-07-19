# -*- encoding: utf-8 -*-
"""
Script to analyze model inference results and count blackout images

This script processes each .pth model file in the pretrain folder, matches it with the
corresponding Test-DATASET-KFOLD folder in the inference directory, and analyzes images
to determine if they're blackout (mostly black) based on a threshold.

Results are saved to a text file showing statistics for each model-dataset-fold combination.
"""

import os
import re
import glob
import argparse

import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from lib import utils, models, testers


def extract_model_info(pth_file):
    """
    Extract model name, dataset, and kfold from .pth filename
    
    Args:
        pth_file (str): Path to .pth file (e.g., 'LiSANet-1000-K1.pth')
        
    Returns:
        tuple: (model_name, dataset, kfold)
    """
    filename = os.path.basename(pth_file)
    pattern = r'([A-Za-z]+)-([0-9]+)-K([0-9]+)\.pth'
    match = re.match(pattern, filename)
    
    if match:
        model_name = match.group(1)
        dataset = match.group(2)
        kfold = match.group(3)
        return model_name, dataset, kfold
    else:
        raise ValueError(f"Could not parse model info from filename: {filename}")


def find_matching_test_folder(dataset, kfold):
    """
    Find the matching Test-DATASET-KFOLD folder in the inference directory
    
    Args:
        dataset (str): Dataset identifier (e.g., '1000')
        kfold (str): K-fold identifier (e.g., '1')
        
    Returns:
        str: Path to matching test folder
    """
    test_folder = os.path.join(os.path.dirname(__file__), 'inference', f'Test-{dataset}-K{kfold}')
    if os.path.exists(test_folder):
        return test_folder
    else:
        raise ValueError(f"Could not find matching test folder: Test-{dataset}-K{kfold}")


def is_blackout_image(image_path, threshold=0.95):
    """
    Determine if an image is a blackout image (more than threshold percentage of black pixels)
    
    Args:
        image_path (str): Path to image file
        threshold (float): Threshold for black pixel percentage (default: 0.95)
        
    Returns:
        bool: True if image is blackout, False otherwise
    """
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is not None:
        # Calculate percentage of black pixels (assuming black is 0)
        total_pixels = img.size
        black_pixels = np.sum(img == 0)
        black_percentage = black_pixels / total_pixels
        
        return black_percentage >= threshold
    
    return False


def process_model(pth_file, params, threshold=0.95):
    """
    Process a model file, perform inference on images, and count blackout images
    
    Args:
        pth_file (str): Path to .pth model file
        params (dict): Model parameters
        threshold (float): Threshold for black pixel percentage
        
    Returns:
        tuple: (model_name, dataset, kfold, blackout_count, total_count)
    """
    # Extract model info from filename
    model_name, dataset, kfold = extract_model_info(pth_file)
    
    # Find matching test folder
    test_folder = find_matching_test_folder(dataset, kfold)
    
    # Update parameters
    params["model_name"] = model_name
    params["pretrain"] = pth_file
    
    # Initialize model
    model = models.get_model(params)
    
    # Initialize tester
    tester = testers.get_tester(params, model)
    
    # Load model weights
    tester.load()
    
    # Get all image files (not segmentation files)
    image_files = [f for f in os.listdir(test_folder) if not f.endswith('_segmentation.jpg') and f.endswith('.jpg')]
    
    blackout_count = 0
    total_count = len(image_files)
    
    for image_file in image_files:
        image_path = os.path.join(test_folder, image_file)
        
        # Perform inference to generate segmentation
        segmentation_path = tester.inference(image_path)
        
        # Check if segmentation is blackout
        if is_blackout_image(segmentation_path, threshold):
            blackout_count += 1
    
    return model_name, dataset, kfold, blackout_count, total_count


def get_params_for_dataset(dataset_name):
    """
    Get parameters for the specified dataset
    
    Args:
        dataset_name (str): Dataset name
        
    Returns:
        dict: Parameters for the dataset
    """
    # ISIC-2018 parameters
    params = {
        # Launch Initialization
        "CUDA_VISIBLE_DEVICES": "0",
        "seed": 1777777,
        "cuda": True,
        "benchmark": False,
        "deterministic": True,
        # Preprocessing
        "resize_shape": (224, 224),
        # Data Augmentation
        "normalize_means": (0.50297405, 0.54711632, 0.71049083),
        "normalize_stds": (0.18653496, 0.17118206, 0.17080363),
        # Data Loading
        "dataset_name": "ISIC-2018",
        "dataset_path": r"./datasets/ISIC-2018",
        "batch_size": 32,
        "num_workers": 2,
        # Model
        "model_name": None,  # Will be set based on .pth file
        "in_channels": 3,
        "classes": 2,
        "index_to_class_dict": {
            0: "background",
            1: "foreground"
        },
        "resume": None,
        "pretrain": None,  # Will be set based on .pth file
        # Device
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    }
    
    return params


def main():
    parser = argparse.ArgumentParser(description="Analyze model inference results and count blackout images")
    parser.add_argument("--threshold", type=float, default=0.95, 
                        help="Threshold for black pixel percentage (default: 0.95)")
    parser.add_argument("--output", type=str, default="blackout_results.txt",
                        help="Output file to save results (default: blackout_results.txt)")
    args = parser.parse_args()
    
    # Set up environment
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # Get all .pth files in pretrain folder
    pretrain_dir = os.path.join(os.path.dirname(__file__), 'pretrain')
    pth_files = glob.glob(os.path.join(pretrain_dir, '*.pth'))
    
    if not pth_files:
        print("No .pth files found in pretrain directory")
        return
    
    # Get parameters for dataset
    params = get_params_for_dataset("ISIC-2018")
    
    # Process each model and collect results
    results = []
    for pth_file in pth_files:
        try:
            print(f"Processing {os.path.basename(pth_file)}...")
            model_name, dataset, kfold, blackout_count, total_count = process_model(
                pth_file, params, args.threshold)
            results.append((model_name, dataset, kfold, blackout_count, total_count))
            print(f"{model_name}-{dataset}-K{kfold}: {blackout_count}/{total_count} images are black")
        except Exception as e:
            print(f"Error processing {os.path.basename(pth_file)}: {str(e)}")
    
    # Save results to file
    output_path = os.path.join(os.path.dirname(__file__), args.output)
    with open(output_path, 'w') as f:
        for model_name, dataset, kfold, blackout_count, total_count in results:
            f.write(f"{model_name}-{dataset}-K{kfold}: {blackout_count}/{total_count} images are black\n")
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()