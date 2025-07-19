import os
import argparse
import glob
import pathlib # Add imports

import torch

from lib import utils, dataloaders, models, metrics, testers

# Parameters remain largely the same, but pretrain/resume will be set in the loop
# ... (Keep params_3D_CBCT_Tooth, params_MMOTU, params_ISIC_2018 definitions as they are) ...
params_3D_CBCT_Tooth = {
    # ——————————————————————————————————————————————    Launch Initialization    —————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing      ————————————————————————————————————————————————————
    "resample_spacing": [0.5, 0.5, 0.5],
    "clip_lower_bound": -1412,
    "clip_upper_bound": 17943,
    "samples_train": 2048,
    "crop_size": (160, 160, 96),
    "crop_threshold": 0.5,
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_probability": 0.3,
    "augmentation_method": "Choice",
    "open_elastic_transform": True,
    "elastic_transform_sigma": 20,
    "elastic_transform_alpha": 1,
    "open_gaussian_noise": True,
    "gaussian_noise_mean": 0,
    "gaussian_noise_std": 0.01,
    "open_random_flip": True,
    "open_random_rescale": True,
    "random_rescale_min_percentage": 0.5,
    "random_rescale_max_percentage": 1.5,
    "open_random_rotate": True,
    "random_rotate_min_angle": -50,
    "random_rotate_max_angle": 50,
    "open_random_shift": True,
    "random_shift_max_percentage": 0.3,
    "normalize_mean": 0.05029342141696459,
    "normalize_std": 0.028477091559295814,
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "3D-CBCT-Tooth",
    "dataset_path": r"./datasets/3D-CBCT-Tooth",
    "create_data": False,
    "batch_size": 1,
    "num_workers": 2,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "PMFSNet",
    "in_channels": 1,
    "classes": 2,
    "index_to_class_dict":
        {
            0: "background",
            1: "foreground"
        },
    "resume": None,
    "pretrain": None, # Will be set in the loop
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "Adam",
    "learning_rate": 0.0005,
    "weight_decay": 0.00005,
    "momentum": 0.8,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "ReduceLROnPlateau",
    "gamma": 0.1,
    "step_size": 9,
    "milestones": [1, 3, 5, 7, 8, 9],
    "T_max": 2,
    "T_0": 2,
    "T_mult": 2,
    "mode": "max",
    "patience": 1,
    "factor": 0.5,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["HD", "ASSD", "IoU", "SO", "DSC"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.00551122, 0.99448878],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "use_amp": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 20,
    "best_dice": 0.60,
    "update_weight_freq": 32,
    "terminal_show_freq": 256,
    "save_epoch_freq": 4,
    # ————————————————————————————————————————————   Testing   ———————————————————————————————————————————————————————
    "crop_stride": [32, 32, 32]
}

params_MMOTU = {
    # ——————————————————————————————————————————————     Launch Initialization    ———————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing       ————————————————————————————————————————————————————
    "resize_shape": (224, 224),
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_p": 0.12097393901893663,
    "color_jitter": 0.4203933474361258,
    "random_rotation_angle": 30,
    "normalize_means": (0.22250386, 0.21844882, 0.21521868),
    "normalize_stds": (0.21923075, 0.21622984, 0.21370508),
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "MMOTU",
    "dataset_path": r"./datasets/MMOTU",
    "batch_size": 32,
    "num_workers": 2,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "PMFSNet",
    "in_channels": 3,
    "classes": 2,
    "index_to_class_dict":
        {
            0: "background",
            1: "foreground"
        },
    "resume": None,
    "pretrain": None, # Will be set in the loop
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "AdamW",
    "learning_rate": 0.01,
    "weight_decay": 0.00001,
    "momentum": 0.7725414416309884,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "CosineAnnealingLR",
    "gamma": 0.8689275449032848,
    "step_size": 5,
    "milestones": [10, 30, 60, 100, 120, 140, 160, 170],
    "T_max": 200,
    "T_0": 10,
    "T_mult": 5,
    "mode": "max",
    "patience": 1,
    "factor": 0.97,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC", "IoU"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.2350689696563569, 1 - 0.2350689696563569],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 2000,
    "best_metric": 0,
    "terminal_show_freq": 8,
    "save_epoch_freq": 500,
}

params_ISIC_2018 = {
    # ——————————————————————————————————————————————     Launch Initialization    ———————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing       ————————————————————————————————————————————————————
    "resize_shape": (224, 224),
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_p": 0.1,
    "color_jitter": 0.37,
    "random_rotation_angle": 15,
    "normalize_means": (0.50297405, 0.54711632, 0.71049083),
    "normalize_stds": (0.18653496, 0.17118206, 0.17080363),
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "ISIC-2018",
    "dataset_path": r"./datasets/ISIC-2018",
    "batch_size": 32,
    "num_workers": 2,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "PMFSNet",
    "in_channels": 3,
    "classes": 2,
    "index_to_class_dict":
        {
            0: "background",
            1: "foreground"
        },
    "resume": None,
    "pretrain": None, # Will be set in the loop
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "AdamW",
    "learning_rate": 0.005,
    "weight_decay": 0.000001,
    "momentum": 0.9657205586290213,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "CosineAnnealingWarmRestarts",
    "gamma": 0.9582311026945434,
    "step_size": 20,
    "milestones": [1, 3, 5, 7, 8, 9],
    "T_max": 100,
    "T_0": 5,
    "T_mult": 5,
    "mode": "max",
    "patience": 20,
    "factor": 0.3,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC", "IoU", "JI", "ACC"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.029, 1 - 0.029],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 150,
    "best_metric": 0,
    "terminal_show_freq": 20,
    "save_epoch_freq": 50,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for all K-folds of a model.") # Update description
    parser.add_argument("--dataset", type=str, default="ISIC-2018", help="Dataset name (e.g., ISIC-2018, MMOTU, 3D-CBCT-Tooth)")
    parser.add_argument("--model", type=str, default="UNet", help="Base model architecture name (e.g., LiSANet, PMFSNet)")
    # Change pretrain_weight to model_prefix
    parser.add_argument("--model_prefix", type=str, default="UNet-Base", help="Prefix for pre-trained weight files in pretrain/ (e.g., LiSANet-1000)")
    parser.add_argument("--dimension", type=str, default="2d", help="Dimension of dataset images and models (e.g., 2d, 3d)")
    parser.add_argument("--scaling_version", type=str, default="BASIC", help="Scaling version of PMFSNet (if applicable)")
    parser.add_argument("--image_path", type=str, default="pretrain/ISIC_0000013.jpg", help="Path of the single image to run inference on")
    # Add output_dir argument
    parser.add_argument("--output_dir", type=str, default="inferences", help="Directory to save the output segmentation images")
    args = parser.parse_args()
    return args


def main():
    # analyse console arguments
    args = parse_args()

    # select the dictionary of hyperparameters used for training
    if args.dataset == "3D-CBCT-Tooth":
        params = params_3D_CBCT_Tooth
    elif args.dataset == "MMOTU":
        params = params_MMOTU
    elif args.dataset == "ISIC-2018":
        params = params_ISIC_2018
    else:
        raise RuntimeError(f"No {args.dataset} dataset available")

    # update the dictionary of hyperparameters used for training
    params["dataset_name"] = args.dataset
    # Construct dataset path relative to workspace
    params["dataset_path"] = os.path.join("datasets", ("NC-release-data-checked" if args.dataset == "3D-CBCT-Tooth" else args.dataset))
    params["model_name"] = args.model
    # Remove pretrain weight setting here, will be set in loop
    # if args.pretrain_weight is None:
    #     raise RuntimeError("model weights cannot be None")
    # params["pretrain"] = args.pretrain_weight
    params["dimension"] = args.dimension
    params["scaling_version"] = args.scaling_version
    if not os.path.exists(args.image_path):
         raise FileNotFoundError(f"Input image not found: {args.image_path}")
    if not os.path.isdir("pretrain"):
         raise FileNotFoundError("Directory 'pretrain/' not found.")

    # launch initialization
    os.environ["CUDA_VISIBLE_DEVICES"] = params["CUDA_VISIBLE_DEVICES"]
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128" # Consider making this configurable if needed
    utils.reproducibility(params["seed"], params["deterministic"], params["benchmark"])

    # get the cuda device
    if params["cuda"]:
        params["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        params["device"] = torch.device("cpu")
    print(f"Using device: {params['device']}")
    print("Complete the initialization of configuration")

    # Find K-fold weight files
    weight_files_pattern = os.path.join("pretrain", f"{args.model_prefix}-K[1-5].pth") # Use pattern for K1-K5
    weight_files = sorted(glob.glob(weight_files_pattern))

    if not weight_files:
        raise FileNotFoundError(f"No weight files found matching pattern: {weight_files_pattern}")

    print(f"Found {len(weight_files)} weight files for prefix '{args.model_prefix}':")
    for wf in weight_files:
        print(f"  - {wf}")

    # Create output directory
    output_dir_path = pathlib.Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir_path.resolve()}")

    # Initialize the model (once, assuming same architecture for all folds)
    model = models.get_model(params)
    print(f"Initialized model: {params['model_name']}")

    # Initialize the tester (once)
    tester = testers.get_tester(params, model)
    print(f"Initialized tester for dataset: {params['dataset_name']}")

    # Loop through weight files, load weights, and run inference
    for weight_file in weight_files:
        print(f"--- Processing weight file: {weight_file} ---")
        params["pretrain"] = weight_file # Set current weight file in params

        # load training weights
        try:
            tester.load() # Assumes tester uses params["pretrain"] internally
            print("Complete loading training weights")
        except Exception as e:
            print(f"Error loading weights from {weight_file}: {e}")
            continue # Skip to next weight file if loading fails

        # Determine output path
        output_filename = pathlib.Path(weight_file).stem + ".jpg" # e.g., LiSANet-1000-K1.jpg
        output_path = output_dir_path / output_filename

        # Run inference for the single image
        try:
            tester.inference(args.image_path, str(output_path)) # Pass output path to tester
        except Exception as e:
            print(f"Error during inference for {weight_file}: {e}")
            continue # Skip to next weight file if inference fails

    print("--- Inference complete for all K-folds ---")


if __name__ == '__main__':
    main() 