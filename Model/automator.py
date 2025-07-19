import os

# Set the base directory where the datasets are storedd
base_dir = './datasets'

# Loop through the folders in the base directory
for i in range(5, 6):
    folder_name = f"ISIC_{i}"
    old_path = os.path.join(base_dir, folder_name)
    new_path = os.path.join(base_dir, "ISIC-2018")
    
    # Rename the folder to "ISIC-2018"
    os.rename(old_path, new_path)
    print(f"Renamed {folder_name} to 'ISIC-2018'.")
    
    # Run the train.py script
    os.system("python train.py")
    
    # Rename the folder back to the original name
    os.rename(new_path, old_path)
    print(f"Renamed 'ISIC-2018' back to {folder_name}.")