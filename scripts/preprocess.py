import os
import shutil
from sklearn.model_selection import train_test_split

# Define the paths
source_dir = "C:/Users/samee/Documents/cubtrain/datasets/cub2002011/cub2002011/CUB_200_2011/images"  # Update this to the correct dataset path
train_dir = "C:/Users/samee/Documents/cubtrain/datasets/cub2002011/images_split/train"  # Path to save training images
test_dir = "C:/Users/samee/Documents/cubtrain/datasets/cub2002011/images_split/test"  # Path to save testing images

# Create directories for train and test sets in the working directory
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# List all subdirectories (each subdirectory is a class)
class_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]

# Split images inside each class into training and testing
for class_folder in class_folders:
    class_folder_path = os.path.join(source_dir, class_folder)

    # List all images in the class folder
    image_files = [f for f in os.listdir(class_folder_path) if os.path.isfile(os.path.join(class_folder_path, f))]

    # Debugging: Print the number of images in the current class
    print(f"Class '{class_folder}' contains {len(image_files)} images.")

    # Only proceed if there are images in the folder
    if len(image_files) > 0:
        # Split the image files into training and testing sets (80% train, 20% test)
        train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)

        # Create subdirectories for each class in the train and test folders
        os.makedirs(os.path.join(train_dir, class_folder), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_folder), exist_ok=True)

        # Copy the train and test images to their respective directories
        for file_name in train_files:
            src_path = os.path.join(class_folder_path, file_name)
            dest_path = os.path.join(train_dir, class_folder, file_name)
            shutil.copy(src_path, dest_path)  # Use copy instead of move

        for file_name in test_files:
            src_path = os.path.join(class_folder_path, file_name)
            dest_path = os.path.join(test_dir, class_folder, file_name)
            shutil.copy(src_path, dest_path)  # Use copy instead of move

        print(f"Class '{class_folder}': {len(train_files)} images for training, {len(test_files)} images for testing")
    else:
        print(f"Warning: Class '{class_folder}' has no images.")

print("Dataset splitting completed!")
