import os

def locate_split_dataset(base_dir):
    """
    Function to locate the training and testing datasets in the split dataset directory.

    Args:
    - base_dir (str): The base directory where the split dataset is located.

    Returns:
    - dict: A dictionary containing the paths of the train and test directories.
    """
    # Define train and test directory paths based on the base directory
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

    # Check if the train and test directories exist
    if os.path.isdir(train_dir) and os.path.isdir(test_dir):
        # List subdirectories (each subdirectory represents a class)
        train_classes = os.listdir(train_dir)
        test_classes = os.listdir(test_dir)

        # Return the paths and class information if directories exist
        return {
            'train_dir': train_dir,
            'test_dir': test_dir,
            'train_classes': train_classes,
            'test_classes': test_classes
        }
    else:
        # If directories don't exist, notify the user
        return {
            'message': 'Train and/or Test directories not found at the specified location.',
            'train_dir': train_dir,
            'test_dir': test_dir
        }


if __name__ == "__main__":
    # Define the base directory for the split dataset
    BASE_DIR = "C:/Users/samee/Documents/cubtrain/datasets/cub2002011/images_split"  # Update this to your split dataset path

    # Locate the split dataset
    split_dataset_location = locate_split_dataset(BASE_DIR)

    # Print the result
    if "message" in split_dataset_location:
        print(split_dataset_location["message"])
    else:
        print(f"Train Directory: {split_dataset_location['train_dir']}")
        print(f"Test Directory: {split_dataset_location['test_dir']}")
        print(f"Train Classes: {split_dataset_location['train_classes']}")
        print(f"Test Classes: {split_dataset_location['test_classes']}")
