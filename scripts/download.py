import subprocess

def download_dataset(dataset_identifier, output_dir):
    """
    Download a Kaggle dataset using the Kaggle CLI.

    Args:
        dataset_identifier (str): Kaggle dataset identifier (e.g., "wenewone/cub2002011").
        output_dir (str): Directory to save the downloaded dataset.
    """
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", dataset_identifier,
        "-p", output_dir
    ])
    print(f"Dataset downloaded to: {output_dir}")

# Example usage
download_dataset("wenewone/cub2002011", "./datasets/cub2002011")
