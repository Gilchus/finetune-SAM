import numpy as np
import argparse
import imageio
import os

def create_npz_dataset(image_dir, save_path):
    all_files = sorted(os.listdir(image_dir))
    original_files = [f for f in all_files if 'original' in f]
    label_files = [f for f in all_files if 'label' in f]

    # Ensure both lists are sorted to match pairs correctly
    original_files.sort()
    label_files.sort()

    assert len(original_files) == len(label_files), "Mismatch in number of original and label images"

    images = []
    labels = []

    # Load the image pairs
    for orig_file, label_file in zip(original_files, label_files):
        original_image = imageio.imread(os.path.join(image_dir, orig_file))
        labeled_image = imageio.imread(os.path.join(image_dir, label_file))

        images.append(original_image)
        labels.append(labeled_image)

    # Convert lists to numpy arrays
    images_array = np.array(images)
    labels_array = np.array(labels)

    # Save the arrays as an NPZ file
    np.savez(save_path, imgs=images_array, gts=labels_array)
    
    print(f"Dataset saved to {save_path}")

if __name__ == "main":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, help="Directory containing images")
    parser.add_argument("--save_path", type=str, help="Path to save the dataset")
    args = parser.parse_args()
    create_npz_dataset(args.image_dir, args.save_path)
    
    

