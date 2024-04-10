import argparse
from scripts.addlogger import generate_datasset
from scripts.augmenter import augment_images
from scripts.datagenerator import detect_and_save_poses

def main(input_folder, output_folder_augmented, augment_count, output_folder_pose_detected, model_path, csv_path):
    # Augment images
    for i in range(2):
        augment_images(f"{input_folder}/{i}", f"{output_folder_augmented}/{i}", augment_count)

    # Detect and save poses
    detect_and_save_poses(f"{output_folder_augmented}/0", f"{output_folder_augmented}/1", output_folder_pose_detected, model_path, csv_path)

    # Generate dataset
    generate_datasset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images, augment them, detect poses, and generate a dataset.")
    
    parser.add_argument('--input_folder', type=str, default='src/data', help='Folder containing the original images')
    parser.add_argument('--output_folder_augmented', type=str, default='src/output/images/augmented', help='Folder to save the augmented images')
    parser.add_argument('--augment_count', type=int, default=5, help='Number of augmented copies for each image')
    parser.add_argument('--output_folder_pose_detected', type=str, default='src/output/images/pose_detected', help='Folder to save the pose-detected images')
    parser.add_argument('--model_path', type=str, default='src/models/PoseNet.tflite', help='Path to the PoseNet TensorFlow Lite model')
    parser.add_argument('--csv_path', type=str, default='src/output/csv/keypoints.csv', help='Path to save the keypoints CSV file')

    args = parser.parse_args()

    main(args.input_folder, args.output_folder_augmented, args.augment_count, args.output_folder_pose_detected, args.model_path, args.csv_path)
