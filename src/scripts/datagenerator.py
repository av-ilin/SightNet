import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.posenet import PoseNetDetector, process_output, draw_poses

def detect_and_save_poses(input_folder_0, input_folder_1, output_folder, model_path, csv_path):
    """
    Detect and save poses from images in the input folders and store the keypoints and labels in a CSV file.

    Parameters:
    - input_folder_0 (str): Path to the first input folder containing images of the first class.
    - input_folder_1 (str): Path to the second input folder containing images of the second class.
    - output_folder (str): Path to the output folder where the pose-detected images will be saved.
    - model_path (str): Path to the TensorFlow Lite model file.
    - csv_path (str): Path to the CSV file where keypoints and labels will be saved.

    Returns:
    - None
    """

    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Lists to store keypoints data and labels
    all_keypoints_data = []
    all_labels = []

    # Initialize PoseNet detector
    posenet_detector = PoseNetDetector(model_path)

    # Process images in the first input folder
    for filename in tqdm(os.listdir(input_folder_0)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder_0, filename)
            image = cv2.imread(image_path)

            # Detect poses in the image
            poses = posenet_detector.process_image(image)

            # Process and save keypoints data and labels
            for pose_data in poses:
                keypoints = process_output(pose_data)
                all_keypoints_data.append(keypoints)
                all_labels.append(0)
            
            # Draw detected poses on the image and save it
            draw_poses(image, poses)
            cv2.imwrite(os.path.join(f"{output_folder}/0", filename), image)

    # Process images in the second input folder
    for filename in tqdm(os.listdir(input_folder_1)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder_1, filename)
            image = cv2.imread(image_path)

            # Detect poses in the image
            poses = posenet_detector.process_image(image)

            # Process and save keypoints data and labels
            for pose_data in poses:
                keypoints = process_output(pose_data)
                all_keypoints_data.append(keypoints)
                all_labels.append(1)
            
            # Draw detected poses on the image and save it
            draw_poses(image, poses)
            cv2.imwrite(os.path.join(f"{output_folder}/1", filename), image)

    # Create a DataFrame to store keypoints data and labels
    df = pd.DataFrame(all_keypoints_data, columns=[
        'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
        'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
        'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
        'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'])

    df['Label'] = all_labels

    # Save the DataFrame to a CSV file
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    # Input and output paths
    input_folder_0 = 'augmented_images_0'
    input_folder_1 = 'augmented_images_1'
    output_folder = 'pose_detected_images'
    model_path = '4.tflite'
    csv_path = 'keypoints_data.csv'

    # Call the main function
    detect_and_save_poses(input_folder_0, input_folder_1, output_folder, model_path, csv_path)
