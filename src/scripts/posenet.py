import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

class PoseNetDetector:
    """
    Class for detecting poses using the PoseNet model.

    Attributes:
    - interpreter: TensorFlow Lite interpreter for running inference.
    """

    def __init__(self, model_path):
        """
        Initialize the PoseNetDetector with the given TensorFlow Lite model.

        Parameters:
        - model_path (str): Path to the TensorFlow Lite model file.

        Returns:
        - None
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

    def process_image(self, image):
        """
        Process an image to detect poses using the PoseNet model.

        Parameters:
        - image (numpy.ndarray): Input image to detect poses from.

        Returns:
        - List of numpy.ndarrays: Detected pose keypoints.
        """
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        input_shape = input_details[0]['shape']
        input_data = np.expand_dims(cv2.resize(image, (input_shape[1], input_shape[2])), axis=0)

        input_data = np.uint8(input_data)

        self.interpreter.set_tensor(input_details[0]['index'], input_data)
        self.interpreter.invoke()

        output_data = [self.interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        return output_data
    
def process_output(output_data):
    """
    Process the output data from the PoseNet model to extract keypoints.

    Parameters:
    - output_data (numpy.ndarray): Output data from the PoseNet model.

    Returns:
    - List of dictionaries: Processed keypoints with 'y', 'x', and 'confidence' values.
    """
    keypoints = output_data[0, 0, :, :]
    processed_keypoints = []
    for keypoint in keypoints:
        y, x, confidence = keypoint
        processed_keypoints.append({'y': y, 'x': x, 'confidence': confidence})
    return processed_keypoints

def draw_poses(image, output_data):
    """
    Draw detected poses on the input image.

    Parameters:
    - image (numpy.ndarray): Input image to draw poses on.
    - output_data (list): Detected pose keypoints.

    Returns:
    - None
    """
    for pose_data in output_data:
        keypoints = process_output(pose_data)
        if keypoints is not None:
            for keypoint in keypoints:
                if keypoint['confidence'] > 0.2: 
                    cv2.circle(image, (int(keypoint['x'] * image.shape[1]), int(keypoint['y'] * image.shape[0])), 5, (0, 255, 0), -1)

def detect_poses(input_folder, output_folder, model_path):
    """
    Detect poses in images from the input folder and save the result in the output folder.

    Parameters:
    - input_folder (str): Path to the input folder containing images.
    - output_folder (str): Path to the output folder where the pose-detected images will be saved.
    - model_path (str): Path to the TensorFlow Lite model file.

    Returns:
    - None
    """
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize PoseNet detector
    posenet_detector = PoseNetDetector(model_path)

    # Process images in the input folder
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # Detect poses in the image
            poses = posenet_detector.process_image(image)

            # Draw detected poses on the image
            draw_poses(image, poses)

            # Save the image with detected poses
            cv2.imwrite(os.path.join(output_folder, filename), image)

            # Print progress information
            print(f"Processed image: {filename}")

if __name__ == "__main__":
    # Input and output paths
    input_folder = 'augmented_images_0'
    output_folder = 'pose_detected_images_0'
    model_path = '4.tflite'

    # Call the main function
    detect_poses(input_folder, output_folder, model_path)
