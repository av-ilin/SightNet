import os
import imgaug.augmenters as iaa
import cv2

def augment_images(input_folder, output_folder, augment_count):
    """
    Augment images in the input folder and save the augmented images to the output folder.

    Parameters:
    - input_folder (str): Path to the folder containing the original images.
    - output_folder (str): Path to the folder where augmented images will be saved.
    - augment_count (int): Number of augmented copies to create for each image.

    Returns:
    - None
    """

    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define the augmentation sequence
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # Random horizontal flip
        iaa.Affine(rotate=(-10, 10)),  # Random rotation between -10 to 10 degrees
        iaa.GaussianBlur(sigma=(0, 1.0)),  # Random blur
        iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),  # Add random Gaussian noise
    ])

    # Iterate through all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # Apply augmentation and save the augmented images
            for i in range(augment_count):
                augmented_image = seq(image=image)
                output_path = os.path.join(output_folder, f"{filename.split('.')[0]}_aug_{i+1}.jpg")
                cv2.imwrite(output_path, augmented_image)

if __name__ == "__main__":
    input_folder = "downloaded_files/source data/1"  # Folder containing the original images
    output_folder = 'augmented_images_1'  # Folder to save the augmented images
    augment_count = 5  # Number of augmented copies for each image

    augment_images(input_folder, output_folder, augment_count)
