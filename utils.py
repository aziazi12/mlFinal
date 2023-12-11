import cv2
import torch


def get_files_from_folder(filename_batch):
    """
    This function takes a batch of filenames and returns a batch of images
    """
    images = []  # Initialize an empty list to store the images
    for filename in filename_batch:  # Loop over each filename in the batch
        image = cv2.imread(f'data/images/{filename}')  # Read the image file
        images.append(image)  # Append the image to the list

    return images  # Return the list of images

def processed_images(images, target_size):
    """
    This function takes a list of images, resizes them to the target size, and then converts them to a tensor
    """
    processed_images = []  # Initialize an empty list to store the processed images
    for image in images:  # Loop over each image in the list
        processed_image = cv2.resize(image, target_size)  # Resize the image to the target size
        processed_image = torch.tensor(processed_image)  # Convert the image to a tensor
        processed_images.append(processed_image)  # Append the processed image to the list

    return processed_images  # Return the list of processed images


