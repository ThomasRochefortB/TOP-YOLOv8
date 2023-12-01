import numpy as np

def dice_coefficient(image1, image2):
    """
    Calculate the Dice Coefficient between two binary images.
    
    Parameters:
    - image1: A binary image (numpy array).
    - image2: Another binary image (numpy array) of the same size as image1.

    Returns:
    - Dice coefficient as a float.
    """

    # Ensure the images are binary
    image1_bool = image1.astype(bool)
    image2_bool = image2.astype(bool)

    # Calculate intersection and union
    intersection = np.logical_and(image1_bool, image2_bool)
    union = np.logical_or(image1_bool, image2_bool)

    # Calculate Dice coefficient
    dice = 2. * intersection.sum() / (image1_bool.sum() + image2_bool.sum())

    return dice

# Example usage:
# dice = dice_coefficient(test1, blank_image)
# print(f"Dice Coefficient: {dice}")