import cv2
import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os

from PIL import Image, ImageOps

def add_border_and_save(input_image_path, output_image_path, border_size):
    """
    Adds a white border to an image and saves the result.

    :param input_image_path: Path to the input image.
    :param output_image_path: Path to save the output image.
    :param border_size: Size of the border to be added.
    """
    # Open the image
    img = Image.open(input_image_path)

    # Add border
    img_with_border = ImageOps.expand(img, border=border_size, fill='white')

    # Save the image
    img_with_border.save(output_image_path)

# Example usage
#add_border_and_save('test_images/test_simp2.png', 'test_images/test_simp2.png', 10)


def save_useful_area_image_with_buffer(input_image_path, output_image_path, buffer=0):
    """
    Function to read an image from the given path, find the bounding box around black pixels,
    optionally add a buffer around this area, create a new image highlighting this area, 
    and save it to the specified output path.

    Args:
    input_image_path (str): Path to the input image.
    output_image_path (str): Path to save the output image.
    buffer (int): Number of pixels to add as a buffer around the bounding box. Default is 0.
    """
    # Read the image
    img = mpimg.imread(input_image_path)

    # Finding the coordinates of black pixels
    black_pixels = np.where(img == 0)
    min_y, max_y = min(black_pixels[0]), max(black_pixels[0])
    min_x, max_x = min(black_pixels[1]), max(black_pixels[1])

    # Adding buffer to the coordinates
    min_y = max(min_y - buffer, 0)
    max_y = min(max_y + buffer, img.shape[0] - 1)
    min_x = max(min_x - buffer, 0)
    max_x = min(max_x + buffer, img.shape[1] - 1)

    # Creating the useful area image based on the original image format
    if len(img.shape) == 2:  # Grayscale image
        useful_area_img = np.ones_like(img) * 255
        useful_area_img[min_y:max_y+1, min_x:max_x+1] = 0

    elif len(img.shape) == 3 and img.shape[2] == 3:  # RGB image
        useful_area_img = np.ones_like(img) * 255
        useful_area_img[min_y:max_y+1, min_x:max_x+1] = 0

    elif len(img.shape) == 3 and img.shape[2] == 4:  # RGBA image
        useful_area_img = np.ones((img.shape[0], img.shape[1], 4)) * 255
        useful_area_img[:, :, 3] = 255  # Set alpha channel to fully opaque
        useful_area_img[min_y:max_y+1, min_x:max_x+1, :3] = 0
        useful_area_img[min_y:max_y+1, min_x:max_x+1, 3] = 255

    print(useful_area_img.max())
    cv2.imwrite(os.path.expanduser(output_image_path), useful_area_img)

# Example usage
#save_useful_area_image_with_buffer('/Users/thomasrochefort/Documents/GitHub/truss-rec-python/test_images/test_simp2.png', 'test_images/test_simp2_contour.png',buffer=0)


def load_and_crop(image1, image2):
    # Load images
    im_topo = cv2.imread(image1)
    im_contour = cv2.imread(image2)

    if im_topo.shape == im_contour.shape:
        # Convert to grayscale
        im_topo = cv2.cvtColor(im_topo, cv2.COLOR_BGR2GRAY)
        im_contour = cv2.cvtColor(im_contour, cv2.COLOR_BGR2GRAY)

        # Binarize images
        _, im_topo = cv2.threshold(im_topo, 230, 255, cv2.THRESH_BINARY_INV)
        _, im_contour = cv2.threshold(im_contour, 230, 255, cv2.THRESH_BINARY_INV)

        # Label connected regions
        labels = label(im_contour)

        # Find largest region
        max_area = 0
        max_region = 0
        for region in regionprops(labels):
            if region.area > max_area:
                max_region = region.label
                max_area = region.area

        max_label = (labels == max_region)

        # Get bounding box of the largest region
        # Replace 'np.int' with Python's built-in 'int'
        prop = regionprops(max_label.astype(int))[0].bbox
        del_margin = 5
        x1, y1, x2, y2 = prop[1] - del_margin, prop[0] - del_margin, prop[3] + del_margin, prop[2] + del_margin

        # Crop images
        i1 = im_topo[y1:y2, x1:x2]
        i2 = im_contour[y1:y2, x1:x2]

        return i1, i2

    else:
        raise ValueError('Images are not of the same size')



import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_branches_with_average_thickness(input_image, adjusted_branches_info, num_points=10):
    """
    Draws branches on the given grayscale image with an average thickness determined by
    analyzing multiple points along each branch.

    Parameters:
    - input_image (numpy.ndarray): A grayscale image of branches.
    - adjusted_branches_info (list): List of tuples, each representing a branch with start and end points.
    - num_points (int): Number of points to interpolate along each branch for thickness calculation.
    - max_thickness (int): Maximum thickness allowed for a branch.

    Returns:
    - numpy.ndarray: An image with branches drawn with average thickness.
    """

    _, binary_image = cv2.threshold(input_image, 128, 255, cv2.THRESH_BINARY)
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)

    # def estimate_local_thickness(dist_transform, point):
    #     x, y = int(point[0]), int(point[1])
    #     return dist_transform[y, x] * 2
    def estimate_local_thickness(dist_transform, point, kernel_size=20):
        """
        Estimates the local thickness at a given point using an average value of the distance transform
        within a specified kernel around the point.

        Parameters:
        - dist_transform (numpy.ndarray): The distance transform of the binary image.
        - point (tuple): The (x, y) coordinates of the point.
        - kernel_size (int): The size of the kernel to consider for averaging.

        Returns:
        - float: The estimated local thickness at the point.
        """
        x, y = int(point[0]), int(point[1])
        half_kernel = kernel_size // 2

        # Ensure the kernel is within the bounds of the image
        x_start, x_end = max(x - half_kernel, 0), min(x + half_kernel + 1, dist_transform.shape[1])
        y_start, y_end = max(y - half_kernel, 0), min(y + half_kernel + 1, dist_transform.shape[0])

        # Calculate the average distance transform value within the kernel
        kernel_values = dist_transform[y_start:y_end, x_start:x_end]
        average_value = np.mean(kernel_values)

        return average_value * 2  # Multiply by 2 to get the diameter as thickness

    blank_image = np.zeros_like(input_image)
    thicknesses = []

    for branch in adjusted_branches_info:
        node1, node2 = branch
        total_thickness = 0

        for i in range(num_points):
            interp_point = (node1[0] + (node2[0] - node1[0]) * i / (num_points - 1),
                            node1[1] + (node2[1] - node1[1]) * i / (num_points - 1))
            total_thickness += estimate_local_thickness(dist_transform, interp_point)

        average_thickness = total_thickness / num_points
        thicknesses.append(average_thickness)

        # Apply thickness limits
        average_thickness = max(np.percentile(thicknesses, 5), average_thickness)

        
        cv2.line(blank_image, (int(node1[0]), int(node1[1])), (int(node2[0]), int(node2[1])), (255, 255, 255), int(average_thickness))

    return blank_image



import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_with_red_skeleton_overlay(original_image, skeleton_image, alpha=0.5):
    """
    Creates an overlay of the skeleton image on the original image and displays it.
    The skeleton is highlighted in red.

    Parameters:
    - original_image (numpy.ndarray): The original grayscale or color image.
    - skeleton_image (numpy.ndarray): The binary image with the skeleton (branches) drawn.
    - alpha (float, optional): Transparency factor for the overlay. Default is 0.5.

    The function displays the overlayed image with a legend indicating the skeleton overlay.
    """

    # Create a color version of the skeleton image to hold the red overlay
    color_skeleton_image = np.zeros((*skeleton_image.shape, 3), dtype=np.uint8)
    color_skeleton_image[skeleton_image > 0] = [255, 0, 0]  # Red color

    # Convert the original image to a 3-channel image if it's grayscale
    if len(original_image.shape) == 2 or original_image.shape[2] == 1:
        original_image_color = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        original_image_color = original_image.copy()

    # Create a red patch as a proxy for the red overlay
    red_patch = mpatches.Patch(color='red', label='Skeleton Overlay')

    # Blend the two images
    overlayed_image = cv2.addWeighted(color_skeleton_image, alpha, original_image_color, 1 - alpha, 0)

    # Display the image
    plt.imshow(overlayed_image)
    plt.axis('off')  # Hide the axis
    plt.legend(handles=[red_patch], loc='upper left')
    plt.show()

# Example usage:
# test1 is your original grayscale or color image
# blank_image is your binary image with branches drawn
# plot_with_red_skeleton_overlay(test1, interp_image, alpha=0.5)