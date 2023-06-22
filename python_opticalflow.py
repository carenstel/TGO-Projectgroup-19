# --------------------------------------------------
# Script: Use optical flow to visualize the lung motion in general.
# Images and a video that give an insight of the lung motion are saved.
# --------------------------------------------------

# Needed packages for the code.
from cmath import sqrt
from statistics import mean
from tkinter import FIRST
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from matplotlib.colors import LinearSegmentedColormap
import glob
import math
import statistics

# Constants.
LK_PARAMS = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) # Parameters for the Lukas-Kanade algorithm.
RIB_COLOR = (0, 255, 0)  # Green color for rib tracking.
POINT_COLOR = (255, 0, 0)  # Blue color for center point.

# Number of frames to skip. 
frames_skip = 1 # The step between two analysed frames. This means that every frame is analysed. Default is 1.
start_frame = 169 # The startframe is the first frame of the DICOM that is analysed. Default is 0.

# Load the images and masks.
general_path = "C:/Users/User/Documents/Technische Geneeskunde/Module 12 TGO/Python/fluoroscopy"

# Set image_path and get the filenames.
image_path = general_path + "/image"
all_images_paths = glob.glob(image_path + "/*.jpg")
# Sort the image file names in a natural numeric order.
all_images_paths = sorted(all_images_paths, key=lambda x: int(''.join(filter(str.isdigit, x))))

# Set overlay_path and get the filenames.
image_mask_overlay_path = general_path + "/image_mask_overlay"
all_images_masks_overlay_path = glob.glob(image_mask_overlay_path + "/*.jpg")
# Sort the overlay file names in a natural numeric order.
all_images_masks_overlay_path = sorted(all_images_masks_overlay_path, key=lambda x: int(''.join(filter(str.isdigit, x))))

# Set mask_path and get the filenames.
mask_path = general_path + "/mask"
all_masks_paths = glob.glob(mask_path + "/*.jpg")
# Sort the mask file names in a natural numeric order.
all_masks_paths = sorted(all_masks_paths, key=lambda x: int(''.join(filter(str.isdigit, x))))

# If the number of images and masks are not the same, the folder is corrupt and therefore break of the code is necessary.
if len(all_images_paths) != len(all_masks_paths):
    print("Number of images and masks are not the same")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Save directories for images and graphs.
opticalflow_general_path = general_path + "/opticalflow"
colorimage_general_path = general_path + "/colorimage"
colorplot_general_path = general_path + "/colorplot"
colorcircle_general_path = general_path + "/colorcircles"

# Running average for color scaling. This is important for scaling of the colors in the colorplot and colorimage to get maximum contrast, but prevent large color differences between frames.
value_running_average = 1 * frames_skip # Default to 1, this means a translation of 1 pixel between 2 frames. If the frames_skip increase, then the running average value must increase, because the motion is also expected to be larger.
length_running_average = int(10 / frames_skip) # How longer the length of the running average list, how slower a change in color scaling occur.
running_average_absolute_max_translation = [value_running_average] * length_running_average # Make a list of length x, containing x times the same number.

# Settings for images.
size_kernel = 3 # How greater the kernel, how more smoothen the image.
neighborhood_size = 1  # Adjust the neighborhood size in pixels as desired. This is the amount of pixels to look around a specific pixelcoordinate.

# Lists for greatest x and y changes. Create them outside the function, because the function is multiple times called in a for loop.
min_negative_x_translations = []
max_positive_x_translations = []
min_negative_y_translations = []
max_positive_y_translations = []

# Global variables to store the coordinates of mouse_click.
x_mouse_click = None
y_mouse_click = None

# Get the color image, color plot and color circle. This function analyses the difference between two frames with optical flow and visualizes it.
def get_colorimage_colorplot_colorcircle(image0_path, image1_path, mask0_path):

    # Open image0, image1 and mask0.
    image0 = Image.open(image0_path)
    image1 = Image.open(image1_path)
    mask0 = Image.open(mask0_path)

    # Convert images and mask to NumPy arrays. By converting the images to a NumPy it is possible to make use of the Numpy functions. For example filtering of the data.
    image0_array = np.array(image0)
    image1_array = np.array(image1)
    mask0_array = np.array(mask0)

    # Assign list to store all the points that the optical flow will follow.
    points_to_follow = []

    # Iterate over the image dimensions with the specified area width and height.
    for x in range(image0.width):
        for y in range(image0.height):

            # Access the pixel value from the mask.
            intensity = mask0_array[y, x]

            # All pixel coordinates that are white on the mask (intensity == 255) correspont with the lung pixel coordinates in the normal image.
            if intensity == 255:

                # The points where the mask is white must be followed, because these correspont with the place of the lungs on the normal image.
                points_to_follow.append((x, y))

    # Convert points to NumPy array.
    points_to_follow = np.array(points_to_follow, dtype=np.float32).reshape(-1, 2)

    # Convert images to grayscale. Where the if and else statement is determined which type of conversion is needed.
    if len(image0_array.shape) == 3 and image0_array.shape[2] == 3: # If the images are RGB, they must be converted to grayscale so all the images have the same format.
        old_gray = cv2.cvtColor(image0_array, cv2.COLOR_RGB2GRAY)
    else:
        old_gray = image0_array

    if len(image1_array.shape) == 3 and image1_array.shape[2] == 3: # If the images are RGB, they must be converted to grayscale so all the images have the same format.
        new_gray = cv2.cvtColor(image1_array, cv2.COLOR_RGB2GRAY)
    else:
        new_gray = image1_array

    # Calculate optical flow using Lucas-Kanade method. p1 is an array with the new coordinates in frame1 of the pixels that are followed in frame0. P1 is the array with the new pixel coordinates.
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, points_to_follow, None, **LK_PARAMS)
    
    # Calculate the translations by subtracting the new and original location.
    translations = p1 - points_to_follow

    # Draw the tracks and center points on the new frame.
    frame = np.copy(image1_array)

    # Convert the frame back to BGR (Blue Green Red) format, because this is needed to save the image with the right color channels.
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Draw the tracks and center points on the BGR frame.
    for i, (new, old) in enumerate(zip(p1, points_to_follow)):
        a, b = new.ravel()
        c, d = old.ravel()
        
        # Draw the track on the BGR frame.
        frame_bgr = cv2.line(frame_bgr, (int(a), int(b)), (int(c), int(d)), RIB_COLOR, 2)

        # Draw a point at the center of the tracked ribs.
        center_x = int((a + c) / 2)
        center_y = int((b + d) / 2)
        frame_bgr = cv2.circle(frame_bgr, (center_x, center_y), 3, POINT_COLOR, -1)

    # Store frame_bgr in the variable opticalflow for later saving the tracked points in overlay with the original image.
    opticalflow = frame_bgr

    # Determine the dimensions of the arrays of the points to follow.
    max_x = int(np.max(points_to_follow[:, 0]))
    max_y = int(np.max(points_to_follow[:, 1]))

    # Initialize the arrays coordinates_x_translation_array and coordinates_y_translation_array filled with zeros. The arrays have the dimensions (max_y+1) and (max_x+1).
    coordinates_x_translation_array = np.zeros((max_y + 1, max_x + 1))
    coordinates_y_translation_array = np.zeros((max_y + 1, max_x + 1))
    
    # Set lists to check what the greatest translation for this index of the for loop is. Iniatilly set 0, to prevent an error when there is in the image no movement in certain direction.
    local_negative_x_translations = [0]
    local_positive_x_translations = [0]
    local_negative_y_translations = [0]
    local_positive_y_translations = [0]
    
    # In this for loop the exterme values are removed. This prevents a disruption of the image, when one point was not properly tracked.
    for (x, y), translation in zip(points_to_follow, translations):

        # Get the x and y translation out of the translation array.
        x_translation = translation[0]
        y_translation = translation[1]

        # Assign the value of the translation to the good list.
        if (x_translation < 0):
            local_negative_x_translations.append(x_translation)
        if (x_translation > 0):
            local_positive_x_translations.append(x_translation)
        if (y_translation < 0):
            local_negative_y_translations.append(y_translation)
        if (y_translation > 0):
            local_positive_y_translations.append(y_translation)

        # Set the x and y translation in a new array.    
        coordinates_x_translation_array[int(y), int(x)] = x_translation
        coordinates_y_translation_array[int(y), int(x)] = y_translation

    # Find and set the greatest translations in four directions to the max lists.
    min_negative_x_translations.append(min(local_negative_x_translations)) # Min because list contains negative values
    max_positive_x_translations.append(max(local_positive_x_translations)) # Max because list contains positive values
    min_negative_y_translations.append(min(local_negative_y_translations)) # Min because list contains negative values
    max_positive_y_translations.append(max(local_positive_y_translations)) # Max because list contains positive values

    # Remove extreme points. The threshold corresponds with the maximum translation in pixels.
    absolute_threshold = neighborhood_size  # the threshold is set to the neighboorhoodsize = 3.

    # Assign the variable num_averaged_pixels to count how many pixels are averaged. With averaged is meant: Indicated as an extreme translation, probably caused by tracking error, and therefore replaced with the median of the neighbors.
    num_averaged_pixels = 0

    # Iterate trough all x and y coordinates.
    for y in range(coordinates_x_translation_array.shape[0]):
        for x in range(coordinates_x_translation_array.shape[1]):

            # Get the x and y translation.
            x_translation = coordinates_x_translation_array[y, x]
            y_translation = coordinates_y_translation_array[y, x]
                            
            # Assign the lists for the translation of neighbor coordinates.
            neighbor_pixel_x_translation_list = []
            neighbor_pixel_y_translation_list = []
            neighbor_pixel_distance_list = []

            # Access the values of the surrounding pixels
            for row in range((2 * neighborhood_size) + 1):
                for col in range((2 * neighborhood_size) + 1):
                    
                    # Calculate the distance of the neighbor pixel to the original pixel.
                    neighbor_x = x - neighborhood_size + col 
                    neighbor_y = y - neighborhood_size + row
                    
                    # Break if (row, col) is the pixel self.
                    if (x == neighbor_x) and (y == neighbor_y):
                        continue # Skip the rest of the code in the inner for loop.

                    # Break if x coordinates are outside the image.
                    elif (neighbor_x < 0) or (neighbor_x >= coordinates_x_translation_array.shape[1]):
                        continue # Skip the rest of the code in the inner for loop.

                    # Break if x coordinates are outside the image.
                    elif (neighbor_y < 0) or (neighbor_y >= coordinates_x_translation_array.shape[0]):
                        continue # Skip the rest of the code in the inner for loop.

                    else:
                        # Get the translation of the neighbor in the x direction and append it to the list.
                        neighbor_x_translation = coordinates_x_translation_array[neighbor_y, neighbor_x]
                        neighbor_pixel_x_translation_list.append(neighbor_x_translation)

                        # Get the translation of the neighbor in the y direction and append it to the list.
                        neighbor_y_translation = coordinates_y_translation_array[neighbor_y, neighbor_x]
                        neighbor_pixel_y_translation_list.append(neighbor_y_translation)

            # Calculate the median of the surrounding pixels.
            median_x = np.median(neighbor_pixel_x_translation_list)
            median_y = np.median(neighbor_pixel_y_translation_list)

            # Calculate the percentage difference between the pixel value and the average pixel value of the neighbors.
            if median_x != 0:
                diff_x = np.abs(x_translation - median_x)
            else:
                diff_x = 0

            if median_y != 0:
                diff_y = np.abs(y_translation - median_y)
            else:
                diff_y = 0

            # Check if the percentage difference exceeds the threshold.
            if diff_x > absolute_threshold or diff_y > absolute_threshold:

                # Set the pixel value to the average of the surrounding pixels.
                coordinates_x_translation_array[y, x] = median_x
                coordinates_y_translation_array[y, x] = median_y
                num_averaged_pixels += 1
    
    # Print the number of pixels that are averaged to monitor the quality of the optical flow.
    print("Number of pixels averaged: ", num_averaged_pixels)
                
    # Create new array of translations, which are based on the movement of the neighbors' pixels.
    coordinates_x_translation_array_weighted = np.zeros_like(coordinates_x_translation_array)
    coordinates_y_translation_array_weighted = np.zeros_like(coordinates_y_translation_array)    

    # Iterate through all x and y coordinates.
    for y in range(coordinates_x_translation_array.shape[0]):
        for x in range(coordinates_x_translation_array.shape[1]):

            # Get the x and y translation.
            x_translation = coordinates_x_translation_array[y, x]
            y_translation = coordinates_y_translation_array[y, x]
                            
            # Assign the lists for the translation of neighbor coordinates.
            neighbor_pixel_x_translation_list = []
            neighbor_pixel_y_translation_list = []
            neighbor_pixel_distance_list = []

            # Access the values of the surrounding pixels.
            for row in range((2 * neighborhood_size) + 1):
                for col in range((2 * neighborhood_size) + 1):

                    # Calculate the distance of the neighbor pixel to the original pixel.
                    neighbor_x = x - neighborhood_size + col 
                    neighbor_y = y - neighborhood_size + row
                    
                    # Break if row, col is the pixel self.
                    if (x == neighbor_x) and (y == neighbor_y):
                        continue # Skip the rest of the code in the inner for loop.

                    # Break if x coordinates are outside the image
                    elif (neighbor_x < 0) or (neighbor_x >= coordinates_x_translation_array.shape[1]):
                        continue # Skip the rest of the code in the inner for loop.

                    elif (neighbor_y < 0) or (neighbor_y >= coordinates_x_translation_array.shape[0]):
                        continue # Skip the rest of the code in the inner for loop.

                    else:
                        # Get the neighbor value and append it to the list.
                        neighbor_x_translation = coordinates_x_translation_array[neighbor_y, neighbor_x]
                        neighbor_pixel_x_translation_list.append(neighbor_x_translation)

                        neighbor_y_translation = coordinates_y_translation_array[neighbor_y, neighbor_x]
                        neighbor_pixel_y_translation_list.append(neighbor_y_translation)
                    
                        # Calculate the distance of the neighbor pixel to the original pixel.
                        distance_neighbor_to_pixel = math.sqrt((neighbor_x - x) ** 2 + (neighbor_y - y) ** 2)
                        neighbor_pixel_distance_list.append(distance_neighbor_to_pixel)

            # Calculate the weighted average of the neighbor translations. The distance of the neighbor pixel to the original pixel is important. The greater the distance to the original pixel, the less effect this neighbor may have in the weighted average.
            translation_x_devided_by_distance_x = np.divide(np.array(neighbor_pixel_x_translation_list), np.array(neighbor_pixel_distance_list), where=np.array(neighbor_pixel_distance_list)!=0)
            sum_translation_x_devided_by_distance_x = np.sum(translation_x_devided_by_distance_x)
            weighted_average_x = sum_translation_x_devided_by_distance_x / len(neighbor_pixel_distance_list)

            # Calculate the weighted average of the neighbor translations. The distance of the neighbor pixel to the original pixel is important. The greater the distance to the original pixel, the less effect this neighbor may have in the weighted average.
            translation_y_devided_by_distance_y = np.divide(np.array(neighbor_pixel_y_translation_list), np.array(neighbor_pixel_distance_list), where=np.array(neighbor_pixel_distance_list)!=0)
            sum_translation_y_devided_by_distance_y = np.sum(translation_y_devided_by_distance_y)
            weighted_average_y = sum_translation_y_devided_by_distance_y / len(neighbor_pixel_distance_list) 

            # Set the weighted average of the neighbors' pixels as the pixels' translation.
            coordinates_x_translation_array_weighted[y, x] = weighted_average_x
            coordinates_y_translation_array_weighted[y, x] = weighted_average_y
            
    # Define the (Kernel) blurring filter.
    kernel = np.ones((size_kernel, size_kernel)) / (size_kernel * size_kernel) # Averaging filter.

    # Apply Kernel blurring filter to smooth the pixel array.
    smoothed_coordinates_x_translation_array = convolve(coordinates_x_translation_array_weighted, kernel)
    smoothed_coordinates_y_translation_array = convolve(coordinates_y_translation_array_weighted, kernel)

    # Assign the smoothed values to the original translations array.
    translations[:, 0] = smoothed_coordinates_x_translation_array[points_to_follow[:, 1].astype(int), points_to_follow[:, 0].astype(int)]
    translations[:, 1] = smoothed_coordinates_y_translation_array[points_to_follow[:, 1].astype(int), points_to_follow[:, 0].astype(int)]

    # Reshape translations array if it does not have the right shape.
    if translations.shape[1] == 1:
        translations = translations.reshape(-1, 2)

    # Calculate the translation lengths of the vector.
    translation_lengths = np.linalg.norm(translations, axis=1)
    
    # Calculate the minimum and maximum translation lengths of the length in the array translation_lengths.
    min_translation_length = np.min(translation_lengths)
    max_translation_length = np.max(translation_lengths)

    # Print the minimum and maximum translation lengths.
    print(min_translation_length)
    print(max_translation_length)

    # Assign an empty list to store colorvalues.
    colors = []

    # Append the max_translation_length value to the running_average_absolute_max_translation_length list.
    running_average_absolute_max_translation.append(max_translation_length)

    # Delete the first item from the running_average_absolute_max_translation_length list to keep the list the same length when adding a new item after deleting.
    del running_average_absolute_max_translation[0]

    # Convert the elements of the list to float (decimal number) and calculate the mean of the numbers in the running average list.
    absolute_max_translation_length = statistics.mean(map(float, running_average_absolute_max_translation))

    # Calculate the mean translation vector to correct for moving of the lung in general.
    mean_translation_vector = np.mean(translations, axis=0)
    print(mean_translation_vector)

    # Iterate over the translations and calculate the color.
    for translation in translations:

        # Correct the translation with the mean translation_vector to correct for moving of the lung in general.
        translation_corrected = translation - mean_translation_vector

        x_translation, y_translation = translation_corrected

        # Calculate the length of the translation vector.
        translation_length = np.sqrt(x_translation ** 2 + y_translation ** 2)

        # Normalize the translation length between 0 and 1.
        intensity = (translation_length) / (absolute_max_translation_length)

        # Correct intensity if it is a greater then 1.
        if intensity > 1:
            intensity == 1

        # Determine the angle of the vector in radians and convert it to the angle in degrees.
        angle_radians = np.arctan2(abs(y_translation), abs(x_translation))
        angle_degrees = np.degrees(angle_radians)

        if (x_translation >= 0) and (y_translation >= 0): # Right, upper quadrant of the unit circle.
            angle = angle_degrees
        
        elif (x_translation <= 0) and (y_translation >= 0): # Left, upper quadrant of the unit circle.
            angle = 90 + (90 - angle_degrees)

        elif (x_translation <= 0) and (y_translation <= 0): # left, lower quadrant of the unit circle.
            angle = 180 + angle_degrees

        elif (x_translation >= 0) and (y_translation <= 0): # Right, lower quadrant of the unit circle.
            angle = 270 + (90 - angle_degrees)

        else:
            angle = 0 # When problems.
        
        # Define the colors in the colorbar gradient.
        color_right = (0.5, 1.0, 0.0)  # Bright lime green.
        color_up = (1.0, 0.0, 0.0)  # Pure red.
        color_left = (0.5, 0.0, 1.0)  # Violet.
        color_down = (0.0, 1.0, 1.0)  # Cyan.
        colors_of_colorbar = [color_right, color_up, color_left, color_down, color_right]

        # Create a custom colormap with linear interpolation between the given colors.
        cmap = LinearSegmentedColormap.from_list('custom_colorbar', colors_of_colorbar)

        # Normalize the translation angle between 0 and 1.
        angle_normalized = angle / 360.0
        
        # Get the color from the colormap based on the normalized angle.
        color_out_of_colorbar = cmap(angle_normalized)
        
        # Check for NaN (Not a Number) in RGB color.
        if np.isnan(color_out_of_colorbar).any():
            color_out_of_colorbar = [0, 0, 0]

        # Scale RGB values to the range of 0-255.
        red = (color_out_of_colorbar[0] * 255)
        green = (color_out_of_colorbar[1] * 255)
        blue = (color_out_of_colorbar[2] * 255)

        # Make the color whiter as the intensity of the vector size decreases.
        red = red + ((255 - red) * (1 - intensity))
        green = green + ((255 - green) * (1 - intensity))
        blue = blue + ((255 - blue) * (1 - intensity))

        # Cap the color values at 255.
        red = min(int(red), int(255))
        green = min(int(green), int(255))
        blue = min(int(blue), int(255))

        # Assign the color values to the list.
        color = (red, green, blue)
        colors.append(color)

    # Normalize the RGB color values and clip them to the values 0 to 1.
    normalized_colors = np.array(colors) / 255.0
    normalized_colors = np.clip(normalized_colors, 0, 1)

    # Get colorplot.
    colorplot = get_colorplot(image0, points_to_follow, normalized_colors)

    # Get colorimage.
    colorimage = get_colorimage(image0, points_to_follow, normalized_colors)

    # Get color circle.
    up_magnitude = max_positive_y_translations[-1] # Get the last value out of the list. This item is the maximum for this index of the for loop.
    up_max_magnitude = max(max_positive_y_translations) # Get the max out of the list, because this is the maximum until now.
    up_color_bar = (0, 0, 255)

    right_magnitude = max_positive_x_translations[-1] # Get the last value out of the list. This item is the maximum for this index of the for loop.
    right_max_magnitude = max(max_positive_x_translations) # Get the max out of the list, because this is the maximum until now.
    right_color_bar = (0, 255, int(255/2))

    down_magnitude = min_negative_y_translations[-1] # Get the last value out of the list. This item is the maximum for this index of the for loop.
    down_max_magnitude = min(min_negative_y_translations) # Get the max out of the list, because this is the maximum until now.
    down_color_bar = (255, 255, 0)

    left_magnitude = min_negative_x_translations[-1] # Get the last value out of the list. This item is the maximum for this index of the for loop.
    left_max_magnitude = min(min_negative_x_translations) # Get the max out of the list, because this is the maximum until now.
    left_color_bar = (255, 0, int(255/2))

    # Get colorcircle.
    colorcircle = get_color_circle(up_magnitude, up_max_magnitude, up_color_bar, right_magnitude, right_max_magnitude, right_color_bar, down_magnitude, down_max_magnitude, down_color_bar, left_magnitude, left_max_magnitude, left_color_bar)

    # Return the opticalflow, colorimage, colorplot and colorcircle.
    return opticalflow, colorimage, colorplot, colorcircle

# This function gets the colorcircle.
def get_colorplot(lung_image, points_to_follow, normalized_colors):
    
    # Reset the plot.
    fig, ax = plt.subplots()

    # Plot the points with color-coded translations.
    x_values = points_to_follow.squeeze()[:, 0]
    y_values = points_to_follow.squeeze()[:, 1]

    # Plot the points with color-coded translations
    plt.scatter(x_values, y_values, c=normalized_colors, s=1)  # Adjust the 's' parameter as desired.

    # Set the x-axis and y-axis limits
    plt.xlim(0, lung_image.width)  # Set the x-axis limits to the width of image0.
    plt.ylim(0, lung_image.height)  # Set the y-axis limits to the height of image0.

    # Invert the y-axis.
    plt.gca().invert_yaxis()

    # Add labels and title to the plot.
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Color-coded Points based on Translations')

    # Assign the plot to the variable colorplot.
    colorplot = plt

    return colorplot

def get_colorimage(background_image, points_to_follow, normalized_colors):
    # Convert grayscale image to three-channel BGR.
    background_image_bgr = cv2.cvtColor(np.uint8(background_image), cv2.COLOR_GRAY2BGR)

    # Plot the points with color-coded translations.
    x_values = points_to_follow.squeeze()[:, 0]
    y_values = points_to_follow.squeeze()[:, 1]

    # Set the colors of specified coordinates
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        # Get the corresponding normalized color.
        color = normalized_colors[i]

        # Convert the color to BGR format.
        bgr_color = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))

        # Set the color at the specified coordinates.
        colorimage_bgr = cv2.circle(background_image_bgr, (int(x), int(y)), 1, bgr_color, -1)

    # Convert the image to RGB format.
    colorimage = cv2.cvtColor(colorimage_bgr, cv2.COLOR_BGR2RGB)
    
    return colorimage

def get_color_circle(up_magnitude, up_max_magnitude, up_color_bar, right_magnitude, right_max_magnitude, right_color_bar, down_magnitude, down_max_magnitude, down_color_bar, left_magnitude, left_max_magnitude, left_color_bar):
    # Load the images and mask.
    colorcircle_path = colorcircle_general_path + "/colorcircle.png"
    colorcircle = cv2.imread(colorcircle_path)

    # Get the image width and height.
    colorcircle_height, colorcircle_width, _ = colorcircle.shape

    # Radius center circle.
    radius_center_circle = 24

    # General parameters colorcircle.
    height_rectangle = 24
    max_length_bar = colorcircle_width * 0.41
    font_scale = 0.7
    
    draw_up_bar_and_label(colorcircle, up_magnitude, up_max_magnitude, up_color_bar, height_rectangle, max_length_bar, font_scale)
    draw_right_bar_and_label(colorcircle, right_magnitude, right_max_magnitude, right_color_bar, height_rectangle, max_length_bar, font_scale)
    draw_down_bar_and_label(colorcircle, down_magnitude, down_max_magnitude, down_color_bar, height_rectangle, max_length_bar, font_scale)
    draw_left_bar_and_label(colorcircle, left_magnitude, left_max_magnitude, left_color_bar, height_rectangle, max_length_bar, font_scale)

    # Draw center circle.
    # Set the coordinates and radius of the circle.
    center_x = int(colorcircle_height / 2)
    center_y = int(colorcircle_width / 2)
    radius = radius_center_circle

    # Set the color of the center circle (BGR format).
    color = (0, 0, 0)

    # Draw the circle on the image.
    cv2.circle(colorcircle, (center_x, center_y), radius, color, thickness=-1)
    
    return colorcircle

def draw_up_bar_and_label(colorcircle, up_magnitude, up_max_magnitude, color_bar, thickness_rectangle, max_length_bar, font_scale):
    # Set the basic parameters.
    color_max_bar = tuple(c // 3 for c in color_bar) # Make the bar darker.

    # Get the image width and height.
    colorcircle_height, colorcircle_width, _ = colorcircle.shape

    # Determine the length. If statement to prevent devided by zero.
    if (up_magnitude == up_max_magnitude):
        factor = 1
    else:
        factor = up_magnitude / up_max_magnitude
    
    # Determine the length of the colored bar.
    true_length_bar = max_length_bar * factor

    radius = int(thickness_rectangle / 2)

    # Set the parameters for the rectangle.
    x_start_rectangle_max_bar = (colorcircle_width / 2) - thickness_rectangle / 2
    x_start_rectangle_bar = (colorcircle_width / 2) - thickness_rectangle / 2

    x_end_rectangle_max_bar = x_start_rectangle_max_bar + thickness_rectangle
    x_end_rectangle_bar = x_start_rectangle_bar + thickness_rectangle

    y_end_rectangle_max_bar = colorcircle_height / 2
    y_end_rectangle_bar = colorcircle_height / 2

    y_start_rectangle_max_bar = y_end_rectangle_max_bar - max_length_bar
    y_start_rectangle_bar = y_end_rectangle_max_bar - true_length_bar

    start_point_max_bar = (int(x_start_rectangle_max_bar), int(y_start_rectangle_max_bar))
    start_point_bar = (int(x_start_rectangle_bar), int(y_start_rectangle_bar))

    end_point_max_bar = (int(x_end_rectangle_max_bar), int(y_end_rectangle_max_bar))
    end_point_bar = (int(x_end_rectangle_bar), int(y_end_rectangle_bar))

    thickness_fill_rectangle = -1  # Negative thickness will fill the rectangle.

    # Draw the rectangle for the max bar on the colorplot.
    cv2.rectangle(colorcircle, start_point_max_bar, end_point_max_bar, color_max_bar, thickness_fill_rectangle)
    # Draw the rectangle on the colorplot.
    cv2.rectangle(colorcircle, start_point_bar, end_point_bar, color_bar, thickness_fill_rectangle)

    # Set the parameters for the circle.
    center_x_max_bar = int(colorcircle_width / 2)
    center_x_bar = int(colorcircle_width / 2)

    center_y_max_bar = int(y_start_rectangle_max_bar)
    center_y_bar = int(y_start_rectangle_bar)

    radius = int(thickness_rectangle / 2)

    # Draw the circle for the max bar on the colorplot.
    cv2.circle(colorcircle, (center_x_max_bar, center_y_max_bar), radius, color_max_bar, thickness=-1)
    # Draw the circle on the colorplot.
    cv2.circle(colorcircle, (center_x_bar, center_y_bar), radius, color_bar, thickness=-1)
    
    # Set the start and end coordinates for the label. The coordinates are manually determined so that the labels fit in the white rounded rectangles of the color circle image.
    start_x = int(0.056 * colorcircle_width)
    end_x = int(0.278 * colorcircle_width)
    start_y = int(0.405 * colorcircle_height)
    end_y = int(0.461 * colorcircle_height)

    # Calculate the center positions for x and y coordinates.
    center_x = (start_x + end_x) // 2
    center_y = (start_y + end_y) // 2

    # Set the label text
    label = str(round(up_magnitude, 2)) + "... / " + str(round(up_max_magnitude, 2)) + "..."

    # Set the font, scale, color, and thickness of the text.
    font = cv2.FONT_HERSHEY_SIMPLEX
    subtraction_color = (255, 255, 255)  # BGR color format (red color).
    thickness = 1

    # Get the size of the label text
    (label_width, label_height), _ = cv2.getTextSize(label, font, font_scale, thickness)

    # Calculate the text position to center it
    text_x = center_x - (label_width // 2)
    text_y = center_y + (label_height // 2)  # Adjust the y-coordinate based on label height.

    # Create a blank image of the same size as colorcircle.
    label_image = np.zeros_like(colorcircle)

    # Draw the label on the image.
    cv2.putText(label_image, label, (text_x, text_y), font, font_scale, subtraction_color, thickness)

    # Rotate the image 90 degrees clockwise.
    label_image_rotated = cv2.rotate(label_image, cv2.ROTATE_90_CLOCKWISE)

    # Overlay the rotated label image on colorcircle.
    cv2.subtract(colorcircle, label_image_rotated, colorcircle)

def draw_right_bar_and_label(colorcircle, right_magnitude, right_max_magnitude, color_bar, height_rectangle, max_length_bar, font_scale):
    # Set the basic parameters.
    color_max_bar = tuple(c // 3 for c in color_bar) # Make the bar darker.

    # Get the image width and height.
    colorcircle_height, colorcircle_width, _ = colorcircle.shape
    
    # Determine the length. If statement to prevent devided by zero.
    if (right_magnitude == right_max_magnitude):
        factor = 1
    else:
        factor = right_magnitude / right_max_magnitude
    
    # Determine the length of the colored bar.
    true_length_bar = max_length_bar * factor

    radius = int(height_rectangle / 2)

    # Set the parameters for the rectangle.
    x_start_rectangle_max_bar = colorcircle_width / 2
    x_start_rectangle_bar = colorcircle_width / 2

    x_end_rectangle_max_bar = x_start_rectangle_max_bar + max_length_bar
    x_end_rectangle_bar = x_start_rectangle_bar + true_length_bar

    y_start_rectangle_max_bar = (colorcircle_height / 2) - height_rectangle / 2
    y_start_rectangle_bar = (colorcircle_height / 2) - height_rectangle / 2

    y_end_rectangle_max_bar = y_start_rectangle_max_bar + height_rectangle
    y_end_rectangle_bar = y_start_rectangle_bar + height_rectangle

    start_point_max_bar = (int(x_start_rectangle_max_bar), int(y_start_rectangle_max_bar))
    start_point_bar = (int(x_start_rectangle_bar), int(y_start_rectangle_bar))

    end_point_max_bar = (int(x_end_rectangle_max_bar), int(y_end_rectangle_max_bar))
    end_point_bar = (int(x_end_rectangle_bar), int(y_end_rectangle_bar))

    thickness = -1  # Negative thickness will fill the rectangle.

    # Draw the rectangle for the max bar on the colorplot.
    cv2.rectangle(colorcircle, start_point_max_bar, end_point_max_bar, color_max_bar, thickness)
    # Draw the rectangle on the colorplot.
    cv2.rectangle(colorcircle, start_point_bar, end_point_bar, color_bar, thickness)

    # Set the parameters for the circle.
    center_x_max_bar = int(x_end_rectangle_max_bar)
    center_x_bar = int(x_end_rectangle_bar)

    center_y_max_bar = int(colorcircle_height / 2)
    center_y_bar = int(colorcircle_height / 2)

    radius = int(height_rectangle / 2)

    # Draw the circle for the max bar on the colorplot.
    cv2.circle(colorcircle, (center_x_max_bar, center_y_max_bar), radius, color_max_bar, thickness=-1)
    # Draw the circle on the colorplot.
    cv2.circle(colorcircle, (center_x_bar, center_y_bar), radius, color_bar, thickness=-1)
    
    # Set the start and end coordinates for the label.
    start_x = int(0.722 * colorcircle_width)
    end_x = int(0.944 * colorcircle_width)
    start_y = int(0.539 * colorcircle_height)
    end_y = int(0.594 * colorcircle_height)

    # Calculate the center positions for x and y coordinates.
    center_x = (start_x + end_x) // 2
    center_y = (start_y + end_y) // 2

    # Set the label text.
    label = str(round(right_magnitude, 2)) + "... / " + str(round(right_max_magnitude, 2)) + "..."

    # Set the font, scale, color, and thickness of the text.
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 0)  # BGR color format (red color).
    thickness = 1

    # Get the size of the label text.
    (label_width, label_height), _ = cv2.getTextSize(label, font, font_scale, thickness)

    # Calculate the text position to center it.
    text_x = center_x - (label_width // 2)
    text_y = center_y + (label_height // 2)  # Adjust the y-coordinate based on label height.

    # Draw the label on the image.
    cv2.putText(colorcircle, label, (text_x, text_y), font, font_scale, color, thickness)

def draw_down_bar_and_label(colorcircle, down_magnitude, down_max_magnitude, color_bar, thickness_rectangle, max_length_bar, font_scale):
    # Set the basic parameters.
    color_max_bar = tuple(c // 3 for c in color_bar) # Make the bar darker.

    # Get the image width and height.
    colorcircle_height, colorcircle_width, _ = colorcircle.shape

    # Determine the length. If statement to prevent devided by zero.
    if (down_magnitude == down_max_magnitude):
        factor = 1
    else:
        factor = down_magnitude / down_max_magnitude
    
    # Determine the length of the colored bar.
    true_length_bar = max_length_bar * factor

    radius = int(thickness_rectangle / 2)

    # Set the parameters for the rectangle.
    x_start_rectangle_max_bar = (colorcircle_width / 2) - thickness_rectangle / 2
    x_start_rectangle_bar = (colorcircle_width / 2) - thickness_rectangle / 2

    x_end_rectangle_max_bar = x_start_rectangle_max_bar + thickness_rectangle
    x_end_rectangle_bar = x_start_rectangle_bar + thickness_rectangle

    y_start_rectangle_max_bar = colorcircle_height / 2
    y_start_rectangle_bar = colorcircle_height / 2

    y_end_rectangle_max_bar = y_start_rectangle_max_bar + max_length_bar
    y_end_rectangle_bar = y_start_rectangle_max_bar + true_length_bar

    start_point_max_bar = (int(x_start_rectangle_max_bar), int(y_start_rectangle_max_bar))
    start_point_bar = (int(x_start_rectangle_bar), int(y_start_rectangle_bar))

    end_point_max_bar = (int(x_end_rectangle_max_bar), int(y_end_rectangle_max_bar))
    end_point_bar = (int(x_end_rectangle_bar), int(y_end_rectangle_bar))

    thickness_fill_rectangle = -1  # Negative thickness will fill the rectangle.

    # Draw the rectangle for the max bar on the colorplot.
    cv2.rectangle(colorcircle, start_point_max_bar, end_point_max_bar, color_max_bar, thickness_fill_rectangle)
    # Draw the rectangle on the colorplot.
    cv2.rectangle(colorcircle, start_point_bar, end_point_bar, color_bar, thickness_fill_rectangle)

    # Set the parameters for the circle.
    center_x_max_bar = int(colorcircle_width / 2)
    center_x_bar = int(colorcircle_width / 2)

    center_y_max_bar = int(y_end_rectangle_max_bar)
    center_y_bar = int(y_end_rectangle_bar)

    radius = int(thickness_rectangle / 2)

    # Draw the circle for the max bar on the colorplot.
    cv2.circle(colorcircle, (center_x_max_bar, center_y_max_bar), radius, color_max_bar, thickness=-1)
    # Draw the circle on the colorplot.
    cv2.circle(colorcircle, (center_x_bar, center_y_bar), radius, color_bar, thickness=-1)
    
    # Set the start and end coordinates for the label.
    start_x = int(0.722 * colorcircle_width)
    end_x = int(0.944 * colorcircle_width)
    start_y = int(0.405 * colorcircle_height)
    end_y = int(0.461 * colorcircle_height)

    # Calculate the center positions for x and y coordinates.
    center_x = (start_x + end_x) // 2
    center_y = (start_y + end_y) // 2

    # Set the label text.
    label = str(round(down_magnitude, 2)) + "... / " + str(round(down_max_magnitude, 2)) + "..."

    # Set the font, scale, color, and thickness of the text.
    font = cv2.FONT_HERSHEY_SIMPLEX
    subtraction_color = (255, 255, 255)  # BGR color format (red color).
    thickness = 1

    # Get the size of the label text.
    (label_width, label_height), _ = cv2.getTextSize(label, font, font_scale, thickness)

    # Calculate the text position to center it.
    text_x = center_x - (label_width // 2)
    text_y = center_y + (label_height // 2)  # Adjust the y-coordinate based on label height.

    # Create a blank image of the same size as colorcircle.
    label_image = np.zeros_like(colorcircle)

    # Draw the label on the image.
    cv2.putText(label_image, label, (text_x, text_y), font, font_scale, subtraction_color, thickness)

    # Rotate the image 90 degrees clockwise.
    label_image_rotated = cv2.rotate(label_image, cv2.ROTATE_90_CLOCKWISE)

    # Overlay the rotated label image on colorcircle.
    cv2.subtract(colorcircle, label_image_rotated, colorcircle)

def draw_left_bar_and_label(colorcircle, left_magnitude, left_max_magnitude, color_bar, height_rectangle, max_length_bar, font_scale):
    # Set the basic parameters.
    color_max_bar = tuple(c // 3 for c in color_bar) # Make the bar darker.

    # Get the image width and height.
    colorcircle_height, colorcircle_width, _ = colorcircle.shape

    # Determine the length. If statement to prevent devided by zero.
    if (left_magnitude == left_max_magnitude):
        factor = 1
    else:
        factor = left_magnitude / left_max_magnitude
    
    # Determine the length of the colored bar.
    true_length_bar = max_length_bar * factor

    radius = int(height_rectangle / 2)

    # Set the parameters for the rectangle.
    x_end_rectangle_max_bar = colorcircle_width / 2
    x_end_rectangle_bar = colorcircle_width / 2

    x_start_rectangle_max_bar = x_end_rectangle_max_bar - max_length_bar
    x_start_rectangle_bar = x_end_rectangle_bar - true_length_bar

    y_start_rectangle_max_bar = (colorcircle_height / 2) - height_rectangle / 2
    y_start_rectangle_bar = (colorcircle_height / 2) - height_rectangle / 2

    y_end_rectangle_max_bar = y_start_rectangle_max_bar + height_rectangle
    y_end_rectangle_bar = y_start_rectangle_bar + height_rectangle

    start_point_max_bar = (int(x_start_rectangle_max_bar), int(y_start_rectangle_max_bar))
    start_point_bar = (int(x_start_rectangle_bar), int(y_start_rectangle_bar))

    end_point_max_bar = (int(x_end_rectangle_max_bar), int(y_end_rectangle_max_bar))
    end_point_bar = (int(x_end_rectangle_bar), int(y_end_rectangle_bar))

    thickness = -1  # Negative thickness will fill the rectangle.

    # Draw the rectangle for the max bar on the colorplot.
    cv2.rectangle(colorcircle, start_point_max_bar, end_point_max_bar, color_max_bar, thickness)
    # Draw the rectangle on the colorplot.
    cv2.rectangle(colorcircle, start_point_bar, end_point_bar, color_bar, thickness)

    # Set the parameters for the circle.
    center_x_max_bar = int(x_start_rectangle_max_bar)
    center_x_bar = int(x_start_rectangle_bar)

    center_y_max_bar = int(colorcircle_height / 2)
    center_y_bar = int(colorcircle_height / 2)

    radius = int(height_rectangle / 2)

    # Draw the circle for the max bar on the colorplot.
    cv2.circle(colorcircle, (center_x_max_bar, center_y_max_bar), radius, color_max_bar, thickness=-1)
    # Draw the circle on the colorplot.
    cv2.circle(colorcircle, (center_x_bar, center_y_bar), radius, color_bar, thickness=-1)
    
    # Set the start and end coordinates for the label.
    start_x = int(0.056 * colorcircle_width)
    end_x = int(0.278 * colorcircle_width)
    start_y = int(0.539 * colorcircle_height)
    end_y = int(0.594 * colorcircle_height)

    # Calculate the center positions for x and y coordinates.
    center_x = (start_x + end_x) // 2
    center_y = (start_y + end_y) // 2

    # Set the label text.
    label = str(round(left_magnitude, 2)) + "... / " + str(round(left_max_magnitude, 2)) + "..."

    # Set the font, scale, color, and thickness of the text.
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 0)  # BGR color format (red color).
    thickness = 1

    # Get the size of the label text.
    (label_width, label_height), _ = cv2.getTextSize(label, font, font_scale, thickness)

    # Calculate the text position to center it.
    text_x = center_x - (label_width // 2)
    text_y = center_y + (label_height // 2)  # Adjust the y-coordinate based on label height.

    # Draw the label on the image.
    cv2.putText(colorcircle, label, (text_x, text_y), font, font_scale, color, thickness)

def save_video():
    # Load the images and mask.
    video_general_path = general_path + "/video"

    # Image paths.
    image_paths = glob.glob(colorimage_general_path + "/*.jpg")
    # Sort the image file names in a natural numeric order.
    image_paths = sorted(image_paths, key=lambda x: int(''.join(filter(str.isdigit, x))))

    # Get the first image to determine the frame size.
    first_image = cv2.imread(image_paths[0])
    height, width, _ = first_image.shape

    # Define the video codec and create a VideoWriter object.
    fps = 15 # Frames per second.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = video_general_path + "/colorimage.mp4"
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Iterate over the image files and write each frame to the video.
    for image_path in image_paths:
        # Read the image.
        frame = cv2.imread(image_path)

        # Write the frame to the video.
        video_writer.write(frame)

    # Release the video writer and close the video file.
    video_writer.release()

    print('Video created successfully.')

def frames_to_colorplot():
    
    # For loop that runs the whole program and get everything frame by frame.
    for i in range(start_frame, len(all_images_paths) - 1, frames_skip):
        image0_path = all_images_paths[i]
        image1_path = all_images_paths[i + frames_skip]
        mask0_path = all_masks_paths[i]

        # Print the frame numbers between which the optical flow movement is calculated.
        print("Image " + str(i) + " -> " + str(i + frames_skip) + " / " + str(len(all_images_paths) - 1))

        # Get opticalflow, color image, color plot and color circle.
        opticalflow, colorimage, colorplot, colorcircle = get_colorimage_colorplot_colorcircle(image0_path, image1_path, mask0_path)

        if frames_skip == 1:
            string_index_name = str(i)
        else:
            string_index_name = str(i) + "-" + str(i + frames_skip)

        dpi = 500  # Set the dpi of the image.
        jpeg_quality = 500  # Calculate JPEG quality based on dpi.

        # Save optical flow image using PIL.
        opticalflow_path = opticalflow_general_path + "/Opticalflow_" + string_index_name + ".jpg"
        opticalflow_pil = Image.fromarray(opticalflow)
        opticalflow_pil.save(opticalflow_path, format='JPEG', quality=jpeg_quality)

        # Save color image using PIL.
        colorimage_path = colorimage_general_path + "/Colorimage_" + string_index_name + ".jpg"
        colorimage_pil = Image.fromarray(colorimage)
        colorimage_pil.save(colorimage_path, format='JPEG', quality=jpeg_quality)

        # Save color plot using Matplotlib and then convert to JPEG using PIL.
        colorplot_path = colorplot_general_path + "/Colorplot_" + string_index_name + ".jpg"
        colorplot.savefig(colorplot_path, format='PNG', dpi=dpi)
        colorplot_pil = Image.open(colorplot_path).convert('RGB')
        colorplot_pil.save(colorplot_path, format='JPEG', quality=jpeg_quality)

        # Save color circle using PIL.
        colorcircle_path = colorcircle_general_path + "/Colorcircle_" + string_index_name + ".jpg"
        colorcircle_rgb = cv2.cvtColor(colorcircle, cv2.COLOR_BGR2RGB)
        colorcircle_pil = Image.fromarray(colorcircle_rgb)
        colorcircle_pil.save(colorcircle_path, format='JPEG', quality=jpeg_quality)

    # Save colorimages to video.
    save_video()

# Mouse callback function.
def mouse_callback(event, mouse_x, mouse_y, flags, param):
    global x_mouse_click, y_mouse_click

    frame = param

    if event == cv2.EVENT_LBUTTONDOWN:
        x_mouse_click = mouse_x
        y_mouse_click = mouse_y

def draw_lesion_on_canvas(canvas, x_center_lesion, y_center_lesion):
    # Load the PNG image with alpha channel.
    png_path = general_path + "/augmented_overlay/augmented.png"
    png_image = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)

    # Resize png to certain dimensions.
    desired_width = 20
    desired_height = 20

    # Coordinates of the left upper corner of the png image on the canvas.
    x_begin = int(x_center_lesion - (desired_width / 2))
    y_begin = int(y_center_lesion - (desired_height / 2))
    x_end = x_begin + desired_width
    y_end = y_begin + desired_height

    # Read a part of the lung image as background for the png image.
    background = canvas[y_begin:y_end, x_begin:x_end]

    # Resize the image.
    resized_png_image = cv2.resize(png_image, (background.shape[1], background.shape[0]))

    # Extract the alpha (transparency) channel from the image.
    alpha = resized_png_image[:, :, 3]

    # Remove the alpha channel from the image.
    png_image_rgb = resized_png_image[:, :, :3]

    # Convert the alpha channel to a mask.
    mask = np.expand_dims(alpha, axis=2) / 255.0

    # Convert the mask to have 3 channels.
    mask = np.dstack([mask] * 3)

    # Apply the mask to the resized PNG image.
    masked_image = cv2.multiply(mask, png_image_rgb.astype(float))

    # Apply the inverse mask to the background image.
    masked_background = cv2.multiply(1.0 - mask, background.astype(float))

    # Combine the masked PNG image and the masked background.
    png_with_background = cv2.add(masked_image, masked_background).astype(np.uint8)

    # Combine the png_with_background and original image.
    canvas[y_begin:y_end, x_begin:x_end] = png_with_background

    return canvas

def frames_to_lesion_track():

    # Read first frame.
    first_frame = cv2.imread(all_images_paths[0])
    first_frame_overlay = cv2.imread(all_images_masks_overlay_path[0])
    
    # Save x and y values.
    x_track = 0
    y_track = 0

    frame_name = "Frames"
    cv2.namedWindow(frame_name)
    cv2.setMouseCallback(frame_name, mouse_callback)
    cv2.imshow(frame_name, first_frame_overlay)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Exit the program if 'q' is pressed.
            break
        elif x_mouse_click is not None and y_mouse_click is not None:
            # Draw the lesion on the frame.
            x_track = x_mouse_click
            y_track = y_mouse_click
            
            draw_lesion_on_canvas(first_frame_overlay, x_track, y_track)
            cv2.imshow(frame_name, first_frame_overlay)
            break

    cv2.waitKey(0) # Wait until a key is pressed.

    tracking_rectangle_size = 10 # Number of pixels left/right/up/down from the center pixel.

    # Creates empty lists.
    list_framenumber = []
    list_start_coordinates = []
    list_average_end_coordinates = []
    list_average_length = []
    list_median_length = []

    # Analyze frame by frame and draw the augmented lesion and tracking rectangle.
    for i in range(len(all_images_paths) - 1):
        image0 = cv2.imread(all_images_paths[i])
        image1 = cv2.imread(all_images_paths[i + 1])
        image1_overlay = cv2.imread(all_images_masks_overlay_path[i + 1])

        # Convert the RGB image to grayscale, because this is needed for optical flow.
        image0_gray = cv2.cvtColor(image0, cv2.COLOR_RGB2GRAY)
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

        # X and y coordinates of center of tracking area.
        x_center_tracking = int(x_track)
        y_center_tracking = int(y_track)

        # Borders of the image.
        height_image0, width_image0 = image0.shape[:2]

        # Define the boundaries of the square.
        x_min = max(0, x_center_tracking - tracking_rectangle_size)  # Ensure x_min is within the image boundaries.
        x_max = min(width_image0 - 1, x_center_tracking + tracking_rectangle_size)  # Ensure x_max is within the image boundaries.
        y_min = max(0, y_center_tracking - tracking_rectangle_size)  # Ensure y_min is within the image boundaries.
        y_max = min(height_image0 - 1, y_center_tracking + tracking_rectangle_size)  # Ensure y_max is within the image boundaries.
        
        # Create meshgrid of coordinates around a certain pixel coordinate.
        x_coords, y_coords = np.meshgrid(range(x_min, x_max + 1), range(y_min, y_max + 1))

        # Flatten the coordinates.
        pixel_coordinates = np.column_stack((x_coords.flatten(), y_coords.flatten()))
        pixel_coordinates = pixel_coordinates.astype(np.float32)

        # Calculate optical flow using Lucas-Kanade method. p1 is an array with the new coordinates in frame1 of the pixels that are followed in frame0.
        p1, st, err = cv2.calcOpticalFlowPyrLK(image0_gray, image1_gray, pixel_coordinates, None, **LK_PARAMS)

        # Calcultate the translation.
        translations = p1 - pixel_coordinates
        x_translations = translations[0]
        y_translations = translations[1]

        # Mean translation.
        mean_x_translation = np.mean(x_translations)
        mean_y_translation = np.mean(y_translations)

        # Median translation.
        median_x_translation = np.median(x_translations)
        median_y_translation = np.median(y_translations)

        # Assign the mean translation to the variable motion.
        motion_x = mean_x_translation
        motion_y = mean_y_translation

        # Update the central coordinate of the augmented lesion.
        x_track = x_track + motion_x
        y_track = y_track + motion_y

        # Draw the lesion on the current frame.
        draw_lesion_on_canvas(image1_overlay, x_track, y_track)

        # Draw rectangle around border of tracking area.
        cv2.rectangle(image1_overlay, (x_min, y_min), (x_max, y_max), (0, 0, 255), thickness=1)

        # Show the image.
        cv2.imshow(frame_name, image1_overlay)
        # Wait 1 ms or until a key is pressed.
        cv2.waitKey(1)

        # Append the values to the created lists.
        list_framenumber.append(str(i) + "->" + str(i+1))
        list_start_coordinates.append("[" + str(x_center_tracking) + " , " + str(y_center_tracking) + "]")
        list_average_end_coordinates.append("[" + str(int(x_track)) + " , " + str(int(y_track)) + "]")
        list_average_length.append(sqrt((mean_x_translation**2) + (mean_y_translation**2)).real)
        list_median_length.append(sqrt((median_x_translation**2) + (median_y_translation**2)).real)

    # Print the values of the list in the output window for data analysis.
    for i in range(len(list_framenumber)):
        
        print(list_framenumber[i])
    
    print("")

    for i in range(len(list_framenumber)):
        print(list_start_coordinates[i])
    
    print("")

    for i in range(len(list_framenumber)):    
        print(list_average_end_coordinates[i])
    
    print("")
    
    for i in range(len(list_framenumber)):
        print(round(list_average_length[i],2))
    
    print("")
    
    for i in range(len(list_framenumber)):
        print(round(list_median_length[i],2))
        
    # Wait until a key is pressed.
    cv2.waitKey(0)

# Booleans to set the function of the program.
frames_to_colorplot_true_false = False
frames_to_lesion_track_true_false = True

# Switch between the both optical flow programs.
if (frames_to_colorplot_true_false == True):
    frames_to_colorplot()

if (frames_to_lesion_track_true_false == True):
    frames_to_lesion_track()

# Wait until a key is pressed.
cv2.waitKey(0)
# Destroy all windows.
cv2.destroyAllWindows()