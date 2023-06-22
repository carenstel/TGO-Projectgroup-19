# --------------------------------------------------
# Script: Uses the deep learning model to create masks.
# After that the masks are analyzed with a Fourier Transform to detect heart beats and lung motion.
# --------------------------------------------------

# Needed packages for the code.
from glob import glob
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import model_from_json, load_model
from lib2to3.pgen2.token import PERCENT
from numpy.fft import fft, ifft
from PIL import Image
from scipy.signal import find_peaks
from skimage import measure, morphology
from tqdm import tqdm
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np 
import os
import os
import pandas as pd
import pydicom
import tensorflow as tf

# Set screen settings.
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_DEVICE_PIXEL_RATIO"] = "1"

# Load the deep learning model.
model_path = "C:/Users/User/source/repos/LungSegmentationDeepLearning/LungSegmentationDeepLearning/lung_segmentation_model.h5"
model = load_model(model_path)

# Choose the number of greatest regions to keep. On each image, the two lungs are visible, so it works to keep only the two greatest white areas. It is also possible to work with a minimum number of pixels to keep an object. This is needed to prevent small white particles.
num_regions_to_keep = 2 # Modify this variable to choose the desired number of regions.

# Specify the path to the DICOM file
dicom_path = 'C:/Users/User/Documents/Technische Geneeskunde/Module 12 TGO/Python/fluoroscopy/dicom_RadboudUMC/patient 3/IM_0003'

# Specify the path to save the images and masks.
general_save_directory = "C:/Users/User/Documents/Technische Geneeskunde/Module 12 TGO/Python/fluoroscopy"
save_image_directory = general_save_directory + "/image"
save_image_contrast_directory = general_save_directory + "/image_contrast"
save_mask_directory = general_save_directory + "/mask"
save_image_mask_overlay_directory = general_save_directory + "/image_mask_overlay"
save_heart_and_lung_signal = general_save_directory + "/breathing_and_heart_signal"

## This function ensures that masks that lie on the edge are a closed figure.
#def connect_lines_with_border(mask):

#    # Determines the dimensions of the mask.
#    height, width = mask.shape

#    # Find contours in the mask.
#    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#    # Iterate through each contour.
#    for contour in contours:
#        x, y, w, h = cv2.boundingRect(contour)

#        # Check if the contour touches the border.
#        if x == 0 or y == 0 or x + w == width or y + h == height:
#            # Get the endpoints of the contour.
#            points = contour.squeeze()

#            # Connect the endpoints with a straight white line.
#            cv2.line(mask, tuple(points[0]), tuple(points[-1]), 255, thickness=1)

#    # Returns the connected mask.
#    return mask

## Function that gets the mask from the image and deep learning model.
#def get_mask(frame):
    
#    # Preprocess the frame.
#    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#    # Invert the image, because the trained image are inverted.
#    grayscale_image = (255 - grayscale_image)

#    # Resize the image to the size the deep learning model images are also trained on.
#    resized_image = cv2.resize(grayscale_image, (512, 512))

#    # Equalize the image (Redistribute the pixel intensities) to get a higher contrast. The deep learning model is also trained with equalized images.
#    equalized_grayscale_image = cv2.equalizeHist(grayscale_image)
#    # Resize the equalized image to the size the deep learning model images are also trained on.
#    resized_image_contrast = cv2.resize(equalized_grayscale_image, (512, 512))
#    # Normalize the intensities of the image. Instead of values between 0-255 values between 0-1.
#    normalized_image_contrast = (resized_image_contrast / 255)
#    input_image_constrast = np.expand_dims(normalized_image_contrast, axis=-1)
#    input_image_constrast = np.expand_dims(input_image_constrast, axis=0)

#    # Predict the mask.
#    mask = model.predict(input_image_constrast)
#    # Remove single dimensional axes. For example: (512,512,1) -> (512,512).
#    mask = mask.squeeze()
#    # All intensities above the half will be displayed as white, others as black.
#    threshold = 0.5
#    # Get the mask with intensities between 0-255. Also convert it to an aray with int numbers.
#    mask = ((mask > threshold) * 255).astype(np.uint8)

#    # Find connected components in the mask.
#    labels = measure.label(mask)
#    regions = measure.regionprops(labels)
#    # Get all white areas of the mask.
#    areas = [region.area for region in regions]
#    # Sort the regions based on the number of pixels inside.
#    sorted_areas = sorted(areas, reverse=True)

#    # Keep the desired number of greatest regions and discard the rest
#    largest_labels = []
#    for region in regions:

#        # If statement which only get the first x regions.
#        if region.area in sorted_areas[:num_regions_to_keep]:

#            # Append only the largest x regions to the list largest_labels.
#            largest_labels.append(region.label)

#    # Create a new mask with only the desired number of greatest regions
#    mask_new = np.zeros_like(mask)
#    # Set all labels in the list largest_labels as white on the new black mask.
#    for label in largest_labels:
#        mask_new[labels == label] = 255

#    # Apply closing to close the lung boundaries. Small gaps in the boundary will be closed.
#    kernel_size = 5  # Adjust the kernel size as needed
#    kernel = np.ones((kernel_size, kernel_size), np.uint8)
#    closed_mask = cv2.morphologyEx(mask_new, cv2.MORPH_CLOSE, kernel)

#    # Connect lines touching the sides
#    connected_mask = connect_lines_with_border(closed_mask)

#    # Get filtered mask.
#    mask_new = get_filtered_mask(connected_mask)
#    # Set the filtered mask to the mask.
#    mask = mask_new

#    # Calculate percentage of white pixels
#    white_pixels = np.sum(mask == 255)
#    total_pixels = mask.size
#    # Calculates the percentage of white pixels on the mask. How greater the percentage, how greater the lungs are.
#    percentage = (white_pixels / total_pixels) * 100

#    # Return the mask, resized_image, resize_image_contrast and percentage.
#    return mask, resized_image, resized_image_contrast, percentage

## Function that filters the mask. Removes any black pieces present in white lung areas.
#def get_filtered_mask(mask):
#    # Increase canvas size on all sides by padding_size pixels. Later in the code, the boundary pixels are moved one pixel further out. At this point, some boundary pixels may be on the edge, making an outward movement impossible.
#    padding_size = 10

#    # Calculate the dimensions of the new canvas.
#    canvas_width = mask.shape[1] + 2 * padding_size
#    canvas_height = mask.shape[0] + 2 * padding_size

#    # Create a new canvas with the calculated dimensions.
#    new_canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

#    # Calculate the coordinates to paste the mask onto the new canvas. This is important to remove at the end of the function the padding.
#    paste_x = padding_size
#    paste_y = padding_size

#    # Copy the mask onto the center of the new canvas.
#    new_canvas[paste_y:paste_y + mask.shape[0], paste_x:paste_x + mask.shape[1]] = mask

#    # Find connected components in the new canvas.
#    labels_white = measure.label(new_canvas)
#    regions_white = measure.regionprops(labels_white)

#    # Sort the regions based on area in descending order.
#    sorted_regions = sorted(regions_white, key=lambda x: x.area, reverse=True)

#    # Choose the x greatest white regions.
#    selected_regions = sorted_regions[:num_regions_to_keep]

#    # Create a new mask with the same shape as the new canvas.
#    new_mask = np.zeros_like(new_canvas, dtype=np.uint8)

#    # Extract and draw the borders of the selected regions.
#    for region in selected_regions:
#        # Get all coordinates of the region.
#        coords = region.coords

#        # Create a binary mask for the current region
#        region_mask = np.zeros_like(new_canvas)
#        region_mask[coords[:, 0], coords[:, 1]] = 1

#        # Expand the region mask by one pixel. This means that the boundary of the region is shifted out by one pixel.
#        dilated_mask = cv2.dilate(region_mask, kernel=None, iterations=1)

#        # Calculate the border pixels by taking the difference.
#        border_mask = np.logical_xor(dilated_mask, region_mask)

#        # Find connected components in the border mask.
#        labels_borders = measure.label(border_mask)

#        # Find contours in the border mask.
#        contours, _ = cv2.findContours(labels_borders.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#        # Combine all contours into a single contour.
#        combined_contour = np.concatenate(contours)

#        # Create a new mask for the combined contour with the shape of the new canvas.
#        contour_mask = np.zeros_like(new_canvas, dtype=np.uint8)

#        # Draw filled contour on the mask. Recoloring the surface within the boundary overrides any black areas within it.
#        cv2.drawContours(contour_mask, [combined_contour], -1, 255, thickness=cv2.FILLED)

#        # Crop the contour mask to remove padding.
#        cropped_mask = contour_mask[paste_y:paste_y + mask.shape[0], paste_x:paste_x + mask.shape[1]]

#        # Set the cropped mask pixels to white on the new mask.
#        new_mask[paste_y:paste_y + mask.shape[0], paste_x:paste_x + mask.shape[1]] = np.logical_or(
#            new_mask[paste_y:paste_y + mask.shape[0], paste_x:paste_x + mask.shape[1]], cropped_mask
#        )

#    # Threshold the new mask to ensure binary values. Each pixel that has a value not equal to 0 is set to 255.
#    new_mask[new_mask > 0] = 255

#    # Calculate the cropping indices after removing padding from all sides.
#    cropped_indices_x = slice(paste_x, paste_x + mask.shape[1])
#    cropped_indices_y = slice(paste_y, paste_y + mask.shape[0])

#    # Crop the new mask to remove padding from all sides.
#    cropped_new_mask = new_mask[cropped_indices_y, cropped_indices_x]

#    # Create a canvas with the original size.
#    original_size_mask = np.zeros_like(mask, dtype=np.uint8)

#    # Paste the cropped mask onto the canvas with the original size.
#    original_size_mask[:cropped_new_mask.shape[0], :cropped_new_mask.shape[1]] = cropped_new_mask

#    # Return the original_size_mask as a mask that is filtered.
#    return original_size_mask

## Read the DICOM file.
#dicom_data = pydicom.dcmread(dicom_path)

## Access the pixel array.
#pixel_array = dicom_data.pixel_array

## Function to overlay the mask with colors over the image.
#def get_segmented_frame(frame, mask):
    
#    # Resize mask.
#    resized_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

#    # Convert frame to 3-channel grayscale image. Needed because the overlay is colorfull.
#    frame_gray = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

#    # Create green mask color.
#    mask_color = np.zeros_like(frame_gray)
#    mask_color[:, :, 1] = resized_mask  # Assign the resized mask to the green channel
    
#    # Create segmented frame.
#    segmented_frame = cv2.addWeighted(frame_gray, 0.7, mask_color, 0.3, 0) # The green mask is mostly transparent to see through it the luungs.

#    # Detect edges of the segmented lungs.
#    edges = cv2.Canny(resized_mask, 30, 100)

#    # Create a mask with red edges.
#    edges_mask = np.zeros_like(frame_gray)
#    edges_mask[:, :, 2] = edges  # Assign the edges to the red channel. The edge of the mask is red.

#    # Add the red edges to the segmented frame.
#    segmented_frame = cv2.addWeighted(segmented_frame, 1, edges_mask, 1, 0)

#    # Return the segmented frame.
#    return segmented_frame

## Function to preprocess the frame.
#def preprocess_frame(frame):
#    # Normalize the intensity values of the frame to the range of 0 to 255.
#    normalized_frame = ((frame - np.min(frame)) / (np.max(frame) - np.min(frame))) * 255

#    # Convert the normalized frame to uint8 data type.
#    normalized_frame = normalized_frame.astype(np.uint8)

#    # Apply noise reduction.
#    blurred_frame = cv2.GaussianBlur(normalized_frame, (5, 5), 0)

#    # Apply histogram equalization.
#    equalized_frame = cv2.equalizeHist(blurred_frame)

#    # Convert the cropped frame back to color (BGR).
#    cropped_frame_bgr = cv2.cvtColor(equalized_frame, cv2.COLOR_GRAY2BGR)

#    # Return cropped_frame_bgr as preprocessed frame.
#    return cropped_frame_bgr

## Set the frame count to 0 and the prev_frame to the first frame.
#frame_count = 0 # The number of frames already analysed is zero.
#prev_frame = pixel_array[0] # Start from frame 0.

## When necessary, convert the colorimage to grayscale.
#if len(prev_frame.shape) == 3:
#    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#else:
#    prev_frame_gray = prev_frame # Assuming prev_frame is already grayscale

## Set the first image and mask outside the for loop.
#prev_frame_preprocessed = preprocess_frame(prev_frame_gray)
#prev_mask, prev_resized_image, prev_image_contrast, prev_percentage = get_mask(prev_frame_preprocessed)

## For loop that writes all images, contrast images, masks and overlays of the dicom file to the folder.
#for frame_index in tqdm(range(pixel_array.shape[0])):
#    frame = pixel_array[frame_index]

#    # Preprocess the frame.
#    preprocessed_frame = preprocess_frame(frame)

#    # Get mask from preprocessed frame.
#    mask, resized_image, resized_image_contrast, percentage = get_mask(preprocessed_frame)

#    # Get segmented frame.
#    segmented_frame = get_segmented_frame(resized_image, mask)

#    # Save mask.
#    mask_path = os.path.join(save_mask_directory, f"mask_{frame_count}.jpg")
#    cv2.imwrite(mask_path, mask)

#    # Save image.
#    image_path = os.path.join(save_image_directory, f"image_{frame_count}.jpg")
#    cv2.imwrite(image_path, resized_image)

#    # Save contrast image.
#    image_contrast_path = os.path.join(save_image_contrast_directory, f"image_{frame_count}.jpg")
#    cv2.imwrite(image_contrast_path, resized_image_contrast)

#    # Save overlay.
#    segmented_path = os.path.join(save_image_mask_overlay_directory, f"segmented_{frame_count}.jpg")
#    cv2.imwrite(segmented_path, segmented_frame)

#    # Add one to the frame_count. The framecount is used for the filenames.
#    frame_count += 1

# Create a list of image file paths using the glob module
mask_files = glob.glob(save_mask_directory + "/*.jpg")

# Define lists and set the frame_count again to zero.
percentage_list = []
frame_number_list = []
frame_count = 0

# Loop through the image files to get the percentage of white pixels.
for mask_file in mask_files:

    # Load each image using the PIL library.
    mask = Image.open(mask_file)
    
    # Convert the image to grayscale.
    mask_gray = mask.convert("L")

    # Convert the image to a numpy array.
    mask_array = np.array(mask_gray)

    # Calculate the total number of pixels.
    total_pixels = mask_array.size

    # Count the number of white pixels.
    white_pixels = np.count_nonzero(mask_array == 255)

    # Calculate the percentage of white pixels.
    percentage_white_pixels = (white_pixels / total_pixels) * 100
    
    # Add the frame_numbers and percentages to the list.
    percentage_list.append(percentage_white_pixels)
    frame_number_list.append(frame_count)

    frame_count += 1

    # Close the image.
    mask.close()

# The original frames per second is known by the settings of the recording device.
fps = 15  # Frames per second of the video.
total_frames = len(percentage_list)
dpi = 500

fontsize_small = 15
fontsize_large = 20

# Compute the Fourier transform for the heart signal.
fft_data = np.fft.fft(percentage_list)
# Calculate the frequencies for all the frames.
freq = np.fft.fftfreq(total_frames, 1/fps)

# Create subplots.
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot the normal signal.
axes[0].plot(frame_number_list, percentage_list)
axes[0].set_xlabel('Frame Number', fontsize=fontsize_small)
axes[0].set_ylabel('Percentage', fontsize=fontsize_small)
axes[0].set_title('Normal Signal', fontsize=fontsize_large)
axes[0].tick_params(axis='both', which='major', labelsize=fontsize_small)

# Plot the fft signal.
axes[1].plot(freq, np.abs(fft_data))
axes[1].set_xlabel('Frequency (Hz)', fontsize=fontsize_small)
axes[1].set_ylabel('Magnitude', fontsize=fontsize_small)
axes[1].set_title('FFT', fontsize=fontsize_large)
axes[1].tick_params(axis='both', which='major', labelsize=fontsize_small)

# Adjust spacing between subplots.
plt.tight_layout()

# Save plot.
save_path_plt_chart = save_heart_and_lung_signal + "/Normal_and_fft_signal"
plt.savefig(save_path_plt_chart, dpi=dpi, bbox_inches='tight', pad_inches=0)

# Show the plot.
plt.show()

# Set threshold values for heart signal.
threshold_low_heart_hz = 1 # Lower cutoff frequency in Hz. 1 Hz = 60 BPM.
threshold_up_heart_hz = 1.3 # Upper cutoff frequency in Hz. 1,3 Hz = 78 BPM.

# Convert threshold values to corresponding indices in the frequency array.
threshold_low_heart_index = np.abs(freq - threshold_low_heart_hz).argmin()
threshold_up_heart_index = np.abs(freq - threshold_up_heart_hz).argmin()

# Filter out small movements for heart signal based on the threshold indices.
filtered_fft_data_heart = fft_data.copy()
filtered_fft_data_heart[:threshold_low_heart_index] = 0
filtered_fft_data_heart[threshold_up_heart_index:] = 0

# After filtering use inverse Fouriers to calculate the filtered heart signal.
filtered_signal_heart = np.fft.ifft(filtered_fft_data_heart)

# Set threshold values for breathing signal.
threshold_low_breathing_hz = 0.15 # Lower cutoff frequency in Hz. 0.15 Hz = 9 breaths / min.
threshold_up_breathing_hz = 0.35 # Upper cutoff frequency in Hz. 0.35 Hz = 21 breaths / min.

# Convert threshold values to corresponding indices in the frequency array.
threshold_low_breathing_index = np.abs(freq - threshold_low_breathing_hz).argmin()
threshold_up_breathing_index = np.abs(freq - threshold_up_breathing_hz).argmin()

# Filter out small movements for breathing signal based on the threshold indices.
filtered_fft_data_breathing = fft_data.copy()
filtered_fft_data_breathing[:threshold_low_breathing_index] = 0
filtered_fft_data_breathing[threshold_up_breathing_index:] = 0

# After filtering use inverse Fouriers to calculate the filtered breathing signal.
filtered_signal_breathing = np.fft.ifft(filtered_fft_data_breathing)

# Find peaks and determine frequency
peaks_heart, _ = find_peaks(filtered_signal_heart)
valleys_heart, _ = find_peaks(-filtered_signal_heart)
peaks_breathing, _ = find_peaks(filtered_signal_breathing)
valleys_breathing, _ = find_peaks(-filtered_signal_breathing)

print("Peaks")
print(peaks_breathing)
print("Valleys")
print(valleys_breathing)

# Frame numbers that are analyzed for the deeplearning validation.
frames_analyzed = [43, 148, 255] # Framenumbers to analyze.

# Calculate the average beats per minute.
beats = (len(peaks_heart) + len(valleys_heart)) / 2
first_frame = peaks_heart[0]
last_frame = peaks_heart[-1]
frames_beating = last_frame - first_frame
average_bps = beats / (frames_beating / fps)
average_bpm = average_bps * 60
print("Heart: " + str(average_bpm) + " beats/min") # Print.

# Calculate the average breaths per minute.
breathings = (len(peaks_breathing) + len(valleys_breathing)) / 2
first_frame = peaks_breathing[0]
last_frame = peaks_breathing[-1]
frames_breathing = last_frame - first_frame
average_brps = breathings / (frames_breathing / fps)
average_brpm = average_brps * 60
print("Lungs: " + str(average_brpm) + " breathings/min") # Print.

# Create subplots.
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot the filtered heart signal.
axes[0].plot(frame_number_list, np.real(filtered_signal_heart))
axes[0].set_xlabel('Frame Number', fontsize=fontsize_small)
axes[0].set_ylabel('Filtered Signal', fontsize=fontsize_small)
axes[0].tick_params(axis='both', which='major', labelsize=fontsize_small)

# Draw circles on peaks and valleys to visualize where the peaks are.
markersize = 5 # Set the size of the circles.
markerborder = 2

# Mark the frames that has been analyzed for the validation of the deep learning model.
for frame_analyzed in frames_analyzed:
    axes[0].plot(frame_number_list[frame_analyzed], np.real(filtered_signal_heart[frame_analyzed]), 'bo', markersize=(markersize + markerborder), markerfacecolor='white', markeredgewidth=markerborder, markeredgecolor='black')

# Mark the peaks with a green circle.
for peak in peaks_heart:
    axes[0].plot(frame_number_list[peak], np.real(filtered_signal_heart[peak]), 'go', markersize=markersize)

# Mark the valleys with a red circle.
for valley in valleys_heart:
    axes[0].plot(frame_number_list[valley], np.real(filtered_signal_heart[valley]), 'ro', markersize=markersize)

# Settings for the heart plot.
title1 = "Heart Signal (Filtered: " + str(threshold_low_heart_hz) + " - " + str(threshold_up_heart_hz) + " Hz)"
title2 = (str(round(average_bpm,1)) + " beats/min")
axes[0].set_title(title1 + "\n" + title2, fontsize=fontsize_large) # Two titles under each other by using next line (\n).

# Plot the filtered breathing signal.
axes[1].plot(frame_number_list, np.real(filtered_signal_breathing))
axes[1].set_xlabel('Frame Number', fontsize=fontsize_small)
axes[1].set_ylabel('Filtered Signal', fontsize=fontsize_small)
axes[1].tick_params(axis='both', which='major', labelsize=fontsize_small)

# Mark the frames that has been analyzed for the validation of the deep learning model.
for frame_analyzed in frames_analyzed:
    axes[1].plot(frame_number_list[frame_analyzed], np.real(filtered_signal_breathing[frame_analyzed]), 'bo', markersize=(markersize + markerborder), markerfacecolor='white', markeredgewidth=markerborder, markeredgecolor='black')

# Mark the peaks with a green circle.
for peak in peaks_breathing:
    axes[1].plot(frame_number_list[peak], np.real(filtered_signal_breathing[peak]), 'go', markersize=markersize)

# Mark the valleys with a red circle.
for valley in valleys_breathing:
    axes[1].plot(frame_number_list[valley], np.real(filtered_signal_breathing[valley]), 'ro', markersize=markersize)

# Settings for the breathing plot.
title1 = "Lung Signal (Filtered: " + str(threshold_low_breathing_hz) + " - " + str(threshold_up_breathing_hz) + " Hz)"
title2 = (str(round(average_brpm,1)) + " breathings/min")
axes[1].set_title(title1 + "\n" + title2, fontsize=fontsize_large)

# Adjust spacing between subplots.
plt.tight_layout()

# Save plot.
save_path_plt_chart = save_heart_and_lung_signal + "/Filtered_heart_and_lung_signal"
plt.savefig(save_path_plt_chart, dpi=dpi, bbox_inches='tight', pad_inches=0)

# Show the plot.
plt.show()