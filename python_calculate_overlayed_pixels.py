# --------------------------------------------------
# Script: Make plots to visualize the percentage overlap and lung size ratio.
# This script analyze the overlap between the deep learning and manually masks and calculates the lung size ratio. The manual masks are made in Adobe Photoshop.
# --------------------------------------------------

# Needed packages for the code.
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io

general_path = "C:/Users/User/Documents/Technische Geneeskunde/Module 12 TGO/Python/fluoroscopy"
mask_general_path = general_path + "/precision_mask"
save_general_path = general_path + "/overlapplot"
frame_numbers = [43, 148, 255] # Framenumbers to analyze.
dpi = 500 # Set the dpi (quality) of the saved images

# Function that calculates the percentage of overlay between two masks.
def calculate_overlay_percentage(mask1, mask2):

    # Convert masks to binary format based on intensity threshold.
    threshold = 255 // 2
    mask1_binary = (mask1 > threshold).astype(np.uint8)
    mask2_binary = (mask2 > threshold).astype(np.uint8)

    # Perform element-wise logical AND operation to get a mask that is the overlay of two masks.
    overlapped_mask = np.logical_and(mask1_binary, mask2_binary)

    # Calculate the number of white pixels in each mask.
    total_white_pixels_mask1 = np.sum(mask1_binary)
    total_white_pixels_mask2 = np.sum(mask2_binary)

    # Calculate the number of white pixels in the overlapped region.
    overlapped_white_pixels = np.sum(overlapped_mask)

    # Calculate the overlay percentage. It uses the mask with smallest number of white pixels as 100%.
    overlay_percentage = (overlapped_white_pixels / min(total_white_pixels_mask1, total_white_pixels_mask2)) * 100

    # Calculate the number of black pixels in each mask.
    total_black_pixels_mask1 = np.sum(mask1_binary == 0)
    total_black_pixels_mask2 = np.sum(mask2_binary == 0)

    lung_size_ratio_mask_1 = total_white_pixels_mask1 / (total_white_pixels_mask1 + total_black_pixels_mask1)
    lung_size_ratio_mask_2 = total_white_pixels_mask2 / (total_white_pixels_mask2 + total_black_pixels_mask2)

    print("Difference x")
    print(abs(lung_size_ratio_mask_1 - lung_size_ratio_mask_2))

    # Return the overlay_percentage, lung_size_ratio_mask_1 and lung_size_ratio_mask_2.
    return overlay_percentage, lung_size_ratio_mask_1, lung_size_ratio_mask_2

# Draw a bar chart to visualize the overlay and lung size ratio.
def draw_bar_chart(y_value, x_value, bar_name, bar_name_main_names, bar_color, bar_color_main_colors, ylabel, xlabel):
    # Set the figure size.
    fig, ax = plt.subplots(figsize=(16, 8))

    # Fontsize.
    fontsize = 30

    # Define the desired width of the bars in pixels.
    relative_bar_width = 5
    
    # Get the difference between the min and max y_values.
    y_diff = max(y_value) - min(y_value)
    y_diff_y = y_diff / 100

    print("Max y")
    print(max(y_value))
    print("Min y")
    print(min(y_value))
    print("Diff y")
    print(y_diff)

    # Calculate the bar_width based on the desired width in pixels and the y-axis range.
    bar_width = relative_bar_width * y_diff_y

    # Create the horizontal bar chart.
    plt.barh(y_value, x_value, height=bar_width, color=bar_color)

    # Set the labels and title.
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)

    # Create the legend handles with small colored rectangles.
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in bar_color_main_colors]

    # Specify the labels for the legend.
    labels = bar_name_main_names

    # Add the legend with handles and labels.
    legend = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=fontsize)

    # Adjust the position of the legend.
    legend.get_frame().set_linewidth(0)  # Remove the border around the legend.

    # Adjust the fontsize of the x and y axis tick labels.
    plt.tick_params(axis='x', labelsize=int(fontsize*0.7))
    plt.tick_params(axis='y', labelsize=int(fontsize*0.7))

    # Adjust the layout to place the legend below the figure.
    plt.subplots_adjust(bottom=0.3)

    # Return the plot.
    return plt

def calculate_intermediate_colors(color1, color2, num_colors):
    # Extract RGB values of color1.
    r1, g1, b1 = color1

    # Extract RGB values of color2.
    r2, g2, b2 = color2

    # Calculate the step size for each color channel.
    r_step = (r2 - r1) / num_colors
    g_step = (g2 - g1) / num_colors
    b_step = (b2 - b1) / num_colors

    # Initialize the list to store intermediate colors.
    intermediate_colors = []

    # Calculate intermediate colors.
    for i in range(num_colors):
        # Calculate the RGB values for the current step.
        r = (r1 + r_step * (i + 1))
        g = (g1 + g_step * (i + 1))
        b = (b1 + b_step * (i + 1))

        # Append the interpolated color to the list.
        intermediate_colors.append((r, g, b))
    
    # Return the list of intermediate colors for the gradient bar.
    return intermediate_colors

# Function to plot the bar graph.
def print_bar_plot_and_mask_overview(int_number_of_frame):
    # Paths to the images and masks.
    number_of_frame = str(int_number_of_frame)
    mask_general_path_and_name = mask_general_path + "/mask_" + number_of_frame
    image_path = mask_general_path + "/image_" + number_of_frame + ".jpg"
    mask_deeplearning_path = mask_general_path_and_name + " deeplearning.jpg"
    mask_drawing1_path = mask_general_path_and_name + " drawing 1.png"
    mask_drawing2_path = mask_general_path_and_name + " drawing 2.png"

    # Load all the images.
    mask_deeplearning = cv2.imread(mask_deeplearning_path, cv2.IMREAD_GRAYSCALE)
    mask_attempt1 = cv2.imread(mask_drawing1_path, cv2.IMREAD_GRAYSCALE)
    mask_attempt2 = cv2.imread(mask_drawing2_path, cv2.IMREAD_GRAYSCALE)

    # Create lists for all options.
    colors = []
    mask1_list = []
    mask1_name_list = []
    mask2_list = []
    mask2_name_list = []

    # Fill the lists for all options.
    colors.append((1.0, 0.0, 0.0))  # red.
    mask1_list.append(mask_attempt1)
    mask1_name_list.append("Attempt 1")
    mask2_list.append(mask_deeplearning)
    mask2_name_list.append("Deep learning")

    colors.append((1.0, 0.65, 0.0))  # orange.
    mask1_list.append(mask_deeplearning)
    mask1_name_list.append("Deep learning")
    mask2_list.append(mask_attempt2)
    mask2_name_list.append("Attempt 1")

    colors.append((1.0, 1.0, 0.0))  # yellow.
    mask1_list.append(mask_attempt2)
    mask1_name_list.append("Attempt 2")
    mask2_list.append(mask_deeplearning)
    mask2_name_list.append("Deep learning")

    colors.append((0.0, 0.5, 0.0))  # green.
    mask1_list.append(mask_deeplearning)
    mask1_name_list.append("Deeplearning")
    mask2_list.append(mask_attempt2)
    mask2_name_list.append("Attempt 2")

    colors.append((0.0, 0.0, 1.0))  # blue.
    mask1_list.append(mask_attempt1)
    mask1_name_list.append("Attempt 1")
    mask2_list.append(mask_attempt2)
    mask2_name_list.append("Attempt 2")

    colors.append((1.0, 0.0, 0.5))  # pink.
    mask1_list.append(mask_attempt2)
    mask1_name_list.append("Attempt 2")
    mask2_list.append(mask_attempt1)
    mask2_name_list.append("Attempt 1")

    # Create list for y_values and x_values and bar_names.
    y_value = []
    x_value = []
    bar_name = []
    bar_name_main_names = [] # Needed for the legend of the plot.
    bar_color = []
    bar_color_main_colors = [] # Needed for the legend of the plot.

    # Set the bars for all possibilities.
    for i in range(0, len(colors), 2):

        # Get all masks, mask_names and mask_colors from the lists.
        mask1 = mask1_list[i]
        mask1_name = mask1_name_list[i]
        mask1_color = colors[i]

        mask2 = mask2_list[i]
        mask2_name = mask2_name_list[i]
        mask2_color = colors[i + 1]

        # Get the overlap ratio and lung size ratios.
        overlap_ratio, lung_size_ratio_mask_1, lung_size_ratio_mask_2 = calculate_overlay_percentage(mask1, mask2)

        # Create the variables for the smallest mask.
        smallest_lung_size_ratio = 0
        smallest_lung_size_name = ""
        smallest_color = ""

        # Create the variables for the greatest mask.
        largest_lung_size_ratio = 0
        largest_lung_size_name = ""
        largest_color = ""

        # Determine which mask is the smallest or largest.
        if lung_size_ratio_mask_1 < lung_size_ratio_mask_2:
            smallest_lung_size_ratio = lung_size_ratio_mask_1
            smallest_lung_size_name = mask1_name + " - " + mask2_name
            smallest_color = mask1_color

            largest_lung_size_ratio = lung_size_ratio_mask_2
            largest_lung_size_name = mask2_name + " - " + mask1_name
            largest_color = mask2_color
        else:
            smallest_lung_size_ratio = lung_size_ratio_mask_2
            smallest_lung_size_name = mask2_name + " - " + mask1_name
            smallest_color = mask2_color

            largest_lung_size_ratio = lung_size_ratio_mask_1
            largest_lung_size_name = mask1_name + " - " + mask2_name
            largest_color = mask1_color
    
        # When smallest and largest bar are too close (difference < 0.01), make a gradient bar.
        if abs(lung_size_ratio_mask_1 - largest_lung_size_ratio) < 0.01:
        
            # Calculate the intermediate colors to create a optical gradient.
            color1 = mask2_color
            color2 = mask1_color
            num_colors = 100
            intermediate_colors = calculate_intermediate_colors(color1, color2, num_colors)

            # Append all intermediate colors to the list.
            for i in range(len(intermediate_colors)):
                factor = 1 - i/len(intermediate_colors)
                y_value.append(overlap_ratio)
                x_value.append(largest_lung_size_ratio * factor)
                bar_name.append(largest_lung_size_name)
                bar_color.append(intermediate_colors[i])

            # Set names for legend.
            bar_name_main_names.append(mask1_name + " & " + mask2_name)
            bar_color_main_colors.append(color2)

            bar_name_main_names.append(mask2_name + " & " + mask1_name)
            bar_color_main_colors.append(color1)

        else:
            # Set largest bar.
            y_value.append(overlap_ratio)
            x_value.append(largest_lung_size_ratio)
            bar_name.append(largest_lung_size_name)
            bar_name_main_names.append(largest_lung_size_name)

            bar_color.append(largest_color)
            bar_color_main_colors.append(largest_color)
        
            # Set smallest bar.
            y_value.append(overlap_ratio)
            x_value.append(smallest_lung_size_ratio)
            bar_name.append(smallest_lung_size_name)
            bar_name_main_names.append(smallest_lung_size_name)

            bar_color.append(smallest_color)
            bar_color_main_colors.append(smallest_color)

    # X and y label of the bar chart.
    xlabel = "Lung Size Ratio"
    ylabel = "Overlap Ratio"

    # Draw the bar chart and save it as an image file.
    plt_bar_chart = draw_bar_chart(y_value, x_value, bar_name, bar_name_main_names, bar_color, bar_color_main_colors, ylabel, xlabel)
    save_path_plt_chart = save_general_path + "/Bar_Chart_" + number_of_frame
    plt.savefig(save_path_plt_chart, dpi=dpi, bbox_inches='tight', pad_inches=0)

    # Print in the output window the mask which is saved to see the progress.
    print("Mask" + str(number_of_frame))

    # Load the images.
    image = plt.imread(image_path)
    mask_deeplearning = plt.imread(mask_deeplearning_path)
    mask_attempt1 = plt.imread(mask_drawing1_path)
    mask_attempt2 = plt.imread(mask_drawing2_path)

    # Create subplots to display the images.
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # Set the fontsize of the titles in the image.
    fontsize = 32

    # Display the images.
    axs[0].imshow(image, cmap='gray')
    title = "Image (Frame: " + number_of_frame + ")"
    axs[0].set_title(title, fontsize=fontsize)

    axs[1].imshow(mask_deeplearning, cmap='gray')
    axs[1].set_title('Deep learning mask', fontsize=fontsize)

    axs[2].imshow(mask_attempt1, cmap='gray')
    axs[2].set_title('Attempt 1 mask', fontsize=fontsize)

    axs[3].imshow(mask_attempt2, cmap='gray')
    axs[3].set_title('Attempt 2 mask', fontsize=fontsize)

    # Remove the axis labels.
    for ax in axs:
        ax.axis('off')

    # Adjust the layout.
    plt.tight_layout()

    save_path_plt = save_general_path + "/Masks_Overview_" + number_of_frame
    # Save the figure as an image file.
    plt.savefig(save_path_plt, dpi=dpi, bbox_inches='tight', pad_inches=0)

# For loop to create and save all bar charts.
for i in range(len(frame_numbers)):
    print_bar_plot_and_mask_overview(frame_numbers[i])
