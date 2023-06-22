# --------------------------------------------------
# Script: Make plot to visualize the loss, validation loss and learning rate during epochs.
# This sript uses the copied date from the output of the deeplearning training script to make a plot.
# --------------------------------------------------

# Needed packages for the code.
import matplotlib.pyplot as plt

# Define saving path of the plot
general_path = "C:/Users/User/Documents/Technische Geneeskunde/Module 12 TGO/Python/fluoroscopy"
save_general_path = general_path + "/deeplearning_model"
dpi = 1000  # Set the dpi of the image

# Data obtained from the output window of the deep learning script.
epoch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
loss = [0.4654, 0.2482, 0.1941, 0.1616, 0.1397, 0.125, 0.1206, 0.1116, 0.1074, 0.1003, 0.1018, 0.0968, 0.0947, 0.0977, 0.0916, 0.0881, 0.0851, 0.0807]
val_loss = [0.2538, 0.1741, 0.1367, 0.1346, 0.1103, 0.0952, 0.1048, 0.0831, 0.0991, 0.0831, 0.0927, 0.0812, 0.0877, 0.0903, 0.0743, 0.0956, 0.0764, 0.0804]
lr = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]

# Create a plot with certain size.
fig, ax1 = plt.subplots(figsize=(8, 4))

# Adjust the fontsize for the title, axis labels and legend labels.
fontsize_title = 18
fontsize_label = 14
fontsize_legend = 14

# Set the data for the first y axis and assign the x and y label.
ax1.plot(epoch, loss, color='blue', label='Loss')
ax1.plot(epoch, val_loss, color='red', label='Validation loss')
ax1.set_xlabel('Epoch Number', fontsize=fontsize_label)
ax1.set_ylabel('Loss and validation loss', fontsize=fontsize_label)
ax1.set_title('Loss, validation loss and learning rate during epochs', fontsize=fontsize_title)

# Creating the secondary y-axis to show the learning rate on.
ax2 = ax1.twinx()
ax2.plot(epoch, lr, color='green', label='Learning rate')
ax2.set_ylabel('Learning rate', fontsize=fontsize_label)

# Set the x-axis ticks to display the epoch values.
ax1.set_xticks(epoch)

# Displaying legend.
lines = ax1.get_lines() + ax2.get_lines()
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper right', fontsize=fontsize_legend)

# Save the plot as a PNG image without white space.
path = save_general_path + "/deeplearning_model.png"
plt.savefig(path, format='png', dpi=dpi, bbox_inches='tight')

# Show the plot.
plt.show()