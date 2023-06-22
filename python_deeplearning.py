# --------------------------------------------------
# Script: Deep learning for training to segment lung images
# This script first gets the images and masks of the deep learning model. These images will be randomly cropped to optimize the deep learning model for segmentation of part of the lungs. This is needed because Radboudumc provides images where not the whole lung are visible. This script then trains the deep learning model.
# --------------------------------------------------

# Needed packages for the code
from cv2 import imread, createCLAHE
from glob import glob
from tensorflow.keras import Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Cropping2D, concatenate
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tkinter import Tk
from tkinter.filedialog import askdirectory
from tqdm import tqdm
from types import new_class
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import shutil
import tensorflow as tf

# Allows the computer to power up the GPU (Graphic processing unit) to better run the code.
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Define the paths to the image and mask files
general_path = "C:/Users/User/Documents/Technische Geneeskunde/Module 12 TGO/Python/fluoroscopy"

general_original_image_path = general_path + "/deeplearning_original_images"
general_original_mask_path = general_path + "/deeplearning_original_masks"

general_cropped_image_path = general_path + "/original_and_cropped_images"
general_cropped_mask_path = general_path + "/original_and_cropped_masks"

# --------------------------------------------------
# Randomly crop part of the code to ensure the deep learning model can also segment the lungs when not the entire lungs are visible.
# --------------------------------------------------

# Function which randomly crops an image and mask.
def random_crop(image, mask):

    # Get the height and width of the original image.
    height, width = image.shape[:2]

    # Crop the image to size with factor 0.4 - 0.8 of the original size.
    factor = random.uniform(0.4, 0.8)

    # Determine the width and height of the cropped image.
    new_height = int(height * factor)
    new_width = int(width * factor)

    # Determine randomly the start x and y coordinate. The new_width and new_height are necessary for setting te maximum x and y.
    x = random.randint(0, width - new_width)
    y = random.randint(0, height - new_height)

    # Crop the image and mask with the same crop settings.
    cropped_image = image[y:y+new_height, x:x+new_width]
    cropped_mask = mask[y:y+new_height, x:x+new_width]

    # Return the cropped image, cropped mask, and crop factor.
    return cropped_image, cropped_mask, factor

# Define emptpy path to assign later
cropped_image_path = ""
cropped_mask_path = ""

# Set to true if the original image and mask must be included in the new folder location.
include_original_image_and_mask = True
# Set the number of times the image and mask must be cropped and saved in the new folder location.
number_of_cropped_copies = 2

# Empty the folders of destination, because two times of running will result in different cropped images and masks, with therefore also other names. The reasons for different names is because the cropping factor is visible in the filename.
shutil.rmtree(general_cropped_image_path)
os.makedirs(general_cropped_image_path)

shutil.rmtree(general_cropped_mask_path)
os.makedirs(general_cropped_mask_path)

# Set the file_list which contain the filenames of the original images.
file_list = os.listdir(general_original_image_path)

# Loop through all files in the folder.
for index in range(len(file_list)):

    # Get the filename from the file_list based on the index.
    imagename = file_list[index]

    # Check if the endswitch is type of an image.
    if imagename.endswith((".jpg", ".jpeg", ".png")):
        
        # Construct the full file paths
        image_path = os.path.join(general_original_image_path, imagename)
        imagename_short, image_extension = os.path.splitext(imagename)
        maskname = imagename_short + "_mask" + image_extension # Assuming mask names are the same as image names.
        maskname_new = imagename_short + image_extension # Assuming mask names are the same as image names.
        mask_path = os.path.join(general_original_mask_path, maskname)

        # Check if image and mask exist.
        if os.path.exists(mask_path):
            
            # Copy the image to folder with original and cropped images and masks.
            if include_original_image_and_mask == True:
                new_original_path = os.path.join(general_cropped_image_path, imagename)
                shutil.copyfile(image_path, new_original_path)

                new_mask_path = os.path.join(general_cropped_mask_path, maskname_new)
                shutil.copyfile(mask_path, new_mask_path)
                
            # Read the image.
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path)

            for i in range(number_of_cropped_copies):
                # Apply random crop.
                cropped_image, cropped_mask, factor = random_crop(image, mask)
                
                # Save the cropped image.
                imagename_short, image_extension = os.path.splitext(imagename)
                cropped_image_path = os.path.join(general_cropped_image_path, imagename_short + "(cropped_" + str(round(factor, 2)) + "x)" + image_extension)
                cv2.imwrite(cropped_image_path, cropped_image)

                # Save the cropped mask.
                maskname_short, mask_extension = os.path.splitext(maskname_new)
                cropped_mask_path = os.path.join(general_cropped_mask_path, maskname_short + "(cropped_" + str(round(factor, 2)) + "x)_mask" + mask_extension)
                cv2.imwrite(cropped_mask_path, cropped_mask)
        
        # If path not excist, continue with the for loop.
        else:
            continue

# Print to know the cropping part is ready.
print("Succesfully randomly cropped images and masks")

# --------------------------------------------------
# Deep learning part of the code.
# --------------------------------------------------

# Get the images and masks from the directory where the original and cropped images and masks are stored.
image_path = general_cropped_image_path
mask_path = general_cropped_mask_path

# Create a list of image and mask files
images = os.listdir(image_path) # Get all the images from the image_path
mask = os.listdir(mask_path) # Get all the masks from the mask_path

# Extract the names of the masks by splitting the name from the extension .png
mask = [fName.split(".png")[0] for fName in mask]

# Extract the names of the images by splitting the name from the suffix "mask". For example the image is named "CHNCXR_0001_0" and the corresponding mask is called "CHNCXR_0001_0_mask". To get the image name from the mask name, the suffix "_mask" must be removed.
image_file_name = [fName.split("_mask")[0] for fName in mask]

# Check if the image and corresponding mask are in the folder based on the filenames
check = [i for i in mask if "mask" in i]
print("Total masks with modified name:", len(check))

# Define the training and testing files based on the available files. Only the images for which a mask is used are used.
testing_files = set(os.listdir(image_path)) & set(os.listdir(mask_path))
training_files = check

# Define a function to load and resize image and mask data
def getData(X_shape, flag="test"):
    # Create empty arrays to save the image and mask arrays into.
    im_array = []
    mask_array = []

    # The if statement for the test data. The test dataset is a separate dataset used to evaluate the quality of the deep learning model. The dataset must be different from the training data, but representative.
    if flag == "test":
        
        # For loop with index i to go trough all testing files
        for i in tqdm(testing_files):

            # Resize the image so the images all have the same dimensions.
            im = cv2.resize(cv2.imread(os.path.join(image_path, i)), (X_shape, X_shape))[:, :, 0]
            
            # Apply histogram equalization for contrast enhancement. The images are therefore more comparable in color composition with each other. The deep learning model is therefore better able to recognize the lungs.
            equalized_image = cv2.equalizeHist(im)  
            im = equalized_image

            # Resize the mask so the masks all have the same dimensions.
            mask = cv2.resize(cv2.imread(os.path.join(mask_path, i)), (X_shape, X_shape))[:, :, 0]

            # Add the image and mask array to the list of images.
            im_array.append(im)
            mask_array.append(mask)

        # Return the function with the image and mask arrays.
        return im_array, mask_array

    # The if statement for the training data. The training dataset is used to train the deep learning model. In this case, the dataset consists of chest X-ray images with the corresponding mask covering the area of the lungs. The deep learning model learns from these examples and uses them to develop a model. This model can then be used to predict a mask for new (untrained) images.
    if flag == "train":

        # For loop with index i to go trough all testing files.
        for i in tqdm(training_files):

            # Resize the image so the images all have the same dimensions.
            im = cv2.resize(cv2.imread(os.path.join(image_path, i.split("_mask")[0] + ".png")), (X_shape, X_shape))[:, :, 0]
            
            # Apply histogram equalization for contrast enhancement. The images are therefore more comparable in color composition with each other. The deep learning model is therefore better able to recognize the lungs.
            equalized_image = cv2.equalizeHist(im)
            im = equalized_image

            # Resize the mask so the masks all have the same dimensions.
            mask = cv2.resize(cv2.imread(os.path.join(mask_path, i + ".png")), (X_shape, X_shape))[:, :, 0]

            # Add the image and mask array to the list of images.
            im_array.append(im)
            mask_array.append(mask)

        # Return the function with the image and mask arrays.
        return im_array, mask_array

# This deep learning model uses the U-Net architecture. This is a model that is particularly suitable for image segmentation, including segmentation of the lungs. The model uses an encoder and decoder part. The encoder part searches for landmarks in the chest X-ray, the decoder part tries to predict the mask. This architecture is capable of very accurately predicting a mask.
def unet(input_size=(256, 256, 1)):

    # Creates a layer with the given dimensions (256,256,1).
    inputs = Input(input_size)

    # The conv2D layers perform convolution operations on the input. More conv2D layers are present in the code to be able to detect increasingly complex characteristics.
    # Activation='relu' specifies the way a decision is made. This can be compared to neurons in the brain. A neuron decides on the basis of input values whether it transmits a signal or not. For 'relu', this decision is quite basic. If the input is positive, then the "neuron" transmits the signal. If the input is negative, the "neuron" silences the signal.
    # Padding='same' is a technique that ensures that the dimensions of the input data and the output data remain the same. The input is converted into an output by the processing of filters. The problem is that filters have a fixed size. When there is a mismatch between the size of the filter and that of the input, the output will be sized differently. To prevent this, the input is, if necessary, enlarged with a padding.
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    
    # MaxPooling2D reduced the spatial dimensions of the input. Here is the size (2, 2) used, which means that both height and width of the input is reduced by a factor 4. The MaxPooling2D layer devides the input in rectanglular areas width certain width and height and selects the local maximum value.
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    # The concatenate function combines layers so that the encode and decoder part can be connected with each other. These connections are called skip connections.
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    # This layer is the last convolutional layer. For each pixel, a value between 0 and 1 indicates the probability that it belongs to the mask. The function "Sigmoid" determines this values for each pixel.
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    # Returns the model with the inputs and the outputs.
    return Model(inputs=[inputs], outputs=[conv10])


# Define the parameters for training.
# X_shape sets the dimensions of the image for training. All images have a resolution of 512x512.
X_shape = 512
# Batch size refers to the number of training examples processed together. It affects memory usage and model performance. The bactch_size is decreased from 16 to 4 due to memory issues.
batch_size = 4
# Number of epochs after which the training will at least stop.
epochs = 50
# When the val_loss not further decrease after 3 epoches, the training is stopped.
epochs_no_improvement_until_stopping = 3

# Load the training data.
X_train, y_train = getData(X_shape, flag="train")

# Normalize the data. Instead of color values from 0-255 set it to values between 0-1.
X_train = np.asarray(X_train) / 255.0
y_train = np.asarray(y_train) / 255.0

# Reshape the data for training. Adds an extra dimension to the array, which is needed for deep learning.
X_train = np.expand_dims(X_train, axis=-1)
y_train = np.expand_dims(y_train, axis=-1)

# Configure GPU memory growth with a specific limit.
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(
        gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=0.80)] # Allowing the pc to use up to 80% of the free memory for running this program.
    )

# Create the U-Net model with dimensions X_shape x X_shape.
model = unet(input_size=(X_shape, X_shape, 1))

# Compile the model.
model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy")

# Define callbacks for model training.
checkpoint = ModelCheckpoint("lung_segmentation_model.h5", monitor="val_loss", verbose=1, save_best_only=True, mode="min")
# Early stopping the code when the val_loss not decrease.
early_stopping = EarlyStopping(monitor="val_loss", patience=epochs_no_improvement_until_stopping, mode="min")
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, min_lr=1e-8, mode="min", verbose=1)

# Train the model.
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[checkpoint, early_stopping, reduce_lr])

# Load the testing data.
X_test, y_test = getData(X_shape, flag="test")

# Normalize the data.
X_test = np.asarray(X_test) / 255.0
y_test = np.asarray(y_test) / 255.0

# Reshape the data for testing.
X_test = np.expand_dims(X_test, axis=-1)
y_test = np.expand_dims(y_test, axis=-1)

# Evaluate the model on the testing data.
loss, accuracy = model.evaluate(X_test, y_test)

# Print the loss and accuracy.
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)