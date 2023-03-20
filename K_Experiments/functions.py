import os
import cv2
import time
import shutil
import numpy as np
import mahotas as mh
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from skimage import measure, filters, exposure

def normalize(temp_dir):
    for z in tqdm(sorted(os.listdir(temp_dir))):# added interactive progressbar to decrease the uncertanity and to increase curiosity :)
        if (z.endswith("tif")): # checking the file ends with tif 
            # Read in the image
            img = mh.imread(os.path.join(temp_dir, z))
            
            # Normalize the image
            img = img.astype(np.float)
            img /= img.max()
            img *= 255
            
            # Save the processed image back to the temporary directory
            mh.imsave(os.path.join(temp_dir, z), img)
            
def crop(temp_dir):            
#For images in temporary folder reading them and cropping them to make it easy for segmentation.
    for z in tqdm(sorted(os.listdir(temp_dir))):
        if (z.endswith("tif")): # checking the file ends with tif
            
            # Read in the image
            img = mh.imread(os.path.join(temp_dir, z))
            # Crop the image
            img_cropped = img[1500:2000, 2500:4000]
            #copying temp images to cropped images folder
            shutil.copy(os.path.join(temp_dir, z), os.path.join(cropped_dir, z))
            # Save the processed image back to the temporary directory
            mh.imsave(os.path.join(cropped_dir, z), img_cropped)
            # Add the image to the list
            slices.append(img_cropped)
            # Display the image
            print(z)
            plt.imshow(img_cropped)
            plt.show()


def show(temp_dir):
    for z in tqdm(sorted(os.listdir(temp_dir))):
        if (z.endswith("tif")): # checking the file ends with tif
            # Read in the image       
            img = mh.imread(os.path.join(temp_dir, z))
            print (img.shape)
            print(z)
            plt.imshow(img)
            plt.show()
        
def npyconversion(tif_dir, npy_path):
    tif_files = [f for f in os.listdir(tif_dir) if f.endswith('.tif')]
    data = []
    for tif_file in tqdm(tif_files):
        img = Image.open(os.path.join(tif_dir, tif_file))
        data.append(np.array(img))
    np.save(npy_path, data)

    
    
def check(dir_path):
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        file_size = os.path.getsize(file_path)
        file_ctime = time.ctime(os.path.getctime(file_path))
        print(f'{file_name} - Size: {file_size} bytes - Created: {file_ctime}')
        
        
#to plot_images and compare between 2 directories example orginal and masks         
def plot_images(original_dir, mask_dir):
    num_images = len([f for f in os.listdir(original_dir) if f.endswith('.tif')])
    fig, axs = plt.subplots(num_images, 2, figsize=(5*2, 5*num_images))
    for i, z in enumerate(tqdm(sorted(os.listdir(original_dir)))):
        if z.endswith("tif"):
            # Read in the original image
            img_original = plt.imread(os.path.join(original_dir, z))
            # Read in the mask image
            img_mask = plt.imread(os.path.join(mask_dir, z))
            
            # Plot the original image on the left
            axs[i][0].imshow(img_original)
            axs[i][0].set_title(z)
            
            # Plot the mask image on the right
            axs[i][1].imshow(img_mask)
            axs[i][0].set_title(z)
    
    plt.show()