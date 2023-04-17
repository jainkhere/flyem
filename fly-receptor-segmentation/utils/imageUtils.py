import os
import time
import math
import shutil
import numpy as np
import mahotas as mh
import matplotlib.pyplot as plt
from skimage import measure, filters


class ImageUtils:
    @staticmethod
    def read_image(path):
        return mh.imread(path)
    
    @staticmethod
    def save_image(path, image_data):
        return mh.imsave(path, image_data)

    @staticmethod
    def crop_image(original_image, crop_size):
        # returns cropped image of crop size at the center of the image
        height, width = original_image.shape
        center_x, center_y = height//2, width//2
        new_height, new_width = crop_size
        start_x, start_y = center_x - (new_height//2), center_y - (new_width//2)
        end_x, end_y = center_x + (new_height//2), center_y + (new_width//2)
        result_image = original_image[start_x:end_x, start_y:end_y]
        return result_image

    @staticmethod
    def normalize_image(original_image):
        # returns normalized image to pixel range of 0 to 255 from the original pixel range
        result_image = original_image.astype(np.float64)
        result_image /= result_image.max()
        return result_image * 255

    @staticmethod
    def get_multiple_crops(original_image, crop_size):
        # returns list of all the cropped images of given crop size
        cropped_images = []
        height, width = original_image.shape
        new_height, new_width = crop_size.shape
        for i in range(0, height, new_height):
            for j in range(0, width, new_width):
                cropped_images.append(original_image[i:i + new_height, j:j + new_width])
        return cropped_images

    @staticmethod
    def get_overlay(original_image, mask_image):
        # returns an overlay image from the original image and mask image
        result_image = np.copy(original_image)
        np.putmask(result_image, mask_image.astype(bool), 0)
        return result_image

    @staticmethod
    def apply_gaussian_filter(original_image, sigma_value):
        # returns the gaussian filtered image of the original image
        result_image = mh.gaussian_filter(original_image, sigma=sigma_value)
        return result_image

    @staticmethod
    def apply_threshold(original_image, threshold_value):
        # returns the threshold image of the original image for the given threshold value
        result_image = original_image.copy()
        result_image[result_image < threshold_value] = 0
        return result_image

    @staticmethod
    def apply_region_labelling(original_image):
        # returns the labeled regions and number of regions
        labeled_result, nr_objects_result = mh.label(original_image)
        labeled_result = mh.labeled.remove_bordering(labeled_result)
        return labeled_result, nr_objects_result

    @staticmethod
    def remove_small_regions(labeled, region_size):
        # return the resultant image after removing small regions
        sizes = mh.labeled.labeled_size(labeled)
        too_small = np.where(sizes < region_size)
        labeled_result = mh.labeled.remove_regions(labeled, too_small)
        return labeled_result
    
    @staticmethod
    def remove_large_regions(labeled, region_size):
        # return the resultant image after removing small regions
        sizes = mh.labeled.labeled_size(labeled)
        too_large = np.where(sizes > region_size)
        labeled_result = mh.labeled.remove_regions(labeled, too_large)
        return labeled_result

    @staticmethod
    def get_binary_mask(labeled):
        # returns a binary mask of the input image
        result_mask = labeled.copy()
        result_mask[result_mask > 0] = 1
        return result_mask

    @staticmethod
    def get_closed_binary_mask(binary_mask):
        # return the closed binary mask of the given binary mask
        result_mask = mh.morph.close(binary_mask)
        return result_mask

    @staticmethod
    def get_binary_image(binary_mask):
        # returns the binary image for the given binary mask
        # threshold_value = mh.otsu(binary_mask)
        # result_image = binary_mask > threshold_value
        # return result_image
        threshold = filters.threshold_otsu(binary_mask)
        result_binary_image = binary_mask > threshold
        return result_binary_image

    @staticmethod
    def verify_image_segmentation(binary_mask):
        # verifies the segmentation of the image is clear or not
        binary_mask_closed = ImageUtils.get_closed_binary_mask(binary_mask)
        labeled_binary, nr_objects_binary = ImageUtils.apply_region_labelling(binary_mask)
        region_sizes = measure.regionprops(labeled_binary, intensity_image=binary_mask_closed)
        large_regions = 0
        min_region_size = 3000
        for region in region_sizes:
            if region.area > min_region_size:
                large_regions += 1
        return nr_objects_binary, large_regions
    
    @staticmethod
    def check_accuracy(binary_mask):
        # count number of rhabdoemeres
        # then calculate accuracy of mask
        # accuracy = number of rhabdomeres / total regions in mask
        # a mask that only has rhabdomeres as regions is a mask with accuracy = 1
        num_circles = 0
        num_sq = 0
        num_circle_sq = 0
        binary_mask_closed = ImageUtils.get_closed_binary_mask(binary_mask)
        labeled_binary, nr_objects_binary = ImageUtils.apply_region_labelling(binary_mask)
        # get all regions in image
        region_sizes = measure.regionprops(labeled_binary, intensity_image=binary_mask_closed)
        # check following two properties of regions
        # 1. circularity
        # 2. squareness of bounding box
        # if a region satifies both circularity and squareness
        # then count it as a rhabdomere
        for region in region_sizes:
            x1, y1 = region.centroid
            minr, minc, maxr, maxc = region.bbox
            x = maxc - minc
            y = maxr - minr
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)
            circularity = (region.perimeter ** 2) / (4 * math.pi * region.area);
            if circularity < 2 and circularity > 0.4:
                num_circles += 1
            if x/y >= .4 and x/y <= 2:
                num_sq += 1
            if circularity < 2 and circularity > 0.4 and x/y >= .4 and x/y <= 2:
                num_circle_sq += 1
        # if region_sizes has no elements then set accuracy as 0.0001
        if (len(region_sizes)) != 0:
            accuracy = num_circle_sq/len(region_sizes)
        else:
            accuracy = .0001
        return accuracy, num_circle_sq

    @staticmethod
    def get_start_time():
        # returns start time of the execution
        start_time = time.time()
        return start_time

    @staticmethod
    def get_execution_duration(start_time):
        # returns the total duration of the execution
        end_time = time.time()
        return end_time - start_time

    @staticmethod
    def create_new_dir(dir_path):
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)
        else:
            os.mkdir(dir_path)

    @staticmethod
    def convert_to_npy(image_data):
        # returns the numpy data of image data
        numpy_data = np.array(image_data)
        return numpy_data

    @staticmethod
    def src2dest_copy(src_dir, dest_dir, file):
        # copies the specified file from source dir to destination dir
        shutil.copy(os.path.join(src_dir, file), os.path.join(dest_dir, file))

    @staticmethod
    def display_image(image_data):
        plt.imshow(image_data)
        plt.show()
        
    @staticmethod
    def display_image_compare(image_data_1, image_data_2):
        f, axarr = plt.subplots(1,2, figsize=(20, 40))
        axarr[0].imshow(image_data_1)
        axarr[1].imshow(image_data_2)
        plt.show()
        plt.close('all')
        
    @staticmethod
    def display_and_save_compare_labeled(original_image, bb_label_image, mask_image, plot_path, show_in_notebook):
        f, axarr = plt.subplots(1,3, figsize=(20, 60))
        axarr[1].imshow(bb_label_image, cmap='gray')
        num_circles = 0
        num_sq = 0
        num_circle_sq = 0
        binary_mask_closed = ImageUtils.get_closed_binary_mask(bb_label_image)
        labeled_binary, nr_objects_binary = ImageUtils.apply_region_labelling(bb_label_image)
        region_sizes = measure.regionprops(labeled_binary, intensity_image=binary_mask_closed)
        for region in region_sizes:
            x1, y1 = region.centroid
            minr, minc, maxr, maxc = region.bbox
            x = maxc - minc
            y = maxr - minr
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)
            circularity = (region.perimeter ** 2) / (4 * math.pi * region.area);
            if circularity < 2 and circularity > 0.4:
                num_circles += 1
                axarr[1].plot(y1, x1, '.r', markersize=3.5)
            if x/y >= 0.4 and x/y <= 2:
                num_sq += 1
                axarr[1].plot(bx, by, '-g', linewidth=2.5)
            if circularity < 2 and circularity > 0.4 and x/y >= 0.4 and x/y <= 2:
                num_circle_sq += 1
                axarr[1].plot(y1, x1, '.b', markersize=3.5)
                axarr[1].plot(bx, by, '-b', linewidth=2.5)
        axarr[0].imshow(original_image)
        axarr[2].imshow(mask_image)
        axarr[0].set_title('Original Image', fontsize=10)
        axarr[1].set_title('Intermediate Image', fontsize=10)
        axarr[2].set_title('Binary mask Image', fontsize=10)
        plot_title = plot_path.split('/')
        plt.title(plot_title[len(plot_title) - 1])
        if show_in_notebook:
            plt.show()
        f.savefig(plot_path)
        plt.close(f)
        
    @staticmethod
    def save_image_compare(image_data_1, image_data_2, plot_path):
        f, axarr = plt.subplots(1,2, figsize=(20, 40))
        axarr[0].imshow(image_data_1)
        axarr[1].imshow(image_data_2)
        plt.savefig(plot_path)
        plt.close('all')