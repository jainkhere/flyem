from image.model import Image
from image.view import ImageView


class ImageController:

    @staticmethod
    def read(filepath):
        return Image.read(filepath)
    
    @staticmethod
    def save(filepath, data):
        return Image.save(filepath, data)

    @staticmethod
    def normalize(data):
        return Image.normalize(data)

    @staticmethod
    def center_crop(data, crop_size=(512, 512)):
        return Image.center_crop(data, crop_size)

    @staticmethod
    def smooth(data, sigma_value=3):
        return Image.smooth(data, sigma_value)

    @staticmethod
    def threshold(data, threshold_value):
        return Image.threshold(data, threshold_value)

    @staticmethod
    def label(data):
        return Image.label(data)

    @staticmethod
    def remove_regions(data, min_region_size, max_region_size):
        return Image.remove_regions(data, min_region_size, max_region_size)

    @staticmethod
    def binary_mask(data):
        return Image.binary_mask(data)

    @staticmethod
    def close_binary_mask(data):
        return Image.close_binary_mask(data)

    @staticmethod
    def binary_image(data):
        return Image.binary_image(data)

    @staticmethod
    def display(data):
        ImageView.display(data)
        
    @staticmethod
    def display_compare(data1, data2):
        ImageView.display_compare(data1, data2)
        
    @staticmethod
    def save_compare(data1, data2, plot_path):
        ImageView.save_compare(data1, data2, plot_path)
    
    @staticmethod
    def display_and_save_compare_labeled(data1, data2, data3, plot_path, show_in_notebook=False):
        ImageView.display_and_save_compare_labeled(data1, data2, data3, plot_path, show_in_notebook)

    @staticmethod
    def verify_image_segmentation(data):
        return Image.verify_image_segmentation(data)
    
    @staticmethod
    def check_accuracy(data):
        return Image.check_accuracy(data)