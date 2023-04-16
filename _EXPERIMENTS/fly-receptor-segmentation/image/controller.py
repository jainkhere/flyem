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
    def select_regions(data, region_size):
        return Image.select_regions(data, region_size)
    
    @staticmethod
    def select_large_regions(data, region_size):
        return Image.select_large_regions(data, region_size)

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
    def display_and_save_compare_labeled(data1, data2, data3, plot_path):
        ImageView.display_and_save_compare_labeled(data1, data2, data3, plot_path)

    @staticmethod
    def classify_image(data):
        return Image.classify_image(data)
    
    @staticmethod
    def classify_image_nina(data):
        return Image.classify_image_nina(data)