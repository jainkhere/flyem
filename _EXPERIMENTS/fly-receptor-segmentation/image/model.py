from utils.imageUtils import ImageUtils


class Image:

    @staticmethod
    def read(filepath):
        return ImageUtils.read_image(filepath)
    
    @staticmethod
    def save(filepath, data):
        return ImageUtils.save_image(filepath, data)

    @staticmethod
    def normalize(data):
        return ImageUtils.normalize_image(data)

    @staticmethod
    def center_crop(data, crop_size):
        return ImageUtils.crop_image(data, crop_size=crop_size)

    @staticmethod
    def smooth(data, sigma_value):
        return ImageUtils.apply_gaussian_filter(data, sigma_value=sigma_value)

    @staticmethod
    def threshold(data, threshold_value):
        return ImageUtils.apply_threshold(data, threshold_value)

    @staticmethod
    def label(data):
        return ImageUtils.apply_region_labelling(data)

    @staticmethod
    def select_regions(data, region_size):
        return ImageUtils.remove_small_regions(data, region_size)
    
    @staticmethod
    def select_large_regions(data, region_size):
        return ImageUtils.remove_large_regions(data, region_size)

    @staticmethod
    def binary_mask(data):
        return ImageUtils.get_binary_mask(data)

    @staticmethod
    def close_binary_mask(data):
        return ImageUtils.get_closed_binary_mask(data)

    @staticmethod
    def binary_image(data):
        return ImageUtils.get_binary_image(data)

    @staticmethod
    def classify_image(data):
        return ImageUtils.verify_image_segmentation(data)
    
    @staticmethod
    def classify_image_nina(data):
        return ImageUtils.verify_image_segmentation_nina(data)