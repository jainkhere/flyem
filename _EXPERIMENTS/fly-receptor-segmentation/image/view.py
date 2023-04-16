from utils.imageUtils import ImageUtils


class ImageView:

    @staticmethod
    def display(data):
        ImageUtils.display_image(data)
        
    @staticmethod
    def display_compare(data1, data2):
        ImageUtils.display_image_compare(data1, data2)
        
    @staticmethod
    def save_compare(data1, data2, plot_path):
        ImageUtils.save_image_compare(data1, data2, plot_path)
        
    @staticmethod
    def display_and_save_compare_labeled(data1, data2, data3, plot_path):
        ImageUtils.display_and_save_compare_labeled(data1, data2, data3, plot_path)