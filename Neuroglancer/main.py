import os
import shutil
import tempfile
import json
import mahotas

########################################################################################################################

""" CREATING TEMP FOLDER """

temp_folder = tempfile.TemporaryDirectory()
temp_folder_path = temp_folder.name
print("Path of Temporary folder one is {}".format(temp_folder_path))

########################################################################################################################

"""PATH HERE SPECIFIES THE INPUT PATH OF THE TIF STACK"""

path = "data/Source/Stacks"
# print(sorted(os.listdir(path)))
images = [f for f in sorted(os.listdir(path)) if '.tif' in f.lower()]

for image in images:
    new_path = temp_folder_path + "/" + image
    shutil.copy(path + "/" + image, new_path)
    print("Image has been copied to the location: {}".format(new_path))
print("All the .TIF images have been copied to the temporary folder")

########################################################################################################################

"""ANOTHER TEMP FOLDER FOR .TIF TO JPEG CONVERSION"""
temp_folder_0 = tempfile.TemporaryDirectory()
temp_folder_path_0 = temp_folder_0.name
print("Path of Temporary folder two is {}".format(temp_folder_path_0))

########################################################################################################################

"""CONVERTING IMAGES USING IMAGE MAGICK"""

path = temp_folder_path
images = [f for f in sorted(os.listdir(path)) if '.tif' in f.lower()]

for i in images:
    img_input_path = path + "/" + i
    img_output_path = temp_folder_path_0 + "/" + i.replace("tif", "jpeg")

    # BASH COMMAMD FOR IMAGEMAGICK, -quiet is used to hide tif tag warning

    os.system("convert -quiet {} {}".format(img_input_path, img_output_path))
    print("The converted image has been saved to {}".format(img_output_path))

########################################################################################################################

path = "data/Source/Stacks"
counter = [f for f in sorted(os.listdir(temp_folder_path)) if '.tif' in f.lower()]

########################################################################################################################

"""EXTRACTING DATA FROM THE INFO FILE"""

images = [f for f in sorted(os.listdir(path)) if '.info' in f.lower()]
# print(images)

with open(path + "/" + images[0]) as f:
    lines = f.readlines()
    da = [i for i in lines if "pixelsize" in i.lower()]
    la = [i for i in lines if "tif" in i.lower()]
    pixel_size = da[0].split()
    pixel_x = pixel_size[1]
    pixel_y = pixel_size[2]
    # print(la)
    z_axis = la[0].split()
    z_value = float(z_axis[1])
    f.close()
# print(z_axis, pixel_x, pixel_y)

with open(path + "/" + images[1]) as f:
    lines = f.readlines()
    la = [i for i in lines if "tif" in i.lower()]
    z_axis = la[0].split()
    z_value_2 = float(z_axis[1])
    volume_data = int(z_value_2 - z_value)
    f.close()
# print(volume_data)

########################################################################################################################


########################################################################################################################

"""GENERATING JSON INFO FILE, USING MAHOTAS TO GET DIMENSIONS, CHANNELS"""

counter = [f for f in sorted(os.listdir(temp_folder_path_0)) if '.jpeg' in f.lower()]

slices = len(counter)
image = mahotas.imread(temp_folder_path_0 + "/" + counter[0])
height = image.shape[0]
width = image.shape[1]

if image.ndim == 2:
    channels = 1  # grayscale

if image.ndim == 3:
    channels = image.shape[-1]

# JSON FILE
array_out = {
    "type": "image",
    "data_type": str(image.dtype),
    "num_channels": channels,
    "scales": [
        {
            "chunk_sizes": [],
            "encoding": "jpeg",
            "key": "full",
            "resolution": [int(pixel_x), int(pixel_y), volume_data],
            "size": [height, width, slices],
            "voxel_offset": [0, 0, 0]
        }
    ]

}

with open('data.json', 'w') as f:
    json.dump(array_out, f)

########################################################################################################################

"""RUNNING NEUROGLANCER SCRIPTS"""

os.system("rm -r output")
os.system("mkdir output")
os.system("generate-scales-info data.json output")
os.system("slices-to-precomputed --input-orientation RPS {} output".format(temp_folder_path_0))

# USE THE BELOW COMMAND TO RUN SCRIPTS ON CONVERTED SAMPLE JPEG STACK IN THE IMAGES FOLDER, DEBUG PURPOSE
# os.system("slices-to-precomputed --input-orientation RPS "+ "images "+ "output")
