from imports import *

###############################################functions
start_time = 0  # Define start_time in the global scope

def starttime():
    global start_time  # Use the global keyword to access the global start_time variable
    start_time = time.time()
    #hint: starttime() - To start timer.
    
def endtime():
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    #hint: endtime() - To end timer

def check(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    #hint: check(path) - To recreate a particular path
            
def npyconversion(tif_dir, npy_path):
    tif_files = [f for f in os.listdir(tif_dir) if f.endswith('.tif')]
    data = []
    for tif_file in tqdm(tif_files):
        img = Image.open(os.path.join(tif_dir, tif_file))
        data.append(np.array(img))
    np.save(npy_path, data)
    #hint: npyconversion(path , npy + '/filename.npy' ) -To create NPY files
            
def a2bcopy(path1, path2):
    for z in tqdm(sorted(os.listdir(path1))):
        if z.endswith("tif"):
            shutil.copy(os.path.join(path1, z), os.path.join(path2, z))
            #hint: a2bcopy(sorce path, dest path) - To copy all images
            
def crop(test):
    for z in tqdm(sorted(os.listdir(test))):
        if (z.endswith("tif")): # checking the file ends with tif
            # Read in the image
            img = mh.imread(os.path.join(test, z))
            img_cropped = img[1000:2500, 2500:4500]
            mh.imsave(os.path.join(test, z), img_cropped)
            print(z)
            #hint: crop(spath) - To crop all images
    
def a2brandom(src_dir, dst_dir, number):
    # Get a list of all image files in the source directory
    image_files = [f for f in tqdm(os.listdir(src_dir)) if f.endswith('.tif')]
    # Randomly select 10 images from the list
    selected_images = random.sample(image_files, number)
    # Copy the selected images to the destination directory
    for image in selected_images:
        src_path = os.path.join(src_dir, image)
        dst_path = os.path.join(dst_dir, image)
        shutil.copy2(src_path, dst_path)
    # Print a message when done
    print('Copied 10 random images to', dst_dir)
    #hint: a2brandom(sorce path, dest path, random number) - To copy n random images
    
def count_files(dir_path):
    if os.path.isdir(dir_path):
        file_count = 0
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            file_size_bytes = os.path.getsize(file_path)
            file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
            #print(f'{file_name} - Size: {file_size_mb} MB')
            file_count += 1
        return file_count
    else:
        print(f"{dir_path} is not a valid directory")
        return 0    
        #hint: count_files(path) - To count number of files in that path
    
def norm(img):
    img = img.astype(np.float64)
    img /= img.max() 
    return img
            
def shape(raw):
    for z in tqdm(sorted(os.listdir(raw))):
        if (z.endswith("tif")):
            img = mh.imread(os.path.join(raw, z))
            print (img.shape)
            #hint: shape(path) _ To print shape of all the images in the path
            
def refresh(experiment: str, directories: dict):
    for key in directories:
        if os.path.exists(directories[key]):
            shutil.rmtree(directories[key])
        os.makedirs(directories[key])
        #hint: refresh("experiment name", directories) - to recreate all directories in that dict
        
def paths(directories):
    for key, value in directories.items():
        globals()[key] = value
    return directories
    #hint: paths(directories) - To call the directories outside the dictionary
            
            
#def del(path):