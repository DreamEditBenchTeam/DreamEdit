import os

def get_file_path(filename):
    """
    search file across whole repo and return abspath
    """
    for root, dirs, files in os.walk(r'.'):
        for name in files:
            if name == filename:
                return os.path.abspath(os.path.join(root, name))
    raise FileNotFoundError(filename, "not found.")
    
def check_format(source_path):
    if(os.path.isdir(source_path)):
        return 'folder'
    IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes
    VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
    if(source_path.split(".")[-1] in IMG_FORMATS):
        return 'image'
    if(source_path.split(".")[-1] in VID_FORMATS):
        return 'video'

    return None

def check_is_image(source_path):
    if(not os.path.isdir(source_path)):
        IMG_FORMATS = 'jpeg', 'jpg', 'png'  # include image suffixes
        if(source_path.split(".")[-1] in IMG_FORMATS):
            return True
    return False