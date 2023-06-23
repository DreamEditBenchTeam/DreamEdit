import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
import os
from PIL import Image


def save_tensor_image(tensor_image, dest_folder, filename: str, filetype="jpg"):
    """
    param:
        image in size [B, 3, H, W]
        dest_folder 
        filename
    return:
        ---
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    save_image(tensor_image, os.path.join(dest_folder, f"{filename}.{filetype}"))


def save_pil_image(pil_image, dest_folder, filename: str, filetype="jpg"):
    """
    param:
        image as PIL Image
        dest_folder
        filename
    return:
        ---
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    pil_image.save(os.path.join(dest_folder, f"{filename}.{filetype}"))


def get_mask_pil_image(mask_tensor):
    """
    param:
        mask_tensor in size [H, W]
    return:
        mask in PIL image
    """
    mask = np.array(mask_tensor).astype('uint8')
    mask = np.squeeze(mask)
    mask_img = Image.fromarray(mask * 255)
    return mask_img


def pil_to_tensor(pil_img):
    """
    param:
        pil_img - Image Object
    return:
        tensor in size [1, C, H, W]
    """
    tensor = transforms.ToTensor()(pil_img).unsqueeze_(0)
    return tensor


def tensor_to_pil(tensor_img):
    """
    param:
        tensor_img in size [1, C, H, W]
    return:
        pil - Image Object
    """
    pil = transforms.ToPILImage()(tensor_img.squeeze_(0))
    return pil


def get_concat_pil_images(images: list, direction: str = 'h'):
    """
    param:
        images - list of pil images
        direction - h for horizonal
    return:
        pil - Image Object
    """
    if direction == 'h':
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        return new_im
    else:
        widths, heights = zip(*(i.size for i in images))
        max_width = max(widths)
        total_height = sum(heights)
        new_im = Image.new('RGB', (max_width, total_height))
        y_offset = 0
        for im in images:
            new_im.paste(im, (0, y_offset))
            y_offset += im.size[1]
        return new_im
