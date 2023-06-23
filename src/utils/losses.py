import torch

def mse(img1, img2):
    h, w = img1.shape[-2:]
    diff = (img1 - img2).abs_().mean(dim=[0, 1])
    err = torch.square(diff).sum()
    # err = err/(float(h*w))
    return torch.log(err)