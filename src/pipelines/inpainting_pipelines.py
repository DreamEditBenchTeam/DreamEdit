from diffusers import StableDiffusionInpaintPipeline
import torch

import os
import torchvision
from diffusers import StableDiffusionGLIGENPipeline

from PIL import Image


# pipe = StableDiffusionGLIGENPipeline.from_pretrained("gligen/diffusers-inpainting-text-box", torch_dtype=torch.float32)
# pipe.to("cuda")

# os.makedirs("images", exist_ok=True)
# os.makedirs("images/output", exist_ok=True)

def inpaint_text_gligen(pipe, prompt, background_path, bounding_box, gligen_phrase, config):
    images = pipe(
        prompt,
        num_images_per_prompt=1,
        gligen_phrases=[gligen_phrase],
        gligen_inpaint_image=Image.open(background_path).convert('RGB'),
        gligen_boxes=[[x / 512 for x in bounding_box]],
        gligen_scheduled_sampling_beta=config.gligen_scheduled_sampling_beta,
        output_type="numpy",
        num_inference_steps=config.num_inference_steps
    ).images
    return images


def select_inpainting_pipeline(name: str, device="cuda"):
    if name == "sd_inpaint":
        return get_stable_diffusion_inpaint_pipeline(device)
    elif name == "gligen_inpaint":
        return get_gligen_inpaint_pipeline(device)
    else:
        return None


def get_stable_diffusion_inpaint_pipeline(device):
    inpainting_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
    ).to(device)
    return inpainting_pipe


def get_gligen_inpaint_pipeline(device):
    inpainting_pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        "gligen/diffusers-inpainting-text-box", torch_dtype=torch.float32
    ).to(device)
    return inpainting_pipe
