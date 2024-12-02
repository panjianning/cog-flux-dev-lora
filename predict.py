# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
os.sys.path.append('/src/PuLID/')

from cog import BasePredictor, Input, Path
import os
import re
import time
import torch
import subprocess
import numpy as np
from PIL import Image
from transformers import pipeline as tpipe
from typing import List
from torchvision import transforms
from weights import WeightsDownloadCache
from transformers import CLIPImageProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker
)
from diffusers.models.controlnet_flux import FluxControlNetModel, FluxMultiControlNetModel

from PIL import Image

import torch.nn as nn

from pipeline_flux_ipa_controlnet import FluxPipeline
from transformer_flux import FluxTransformer2DModel
import os

from typing import Callable, List, Optional, Union, Dict, Any

from PIL import Image
import requests
import io

from infer_flux_ipa_siglip import IPAdapter
import cv2
import numpy as np

from PuLID.pulid.pipeline_flux import PuLIDPipeline


def canny_processor(image, low_threshold=100, high_threshold=200):
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

MAX_IMAGE_SIZE = 1440
MODEL_CACHE = "Hyper-FLUX.1-dev"
SAFETY_CACHE = "safety-cache"
SIGLIP_CACHE = "siglip-cache"
FEATURE_EXTRACTOR = "/src/feature-extractor"
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"
# MODEL_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
MODEL_URL = "https://weights.replicate.delivery/default/ByteDance/Hyper-FLUX.1-dev-8steps/model.tar"

ipadapter_path = "/src/FLUX.1-dev-IP-Adapter/ip-adapter.bin"   
image_encoder_path = "/src/siglip-so400m-patch14-384"
# controlnet_path = "InstantX/FLUX.1-dev-Controlnet-Canny"
# image_encoder_path = "google/siglip-so400m-patch14-384"

controlnet_path = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro"

ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "21:9": (1536, 640),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "4:5": (896, 1088),
    "5:4": (1088, 896),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
    "9:16": (768, 1344),
    "9:21": (640, 1536),
}

def download_weights(url, dest, file=False):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    if not file:
        subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    else:
        subprocess.check_call(["pget", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)
    
def resize_image_center_crop(image_path_or_url, target_width, target_height):
    """
    Resizes an image while maintaining aspect ratio using center cropping.
    Handles both local file paths and URLs.

    Args:
        image_path_or_url: Path to the image file or URL.
        target_width: Desired width of the output image.
        target_height: Desired height of the output image.

    Returns:
        A PIL Image object with the resized image, or None if there's an error.
    """
    # if image_path_or_url.startswith(('http://', 'https://')):  # Check if it's a URL
    #     response = requests.get(image_path_or_url, stream=True, timeout=5)
    #     response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    #     img = Image.open(io.BytesIO(response.content))
    # else:  # Assume it's a local file path
    img = Image.open(image_path_or_url)

    img_width, img_height = img.size

    # Calculate aspect ratios
    aspect_ratio_img = img_width / img_height
    aspect_ratio_target = target_width / target_height

    # Determine cropping box
    if aspect_ratio_img > aspect_ratio_target:  # Image is wider than target
        new_width = int(img_height * aspect_ratio_target)
        left = (img_width - new_width) // 2
        right = left + new_width
        top = 0
        bottom = img_height
    else:  # Image is taller or equal to target
        new_height = int(img_width / aspect_ratio_target)
        left = 0
        right = img_width
        top = (img_height - new_height) // 2
        bottom = top + new_height

    # Crop the image
    cropped_img = img.crop((left, top, right, bottom))

    # Resize to target dimensions
    resized_img = cropped_img.resize((target_width, target_height), Image.LANCZOS)

    return resized_img


def resize_numpy_image_long(image, resize_long_edge=768):
    h, w = image.shape[:2]
    if max(h, w) <= resize_long_edge:
        return image
    k = resize_long_edge / max(h, w)
    h = int(h * k)
    w = int(w * k)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()
        


        self.weights_cache = WeightsDownloadCache()
        self.last_loaded_lora = None

        print("Loading safety checker...")
        if not os.path.exists(SAFETY_CACHE):
            download_weights(SAFETY_URL, SAFETY_CACHE)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_CACHE, torch_dtype=torch.float16
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)
        
        print("Loading Flux txt2img Pipeline")
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
            
        transformer = FluxTransformer2DModel.from_pretrained(
            MODEL_CACHE, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        transformer = transformer.to("cuda")

        self.pulid_model = PuLIDPipeline(transformer, device="cuda", weight_dtype=torch.bfloat16,
                    onnx_provider='gpu')
        self.pulid_model.load_pretrain()

        controlnet_union = FluxControlNetModel.from_pretrained(controlnet_path, 
                                                         torch_dtype=torch.bfloat16).to('cuda')
        controlnet = FluxMultiControlNetModel([controlnet_union])

        pipe = FluxPipeline.from_pretrained(
            MODEL_CACHE, transformer=transformer, controlnet=controlnet, torch_dtype=torch.bfloat16
        )
        pipe = pipe.to("cuda")
        
        self.ip_model = IPAdapter(pipe, image_encoder_path, ipadapter_path, device="cuda", num_tokens=128)


        self.depth_pipe = tpipe(task="depth-estimation", device='cuda', model="depth-anything/Depth-Anything-V2-Small-hf")
        self.nsfw_classifier = tpipe("image-classification", device='cuda', model="Falconsai/nsfw_image_detection")

        print("setup took: ", time.time() - start)

    @torch.amp.autocast('cuda')
    def run_safety_checker(self, image):
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to("cuda")
        np_image = [np.array(val) for val in image]
        image, has_nsfw_concept = self.safety_checker(
            images=np_image,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept


    def run_nsfw_checker(self, images):
        result = []
        for image in images:
            items = self.nsfw_classifier(image)
            print(items)
            nsfw = False
            for item in  items:
                if item['label'] == 'nsfw' and item['score'] >= 0.5:
                    nsfw = True
                    break
            result.append(nsfw)
        return result


    def aspect_ratio_to_width_height(self, aspect_ratio: str) -> tuple[int, int]:
        return ASPECT_RATIOS[aspect_ratio]

    def get_image(self, image: str):
        image = Image.open(image).convert("RGB")
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 2.0 * x - 1.0),
            ]
        )
        img: torch.Tensor = transform(image)
        return img[None, ...]

    @staticmethod
    def make_multiple_of_16(n):
        return ((n + 15) // 16) * 16

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Prompt for generated image"),
        aspect_ratio: str = Input(
            description="Aspect ratio for the generated image",
            choices=list(ASPECT_RATIOS.keys()),
            default="1:1"
        ),
        image: Path = Input(
            description="Input image for Ip Adapter",
            default=None,
        ),
        image_strength: float = Input(
            description="Ip Adatater Image strength. 0.0 corresponds to no ip adapter.",
            ge=0,le=1,default=0.5,
        ),
        control_image: Path = Input(
            description="Input image for Depth ControlNet",
            default=None,
        ),
        control_strength: float = Input(
            description="Depth Control Image strength. 0.0 corresponds no depth control",
            ge=0,le=1,default=0.7,
        ),
        face_image:  Path = Input(
            description="Face image for Pulid",
            default=None,
        ),
        face_strength: float = Input(
            description="Face Image strength. 0.0 corresponds no face control",
            ge=0,le=1,default=1.0,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=1,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps.",
            ge=1,le=50,default=8,
        ),
        guidance_scale: float = Input(
            description="Guidance scale for the diffusion process",
            ge=0,le=10,default=3.5,
        ),
        seed: int = Input(description="Random seed. Set for reproducible generation", default=None),
        output_format: str = Input(
            description="Format of the output images",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
            default=80,
            ge=0,
            le=100,
        ),
        disable_sc: bool = Input(
            description="Disable safety checker for generated images. This feature is only available through the API. See [https://replicate.com/docs/how-does-replicate-work#safety](https://replicate.com/docs/how-does-replicate-work#safety)",
            default=False,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        width, height = self.aspect_ratio_to_width_height(aspect_ratio)
        
        print(f"Prompt: {prompt}")

        if not image:
            image = '/src/default_ref.jpeg'
            image_strength = 0

        pil_image = resize_image_center_crop(image_path_or_url=image, target_width=width, target_height=height).convert('RGB')
        
        pil_control_image = None
        if control_image:
            pil_control_image = resize_image_center_crop(image_path_or_url=control_image, target_width=width, target_height=height)
            pil_control_image = self.depth_pipe(pil_control_image)["depth"].convert('RGB')
            # pil_control_image = canny_processor(pil_control_image)
        
        id_embeddings = None
        if face_image:
            pil_face_image = resize_image_center_crop(image_path_or_url=face_image, target_width=width, target_height=height)
            face_image_arr = resize_numpy_image_long(np.array(pil_face_image),1024)
            id_embeddings, uncond_id_embeddings = self.pulid_model.get_id_embedding(face_image_arr)

        ip_args = {
            "prompt": prompt,
            "guidance_scale": guidance_scale,
            "seed":seed,
            
            "control_image": [pil_control_image] if pil_control_image is not None else None,
            "controlnet_conditioning_scale": [control_strength],
            "control_mode": [2],
            
            "id": id_embeddings,
            "id_weight": face_strength,

            "num_inference_steps": num_inference_steps,
            "num_samples":num_outputs,
            "pil_image": pil_image,
            "scale": image_strength,
            "width": width, 
            "height": height
        }

        images = self.ip_model.generate(**ip_args)

        if not disable_sc:
            has_nsfw_content = self.run_nsfw_checker(images)

        output_paths = []
        for i, image in enumerate(images):
            if not disable_sc and has_nsfw_content[i]:
                print(f"NSFW content detected in image {i}")
                continue
            output_path = f"/tmp/out-{i}.{output_format}"
            if output_format != 'png':
                image.save(output_path, quality=output_quality, optimize=True)
            else:
                image.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception("NSFW content detected. Try running it again, or try a different prompt.")

        return output_paths
    
