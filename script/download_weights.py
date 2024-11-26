#!/usr/bin/env python

import os
import shutil
import sys

from transformers import AutoProcessor, SiglipVisionModel

# append project directory to path so predict.py can be imported
sys.path.append('.')

from predict import SIGLIP_CACHE

if os.path.exists(SIGLIP_CACHE):
    shutil.rmtree(SIGLIP_CACHE)
os.makedirs(SIGLIP_CACHE, exist_ok=True)

siglip = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384", cache_dir=SIGLIP_CACHE)
