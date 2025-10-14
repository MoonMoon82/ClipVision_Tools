from typing import Any
import numpy as np
from .utils import (get_image_clip_embeddings)

class GetImageEmbeds:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "clip_vision": ("CLIP_VISION",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("img_emb", )
    RETURN_NAMES = ("IMG_EMB", )
    FUNCTION = "GetImgEmbeds"
    CATEGORY = "ClipVisionTools"

    def GetImgEmbeds(self, clip_vision, image):
        return get_image_clip_embeddings(clip_vision, image),

class ScaleEmbeds:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "img_emb1": ("img_emb",),
                "scale": ("FLOAT", {
                    "default": 1, "step": 0.01
                }),
            },
        }

    RETURN_TYPES = ("img_emb", )
    RETURN_NAMES = ("IMG_EMBEDS", )
    FUNCTION = "Scale_Embeds"
    CATEGORY = "ClipVisionTools"

    def Scale_Embeds(self,  img_emb1, scale):
        image_embeds = (np.array(img_emb1) * scale)
        return image_embeds,

#These algorithms are experimental. Some of them seem to work, some of them seem to have unexpected results.
class CalcEmbeds:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "calculation": (["normalize", "add", "subtract", "most common", "remove image2 from image1", "average of both images", "or", "multiply"], {"default": "subtract"}),
                "img_emb1": ("img_emb",),
            },
            "optional": {
                    "img_emb2": ("img_emb",),
            }
        }

    RETURN_TYPES = ("img_emb", )
    RETURN_NAMES = ("IMG_EMBEDS", )
    FUNCTION = "Calc_Embeds"
    CATEGORY = "ClipVisionTools"

    def Calc_Embeds(self, calculation, img_emb1, img_emb2=None):
        if calculation == "normalize":
            if img_emb2 is not None:
                a_min, a_max = np.min(np.array(img_emb1)), np.max(np.array(img_emb1))
                b_min, b_max = np.min(np.array(img_emb2)), np.max(np.array(img_emb2))
                if np.isclose(a_max, a_min):
                    image_embeds = np.full_like(np.array(img_emb1), (b_min + b_max) / 2)                
                else:
                    image_embeds = (np.array(img_emb1) - a_min) / (a_max - a_min) * (b_max - b_min) + b_min
            else:
                x = np.array(img_emb1, dtype=float)
                min_x = np.min(x)
                max_x = np.max(x)
                if max_x == min_x:
                    image_embeds = np.zeros_like(x)
                else:
                    # scale to [0, 1]
                    norm = (x - min_x) / (max_x - min_x)
                    # convert to [-1, 1]
                    image_embeds = norm * 2 - 1

        if calculation == "add":
            image_embeds = (np.array(img_emb1) + np.array(img_emb2))

        if calculation == "subtract":
            image_embeds = (np.array(img_emb1) - np.array(img_emb2))

        if calculation == "most common":
            cos_sim = np.dot(np.array(img_emb1), np.array(img_emb2)) / (np.linalg.norm(np.array(img_emb1)) * np.linalg.norm(np.array(img_emb2)))
            image_embeds = cos_sim * (np.array(img_emb1) + np.array(img_emb2)) / 2

        if calculation == "remove image2 from image1":
            image_embeds = (np.array(img_emb1) - np.array(img_emb2))
            image_embeds = image_embeds / np.linalg.norm(image_embeds)

        if calculation == "average of both images":
            image_embeds = (np.array(img_emb1) + np.array(img_emb2)) / 2

        if calculation == "or":
            image_embeds = np.minimum(np.array(img_emb1), np.array(img_emb2)) 

        if calculation == "multiply":
            image_embeds = (np.array(img_emb1) * np.array(img_emb2))
        return image_embeds,

class CompareEmbeds:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "img_emb1": ("img_emb",),
                "img_emb2": ("img_emb",),
            }
        }

    RETURN_TYPES = ("FLOAT", )
    RETURN_NAMES = ("SCORE", )
    FUNCTION = "Compare_Embeds"
    CATEGORY = "ClipVisionTools"

    def Compare_Embeds(self, img_emb1, img_emb2):
        image_norm = np.linalg.norm(img_emb1)
        feature_norm = np.linalg.norm(img_emb2)
        dot_products = np.dot(img_emb1, img_emb2)  # Matrix-Vector-Product
        score = dot_products / (image_norm * feature_norm)
        return score,
