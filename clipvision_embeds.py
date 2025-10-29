# from typing import Any
import numpy as np
import torch
from comfy.clip_vision import Output
from comfy.comfy_types import IO

class EmbedsInfo:
    """
    Extracts information about CLIP vision output.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("CLIP TEXT Info",)
    FUNCTION = "cond_to_embeds"
    CATEGORY = "ClipVisionTools/experimental"

    def cond_to_embeds(self,  clip_vision_output):
        info = clip_vision_output["image_embeds"].shape
        return info,

class Cond2Embeds:
    """
    Converts conditioning information to CLIP vision output.
    """    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "conditioning": (IO.CONDITIONING,),
            },
        }

    RETURN_TYPES = ("CLIP_VISION_OUTPUT", )
    RETURN_NAMES = ("CLIP_VISION_OUTPUT", )
    FUNCTION = "cond_to_embeds"
    CATEGORY = "ClipVisionTools"

    def cond_to_embeds(self,  conditioning):
        tmp = Output()
        cond = conditioning[0][1].get("pooled_output", None)
        tmp["image_embeds"] = cond
        return tmp,

class ScaleEmbeds:
    """
    Scales the image embeddings in the CLIP vision output.
    """    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "scale": ("FLOAT", {
                    "default": 1, "step": 0.01
                }),
            },
        }

    RETURN_TYPES = ("CLIP_VISION_OUTPUT", )
    RETURN_NAMES = ("clip_vision_output", )
    FUNCTION = "Scale_Embeds"
    CATEGORY = "ClipVisionTools/experimental"

    def Scale_Embeds(self,  clip_vision_output, scale):
        tmp = Output()
        cv1 = clip_vision_output["image_embeds"].numpy().flatten().tolist()
        oshape = clip_vision_output["image_embeds"].shape
        image_embeds = np.array(cv1) * scale
        tmp["image_embeds"] = torch.tensor(np.array(image_embeds).reshape(oshape))
        return tmp,

class CalcEmbeds:
    """
    Scales the image embeddings in the CLIP vision output.
    These algorithms are experimental. Some of them seem to work, some of them seem to have unexpected results.
    """    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "calculation": (["normalize", "add", "subtract", "most common", "remove image2 from image1", "average of both images", "or", "multiply"], {"default": "subtract"}),
                "clip_vision_output1": ("CLIP_VISION_OUTPUT",),
            },
            "optional": {
                    "clip_vision_output2": ("CLIP_VISION_OUTPUT",),
            }
        }

    RETURN_TYPES = ("CLIP_VISION_OUTPUT", )
    RETURN_NAMES = ("clip_vision_output", )
    FUNCTION = "Calc_Embeds"
    CATEGORY = "ClipVisionTools/experimental"

    def Calc_Embeds(self, calculation, clip_vision_output1, clip_vision_output2=None):
        tmp = Output()
        cv1 = clip_vision_output1["image_embeds"].numpy().flatten().tolist()
        oshape = clip_vision_output1["image_embeds"].shape

        if clip_vision_output2 is not None:
            cv2 = clip_vision_output2["image_embeds"].numpy().flatten().tolist()

        if calculation == "normalize":
            if cv2 is not None:
                a_min, a_max = np.min(np.array(cv1)), np.max(np.array(cv1))
                b_min, b_max = np.min(np.array(cv2)), np.max(np.array(cv2))
                if np.isclose(a_max, a_min):
                    image_embeds = np.full_like(np.array(cv1), (b_min + b_max) / 2)                
                else:
                    image_embeds = (np.array(cv1) - a_min) / (a_max - a_min) * (b_max - b_min) + b_min
            else:
                x = np.array(cv1, dtype=float)
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
            image_embeds = (np.array(cv1) + np.array(cv2))

        if calculation == "subtract":
            image_embeds = (np.array(cv1) - np.array(cv2))

        if calculation == "most common":
            cos_sim = np.dot(np.array(cv1), np.array(cv2)) / (np.linalg.norm(np.array(cv1)) * np.linalg.norm(np.array(cv2)))
            image_embeds = cos_sim * (np.array(cv1) + np.array(cv2)) / 2

        if calculation == "remove image2 from image1":
            image_embeds = (np.array(cv1) - np.array(cv2))
            image_embeds = image_embeds / np.linalg.norm(image_embeds)

        if calculation == "average of both images":
            image_embeds = (np.array(cv1) + np.array(cv2)) / 2

        if calculation == "or":
            image_embeds = np.minimum(np.array(cv1), np.array(cv2)) 

        if calculation == "multiply":
            image_embeds = (np.array(cv1) * np.array(cv2))

        tmp["image_embeds"] = torch.tensor(np.array(image_embeds).reshape(oshape))
        return tmp,

class CompareEmbeds:
    """
    Compares two sets of image embeddings.
    """    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "clip_vision_output1": ("CLIP_VISION_OUTPUT",),
                "clip_vision_output2": ("CLIP_VISION_OUTPUT",),
            }
        }

    RETURN_TYPES = ("FLOAT", )
    RETURN_NAMES = ("SCORE", )
    FUNCTION = "Compare_Embeds"
    CATEGORY = "ClipVisionTools"

    def Compare_Embeds(self, clip_vision_output1, clip_vision_output2):
        img_emb1 = clip_vision_output1["image_embeds"].numpy().flatten().tolist()
        img_emb2 = clip_vision_output2["image_embeds"].numpy().flatten().tolist()
        image_norm = np.linalg.norm(img_emb1)
        feature_norm = np.linalg.norm(img_emb2)
        dot_products = np.dot(img_emb1, img_emb2)  # Matrix-Vector-Product
        score = dot_products / (image_norm * feature_norm)
        return score,
