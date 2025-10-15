from typing import Any
import numpy as np
from .clipvision_db import LoadDB
from .utils import (get_image, repairImage)
from nodes import (ImageBatch)
class results_class:
    distances: Any = None
    clip_features: Any = None

    def __init__(self, d, c):
        self.distances = d
        self.clip_features = c

class ImageSearcher:
    LOADED_DB: Any = None
    LOADED_clip_features: Any = None

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "img_db": ("LoadDB",),
                "img_emb": ("img_emb",),
            },
        }

    RETURN_TYPES = ("SRESULTS",)
    RETURN_NAMES = ("RESULTS",)
    FUNCTION = "get_similar_images"
    CATEGORY = "ClipVisionTools"

    def get_similar_images(self, img_db: LoadDB, img_emb):
        distances = []
        image_embeds = img_emb
        
        clip_features = img_db.LOADED_DB
        self.LOADED_clip_features = clip_features

        filenames, features_list = zip(*clip_features)
        # 2D NumPy array: shape (n_samples, n_features)
        features = np.stack(features_list)
        # Norm image_embeds
        image_norm = np.linalg.norm(image_embeds)
        # Norm features
        features_norms = np.linalg.norm(features, axis=1)
        # scalscalar product of features and image_embeds at once
        dot_products = features @ image_embeds  # Matrix vector product
        # Cosine-similarity:
        distances = dot_products / (image_norm * features_norms)
        result = results_class(distances, self.LOADED_clip_features)
        return result,

class ResultCombiner:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "results1": ("SRESULTS",),
                "results2": ("SRESULTS",),
            },
        }

    RETURN_TYPES = ("SRESULTS",)
    RETURN_NAMES = ("RESULTS",)
    FUNCTION = "combine_results"
    CATEGORY = "ClipVisionTools"

    def combine_results(self, results1, results2):
        distances1 = results1.distances
        distances2 = results2.distances
        tmp_dist = [min(a, b) for a, b in zip(distances1, distances2)]
        result = results_class(tmp_dist,results1.clip_features)
        return result,

class ResultBrowser:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "results": ("SRESULTS",),
                "image_index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff }),
                "match": (["first", "last"], {"default": "first"})
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT")
    RETURN_NAMES = ("IMAGE", "FILENAME", "SCORE")
    FUNCTION = "Result_Browser"
    CATEGORY = "ClipVisionTools"

    def Result_Browser(self, results, image_index, match):
        distances1 = results.distances
        clip_features = results.clip_features

        if match == "first":
            indices = np.argsort(distances1)[::-1]
            distances_sorted = [(i, distances1[i]) for i in indices]

        if match == "last":
            indices = np.argsort(distances1)[::1]
            distances_sorted = [(i, distances1[i]) for i in indices]

        idx, score = distances_sorted[image_index]
        file_name, _ = clip_features[idx]
        t_image = repairImage(get_image(file_name))
        return t_image, file_name, score

class ResultBrowserAdvanced:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "results": ("SRESULTS",),
                "offset_index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff }),
                "image_count": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff }),
                "match": (["first", "last"], {"default": "first"})
            },
            "optional": { 
                "batch_frame_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT")
    RETURN_NAMES = ("IMAGE", "FILENAME", "SCORE")
    FUNCTION = "AdvResult_Browser"
    CATEGORY = "ClipVisionTools"

    def AdvResult_Browser(self, results, offset_index, image_count, match, batch_frame_image=None ):
        scores = []
        filenames = []
        distances1 = results.distances
        clip_features = results.clip_features

        remove_first_image = batch_frame_image is not None

        if match == "first":
            indices = np.argsort(distances1)[::-1]
            distances_sorted = [(i, distances1[i]) for i in indices]

        if match == "last":
            indices = np.argsort(distances1)[::1]
            distances_sorted = [(i, distances1[i]) for i in indices]

        start_idx = offset_index
        end_idx = start_idx + image_count
        
        for i in range(start_idx, end_idx):
            idx, score = distances_sorted[i]
            file_name, _ = clip_features[idx]
            t_image = repairImage(get_image(file_name))
            if batch_frame_image is None:
                #print("init_frame image")
                batch_frame_image = t_image
            else:
                #print("batching image")
                batch_frame_image = ImageBatch().batch(batch_frame_image, t_image )[0]
            scores.append(scores)
            filenames.append(file_name)
            #print("ENDING")
        #print("datatype= ", type(batch_frame_image))
            
        if remove_first_image:
            batch_frame_image = batch_frame_image[1:]
            #batch_frame_image.pop(0)
           # batch_frame_image

        return batch_frame_image, filenames, scores
#ImageBatch

