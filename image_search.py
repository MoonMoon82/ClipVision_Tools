from typing import Any
import numpy as np
from .clipvision_db import LoadDB
from .utils import (get_image, repairImage)

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
    #last_match = ""
    #distances_sorted = []
    #distances_sorted2 = []
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
