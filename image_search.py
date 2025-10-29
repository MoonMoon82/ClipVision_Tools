from typing import Any
import numpy as np
from .clipvision_db import LoadDB
from .utils import (repairImage, image_to_tensor, rotate_image_to_exif)
from PIL import Image, ImageFile, UnidentifiedImageError, ExifTags
from nodes import (ImageBatch)
from pillow_heif import register_heif_opener
from comfy.utils import ProgressBar
from pathlib import Path

import datetime

import fnmatch

register_heif_opener()

class results_class:
    """
    Store the results of image search, including distances and clip features.
    """
    distances: Any = None
    clip_features: Any = None

    def __init__(self, distances, clip_features):
        self.distances = distances
        self.clip_features = clip_features

class ImageSearcher:
    """
    Search for similar images based on clip embeddings.
    """    
    LOADED_DB: Any = None
    LOADED_clip_features: Any = None

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "img_db": ("LoadDB",),
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
            },
        }

    RETURN_TYPES = ("SRESULTS",)
    RETURN_NAMES = ("RESULTS",)
    FUNCTION = "get_similar_images"
    CATEGORY = "ClipVisionTools"

    def get_similar_images(self, img_db: LoadDB, clip_vision_output):
        pbar = ProgressBar(7)
        distances = []

        image_embeds = clip_vision_output["image_embeds"].numpy().flatten().tolist()
        pbar.update(1)
        clip_features = img_db.LOADED_DB
        self.LOADED_clip_features = clip_features

        filenames, features_list = zip(*clip_features)
        pbar.update(1)
        # 2D NumPy array: shape (n_samples, n_features)
        #features = np.stack(features_list)
        features = np.array(features_list)
        pbar.update(1)
        # Norm image_embeds
        image_norm = np.linalg.norm(image_embeds)
        pbar.update(1)
        # Norm features
        features_norms = np.linalg.norm(features, axis=1)
        pbar.update(1)
        # scalscalar product of features and image_embeds at once
        dot_products = features @ image_embeds  # Matrix vector product
        pbar.update(1)
        # Cosine-similarity:
        distances = dot_products / (image_norm * features_norms)
        pbar.update(1)
        result = results_class(distances, self.LOADED_clip_features)
        return result,

class ResultCombiner:
    """
    Combines the results of two image searches.
    """    
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
        result = results_class(np.array(tmp_dist),results1.clip_features)
        return result,

class ResultSubtract:
    """
    Combines the results of two image searches.
    """    
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
    FUNCTION = "subtract_results"
    CATEGORY = "ClipVisionTools"

    def subtract_results(self, results1, results2):
        distances1 = results1.distances
        distances2 = results2.distances
        tmp_dist = [max(-0.99, min(0.99, a - b)) for a, b in zip(distances1, distances2)]
        result = results_class(np.array(tmp_dist),results1.clip_features)
        return result,

class ResultBrowser:
    """
    Browse the results of an image search.
    """    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "results": ("SRESULTS",),
                "image_index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "first result to output"}),
                "match": (["first", "last"], {"default": "first", "tooltip": "Sort order of images to output"})
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT")
    RETURN_NAMES = ("IMAGE", "FILENAME", "SCORE")
    FUNCTION = "Result_Browser"
    CATEGORY = "ClipVisionTools"

    def Result_Browser(self, results, image_index, match):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        distances1 = results.distances
        clip_features = results.clip_features

        pbar = ProgressBar(len(distances1))

        if match == "first":
            indices = np.argsort(distances1)[::-1]
            #distances_sorted = [pbar.update(1) or (i, distances1[i]) for i in indices]
                               #[print(f"Bearbeite {i}") or (i * 2) for i in range(5)]

        if match == "last":
            indices = np.argsort(distances1)[::1]

        #distances_sorted = [(i, distances1[i]) for i in indices]

        distances_sorted = np.column_stack((indices, distances1[indices]))

        idx, score = distances_sorted[image_index]
        file_name, _ = clip_features[int(idx)]
        img = Image.open(file_name)
        img = rotate_image_to_exif(img)
        t_image = repairImage(image_to_tensor(img))
        return t_image, file_name, score

class ResultBrowserAdvanced:
    """
    Browse the results of an image search to get image batches as output instead of single images.
    """    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "results": ("SRESULTS",),
                "offset_index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "first result to output" }),
                "image_count": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff , "tooltip": "Maximum amount of images to output"}),
                "match": (["first", "last"], {"default": "first", "tooltip": "Sort order of images to output"}),
                "threshold": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.001, "tooltip": "Similarity threshold for output images"}),
            },
            "optional": { 
                "batch_frame_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT")
    RETURN_NAMES = ("IMAGE", "FILENAME", "SCORE")
    FUNCTION = "AdvResult_Browser"
    CATEGORY = "ClipVisionTools"

    def AdvResult_Browser(self, results, offset_index, image_count, match, threshold, batch_frame_image=None ):
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        scores = []
        filenames = []
        distances1 = results.distances
        clip_features = results.clip_features

        remove_first_image = batch_frame_image is not None

        if match == "first":
            indices = np.argsort(distances1)[::-1]
            #distances_sorted = [(i, distances1[i]) for i in indices]

        if match == "last":
            indices = np.argsort(distances1)[::1]

        #distances_sorted = [(i, distances1[i]) for i in indices]
        distances_sorted = np.column_stack((indices, distances1[indices]))

        start_idx = offset_index
        end_idx = start_idx + image_count
        
        for i in range(start_idx, end_idx):
            if i >= len(distances_sorted):
                break
            idx, score = distances_sorted[i]
            if score < threshold:
                break

            file_name, _ = clip_features[int(idx)]

            img = Image.open(file_name)
            img = rotate_image_to_exif(img).convert("RGB")

            t_image = repairImage(image_to_tensor(img))

            if batch_frame_image is None:
                batch_frame_image = t_image
            else:
                batch_frame_image = ImageBatch().batch(batch_frame_image, t_image )[0]
            scores.append(score)
            filenames.append(file_name)

        if remove_first_image:
            batch_frame_image = batch_frame_image[1:]

        return batch_frame_image, filenames, scores

class EditResults:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {           
            "required": { 
                "results": ("SRESULTS",),
                "method": (["exclude", "filter", "replace"], {"default": "exclude", "tooltip": "Method to edit results"}),
                "edit_text": ("STRING", {
                    "multiline": False,
                    "default": "remove*images.jpg",
                    "tooltip": "Use wildcards (*) to match filenames or paths"
                },),                            
                "replace_text": ("STRING", {
                    "multiline": False,
                    "default": "/newpath/",
                    "tooltip": "Text to replace with (only for 'replace' method)"
                },),                            
            }
        }

    RETURN_TYPES = ("SRESULTS", )
    RETURN_NAMES = ("results", )
    FUNCTION = "Edit_Results"
    CATEGORY = "ClipVisionTools"

    def Edit_Results(self, results, method, edit_text, replace_text):

        distances1 = results.distances
        clip_features = results.clip_features
        new_results = results_class([], [])

        if isinstance(edit_text, str):
            edit_text = [edit_text]

        all = list(zip(clip_features, distances1))
        for cf, score in all:
            file_name, embeddings = cf
            if method == "exclude":
                if not any(fnmatch.fnmatch(file_name, pat) for pat in edit_text):
                    new_results.clip_features.append((file_name, embeddings))
                    new_results.distances.append(score)

            if method == "filter":
                if any(fnmatch.fnmatch(file_name, pat) for pat in edit_text):
                    new_results.clip_features.append((file_name, embeddings))
                    new_results.distances.append(score)

            if method == "replace":
                new_name = file_name
                for pat in edit_text:
                    if fnmatch.fnmatch(file_name, pat):
                        new_name = file_name.replace(pat.strip("*"), replace_text)
                new_results.clip_features.append((new_name, embeddings))
                new_results.distances.append(score)

        new_results.distances = np.array(new_results.distances)
        return new_results, 

class FolderScores:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {           
            "required": { 
                "results": ("SRESULTS",),
                "offset_index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "first result to show" }),
                "folder_count": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff, "tooltip": "Maximum amount of folders to show" }),
                "match": (["first", "last"], {"default": "first", "tooltip": "Sort order of folders to show"}),
                "weighted_threshold": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.001, "tooltip": "Similarity threshold for weighted scoring each image"}),
                "min_folder_level": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "minimum folder depth to include in scoring"}),
                "max_folder_level": ("INT", {"default": 100, "min": 0, "max": 0xffffffffffffffff, "tooltip": "maximum folder depth to include in scoring"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING" )
    RETURN_NAMES = ("folderstats", "filterstring")
    FUNCTION = "FolderScores"
    CATEGORY = "ClipVisionTools"

    def FolderScores(self, results, offset_index, folder_count, match, weighted_threshold, min_folder_level, max_folder_level):

        distances1 = results.distances
        clip_features = results.clip_features

        all = list(zip(clip_features, distances1))
        AllFolders = {}
        
        #print("step1 = ", datetime.datetime.now())
        
        for cf, score in all:
            file_name, embeddings = cf
            p1 = Path(file_name)
            rfs = 0
            for part in p1.parents[::-1]:
                rfs = rfs + 1
                if rfs > min_folder_level and rfs <= max_folder_level:
                    p = str(part)
                    if not p in AllFolders:
                        AllFolders[p] = np.array([])
                    AllFolders[p] = np.append(AllFolders[p], (1+ (score - weighted_threshold)) * score)


        AllFoldersAverage = {}
        for k in AllFolders:
            histogram, bins = np.histogram(AllFolders[k], bins=512, range=(-1, 1))
            mode_index = np.argmax(histogram)
            AllFoldersAverage[k] = bins[mode_index]

        npSorting = np.array(list(AllFoldersAverage.values()))

        if match == "first":
            indices = np.argsort(npSorting)[::-1]

        if match == "last":
            indices = np.argsort(npSorting)[:: 1]

        folders_sorted = np.column_stack((indices, np.array(npSorting[indices])))

        Output = ""
        filterstring = ""
        filterstring = []
        start_idx = offset_index
        end_idx = start_idx + folder_count
        
        for i in range(start_idx, end_idx):
            if i >= len(AllFoldersAverage):
                break
            idx, avg_score = folders_sorted[i]
            folder_name = list(AllFoldersAverage.keys())[int(idx)]
            Output = Output + folder_name + " -> " + str(avg_score) + "\r\n"
            filterstring.append(folder_name + "*")

        #print("step7 = ", datetime.datetime.now())

        return Output, filterstring