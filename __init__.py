from .clipvision_embeds import GetImageEmbeds
from .clipvision_embeds import CalcEmbeds
from .clipvision_embeds import ScaleEmbeds
from .clipvision_embeds import CompareEmbeds

from .image_search import ImageSearcher
from .image_search import ResultBrowser
from .image_search import ResultBrowserAdvanced
from .image_search import ResultCombiner

from .clipvision_db import LoadDB
from .clipvision_db import GenerateDB

import folder_paths
import os

folder_paths.folder_names_and_paths["EmbDBs"] = ([os.path.join(folder_paths.models_dir, "EmbDBs")], [".json"])

NODE_CLASS_MAPPINGS = {
    "GetImageEmbeds": GetImageEmbeds,
    "ScaleEmbeds": ScaleEmbeds,
    "CalcEmbeds": CalcEmbeds,
    "CompareEmbeds": CompareEmbeds,
    "ImageSearcher": ImageSearcher,
    "RBrowser": ResultBrowser,
    "AdvRBrowser": ResultBrowserAdvanced,
    "RCombiner": ResultCombiner,
    "LoadDB": LoadDB,
    "GenerateDB": GenerateDB,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "GetImageEmbeds": "Get Image Embeddings",
    "ScaleEmbeds": "Scale Embeddings (experimental)",
    "CalcEmbeds": "Calculate Embeddings (experimental)",
    "CompareEmbeds": "Compare Embeddings",
    "ImageSearcher": "Image Searcher",
    "RBrowser": "Result Browser",
    "AdvRBrowser": "Advanced Result Browser",
    "RCombiner": "Combine Results",
    "DBLoader": "Embeddings DB Loader",
    "GenerateDB": "Generate Image Database"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
