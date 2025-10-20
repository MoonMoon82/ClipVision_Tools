from .clipvision_embeds import EmbedsInfo
from .clipvision_embeds import Cond2Embeds
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
    "EmbedsInfo": EmbedsInfo,
    "Cond2Embeds": Cond2Embeds,
    "ScaleEmbeds": ScaleEmbeds,
    "CalcEmbeds": CalcEmbeds,
    "CompareEmbeds": CompareEmbeds,
    "ImageSearcher": ImageSearcher,
    "ResultBrowser": ResultBrowser,
    "ResultBrowserAdvanced": ResultBrowserAdvanced,
    "ResultCombiner": ResultCombiner,
    "LoadDB": LoadDB,
    "GenerateDB": GenerateDB,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "EmbedsInfo": "Embeddings Shape Info (experimental)",
    "Cond2Embeds": "Condition 2 Embeddings (experimental)",
    "ScaleEmbeds": "Scale Embeddings (experimental)",
    "CalcEmbeds": "Calculate Embeddings (experimental)",
    "CompareEmbeds": "Compare Embeddings",
    "ImageSearcher": "Image Searcher",
    "ResultBrowser": "Result Browser",
    "ResultBrowserAdvanced": "Advanced Result Browser",
    "ResultCombiner": "Combine Results",
    "DBLoader": "Embeddings DB Loader",
    "GenerateDB": "Generate Image Database"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
