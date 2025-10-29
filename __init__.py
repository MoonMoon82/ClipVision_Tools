from .clipvision_embeds import EmbedsInfo
from .clipvision_embeds import Cond2Embeds
from .clipvision_embeds import CalcEmbeds
from .clipvision_embeds import ScaleEmbeds
from .clipvision_embeds import CompareEmbeds

#from .testnodes import testnode1

from .image_search import ImageSearcher
from .image_search import ResultBrowser
from .image_search import ResultBrowserAdvanced
from .image_search import ResultCombiner
from .image_search import ResultSubtract
from .image_search import EditResults
from .image_search import FolderScores

from .clipvision_db import LoadDB
from .clipvision_db import GenerateDB
from .clipvision_db import EditDB
import folder_paths
import os

folder_paths.folder_names_and_paths["EmbDBs"] = ([os.path.join(folder_paths.models_dir, "EmbDBs")], [".json"])

NODE_CLASS_MAPPINGS = {
    #"testnode1": testnode1,
    "EmbedsInfo": EmbedsInfo,
    "Cond2Embeds": Cond2Embeds,
    "ScaleEmbeds": ScaleEmbeds,
    "CalcEmbeds": CalcEmbeds,
    "CompareEmbeds": CompareEmbeds,
    "ImageSearcher": ImageSearcher,
    "ResultBrowser": ResultBrowser,
    "ResultBrowserAdvanced": ResultBrowserAdvanced,
    "ResultCombiner": ResultCombiner,
    "ResultSubtract": ResultSubtract,
    "EditResults": EditResults,
    "LoadDB": LoadDB,
    "GenerateDB": GenerateDB,
    "EditDB": EditDB,
    "FolderScores": FolderScores,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {    
    #"testnode1": "Testnode (experimental)",
    "EmbedsInfo": "Embeddings Shape Info (experimental)",
    "ScaleEmbeds": "Scale Embeddings (experimental)",
    "CalcEmbeds": "Calculate Embeddings (experimental)",
    "Cond2Embeds": "Condition 2 Embeddings",
    "CompareEmbeds": "Compare Embeddings",
    "ImageSearcher": "Image Searcher",
    "ResultBrowser": "Result Browser",
    "ResultBrowserAdvanced": "Advanced Result Browser",
    "ResultCombiner": "Combine Results",
    "ResultSubtract": "Subtract Results",
    "EditResults": "Edit Results",
    "DBLoader": "Load Embeddings Database",
    "GenerateDB": "Generate Embeddings Database",
    "EditDB": "Edit Embeddings Database",
    "FolderScores": "Folder Scores",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
