import os
import json
from pathlib import Path
import folder_paths
import orjson

from .utils import (generate_clip_features_json)

class LoadDB:
    MY_LOADED_DB: list = None
    LOADED_DB: list = None
    LOADED_DB_Name = ""
    images_folder = ""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {           
            "required": { 
                "db_name": (folder_paths.get_filename_list("EmbDBs"),) ,
                "path_to_images_folder": ("STRING", {
                    "multiline": False,
                    "default": "/path/to/folder/with/images"
                }), 
            },
                "optional": {
                "img_db": ("LoadDB",),
            }                            
        }

    RETURN_TYPES = ("LoadDB", )
    RETURN_NAMES = ("IMG_DB", )
    FUNCTION = "load_DB"
    CATEGORY = "ClipVisionTools"

    def load_DB(self, db_name, path_to_images_folder, img_db=None):
        db_path = folder_paths.get_full_path_or_raise("EmbDBs", db_name)
        if self.LOADED_DB_Name != db_path or path_to_images_folder != self.images_folder:
            with open(db_path, "rb") as f:
                tmp = orjson.loads(f.read())

            self.MY_LOADED_DB = [
                (os.path.join(path_to_images_folder, filename), vector)
                for filename, vector in tmp
            ]
            self.LOADED_DB_Name = db_path
            self.images_folder = path_to_images_folder

        if img_db is not None:
            self.LOADED_DB = self.MY_LOADED_DB + img_db.LOADED_DB
        else:
            self.LOADED_DB = self.MY_LOADED_DB
        return self, 

class GenerateDB:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "clip_vision": ("CLIP_VISION",),
                "path_to_images_folder": ("STRING", {
                    "multiline": False,
                    "default": "path/to/folder/with/images"
                }),
                "new_db_name": ("STRING", {
                    "multiline": False,
                    "default": "new_img_db.json"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ERRORS",)
    OUTPUT_NODE = True

    FUNCTION = "StartGenDB"
    CATEGORY = "ClipVisionTools"
    def StartGenDB(self, clip_vision, path_to_images_folder, new_db_name):
        path_to_images = Path(path_to_images_folder)
        path_to_database = Path(folder_paths.get_folder_paths("EmbDBs")[0] + "\\" + new_db_name)
        if path_to_database.parent.exists() == False:
            path_to_database.parent.mkdir()
        errors = ""
        if not path_to_database.exists():
            errors = generate_clip_features_json(clip_vision, path_to_images, path_to_database)

        return errors,
