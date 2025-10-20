import orjson
import os
from pathlib import Path
import numpy as np
from nodes import CLIPTextEncode
import comfy.model_management as model_management
import torch
from comfy.clip_vision import Output, clip_preprocess, ClipVisionModel
from PIL import Image, ImageFile, UnidentifiedImageError
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm


def generate_clip_features_json(clip_vision: ClipVisionModel, path_to_images_folder: Path,
                                output_json_path: Path):
    clip_features = []
    errors = []

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    image_path_list = list(path_to_images_folder.glob('**/*.*'))
    imagetypes = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp"}
    
    image_tqdm = tqdm(image_path_list)

    for image_path in image_tqdm:
        if os.path.splitext(str(image_path.name).lower())[1] in imagetypes:
            image_tqdm.set_description("Processing " + str(image_path.name))
            error = ""

            try:
                image = Image.open(image_path)
                image.load()
            except FileNotFoundError:
                error = "Error: File '{image_path}' not found!" + "\r\n"
                image = None
            except UnidentifiedImageError:
                error = "Error: File '{image_path}' seems not to be an image or is corrupted!"
                image = None
            except PermissionError:
                error = "Error: Missing permissions to read '{image_path}'!"
                image = None
            except Exception as e:
                error = "Unknown error occured while opening file '{image_path}': {e}"
                image = None

            if image:
                image_embeds = get_image_clip_embeddings(clip_vision, image)
                relative = str(image_path)[len(str(path_to_images_folder))+1:]
                clip_features.append((relative, image_embeds))
            else:
                print(error)
            if error != "":
                errors = errors + error + "\r\n"

    with open(output_json_path, "wb") as f:  # Binary-Mode!
        f.write(orjson.dumps(clip_features))

    return errors

def get_image_clip_embeddings(clip_vision:ClipVisionModel, image: Image):
    with torch.no_grad():  # Disable gradient calculation during inference
        cv_image =  clip_vision.encode_image(image_to_tensor(image), crop=False)
    return cv_image["image_embeds"].numpy().flatten().tolist()


#def get_image(file_name: str):
#    image = image_to_tensor(Image.open(file_name))
#    return image

def image_to_tensor(image):
    tensor = torch.clamp(pil_to_tensor(image).float() / 255., 0, 1)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.permute(0, 2, 3, 1)
    return tensor

# repair an damaged image
def repairImage(image):
    i1 = Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    i2 = torch.from_numpy(np.array(i1).astype(np.float32) / 255.0).unsqueeze(0)
    return i2

