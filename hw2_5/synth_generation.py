import cv2
import os
import json
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
from PIL import Image
from pycocotools.coco import COCO

device = "cuda"

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to(device)

pipe.enable_xformers_memory_efficient_attention()

SOURCE_DIR = "data/coco_subset/train2017"
SAVE_DIR = "data/coco_subset/train2017_synthetic"
ANNOTATIONS_PATH = "data/coco_subset/annotations/instances_train2017.json"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load COCO annotations
coco = COCO(ANNOTATIONS_PATH)

# Load annotations JSON for updating
with open(ANNOTATIONS_PATH, 'r') as f:
    annotations_data = json.load(f)

num_images = 2000

images = os.listdir(SOURCE_DIR)

# Track synthetic images and annotations
synthetic_images = []
synthetic_annotations = []
max_img_id = max([img['id'] for img in annotations_data['images']]) if annotations_data['images'] else 0
max_ann_id = max([ann['id'] for ann in annotations_data['annotations']]) if annotations_data['annotations'] else 0

for idx in range(num_images):
    img_filename = images[idx % len(images)]
    img_path = os.path.join(SOURCE_DIR, img_filename)
    
    # Get image ID from filename
    img_info = None
    for img_id in coco.getImgIds():
        img_data = coco.loadImgs([img_id])[0]
        if img_data['file_name'] == img_filename:
            img_info = img_data
            break
    
    # Get annotations for this image
    anns = []
    if img_info:
        ann_ids = coco.getAnnIds(imgIds=[img_info['id']])
        anns = coco.loadAnns(ann_ids)
        
        # Get category name from first annotation
        if anns:
            cat_id = anns[0]['category_id']
            cat_info = coco.loadCats([cat_id])[0]
            cat_name = cat_info['name']
        else:
            # Fallback to filename if no annotations
            cat_name = img_filename.split(".")[0]
    else:
        # Fallback if image not found in annotations
        cat_name = img_filename.split(".")[0]
    
    img = cv2.imread(img_path)
    original_height, original_width = img.shape[:2]
    img = cv2.Canny(img, 100, 200)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = Image.fromarray(img)
    print(cat_name)
    result = pipe(
        prompt=f"a photo of a {cat_name}, high quality, realistic ",
        image=img,
        num_inference_steps=30
    ).images[0]

    synthetic_filename = f"synthetic_{idx}.png"
    result.save(f"{SAVE_DIR}/{synthetic_filename}")
    
    # Get synthetic image dimensions
    synthetic_width, synthetic_height = result.size
    
    # Create synthetic image entry
    synthetic_img_id = max_img_id + idx + 1
    synthetic_image_entry = {
        "id": synthetic_img_id,
        "file_name": synthetic_filename,
        "height": synthetic_height,
        "width": synthetic_width,
        "license": 0,
        "coco_url": "",
        "flickr_url": "",
        "date_captured": ""
    }
    synthetic_images.append(synthetic_image_entry)
    
    # Copy annotations from original image
    if img_info and anns:
        for ann in anns:
            max_ann_id += 1
            # Create synthetic annotation
            # For synthetic images, we'll use full image as bbox or copy original bbox
            # Using full image bbox for simplicity
            synthetic_ann = {
                "id": max_ann_id,
                "image_id": synthetic_img_id,
                "category_id": ann['category_id'],
                "bbox": [0, 0, synthetic_width, synthetic_height],  # Full image bbox
                "area": synthetic_width * synthetic_height,
                "iscrowd": 0,
                "segmentation": []  # No segmentation for synthetic images
            }
            synthetic_annotations.append(synthetic_ann)

# Update annotations with synthetic data
annotations_data['images'].extend(synthetic_images)
annotations_data['annotations'].extend(synthetic_annotations)

# Save updated annotations
SYNTHETIC_ANNOTATIONS_PATH = "data/coco_subset/annotations/instances_train2017_synthetic.json"
with open(SYNTHETIC_ANNOTATIONS_PATH, 'w') as f:
    json.dump(annotations_data, f)

print(f"Synthetic images generated! Added {len(synthetic_images)} images and {len(synthetic_annotations)} annotations.")
print(f"Updated annotations saved to {SYNTHETIC_ANNOTATIONS_PATH}")
