import os
import json
import shutil
from tqdm import tqdm
from pycocotools.coco import COCO

# ======================
# CONFIG
# ======================
COCO_ROOT = "/home/dasha/hw2/data/coco"  # где лежит оригинальный COCO
OUT_ROOT = "/home/dasha/hw2/coco_subset"

SPLITS = ["train2017", "val2017"]

SELECTED_CLASSES = [
    "dog", "cat", "bird", "horse", "sheep",
    "cow", "elephant", "bear", "airplane", "train"
]

# ======================
# UTILS
# ======================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def filter_coco_split(split):
    print(f"\nProcessing {split}")

    ann_path = f"{COCO_ROOT}/annotations/instances_{split}.json"
    img_dir = f"{COCO_ROOT}/images/{split}"

    coco = COCO(ann_path)

    # category ids
    cat_ids = coco.getCatIds(catNms=SELECTED_CLASSES)
    cats = coco.loadCats(cat_ids)

    # image ids containing selected classes
    img_ids = set()
    for cid in cat_ids:
        img_ids.update(coco.getImgIds(catIds=[cid]))

    img_ids = list(img_ids)
    print(f"Images selected: {len(img_ids)}")

    images = coco.loadImgs(img_ids)

    # annotations
    anns = []
    ann_id = 1
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=cat_ids, iscrowd=None)
        img_anns = coco.loadAnns(ann_ids)
        for ann in img_anns:
            ann["id"] = ann_id
            ann_id += 1
            anns.append(ann)

    # re-map category ids (important for DETR)
    new_categories = []
    cat_id_map = {}
    for i, cat in enumerate(cats):
        new_id = i + 1
        cat_id_map[cat["id"]] = new_id
        new_categories.append({
            "id": new_id,
            "name": cat["name"],
            "supercategory": cat["supercategory"]
        })

    for ann in anns:
        ann["category_id"] = cat_id_map[ann["category_id"]]

    # ======================
    # SAVE
    # ======================
    out_img_dir = f"{OUT_ROOT}/{split}"
    out_ann_dir = f"{OUT_ROOT}/annotations"

    ensure_dir(out_img_dir)
    ensure_dir(out_ann_dir)

    for img in tqdm(images, desc="Copy images"):
        src = f"{img_dir}/{img['file_name']}"
        dst = f"{out_img_dir}/{img['file_name']}"
        shutil.copyfile(src, dst)

    out_json = {
        "images": images,
        "annotations": anns,
        "categories": new_categories
    }

    with open(f"{out_ann_dir}/instances_{split}.json", "w") as f:
        json.dump(out_json, f)

    print(f"Saved {split} subset")


# ======================
# MAIN
# ======================
if __name__ == "__main__":
    for split in SPLITS:
        filter_coco_split(split)

    print("\n✅ COCO subset prepared for DETR")
