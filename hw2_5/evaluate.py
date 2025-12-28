from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import numpy as np
from torchvision import transforms
import torch
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(model, img_folder, ann_file):
    coco_gt = COCO(ann_file)
    model.eval()
    results = []

    for img_id in coco_gt.imgs.keys():
        img_info = coco_gt.loadImgs(img_id)[0]
        path = f"{img_folder}/{img_info['file_name']}"

        img = transforms.ToTensor()(Image.open(path).convert("RGB")).to(DEVICE)
        with torch.no_grad():
            out = model([img])[0]

        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            x1,y1,x2,y2 = box
            w = x2-x1
            h = y2-y1

            results.append({
                "image_id": img_id,
                "category_id": int(label),
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(score)
            })

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats
