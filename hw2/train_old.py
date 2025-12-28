import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    DetrForObjectDetection,
    DetrImageProcessor
)

from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

# ================= CONFIG =================
DATA_ROOT = "/home/dasha/hw/hw2/data/coco_subset"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE =", DEVICE)

NUM_CLASSES = 10
BATCH_SIZE = 4          # можно попробовать 4 или 8
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 20
NUM_WORKERS = 4

LOG_DIR = "runs/detr_hf_fast"
CKPT_DIR = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
# ==========================================


# -------- COCO class remapping --------
COCO_ID_TO_CONTIG = {
    15: 0, 16: 1, 17: 2, 18: 3, 19: 4,
    20: 5, 21: 6, 22: 7, 23: 8, 24: 9
}
CONTIG_TO_COCO_ID = {v: k for k, v in COCO_ID_TO_CONTIG.items()}


# -------- Dataset --------
class CocoSubset(CocoDetection):
    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)

        orig_w, orig_h = img.size

        filtered_anns = []
        for a in anns:
            if a["category_id"] not in COCO_ID_TO_CONTIG:
                continue

            x, y, w, h = a["bbox"]

            filtered_anns.append({
                "bbox": [x, y, w, h],
                "category_id": COCO_ID_TO_CONTIG[a["category_id"]],
                "area": w * h,
                "iscrowd": 0
            })

        target = {
            "image_id": self.ids[idx],
            "annotations": filtered_anns,
            "orig_size": (orig_h, orig_w)
        }


        img = F.to_tensor(img)
        return img, target


def collate_fn(batch):
    imgs, targets = list(zip(*batch))
    return list(imgs), list(targets)


# -------- Model --------
def build_model():
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )
    return model


# -------- Evaluation --------
@torch.no_grad()
def evaluate(model, processor, dataloader, coco_gt):
    model.eval()
    results = []

    for imgs, targets in tqdm(dataloader, desc="Evaluating"):
        encoding = processor(
            images=imgs,
            return_tensors="pt"
        ).to(DEVICE)

        outputs = model(**encoding)

        logits = outputs.logits.softmax(-1)
        boxes = outputs.pred_boxes

        for i in range(len(imgs)):
            scores, labels = logits[i, :, :-1].max(-1)
            bboxes = boxes[i]

            h, w = targets[i]["orig_size"]

            for s, l, b in zip(scores, labels, bboxes):
                if s < 0.05:
                    continue

                cx, cy, bw, bh = b
                x = (cx - bw / 2) * w
                y = (cy - bh / 2) * h
                bw = bw * w
                bh = bh * h

                results.append({
                    "image_id": targets[i]["image_id"],
                    "category_id": CONTIG_TO_COCO_ID[l.item()],
                    "bbox": [x.item(), y.item(), bw.item(), bh.item()],
                    "score": s.item()
                })

    coco_dt = coco_gt.loadRes(results)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    return evaluator.stats[0], evaluator.stats[1]


# -------- Training --------
def main():
    writer = SummaryWriter(LOG_DIR)

    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50",
        size=800,
        max_size=800
    )

    train_set = CocoSubset(
        f"{DATA_ROOT}/images/train2017",
        f"{DATA_ROOT}/annotations/instances_train_subset.json"
    )
    val_set = CocoSubset(
        f"{DATA_ROOT}/images/val2017",
        f"{DATA_ROOT}/annotations/instances_val_subset.json"
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True
    )

    model = build_model().to(DEVICE)

    # ---------- Freeze backbone ----------
    for p in model.model.backbone.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scaler = torch.cuda.amp.GradScaler()

    coco_gt = val_set.coco
    best_map = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch}"):
            encoding = processor(
                images=imgs,
                annotations=targets,
                return_tensors="pt"
            )

            encoding["pixel_values"] = encoding["pixel_values"].to(DEVICE)
            encoding["pixel_mask"] = encoding["pixel_mask"].to(DEVICE)

            for t in encoding["labels"]:
                for k in t:
                    t[k] = t[k].to(DEVICE)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(**encoding)
                loss = outputs.loss
                loss_dict = outputs.loss_dict

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        writer.add_scalar("train/loss_total", avg_loss, epoch)
        writer.add_scalar("train/loss_ce", loss_dict["loss_ce"], epoch)
        writer.add_scalar("train/loss_bbox", loss_dict["loss_bbox"], epoch)
        writer.add_scalar("train/loss_giou", loss_dict["loss_giou"], epoch)

        print(f"Epoch {epoch}: loss={avg_loss:.4f}")

        # ----- Eval only every 5 epochs -----
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            mAP, mAP50 = evaluate(model, processor, val_loader, coco_gt)
            writer.add_scalar("val/mAP", mAP, epoch)
            writer.add_scalar("val/mAP50", mAP50, epoch)

            if mAP > best_map:
                best_map = mAP
                torch.save(model.state_dict(), f"{CKPT_DIR}/best_map.pth")

        torch.save(model.state_dict(), f"{CKPT_DIR}/last.pth")

    writer.close()


if __name__ == "__main__":
    main()
