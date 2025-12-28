import os
from pycocotools.coco import COCO
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.ops import box_iou
from torch.profiler import profile, record_function, ProfilerActivity


# ================= CONFIG =================
DATA_DIR = "/home/dasha/hw/hw2/data/coco_subset"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 10
EPOCHS = 20
BATCH_SIZE = 8
NUM_WORKERS = 2              
LR = 1e-4
BACKBONE_LR = 1e-5
WEIGHT_DECAY = 1e-4

CHECKPOINT_DIR = "ckpts_clean"
LOG_DIR = "logs_clean"
PROFILE_DIR = "profile_trace"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PROFILE_DIR, exist_ok=True)
# ==========================================


# -------- Dataset --------
class CocoSubsetDataset(Dataset):
    def __init__(self, root, annFile, processor, target_cat_ids, original_id_to_new_id):
        self.root = root
        self.coco = COCO(annFile)
        self.processor = processor
        self.target_cat_ids = target_cat_ids
        self.map = original_id_to_new_id

        self.img_ids = []
        for cid in target_cat_ids:
            self.img_ids.extend(self.coco.getImgIds(catIds=[cid]))
        self.img_ids = list(set(self.img_ids))

        print(f"Loaded {len(self.img_ids)} images")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.root, img_info["file_name"])

        image = Image.open(path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        anns = [a for a in anns if a["category_id"] in self.target_cat_ids]

        target = {
            "image_id": img_id,
            "annotations": [
                {
                    "bbox": a["bbox"],
                    "category_id": a["category_id"],
                    "area": a["bbox"][2] * a["bbox"][3],
                    "iscrowd": a.get("iscrowd", 0),
                }
                for a in anns
            ],
        }

        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"][0]

        if len(labels["class_labels"]) > 0:
            labels["class_labels"] = torch.tensor(
                [self.map[int(x)] for x in labels["class_labels"]], dtype=torch.long
            )

        return pixel_values, labels


def collate_fn(batch):
    pixel_values = [b[0] for b in batch]
    labels = [b[1] for b in batch]

    max_h = max(im.shape[1] for im in pixel_values)
    max_w = max(im.shape[2] for im in pixel_values)

    padded = []
    for img in pixel_values:
        c, h, w = img.shape
        t = torch.zeros((c, max_h, max_w))
        t[:, :h, :w] = img
        padded.append(t)

    return {
        "pixel_values": torch.stack(padded),
        "labels": labels,
    }


# -------- Load COCO and classes --------
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

ann_train = os.path.join(DATA_DIR, "annotations/instances_train_subset.json")
ann_val = os.path.join(DATA_DIR, "annotations/instances_val_subset.json")

coco_train = COCO(ann_train)

target_cats = ["bench", "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra"]
target_cat_ids = coco_train.getCatIds(catNms=target_cats)
cats_info = coco_train.loadCats(target_cat_ids)

id2label = {i: c["name"] for i, c in enumerate(cats_info)}
label2id = {v: k for k, v in id2label.items()}
original_to_new = {cat_id: i for i, cat_id in enumerate(target_cat_ids)}

print("Class mapping:", original_to_new)


# -------- Model --------
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=NUM_CLASSES,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
).to(DEVICE)


# -------- Data --------
train_dataset = CocoSubsetDataset(
    os.path.join(DATA_DIR, "images/train2017"),
    ann_train,
    processor,
    target_cat_ids,
    original_to_new
)

val_dataset = CocoSubsetDataset(
    os.path.join(DATA_DIR, "images/val2017"),
    ann_val,
    processor,
    target_cat_ids,
    original_to_new
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=False
)


# -------- Optimizer --------
backbone_params, other_params = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    (backbone_params if "backbone" in name else other_params).append(p)

optimizer = AdamW(
    [
        {"params": other_params, "lr": LR},
        {"params": backbone_params, "lr": BACKBONE_LR},
    ],
    weight_decay=WEIGHT_DECAY,
)

scaler = GradScaler()
writer = SummaryWriter(LOG_DIR)


# -------- Error analysis helpers --------
def error_analysis(pred_logits, pred_boxes, gt_labels, gt_boxes, iou_thr=0.5):
    cls_errors = 0
    loc_errors = 0
    if len(gt_boxes) == 0:
        return 0, 0

    probs = pred_logits.softmax(-1)[..., :-1]
    scores, pred_cls = probs.max(-1)
    ious = box_iou(pred_boxes, gt_boxes)

    for i in range(len(pred_boxes)):
        max_iou, j = (ious[i].max().item(), ious[i].argmax().item())
        if max_iou < iou_thr:
            loc_errors += 1
        else:
            if pred_cls[i].item() != gt_labels[j].item():
                cls_errors += 1

    return cls_errors, loc_errors


def evaluate(epoch):
    model.eval()
    total_loss = 0
    cls_err_total = 0
    loc_err_total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            imgs = batch["pixel_values"].to(DEVICE)
            labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]

            out = model(pixel_values=imgs, labels=labels)
            total_loss += out.loss.item()

            outputs = model(pixel_values=imgs)
            pred_logits = outputs.logits.cpu()
            pred_boxes = outputs.pred_boxes.cpu()

            # ограничим до 20 самых вероятных боксов
            probs = pred_logits.softmax(-1)[..., :-1]
            scores, _ = probs.max(-1)
            topk = min(20, pred_boxes.shape[1])

            for i, l in enumerate(labels):
                top_idx = scores[i].topk(topk).indices
                pl = pred_logits[i][top_idx]
                pb = pred_boxes[i][top_idx]

                gt_boxes = l["boxes"].cpu()
                gt_labels = l["class_labels"].cpu()

                c, b = error_analysis(pl, pb, gt_labels, gt_boxes)
                cls_err_total += c
                loc_err_total += b

    total_loss /= len(val_loader)
    print(f"Val loss: {total_loss:.4f}")

    writer.add_scalar("Loss/Val", total_loss, epoch)
    writer.add_scalar("Error/Classification", cls_err_total, epoch)
    writer.add_scalar("Error/Localization", loc_err_total, epoch)

    model.train()


# -------- TRAIN LOOP ----------
train_loss_history = []
cls_loss_history = []
bbox_loss_history = []

print("START TRAINING")

use_profiler = True
prof = None

if use_profiler:
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
        record_shapes=False,
        with_stack=False,
        profile_memory=False
    )
    prof.__enter__()

for epoch in range(EPOCHS):
    model.train()
    total = 0
    comp = {}

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        imgs = batch["pixel_values"].to(DEVICE)
        labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            with record_function("forward"):
                out = model(pixel_values=imgs, labels=labels)

            loss = out.loss
            loss_dict = out.loss_dict

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        scaler.step(optimizer)
        scaler.update()

        total += loss.item()
        for k, v in loss_dict.items():
            comp[k] = comp.get(k, 0) + v.item()

        if prof:
            prof.step()

    total /= len(train_loader)
    for k in comp:
        comp[k] /= len(train_loader)

    print(f"Epoch {epoch+1}: loss={total:.4f}")

    writer.add_scalar("Loss/Train", total, epoch)
    for k, v in comp.items():
        writer.add_scalar(f"Loss/{k}", v, epoch)

    train_loss_history.append(total)
    cls_loss_history.append(comp.get("loss_ce", 0))
    bbox_loss_history.append(comp.get("loss_bbox", 0))

    evaluate(epoch)

    torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/epoch_{epoch+1}.pth")
    print("Checkpoint saved")

if prof:
    prof.export_chrome_trace(os.path.join(PROFILE_DIR, "trace.json"))
    prof.__exit__(None, None, None)
    print("Profiler trace saved")


# -------- PLOT LOSS CURVES ----------
plt.figure()
plt.plot(train_loss_history, label="total")
plt.plot(cls_loss_history, label="classification")
plt.plot(bbox_loss_history, label="bbox regression")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("loss_plots.png")
plt.close()

print("Loss plots saved")
