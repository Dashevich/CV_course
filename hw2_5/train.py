import torch
from torchvision.datasets import CocoDetection
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

transform = transforms.Compose([
    transforms.ToTensor()
])

def get_coco_dataset(img_folder, ann_file):
    return CocoDetection(
        root=img_folder,
        annFile=ann_file,
        transform=transform
    )


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="COCO_V1"
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = \
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features,
            num_classes
        )
    return model



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
LR = 1e-4
BATCH = 4

def train(model, dataloader):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        total_loss = 0
        for images, targets in tqdm(dataloader):
            images = [img.to(DEVICE) for img in images]

            new_targets = []
            for t in targets:
                boxes = torch.tensor([obj["bbox"] for obj in t], dtype=torch.float32)
                boxes[:, 2] += boxes[:, 0]
                boxes[:, 3] += boxes[:, 1]

                labels = torch.tensor([obj["category_id"] for obj in t], dtype=torch.int64)

                new_targets.append({
                    "boxes": boxes,
                    "labels": labels
                })

            new_targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in new_targets]

            loss_dict = model(images, new_targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, loss={total_loss/len(dataloader):.4f}")
