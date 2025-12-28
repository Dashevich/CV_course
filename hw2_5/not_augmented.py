import torch
from torchvision.datasets import CocoDetection
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from train import get_coco_dataset, get_model, train
from evaluate import evaluate

transform = transforms.Compose([
    transforms.ToTensor()
])

BATCH = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = "data/coco_subset"

train_data = get_coco_dataset(
    f"{DATASET_PATH}/train2017/",
    f"{DATASET_PATH}/annotations/instances_train2017.json"
)

train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True,
                          collate_fn=lambda x: tuple(zip(*x)))

num_classes = len(train_data.coco.cats) + 1
model = get_model(num_classes).to(DEVICE)

#train(model, train_loader)

baseline_scores = evaluate(
    model,
    f"{DATASET_PATH}/val2017",
    f"{DATASET_PATH}/annotations/instances_val2017.json"
)
print(baseline_scores)