from train import get_coco_dataset, get_model, train
from evaluate import evaluate
from torch.utils.data import DataLoader
import torch

BATCH = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = "data/coco_subset"
train_data_syn = get_coco_dataset(
    f"{DATASET_PATH}/train2017_synthetic/",
    f"{DATASET_PATH}/annotations/instances_train2017_synthetic.json"
)
num_classes = len(train_data_syn.coco.cats) + 1

train_loader_syn = DataLoader(train_data_syn, batch_size=BATCH, shuffle=True,
                              collate_fn=lambda x: tuple(zip(*x)))

model2 = get_model(num_classes).to(DEVICE)

train(model2, train_loader_syn)

synthetic_scores = evaluate(
    model2,
    f"{DATASET_PATH}/val2017",
    f"{DATASET_PATH}/annotations/instances_val2017.json"
)
