from schema.model import VimmsdModel
from schema.dataset import VimmsdDataset
from schema.trainer import infer_collate_fn
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str)
arguments = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ["not-sarcasm", "image-sarcasm", "text-sarcasm", "multi-sarcasm"]
model = VimmsdModel(device=device).to(device=device)
model.load_state_dict(
    torch.load(arguments.model_path, map_location=device, weights_only=True)
)
model.eval()

infer_dataset = VimmsdDataset(
    data_file="/kaggle/input/vimmsd-uit2024/vimmsd-public-test.json",
    images_dir="/kaggle/input/vimmsd-uit2024/public-test-images/dev-images",
    class_names=class_names,
    task="infer",
)
infer_dataloader = DataLoader(infer_dataset, batch_size=1, collate_fn=infer_collate_fn)

results = {}
for id, batch in (progress_bar := tqdm(enumerate(infer_dataloader), desc="infer")):
    features = batch["features"]
    logits = model(**features)
    predictions = F.softmax(logits, dim=1).argmax(dim=1)
    label = class_names[predictions[0]]
    progress_bar.set_postfix({"id": f"{id}", "label": f"{label}"})
    results[str(id)] = label

results = {
    "results": results,
    "phase": "dev",
}

with open("/kaggle/working/results.json", "w") as f:
    json.dump(results, f, indent=2)
