import json
from schema import dataset, trainer, model
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.transforms.v2 import Compose, ToImage, ToDtype
from transformers import AutoTokenizer, AutoImageProcessor
from transformers import TrainingArguments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_size = [256, 256]
image_transforms = Compose([ToImage(), ToDtype(torch.float32, scale=True)])
class_names = ["not-sarcasm", "image-sarcasm", "text-sarcasm", "multi-sarcasm"]

text_tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3")
image_processor = AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224-in21k"
)

train_dataset = dataset.VimmsdDataset(
    data_file="/kaggle/input/vimmsd-uit2024/vimmsd-train.json",
    images_dir="/kaggle/input/vimmsd-uit2024/training-images/train-images",
    class_names=class_names,
    tokenizer=text_tokenizer,
    image_transforms=image_transforms,
    image_processor=image_processor,
    task="train",
)

vimmsd_model = model.VimmsdModel()
vimmsd_model(**train_dataset[0]["features"])

training_args = TrainingArguments(
    output_dir="/kaggle/working/model",
    overwrite_output_dir=True,
    per_device_train_batch_size=64,
    num_train_epochs=15,
    log_level="debug",
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="no",
    report_to="none",
)

model_trainer = trainer.Trainer(
    model=vimmsd_model,
    args=training_args,
    train_dataset=train_dataset,
)

model_trainer.train()
model_trainer.save_model()

infer_dataset = dataset.VimmsdDataset(
    data_file="/kaggle/input/vimmsd-uit2024/vimmsd-public-test.json",
    images_dir="/kaggle/input/vimmsd-uit2024/public-test-images/dev-images",
    class_names=class_names,
    tokenizer=text_tokenizer,
    image_transforms=image_transforms,
    image_processor=image_processor,
    task="infer",
)
infer_dataloader = DataLoader(infer_dataset, batch_size=1)

results = {}
for id, batch in enumerate(infer_dataloader):
    features = batch["features"]
    logits = vimmsd_model(**features)
    predictions = F.softmax(logits, dim=1).argmax(dim=1)
    label = class_names[predictions[0]]
    print(f"id={id} label={label}")
    results[str(id)] = label

results = {
    "results": results,
    "phase": "dev",
}

with open("/kaggle/working/results.json", "w") as f:
    json.dump(results, f, indent=2)
