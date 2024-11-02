import json
import lightning as L
import numpy as np
import torch
import easyocr
from pathlib import Path
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional import f1_score
from lightning.pytorch import callbacks as L_callbacks
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer
from tqdm import tqdm
from PIL import Image

DATA_DIR = "/kaggle/input/vimmsd-uit2024"
CLASS_NAMES = ["not-sarcasm", "image-sarcasm", "text-sarcasm", "multi-sarcasm"]
IMAGE_PROCESSOR = AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224", use_fast=True
)
TEXT_TOKENIZER = AutoTokenizer.from_pretrained("uitnlp/CafeBERT", use_fast=True)
OCR = easyocr.Reader(["vi", "en"])


class DscTrainDataset(Dataset):
    def __init__(self, data_dir: str = DATA_DIR, class_names: list[str] = CLASS_NAMES):
        super().__init__()

        self.data_dir = Path(data_dir)
        images_dir = self.data_dir.joinpath("training-images", "train-images")
        data_file = self.data_dir.joinpath("vimmsd-train.json")

        label2id = dict([(class_name, id) for id, class_name in enumerate(class_names)])
        self.label2id = label2id

        with open(data_file, "r") as f:
            data = json.load(f)

        mapped_data = []

        for key in tqdm(
            data, desc="Mapping image file name to file path", total=len(data)
        ):
            data_i = data[key]

            image_path = images_dir.joinpath(data_i["image"]).as_posix()
            caption = data_i["caption"]
            label = data_i["label"]

            mapped_data.append(
                {"image": image_path, "caption": caption, "label": label2id[label]}
            )

        self.data = mapped_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return item


class DscPredictDataset(Dataset):
    def __init__(self, data_dir: str = DATA_DIR):
        super().__init__()

        self.data_dir = Path(data_dir)
        images_dir = self.data_dir.joinpath("public-test-images", "dev-images")
        data_file = self.data_dir.joinpath("vimmsd-public-test.json")

        with open(data_file, "r") as f:
            data = json.load(f)

        mapped_data = []

        for key in tqdm(
            data, desc="Mapping image file name to file path", total=len(data)
        ):
            data_i = data[key]

            image_path = images_dir.joinpath(data_i["image"]).as_posix()
            caption = data_i["caption"]

            mapped_data.append({"image": image_path, "caption": caption})

        self.data = mapped_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return item


def collate_fn(
    batch: list[dict[str, any]],
    image_processor=IMAGE_PROCESSOR,
    text_tokenizer=TEXT_TOKENIZER,
    ocr=OCR,
):
    images = []
    image_texts = []
    captions = []
    labels = []

    for item in batch:
        image = Image.open(item["image"]).convert("RGB")
        image = np.array(image)
        images.append(image)

        image_text = "\n".join(list(map(lambda result: result[1], ocr.readtext(image))))

        image_texts.append(image_text)
        captions.append(item["caption"])
        labels.append(item.get("label"))

    images = image_processor(images, return_tensors="pt")
    image_texts = text_tokenizer(
        image_texts,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    captions = text_tokenizer(
        captions,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    return (
        {"images": images, "image_texts": image_texts, "captions": captions},
        labels,
    )


class CombinedSarcasmClassifier(L.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        text_encoder = AutoModel.from_pretrained("uitnlp/CafeBERT")

        for param in text_encoder.parameters():
            param.requires_grad = False
        for param in text_encoder.pooler.parameters():
            param.requires_grad = True

        self.text_encoder = text_encoder

        image_encoder = AutoModel.from_pretrained("google/vit-base-patch16-224")

        for param in image_encoder.parameters():
            param.requires_grad = False
        for param in image_encoder.pooler.parameters():
            param.requires_grad = True

        self.image_encoder = image_encoder

        hidden_size = (
            image_encoder.config.hidden_size + text_encoder.config.hidden_size * 2
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 4),
        )

    def setup(self, stage=None):
        self.text_encoder.to(self.device, self.dtype)
        self.image_encoder.to(self.device, self.dtype)

    def forward(self, images, image_texts, captions):
        with torch.no_grad():
            images = self.image_encoder(**images).pooler_output
            image_texts = self.text_encoder(**image_texts).pooler_output
            captions = self.text_encoder(**captions).pooler_output

        embeddings = torch.cat([images, image_texts, captions], dim=1)
        logits = self.fc(embeddings)

        return logits

    def training_step(self, batch, _):
        features, targets = batch
        logits = self.forward(**features)
        targets = torch.tensor(targets, device=self.device)
        loss = F.cross_entropy(logits, targets)
        f1 = f1_score(
            F.softmax(logits, dim=1),
            targets,
            task="multiclass",
            num_classes=len(CLASS_NAMES),
        )

        print(f"train_loss={loss.item():.4f} train_f1={f1.item():.4f}")
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, _):
        features, targets = batch
        targets = torch.tensor(targets, device=self.device)
        logits = self.forward(**features)
        loss = F.cross_entropy(logits, targets)
        f1 = f1_score(
            F.softmax(logits, dim=1),
            targets,
            task="multiclass",
            num_classes=len(CLASS_NAMES),
        )
        print(f"val_loss={loss.item():.4f} val_f1={f1.item():.4f}")
        self.log("val_loss", loss)

    def predict_step(self, batch, _):
        features, _ = batch
        logits = self.forward(**features)
        predictions = F.softmax(logits, dim=1).argmax(dim=1)
        return predictions

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), 5e-5)
        return optimizer


train_ds = DscTrainDataset()
train_ds, val_ds = torch.utils.data.random_split(train_ds, [0.85, 0.15])
train_dataloader = DataLoader(train_ds, collate_fn=collate_fn, batch_size=128)
val_dataloader = DataLoader(val_ds, collate_fn=collate_fn, batch_size=128)

model = CombinedSarcasmClassifier()

callbacks = [L_callbacks.EarlyStopping(monitor="val_loss", mode="min", min_delta=5e-3)]
trainer = L.Trainer(
    accelerator="gpu",
    devices=2,
    strategy="ddp_find_unused_parameters_true",
    max_epochs=5,
    callbacks=callbacks,
    log_every_n_steps=1,
)
trainer.fit(model, train_dataloader, val_dataloader)

predict_ds = DscPredictDataset()
predict_dataloader = DataLoader(predict_ds, collate_fn=collate_fn, batch_size=128)
predictions = trainer.predict(dataloaders=predict_dataloader, ckpt_path="best")

results = {}
for step, batch in (prog_bar := tqdm(enumerate(predictions), desc="Predicting")):
    for index, prediction in enumerate(batch):
        index = str((step * batch.shape[0]) + index)
        prediction = CLASS_NAMES[prediction]
        results.update({index: prediction})
        prog_bar.set_postfix({"label": prediction})
        prog_bar.refresh()
results = {"results": results, "phase": "dev"}

with open("/kaggle/working/results.json", "w") as f:
    json.dump(results, f, indent=2)
