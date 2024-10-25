import torch
from torchvision.transforms.v2 import Compose, ToImage, ToDtype
from transformers import AutoTokenizer, AutoImageProcessor

image_transforms = Compose([ToImage(), ToDtype(torch.float32, scale=False)])
text_tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3")
image_processor = AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224-in21k"
)


def train_collate_fn(batch):
    features = {
        "image": image_processor(
            images=[image_transforms(item["features"]["image"]) for item in batch],
            return_tensors="pt",
        ),
        "text": [item["features"]["text"] for item in batch],
    }
    targets = [item["target"] for item in batch]

    return {"features": features, "target": targets}


from transformers import Trainer as _Trainer
from torch.nn import functional as _F
from torch import optim as _optim
from torch.optim import lr_scheduler as _lr_scheduler


class Trainer(_Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        features, targets = inputs["features"], inputs["target"]
        logits = model(**features)
        predictions = _F.softmax(logits, dim=1)
        loss = _F.cross_entropy(predictions, targets)
        return (loss, logits) if return_outputs else loss

    def create_optimizer(self):
        return _optim.AdamW(self.model.parameters())

    def create_scheduler(self, num_training_steps, optimizer=None):
        return _lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=num_training_steps
        )

    def create_optimizer_and_scheduler(self, num_training_steps):
        optimizer = self.create_optimizer()
        scheduler = self.create_scheduler(num_training_steps, optimizer)
        return (optimizer, scheduler)
