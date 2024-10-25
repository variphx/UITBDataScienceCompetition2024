import json as _json
import pathlib as _pathlib
from PIL import Image
from torch.utils.data import Dataset as _Dataset
from torchvision.transforms.v2 import Compose as _Compose
from transformers import (
    PreTrainedTokenizer as _PretrainedTokenizer,
    BaseImageProcessor as _BaseImageProcessor,
)


class VimmsdDataset(_Dataset):
    def __init__(
        self,
        data_file: str,
        images_dir: str,
        class_names: list[str],
        tokenizer: _PretrainedTokenizer,
        image_transforms: _Compose,
        image_processor: _BaseImageProcessor,
        task: str,
    ):
        super().__init__()
        assert task == "train" or task == "infer"

        self._task = task
        self._class_names = class_names
        self._label2id = {label: id for id, label in enumerate(self._class_names)}

        with open(data_file, "r") as f:
            self._data: dict[str, dict[str, str]] = _json.load(f)
        self._images_dir = _pathlib.PurePath(images_dir)
        self._tokenizer = tokenizer
        self._image_transforms = image_transforms
        self._image_processor = image_processor

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        item = self._data[str(index)]
        features = {
            "image": self._image_processor(
                self._image_transforms(
                    Image.open(self._images_dir.joinpath(item["image"])).convert("RGB")
                ),
                return_tensors="pt",
            ),
            "text": self._tokenizer(item["caption"]),
        }
        if self._task == "train":
            target = self._label2id[item["label"]]
            return {"features": features, "target": target}
        else:
            return {"features": features}
