import torch as _torch
from torch import nn as _nn
from transformers import AutoModel as _AutoModel
from peft import LoraConfig as _LoraConfig, get_peft_model as _get_peft_model


class TextModel(_nn.Module):
    def __init__(self, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        base_model = _AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v3", trust_remote_code=True
        )
        lora_config = _LoraConfig(
            init_lora_weights="olora", target_modules=["out_proj", "dense"]
        )
        self._device = device
        self._model = _get_peft_model(base_model, lora_config).to(self._device)

    def forward(self, x):
        return self._model.encode(x)


class ImageModel(_nn.Module):
    def __init__(self, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        base_model = _AutoModel.from_pretrained("google/vit-base-patch16-224-in21k")
        lora_config = _LoraConfig(
            init_lora_weights="olora", target_modules=["query", "value"]
        )
        self._device = device
        self._model = _get_peft_model(base_model, lora_config)

    def forward(self, x):
        x = x.to(self._device)
        outputs = self._model(**x)
        return outputs.last_hidden_state[:, 0, :]


class VimmsdModel(_nn.Module):
    def __init__(self, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._device = device
        self._image_model = ImageModel()
        self._text_model = TextModel()
        self._fc = _nn.Sequential(
            _nn.LazyLinear(1024),
            _nn.GELU(),
            _nn.Dropout(0.2),
            _nn.Linear(1024, 4096),
            _nn.GELU(),
            _nn.Dropout(0.2),
            _nn.Linear(4096, 64),
            _nn.Tanh(),
            _nn.Dropout(0.2),
            _nn.Linear(64, 4),
        ).to(self._device)

    def forward(self, image, text):
        image_outputs = self._image_model(image)
        image_outputs = image_outputs.reshape([image_outputs.shape[0], -1])
        image_outputs = _torch.as_tensor(image_outputs, device=self._device)
        print(image_outputs)

        text_outputs = self._text_model(text)
        text_outputs = text_outputs.reshape([text_outputs.shape[0], -1])
        text_outputs = _torch.as_tensor(text_outputs, device=self._device)
        print(text_outputs)

        combined = _torch.cat([image_outputs, text_outputs], dim=1)
        logits = self._fc(combined)
        return logits
