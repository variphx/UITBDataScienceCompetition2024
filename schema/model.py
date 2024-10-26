import torch as _torch
from torch import nn as _nn
from torch.nn import functional as _F
from transformers import (
    AutoModel as _AutoModel,
    BitsAndBytesConfig as _BitsAndBytesConfig,
)
from peft import (
    LoraConfig as _LoraConfig,
    prepare_model_for_kbit_training as _prepare_model_for_kbit_training,
    get_peft_model as _get_peft_model,
)


class TextModel(_nn.Module):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        quantization_config = _BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=_torch.bfloat16
        )
        model_4bit = _AutoModel.from_pretrained(
            "uitnlp/CafeBERT",
            device_map=device,
            quantization_config=quantization_config,
        )
        model_4bit = _prepare_model_for_kbit_training(model_4bit)

        lora_config = _LoraConfig(
            r=16,
            lora_alpha=8,
            lora_dropout=0.05,
            init_lora_weights="olora",
            target_modules="all-linear",
        )

        self._device = device
        self._model = _get_peft_model(model_4bit, lora_config)

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return _torch.sum(token_embeddings * input_mask_expanded, 1) / _torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, encoding):
        encoding = encoding.to(self._device)
        with _torch.no_grad():
            outputs = self._model(**encoding)
        embeddings = self.mean_pooling(outputs, encoding["attention_mask"])
        embeddings = _F.normalize(embeddings, p=2, dim=1)
        return embeddings


class ImageModel(_nn.Module):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)

        quantization_config = _BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=_torch.bfloat16
        )
        model_4bit = _AutoModel.from_pretrained(
            "facebook/deit-base-distilled-patch16-224",
            device_map=device,
            quantization_config=quantization_config,
        )
        model_4bit = _prepare_model_for_kbit_training(model_4bit)

        lora_config = _LoraConfig(
            r=16,
            lora_alpha=8,
            lora_dropout=0.05,
            init_lora_weights="olora",
            target_modules="all-linear",
        )

        self._device = device
        self._model = _get_peft_model(model_4bit, lora_config)

    def forward(self, encoding):
        encoding = encoding.to(self._device)
        with _torch.no_grad():
            outputs = self._model(**encoding)
        return outputs.last_hidden_state[:, 0, :]


class VimmsdModel(_nn.Module):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._device = device
        self._image_model = ImageModel(device=device)
        self._text_model = TextModel(device=device)
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
        )

    def forward(self, image, text):
        image_outputs = self._image_model(image)
        image_outputs = image_outputs.reshape([image_outputs.shape[0], -1])
        image_outputs = _torch.as_tensor(image_outputs, device=self._device)
        image_outputs = _F.tanh(image_outputs)

        text_outputs = self._text_model(text)
        text_outputs = text_outputs.reshape([text_outputs.shape[0], -1])
        text_outputs = _torch.as_tensor(text_outputs, device=self._device)
        text_outputs = _F.tanh(text_outputs)

        combined = _torch.cat([image_outputs, text_outputs], dim=1)
        logits = self._fc(combined)
        return logits
