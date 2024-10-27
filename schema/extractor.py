import torch
import transformers
import PIL.Image as Image
import torchvision.transforms.v2 as transforms
import sentence_transformers as sbert
import torch.nn as nn


class ImageEmbedder(nn.Module):
    def __init__(
        self,
        model: str = "google/vit-base-patch16-224",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.image_transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.Resize([256, 256]),
                transforms.CenterCrop([224, 224]),
                transforms.Normalize(),
                transforms.ToDtype(torch.float, scale=False),
            ]
        )
        self.processor = transformers.AutoImageProcessor.from_pretrained(model)
        self.model = transformers.AutoModel.from_pretrained(model)

    def forward(self, images: list[Image.Image]):
        inputs = self.image_transforms(images)
        inputs = self.processor(inputs)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]


class TextEmbedder(nn.Module):
    def __init__(
        self,
        model: str = "hiieu/halong_embedding",
        text_generator: str = "5CD-AI/visocial-T5-base",
        query: str = "Is this sarcasm?",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = sbert.SentenceTransformer(model)
        self.generator = transformers.pipeline("text2text-generation", text_generator)
        self.query = query

    def forward(self, texts: list[str]):
        for index, text in enumerate(texts):
            brief = self.generator(
                f"quesstion: Is this text sarcastic: '{text}'? context: this text is a caption to a comment or a post on social media",
            )
            texts[index] += f"\nbrief: {brief}"

        return self.model.encode(
            texts,
            prompt=self.query,
            convert_to_tensor=True,
            device=self.device,
        )
