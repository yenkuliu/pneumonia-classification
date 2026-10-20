"""Shared model construction, fine-tuning stages, and image preprocessing."""

from PIL import ImageOps
import torch
from torchvision import models, transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class Contrast:
    def __call__(self, image):
        return ImageOps.autocontrast(ImageOps.grayscale(image), cutoff=1).convert("RGB")


def network(name, classes=2, dropout=0.2, pretrained=True):
    builders = {
        "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
        "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
        "densenet121": (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1),
    }
    if name not in builders:
        raise ValueError(f"Unsupported model: {name}")
    builder, weights = builders[name]
    model = builder(weights=weights if pretrained else None)
    if name.startswith("resnet"):
        features = model.fc.in_features
    elif name == "densenet121":
        features = model.classifier.in_features
    else:
        features = model.classifier[-1].in_features
    classifier = torch.nn.Sequential(torch.nn.Dropout(dropout), torch.nn.Linear(features, classes))
    torch.nn.init.xavier_uniform_(classifier[-1].weight)
    torch.nn.init.zeros_(classifier[-1].bias)
    if name.startswith("resnet"):
        model.fc = classifier
    else:
        model.classifier = classifier
    return model


def head(model):
    return model.fc if hasattr(model, "fc") else model.classifier


def parameter_groups(model, config):
    head_ids = {id(p) for p in head(model).parameters()}
    return [dict(params=[p for p in model.parameters() if id(p) not in head_ids], lr=config["learning_rate"]),
            dict(params=list(head(model).parameters()), lr=config["head_learning_rate"])]


def stage(model, name, epoch, warmup_epochs, tuning_scope):
    if tuning_scope not in {"all", "last", "head"}:
        raise ValueError(tuning_scope)
    for p in model.parameters():
        p.requires_grad_(False)
    if epoch > warmup_epochs and tuning_scope != "head":
        if tuning_scope == "all":
            modules = [model]
        elif name.startswith("resnet"):
            modules = [model.layer4]
        elif name == "densenet121":
            modules = [model.features.denseblock4, model.features.norm5]
        else:
            modules = [model.features[-2:]]
        for module in modules:
            for p in module.parameters():
                p.requires_grad_(True)
    for p in head(model).parameters():
        p.requires_grad_(True)


def image_transforms(config):
    size, resize = config["image_size"], config["resize_size"]
    prefix = [Contrast()] if config.get("contrast") == "autocontrast" else []
    policy = config["preprocessing"]
    if policy == "full_frame":
        training = [transforms.Resize((size, size))]
        evaluation = [transforms.Resize((size, size))]
    elif policy == "crop":
        training = [transforms.Resize(resize), transforms.RandomResizedCrop(size, scale=(.86, 1.0))]
        evaluation = [transforms.Resize(resize), transforms.CenterCrop(size)]
    else:
        raise ValueError(policy)
    aug = config["augmentation"]
    training += [transforms.RandomHorizontalFlip(aug["horizontal_flip_p"]),
                 transforms.RandomRotation(aug["rotation_degrees"]),
                 transforms.ColorJitter(brightness=aug["color_jitter_brightness"],
                                        contrast=aug["color_jitter_contrast"])]
    tail = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(prefix + training + tail), transforms.Compose(prefix + evaluation + tail)
