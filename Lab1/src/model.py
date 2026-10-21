
import torch


class ConvNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers = []
        in_channels = 3
        for width in (64, 128, 256, 512):
            for _ in range(2):
                layers.extend([
                    torch.nn.Conv2d(in_channels, width, 3, padding=1),
                    torch.nn.BatchNorm2d(width),
                    torch.nn.ReLU(inplace=True),
                ])
                in_channels = width
            if width != 512:
                layers.append(torch.nn.MaxPool2d(2))
        self.features = torch.nn.Sequential(*layers)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.classifier = torch.nn.Linear(512, 100)
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                torch.nn.init.zeros_(layer.bias)
        return
        raise NotImplementedError()

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))
        raise NotImplementedError()


# Shared utilities live in this existing file so training and pruning use exactly
# the same architecture, split, preprocessing, and evaluation protocol.
import hashlib
import json
import os
import random
from pathlib import Path


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def make_loaders(data_dir, split_path, batch_size, seed, download=False,
                 create_split=False):
    from torchvision import transforms
    from torchvision.datasets import CIFAR100
    from torch.utils.data import DataLoader, Subset

    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                     (0.2675, 0.2565, 0.2761))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize,
    ])
    eval_transform = transforms.Compose([transforms.ToTensor(), normalize])
    train_data = CIFAR100(data_dir, train=True, transform=train_transform, download=download)
    val_data = CIFAR100(data_dir, train=True, transform=eval_transform, download=False)
    test_data = CIFAR100(data_dir, train=False, transform=eval_transform, download=download)
    fingerprint = hashlib.sha256(bytes(train_data.targets)).hexdigest()
    split_path = Path(split_path)
    if split_path.exists():
        split = torch.load(split_path, map_location="cpu", weights_only=True)
    elif create_split:
        indices = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(seed))
        boundary = int(0.8 * len(train_data))
        split = {"train": indices[:boundary].tolist(), "val": indices[boundary:].tolist(),
                 "seed": seed, "dataset": "CIFAR100", "fingerprint": fingerprint}
        split_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(split, split_path)
    else:
        raise FileNotFoundError(f"Missing saved training split: {split_path}. Run train.py first.")
    joined = split["train"] + split["val"]
    if (split.get("dataset") != "CIFAR100" or split.get("fingerprint") != fingerprint
            or split.get("seed") != seed or sorted(joined) != list(range(len(train_data)))
            or len(split["train"]) != int(0.8 * len(train_data))):
        raise ValueError("Saved split does not match this dataset, seed, or the required 80/20 split.")
    common = dict(batch_size=batch_size, num_workers=0, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(Subset(train_data, split["train"]), shuffle=True,
                              generator=torch.Generator().manual_seed(seed), **common)
    val_loader = DataLoader(Subset(val_data, split["val"]), shuffle=False, **common)
    test_loader = DataLoader(test_data, shuffle=False, **common)
    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss_sum += torch.nn.functional.cross_entropy(logits, targets, reduction="sum").item()
        correct += (logits.argmax(1) == targets).sum().item()
        count += targets.numel()
    if not count:
        raise ValueError("Cannot evaluate an empty dataset.")
    return {"loss": loss_sum / count, "accuracy": correct / count}


def load_model(filename, device="cpu"):
    from torch.nn.utils import prune
    model = ConvNet().to(device)
    state = torch.load(filename, map_location=device, weights_only=True)
    # Masked checkpoints need their reparameterizations restored before loading.
    for name, module in model.named_modules():
        prefix = name + "." if name else ""
        for parameter in ("weight", "bias"):
            if prefix + parameter + "_mask" in state:
                prune.identity(module, parameter)
    model.load_state_dict(state)
    return model


def prunable_parameters(model):
    # Include biases, as required by Exercise 1; BatchNorm statistics are buffers.
    return [(module, name) for module in model.modules()
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d))
            for name in ("weight", "bias") if getattr(module, name, None) is not None]


def remove_pruning(model):
    from torch.nn.utils import prune
    for module, name in prunable_parameters(model):
        if hasattr(module, name + "_orig"):
            prune.remove(module, name)


@torch.no_grad()
def sparsity_stats(model):
    total, zeros = 0, 0
    layers = {}
    parameters = prunable_parameters(model)
    for layer_name, module in model.named_modules():
        layer_total, layer_zeros = 0, 0
        for candidate, name in parameters:
            if candidate is module:
                # Compute from orig * mask: cached weight can be stale after an update.
                value = (getattr(module, name + "_orig") * getattr(module, name + "_mask")
                         if hasattr(module, name + "_mask") else getattr(module, name))
                layer_total += value.numel()
                layer_zeros += (value == 0).sum().item()
        if layer_total:
            layers[layer_name] = layer_zeros / layer_total
            total += layer_total
            zeros += layer_zeros
    return {"sparsity": zeros / total, "zeros": zeros, "parameters": total, "layers": layers}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def environment_info():
    import platform
    import torchvision
    import importlib.metadata
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "packages": {name: importlib.metadata.version(name) for name in
                         ("numpy", "pillow", "torchmetrics", "matplotlib")},
            "source_sha256": {str(path.name): file_sha256(path) for path in
                              (Path(__file__), Path(__file__).parent.parent / "train.py",
                               Path(__file__).parent.parent / "pruning.py")}}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
