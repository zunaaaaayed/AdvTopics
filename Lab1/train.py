"""
Note: a pre-configured virtual environment is available on the CREMI computer here:
/net/ens/DeepLearning/python/tensorflow2
"""
import torch
import torchvision.transforms
from torchvision.datasets import CIFAR100
from torch.utils.data.dataloader import DataLoader
from torchmetrics.classification import MulticlassAccuracy

import argparse
import csv
import json
import math
import random
import numpy as np
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from model import ConvNet
from model import (file_sha256, environment_info, evaluate, load_model, make_loaders,
                   seed_everything, write_json)

"""
Exercise 0: train a convolution network on Cifar100.
Implement the whole training loop (and the base model) for the image classification task on Cifar100.


TIPS:
1) Fulfil the following TODOs if you have unsure how to do it.
2) Don't use a too complex and too large network as baseline because the pruning procedure will be harder to implement. I advice to limit
    yourself to few layers (conv, bn, relu, maxpool, linear) and to a "small" architecture. Try to aim accuracy al least around 75%.
    Basically, MobileNet-like or ResNet18 architecture should be enough.

[VERY IMPORTANT] save the train-val split so you can use the same split whe re-training the network.
    Otherwise you could have data leakage leading to overestimate your model performances.
    
    
TIPS:
1) Simple data augmentation is enough
2) Save the model and backup the file in case of you override the original model. You will have to retrain all models otherwise...
3) Learning rate scheduling is good to improve performances, but should  be necessary
4) if you're struggling to get good enough accuracy, switch the dataset to Cifar10 (performances should be around 90% easily)
"""



def rng_state(train_loader):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy_state[0],
            numpy_state[1].tolist(), numpy_state[2], numpy_state[3], numpy_state[4]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": train_loader.generator.get_state()}


def restore_rng(state, train_loader):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32),
                         *numpy_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    train_loader.generator.set_state(state["loader"].cpu())


def save_training_checkpoint(path, model, optimizer, scheduler, epoch, best_acc,
                             history, train_loader, best_model_path, args, split_path,
                             recovery):
    # Rename only after a successful write so interruption preserves the previous epoch.
    state = {"epoch": epoch, "model": model.state_dict(),
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "best_acc": best_acc, "history": history, "rng": rng_state(train_loader),
             "best_model": torch.load(best_model_path, map_location="cpu", weights_only=True),
             "args": vars(args), "split_sha256": file_sha256(split_path),
             "environment": environment_info(), "recovery": recovery}
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Exercise 0: reproducible CIFAR100 baseline")
    root = Path(__file__).resolve().parent
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default=str(root / "data"))
    parser.add_argument("--output-dir", default=str(root / "artifacts" / "baseline"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Resume last full checkpoint, or recover legacy best weights")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size and lr must be positive")

    if args.resume and args.evaluate_only:
        parser.error("--resume and --evaluate-only cannot be combined")

    # Hyper-parameters
    n_epoch = 1
    batch_size = 32
    lr = 1e-4
    device = "cuda" if torch.cuda.is_available() else "gpu"
    best_model_params_path = 'best_model_params.pt'

    # Override the provided demonstration settings without changing the skeleton.
    n_epoch, batch_size, lr = args.epochs, args.batch_size, args.lr
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_params_path = str(output_dir / best_model_params_path)
    split_path = output_dir / "train_val_split.pt"
    seed_everything(args.seed)
    last_checkpoint_path = output_dir / "last_checkpoint.pt"
    previous_config_path = output_dir / "config.json"
    previous_config = (json.loads(previous_config_path.read_text())
                       if previous_config_path.exists() else None)
    if args.resume:
        if previous_config is None or not Path(best_model_params_path).exists():
            parser.error("Resume requires the original config.json and best_model_params.pt")
        for key in ("epochs", "batch_size", "lr", "seed"):
            if previous_config[key] != getattr(args, key):
                parser.error(f"Resume requires original --{key.replace('_', '-')}={previous_config[key]}")
        if (previous_config["environment"]["source_sha256"]["model.py"] !=
                file_sha256(Path(__file__).resolve().parent / "src" / "model.py")):
            parser.error("Model source differs from the original training run")
    if Path(best_model_params_path).exists() and not args.evaluate_only:
        backup_dir = output_dir / ("backup_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        backup_dir.mkdir()
        for artifact in output_dir.iterdir():
            if artifact.is_file():
                shutil.copy2(artifact, backup_dir / artifact.name)


    # TODO - Step 1: Define Pre-processing or input data and target labels:
    #   - Use the data augmentation classes from 'torchvision.transforms' package. The minimal augmentation required is 'ToTensor'
    #   - Consider adding normalization (between -1 and +1), and other data augmentation is the performances are a bit low.

    input_transform = None
    target_transform = None

    # TODO - Step 2: Load the data
    #   - Load Cifar100 dataset from torchvision
    #   - Retrieve both train and validation sets. We will use the validation set for our last model evaluation (test set)
    #       and split the training set into two subsets (80%-20%): 1 for training, and a second one for validation.




    train_loader, val_loader, test_loader = make_loaders(
        args.data_dir, split_path, batch_size, args.seed, args.download,
        create_split=not (args.evaluate_only or args.resume),
    )
    input_transform = train_loader.dataset.dataset.transform


    # TODO - Step 3: Implement the model
    #   - In "model.py" file, implement the BasicConvNet class
    model = ConvNet().to(device=device)
    if args.evaluate_only:
        model = load_model(best_model_params_path, device)
        print({"validation": evaluate(model, val_loader, device),
               "test": evaluate(model, test_loader, device)})
        return





    # TODO - Step 4: Training configuration
    #   - Define the loss function (cf "CrossEntropyLoss" class)
    #   - Define the optimiser (SGD with learning rate of 10e-2)
    #   - Define the metrics (Accuracy). You can use the package 'torchmetrics' if available.






    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)
    accuracy = MulticlassAccuracy(num_classes=100, average="micro").to(device)
    history = []
    start_epoch = 1
    resumed_best_acc = None
    recovery = None
    if args.resume:
        if last_checkpoint_path.exists():
            checkpoint = torch.load(last_checkpoint_path, map_location="cpu", weights_only=True)
            if checkpoint["split_sha256"] != file_sha256(split_path):
                raise ValueError("Resume checkpoint and saved train/validation split differ")
            for key in ("epochs", "batch_size", "lr", "seed"):
                if checkpoint["args"][key] != getattr(args, key):
                    raise ValueError(f"Resume checkpoint has different {key}")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            history = checkpoint["history"]
            resumed_best_acc = checkpoint["best_acc"]
            start_epoch = checkpoint["epoch"] + 1
            recovery = checkpoint.get("recovery")
            torch.save(checkpoint["best_model"], best_model_params_path)
            restore_rng(checkpoint["rng"], train_loader)
            print(f"Resuming full checkpoint after epoch {start_epoch - 1}.", flush=True)
        else:
            history_path = output_dir / "history.csv"
            if not history_path.exists():
                raise ValueError("Legacy recovery requires the saved history.csv")
            with history_path.open(newline="") as handle:
                old_history = [{key: (int(value) if key == "epoch" else float(value))
                                for key, value in row.items()} for row in csv.DictReader(handle)]
            if not old_history:
                raise ValueError("Legacy history contains no completed epochs")
            best_row = max(old_history, key=lambda row: row["val_accuracy"])
            model.load_state_dict(torch.load(best_model_params_path, map_location="cpu", weights_only=True))
            measured = evaluate(model, val_loader, device)["accuracy"]
            if abs(measured - best_row["val_accuracy"]) > 1e-8:
                raise ValueError("Saved weights do not match the best validation accuracy in history")
            recovered_epoch = best_row["epoch"]
            start_epoch = recovered_epoch + 1
            resumed_best_acc = best_row["val_accuracy"]
            history = [row for row in old_history if row["epoch"] <= recovered_epoch]
            # The legacy run did not save momentum or RNG. Continue its cosine
            # schedule but explicitly record this as a weights-only recovery.
            next_lr = lr * (1.0 + math.cos(math.pi * recovered_epoch / n_epoch)) / 2.0
            optimizer.param_groups[0]["lr"] = next_lr
            scheduler.last_epoch = recovered_epoch
            scheduler._last_lr = [next_lr]
            scheduler._step_count = recovered_epoch + 1
            recovery = {"kind": "legacy_weights_only", "recovered_epoch": recovered_epoch,
                        "last_logged_epoch": old_history[-1]["epoch"],
                        "optimizer_momentum_reset": True, "rng_restarted": True,
                        "checkpoint_sha256": file_sha256(best_model_params_path),
                        "original_config": previous_config}
            print(f"Recovering best weights from epoch {recovered_epoch}: "
                  f"validation accuracy {resumed_best_acc:.2%}. Momentum/RNG reset; "
                  "this is not an exact continuation of the interrupted run.", flush=True)
        write_json(output_dir / "recovery.json", recovery or {"kind": "full_checkpoint"})
    write_json(output_dir / "config.json", {**vars(args), "environment": environment_info(),
               "architecture": "ConvNet: VGG-style, 8 convolutions, 3 pooling layers",
               "optimizer": "SGD", "momentum": 0.9, "weight_decay": 5e-4,
               "scheduler": "CosineAnnealingLR", "split": str(split_path), "recovery": recovery})

    # TODO - Step 5: Implement the training loop and the validation loop
    #   - Fill the missing TODOs in the following training loop
    if not args.resume:
        torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.
    best_acc = -1.0
    if resumed_best_acc is not None:
        best_acc = resumed_best_acc
    if not last_checkpoint_path.exists() or not args.resume:
        save_training_checkpoint(last_checkpoint_path, model, optimizer, scheduler,
                                 start_epoch - 1, best_acc, history, train_loader,
                                 best_model_params_path, args, split_path, recovery)

    # Keep the original loop bounds statement and skip epochs already checkpointed.
    for epoch in range(1, n_epoch + 1):
        if epoch < start_epoch:
            continue
        print(f'Epoch {epoch:02d}/{n_epoch:02d}')
        print('-' * 10)

        # TODO - 5.1: Training loop
        #   - Implement each of the following steps
        model.train()
        train_ma_loss = 0.
        train_ma_acc = 0.
        best_val_acc = 0.
        best_val_acc = best_acc
        train_loss_sum, train_count = 0.0, 0
        accuracy.reset()
        print("Training starts...")
        for step, (inp, out) in enumerate(train_loader):
            # TODO: put data to the correct device
            inp, out = inp.to(device), out.to(device)

            # TODO: Reset gradients stored in the optimiser
            optimizer.zero_grad(set_to_none=True)

            # TODO: Perform forward pass
            with torch.set_grad_enabled(model.training):
                predictions = model(inp)

            # TODO: Compute loss function and accuracy between predictions and ground truth
            loss = criterion(predictions, out)
            accuracy.update(predictions.detach(), out)

            # TODO: Do backpropagation
            loss.backward()

            # TODO: Update optimiser internal parameters and metrics
            optimizer.step()

            # TODO: Update moving average loss and accuracy
            count = out.numel()
            train_count += count
            train_loss_sum += loss.item() * count
            train_ma_loss = train_loss_sum / train_count
            train_ma_acc = accuracy.compute().item()
            print(f"\r{step+1:03d}/{len(train_loader):03d}: loss = {train_ma_loss:.4f} - acc (train) = {train_ma_acc:.4f}", end="")
        print()  # new line

        val_loss = 0.
        val_acc = 0.
        val_loss_sum, val_correct, val_count = 0.0, 0, 0
        print("Validation starts...")
        model.eval()
        # TODO - 5.2: Validation loop
        for step, (inp, out) in enumerate(val_loader):
            # TODO: put data to the correct device
            inp, out = inp.to(device), out.to(device)

            # TODO: Perform forward pass
            with torch.set_grad_enabled(model.training):
                predictions = model(inp)

            # TODO: Compute loss function between predictions and ground truth
            loss = criterion(predictions, out)

            # TODO: Update moving average loss and accuracy
            count = out.numel()
            val_count += count
            val_loss_sum += loss.item() * count
            val_correct += (predictions.argmax(1) == out).sum().item()
            val_loss = val_loss_sum / val_count
            val_acc = val_correct / val_count

            print(f"\r{step+1:03d}/{len(val_loader):03d}: loss = {val_loss:.4f} - acc (train) = {val_acc:.4f}", end="")
        print()  # new line

        if best_val_acc < val_acc:
            torch.save(model.state_dict(), best_model_params_path)
            print(f"Saving model with better perfs ({best_val_acc} -> {val_acc}")
            best_val_acc = val_acc

        best_acc = best_val_acc
        history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                        "train_loss": train_ma_loss, "train_accuracy": train_ma_acc,
                        "val_loss": val_loss, "val_accuracy": val_acc})
        with (output_dir / "history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        scheduler.step()
        save_training_checkpoint(last_checkpoint_path, model, optimizer, scheduler,
                                 epoch, best_acc, history, train_loader,
                                 best_model_params_path, args, split_path, recovery)
        print(f"Checkpoint saved after epoch {epoch}.", flush=True)

    model = load_model(best_model_params_path, device)
    metrics = {"best_validation_accuracy": best_acc,
               "test": evaluate(model, test_loader, device)}
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "baseline_metadata.json", {
        "checkpoint_sha256": file_sha256(best_model_params_path),
        "split_sha256": file_sha256(split_path), "seed": args.seed})
    print(metrics)


if __name__ == "__main__":
    main()
