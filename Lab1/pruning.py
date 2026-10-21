
import torch
from torch.nn.utils import prune

import argparse
import copy
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from model import ConvNet
from model import (file_sha256, environment_info, evaluate, load_model, make_loaders,
                   prunable_parameters, remove_pruning, seed_everything,
                   sparsity_stats, write_json)
"""
Exercise 1: Pruning a pre-trained model

Given a pre-trained model, we will perform model pruning with the following properties:
- granularity: unconstrained pruning
- layer-by-layer pruning
- weight selection: L1 norm

PyTorch provides few utility functions to prune models, especially the most common one. Especially in our case, the
function is already implemented:
>> torch.nn.utils.prune.l1_unstructured
https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.prune.l1_unstructured.html

To use l1_unstructured, we need to provide the layer with the weights to prune, but also the weight's name. After loading
the pre-trained model, here are the steps that you should follow to perform pruning:

1) retrieve a list of all layers
2) for each layer, get the different weights
3) prune the weights using l1_unstructured (bias included) for a given sparsity ratio
"""

def fresh_baseline(path, device, seed):
    seed_everything(seed)
    model = load_model(path, device)
    if any(name.endswith("_mask") for name, _ in model.named_buffers()):
        raise ValueError("Use the unpruned baseline for independent pruning experiments.")
    return model


def save_record(stage, ratio, model, loader, device, output_dir, rows, checkpoint=None):
    metrics = evaluate(model, loader, device)
    stats = sparsity_stats(model)
    checkpoint = checkpoint or f"{stage}_{ratio}.pt"
    # Preserve masks in checkpoints used for retraining; keep files CPU-portable.
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    torch.save(state, output_dir / checkpoint)
    row = {"requested_sparsity": ratio, "actual_sparsity": stats["sparsity"],
           "validation_accuracy": metrics["accuracy"], "validation_loss": metrics["loss"],
           "checkpoint": checkpoint}
    rows.append(row)
    write_json(output_dir / f"{stage}_results.json", rows)
    write_json(output_dir / (Path(checkpoint).stem + "_layers.json"), stats)
    with (output_dir / f"{stage}_curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)
    print(stage, row, flush=True)


def fine_tune(model, train_loader, val_loader, device, args, output_dir, label):
    seed_everything(args.seed)
    train_loader.generator.manual_seed(args.seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.finetune_lr,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.finetune_epochs)
    best_accuracy = evaluate(model, val_loader, device)["accuracy"]
    best_state = copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(1, args.finetune_epochs + 1):
        model.train()
        loss_sum, correct, count = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            count += targets.numel()
            loss_sum += loss.item() * targets.numel()
            correct += (logits.detach().argmax(1) == targets).sum().item()
        metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": loss_sum / count,
                        "train_accuracy": correct / count, "validation": metrics})
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_state = copy.deepcopy(model.state_dict())
        write_json(output_dir / f"{label}_history.json", history)
        scheduler.step()
        print(label, "epoch", epoch, metrics, flush=True)
    model.load_state_dict(best_state)
    model.eval()


def finish_curve(stage, rows, baseline_metrics, args, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    axis.plot([row["actual_sparsity"] for row in rows],
              [row["validation_accuracy"] for row in rows], "o-", label=stage)
    axis.axhline(baseline_metrics["accuracy"], linestyle="--", label="baseline")
    axis.set(xlabel="Actual fraction of zero parameters", ylabel="Validation top-1 accuracy",
             title=f"CIFAR100: {stage}", xlim=(0, 1), ylim=(0, 1))
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"{stage}_curve.png", dpi=160)
    plt.close(figure)
    eligible = [row for row in rows if row["validation_accuracy"] >=
                baseline_metrics["accuracy"] - args.accuracy_drop]
    selected = (max(eligible, key=lambda row: (row["actual_sparsity"], row["validation_accuracy"]))
                if eligible else None)
    # This explicit policy answers which model to retain using measured validation
    # results. No test data is used for tuning or selection.
    selection = {"allowed_accuracy_drop": args.accuracy_drop, "selected": selected,
                 "policy": "Highest measured sparsity within the allowed baseline accuracy drop",
                 "comment": ("No measured model satisfies the tolerance; retain the baseline."
                             if selected is None else
                             f"Retain {selected['checkpoint']}: validation accuracy "
                             f"{selected['validation_accuracy']:.4f}, actual sparsity "
                             f"{selected['actual_sparsity']:.4f}." )}
    best_accuracy = max(rows, key=lambda row: row["validation_accuracy"])
    selection["best_measured_accuracy"] = best_accuracy["validation_accuracy"]
    selection["best_accuracy_sparsity"] = best_accuracy["actual_sparsity"]
    report = (
        f"{stage}: measured results\n\n"
        f"Baseline validation accuracy: {baseline_metrics['accuracy']:.2%}.\n"
        f"Best measured validation accuracy: {best_accuracy['validation_accuracy']:.2%} "
        f"at actual sparsity {best_accuracy['actual_sparsity']:.2%}.\n"
        f"Allowed accuracy drop: {args.accuracy_drop:.2%} (absolute accuracy).\n\n"
        f"{selection['comment']}\n\n"
        f"Across the measured sparsities, accuracy ranges from "
        f"{min(row['validation_accuracy'] for row in rows):.2%} to "
        f"{max(row['validation_accuracy'] for row in rows):.2%}. "
        "This conclusion applies to the measured ratios; inspect the curve before "
        "making a claim about intermediate ratios.\n\n"
        "See the curve CSV/PNG and the selected checkpoint's *_layers.json for the "
        "measured trend and sparsity of each layer. Selection uses validation data only.\n"
    )
    (output_dir / f"{stage}_answers.md").write_text(report)
    write_json(output_dir / f"{stage}_selection.json", selection)
    print(selection["comment"])
    return selection


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="CIFAR100 pruning exercises 1-5 and bonus")
    parser.add_argument("--exercise", choices=("all", "1", "2", "3", "4", "5", "bonus", "evaluate"), default="all")
    parser.add_argument("--baseline", default=str(root / "artifacts/baseline/best_model_params.pt"))
    parser.add_argument("--split", default=str(root / "artifacts/baseline/train_val_split.pt"))
    parser.add_argument("--data-dir", default=str(root / "data"))
    parser.add_argument("--output-dir", default=str(root / "artifacts/pruning"))
    parser.add_argument("--global-dir", help="Exercise 3 directory to reuse for Exercise 4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 0.9])
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--finetune-lr", type=float, default=0.01)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--iterative-amount", type=float, default=0.2)
    parser.add_argument("--accuracy-drop", type=float, default=0.01)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if (args.batch_size < 1 or args.finetune_epochs < 1 or args.iterations < 1
            or args.finetune_lr <= 0 or not 0 <= args.accuracy_drop <= 1
            or any(not 0 <= ratio <= 1 for ratio in args.ratios)
            or not 0 <= args.iterative_amount <= 1 or len(set(args.ratios)) != len(args.ratios)):
        parser.error("Use positive training settings and unique sparsity ratios in [0, 1].")
    if args.exercise == "4" and not args.global_dir:
        parser.error("Exercise 4 requires --global-dir pointing to saved Exercise 3 models.")
    args.ratios.sort()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)
    split_path = Path(args.split)
    train_loader, val_loader, test_loader = make_loaders(
        args.data_dir, split_path, args.batch_size, args.seed, args.download)
    output_dir = Path(args.output_dir)
    if args.exercise != "evaluate":
        # Never overwrite previously measured experiments accidentally.
        if output_dir.exists() and any(output_dir.iterdir()):
            parser.error("Output directory is nonempty; choose a new --output-dir for this run.")
        output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: load the-pretrained model
    model = ConvNet()
    filename = ""
    filename = str(args.baseline)
    # Load on CPU first so CUDA-trained checkpoints are portable.
    model = load_model(filename, "cpu")
    # The original skeleton load is superseded by load_model's map_location handling.
    if False:
        model.load_state_dict(torch.load(filename, weights_only=True))
    model.eval()

    sparsity_ratio = 0.2
    model.to(device)
    if args.exercise == "evaluate":
        print({"validation": evaluate(model, val_loader, device),
               "test": evaluate(model, test_loader, device), "stats": sparsity_stats(model)})
        return
    metadata_path = Path(filename).parent / "baseline_metadata.json"
    if metadata_path.exists():
        import json
        metadata = json.loads(metadata_path.read_text())
        if (metadata["checkpoint_sha256"] != file_sha256(filename)
                or metadata["split_sha256"] != file_sha256(split_path)
                or metadata["seed"] != args.seed):
            raise ValueError("Baseline checkpoint and saved split do not match their training metadata.")
    if any(name.endswith("_mask") for name, _ in model.named_buffers()):
        raise ValueError("Pruning experiments require the unpruned training baseline.")
    baseline_metrics = evaluate(model, val_loader, device)
    results = {}
    write_json(output_dir / "config.json", {**vars(args), "environment": environment_info(),
               "baseline_validation": baseline_metrics, "split": str(split_path),
               "checkpoint_sha256": file_sha256(filename), "split_sha256": file_sha256(split_path),
               "pruned_parameters": "Conv2d, Linear and BatchNorm2d weights and biases"})




    # TODO: implement the pruning loop:
    #   tips: you can check which indexes of the original weights have been retained by printing 'model.named_buffers()'




    if args.exercise in ("all", "1"):
        for layer, name in prunable_parameters(model):
            prune.l1_unstructured(layer, name=name, amount=sparsity_ratio)
        print("Retained masks:", [(name, tuple(buffer.shape))
                                   for name, buffer in model.named_buffers() if name.endswith("_mask")])

    # TODO: compress the model weight file.
    #   Pruning using the prune package are performed on the original model, but the original weights are not removed. Basically,
    #   it simply copies the original weights, and apply the pruning on these weights. To permanently remove the original weights,
    #   you need to call on each layer the function 'prune.remove' that takes as input the layer and the weight to remove.






    if args.exercise in ("all", "1"):
        remove_pruning(model)
        save_record("exercise1", sparsity_ratio, model, val_loader, device, output_dir, [])

    """
    Exercise 2: Accuracy / sparsity ratio curve.

    In order to find the optimal compression ratio, we need to evaluate our pruned model on the validation set. So, for
    different sparsity ratio, prune the network, and evaluate the pruned model on Cifar100 validation set.

    <<Warning>> be sure that the random train/val split is the same split you used for training the original model!

    Tips:
    - move Exercise 1 code into a function, and call that function with different sparsity ratio.
    - Remember that the original model is lost, so you have to reload the original weights before doing another pruning procedure.
    - Save the tuple (sparsity ratio , validation accuracy) for each model (and maybe the pruned weights)  to avoid a long retraining procedure in case the script fails.

    [QUESTION] Comments the obtained curve. which sparsity ratio would you keep and why?
    """









    if args.exercise in ("all", "2"):
        rows = []
        for ratio in args.ratios:
            candidate = fresh_baseline(args.baseline, device, args.seed)
            for layer, name in prunable_parameters(candidate):
                prune.l1_unstructured(layer, name=name, amount=ratio)
            remove_pruning(candidate)
            save_record("layerwise", ratio, candidate, val_loader, device, output_dir, rows)
        results["layerwise"] = finish_curve("layerwise", rows, baseline_metrics, args, output_dir)


    """
    Exercise 3: Global pruning

    Instead of considering each layer individually, we can prune the whole network by removing the lowest weights in the entire model.
    This is achieved by calling the function 'global_unstructured'.
    https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.prune.global_unstructured.html

    This function takes three arguments:
    1) :parameter: a list of tuples that contains the layer and the weight's name (similar to those given in 'l1_unstructured'
    2) :pruning_method: the function to be called for pruning. Here, we will use 'prune.L1Unstructured'
    3) :amount: the sparsity ratio.

    [QUESTION] Similarly to the exercise 2, draw the accuracy / sparsity ratio curve and comments the obtained results. For the best
    model, what is the sparsity ratio of each layer?

    Tips: Save the intermediate models to accelerate the next exercise.
    """








    if args.exercise in ("all", "3"):
        rows = []
        for ratio in args.ratios:
            candidate = fresh_baseline(args.baseline, device, args.seed)
            prune.global_unstructured(prunable_parameters(candidate),
                                      pruning_method=prune.L1Unstructured, amount=ratio)
            # Save masks for Exercise 4, plus a plain state_dict for independent evaluation.
            save_record("global", ratio, candidate, val_loader, device, output_dir, rows)
            # Pruned modules contain non-leaf tensors and cannot be deep-copied.
            exported = load_model(output_dir / f"global_{ratio}.pt", "cpu")
            remove_pruning(exported)
            torch.save(exported.cpu().state_dict(), output_dir / f"global_{ratio}_dense.pt")
        results["global"] = finish_curve("global", rows, baseline_metrics, args, output_dir)


    """
    Exercise 4: Re-training a pruned model

    For each model saved in Exercise 3, re-train the network on the training set for a few epoch with a lower learning rate.

    [QUESTION 1] Similarly, draw the accuracy / sparsity ratio curve and comments the obtained results. Which model would you keep and why ?
    """










    if args.exercise in ("all", "4"):
        rows = []
        source_dir = Path(args.global_dir) if args.exercise == "4" else output_dir
        import json
        source_config = json.loads((source_dir / "config.json").read_text())
        if (source_config["checkpoint_sha256"] != file_sha256(args.baseline)
                or source_config["split_sha256"] != file_sha256(split_path)):
            raise ValueError("Exercise 3 checkpoints were produced from a different baseline or split.")
        for ratio in args.ratios:
            candidate = load_model(source_dir / f"global_{ratio}.pt", device)
            fine_tune(candidate, train_loader, val_loader, device, args, output_dir,
                      f"finetuned_{ratio}")
            save_record("finetuned", ratio, candidate, val_loader, device, output_dir, rows)
        results["finetuned"] = finish_curve("finetuned", rows, baseline_metrics, args, output_dir)


    """
    Exercise 5: Iterative training and pruning

    Instead of doing 1 pruning and fine-tuning step, you can perform it many times.

    [QUESTION 1] Similarly, draw the accuracy / sparsity ratio curve and comments the obtained results. Which model would you keep and why ?

    [TIPS]
    1) move exercise 4 into a function and iterate over it.  
    """







    if args.exercise in ("all", "5"):
        # A single cumulative model is used here, unlike the independent sweeps.
        candidate = fresh_baseline(args.baseline, device, args.seed)
        rows = []
        save_record("iterative", 0.0, candidate, val_loader, device, output_dir, rows)
        for step in range(1, args.iterations + 1):
            prune.global_unstructured(prunable_parameters(candidate),
                                      pruning_method=prune.L1Unstructured,
                                      amount=args.iterative_amount)
            fine_tune(candidate, train_loader, val_loader, device, args, output_dir,
                      f"iterative_step_{step}")
            cumulative = 1.0 - (1.0 - args.iterative_amount) ** step
            save_record("iterative", cumulative, candidate, val_loader, device, output_dir,
                        rows, checkpoint=f"iterative_step_{step}.pt")
        results["iterative"] = finish_curve("iterative", rows, baseline_metrics, args, output_dir)

    if args.exercise == "bonus":
        candidate = fresh_baseline(args.baseline, device, args.seed)
        for layer, name in prunable_parameters(candidate):
            if name == "weight" and layer.weight.ndim > 2:
                MyPruningMethod.apply(layer, name, amount=args.iterative_amount)
        save_record("bonus", args.iterative_amount, candidate, val_loader, device, output_dir, [])
    write_json(output_dir / "selections.json", results)


    """
    Bonus Exercise: Custom weight selection function

    Instead of using the provided function for the weight selection you can also implement your own by inheriting from the
    'BasePruningMethod' class.
    https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod


    Implement the structured granularity to remove the 'amount' lowest columns (using L1 norm) for the weight with dimensionality > 2.
    """

class MyPruningMethod(prune.BasePruningMethod):
    """Prune every other entry in a tensor
    """
    PRUNING_TYPE = 'unstructured'
    # The supplied class description/type are placeholders; the requested bonus
    # removes complete input-channel columns (dimension 1) by their L1 norm.
    PRUNING_TYPE = "structured"

    def __init__(self, amount=0.2):
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise TypeError("amount must be an integer count or floating-point fraction")
        if amount < 0 or (isinstance(amount, float) and not 0 <= amount <= 1):
            raise ValueError("amount must be nonnegative and fractions must be in [0, 1]")
        self.amount = amount
        self.dim = 1

    def compute_mask(self, t, default_mask):
        """

        :param t: default tensor (weights)
        :param default_mask: mask obtained from a previous pruning procedure.
        :return: a mask with values in {0, 1} with the same dims as t.
        """
        mask = default_mask.clone()  # always clone to avoid memory corruption
        # TODO
        if t.ndim <= 2:
            return mask
        dimensions = tuple(index for index in range(t.ndim) if index != self.dim)
        active = default_mask.sum(dim=dimensions) > 0
        columns = active.nonzero(as_tuple=False).flatten()
        count = (round(self.amount * columns.numel()) if isinstance(self.amount, float)
                 else self.amount)
        if count > columns.numel():
            raise ValueError("amount exceeds the number of remaining columns")
        if count:
            scores = (t * default_mask).abs().sum(dim=dimensions)
            selected = columns[torch.argsort(scores[columns], stable=True)[:count]]
            mask.index_fill_(self.dim, selected, 0)

        

        return mask

if __name__ == "__main__":
    main()
