# Lab 1: reproducible CIFAR100 model pruning

Run these commands from the repository root. The assignment instructions are
in `Lab1/Instructions.md`. Only the three existing Python files implement the work;
no new Python files are needed. The supplied skeleton statements, TODO comments,
and exercise text are retained. Added overrides correct the demonstration
settings and CPU device typo; the original `NotImplementedError` placeholders
and original checkpoint-load statement are retained but bypassed.

A local `cifar100-pruning-lab` reference (intentionally excluded) inspired the experiment organization,
saved-split protocol, and validation-based selection policy. Its models,
measurements, report, and architecture are not used as results of this solution.

## Environment

Use Python 3.13 with a matching PyTorch/torchvision installation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r Lab1/requirements.txt
```

For a GPU, install the appropriate matching PyTorch wheels for that machine.
Every run records Python, package and CUDA versions, device, command arguments,
and hashes of the three Python sources in `config.json`. The requirements use
NumPy 2.2.4 to remain compatible with the SciPy version on the course machines.

## Train and evaluate the baseline (Exercise 0)

```bash
python Lab1/train.py --download --epochs 200 --batch-size 128 --lr 0.1 --seed 42
python Lab1/train.py --evaluate-only
```

The model is an eight-convolution VGG-style network with batch normalization,
three max-pooling layers and global average pooling. It produces 100 logits for
32x32 RGB inputs. Training uses SGD (momentum 0.9, weight decay 0.0005), cosine
learning-rate decay, random crops and horizontal flips. Normalization uses mean
(0.5071, 0.4867, 0.4408) and standard deviation (0.2675, 0.2565, 0.2761).
Validation and testing use tensor conversion and normalization only.

The official 50,000 training images are split into 40,000 training and 10,000
validation images. The official 10,000 test images are used for final evaluation.
Indices, seed and a label-order fingerprint are saved in
`Lab1/artifacts/baseline/train_val_split.pt`; pruning refuses to create a new
split. The split is checked for duplicates, overlap, coverage and seed mismatch.
Best-model selection uses validation accuracy across all epochs, and the test
set is evaluated after reloading that model.

Baseline outputs include `best_model_params.pt`, `history.csv`, `metrics.json`,
`config.json` and `baseline_metadata.json` (checkpoint and split SHA256 hashes).
An existing baseline and its accompanying files are backed up before retraining.
Use a different `--output-dir` for a separate experiment, and pass its checkpoint
and split explicitly to pruning.

## Recover after an SSH disconnect

Run on the original GPU host, using the same environment and training settings:

```bash
python Lab1/train.py --resume --epochs 200 --batch-size 128 --lr 0.1 --seed 42
```

`last_checkpoint.pt` is written atomically at initialization and after every
completed epoch. It contains current and best weights, SGD momentum, scheduler,
training history, train-loader generator state, and Python/NumPy/PyTorch CPU/CUDA
random states. On the same software and hardware, resuming continues from the
next epoch. A partially completed epoch is replayed. The original epoch total
is required because changing it changes the cosine schedule.

For old runs that only saved `best_model_params.pt`, recovery verifies those
weights against the best row in `history.csv`, restarts from that epoch, and
restores the cosine schedule's position. Momentum and random state were not
saved and must restart. This is a weights-only recovery, not an exact continuation;
`recovery.json` and `config.json` record the distinction and the original run.
Existing artifacts are backed up before either recovery path.

Keep training alive when the SSH terminal closes:

```bash
tmux new-session -s lab1-train
.venv/bin/python -u -B Lab1/train.py --resume > Lab1/artifacts/baseline/resume.log 2>&1
# Press Ctrl-b then d to detach. Reconnect on the same host with:
tmux attach -t lab1-train
```

The recovery implementation was checked with a simulated interruption: full
checkpoint recovery reproduced uninterrupted CPU weights, optimizer momentum,
scheduler, random states and history exactly on synthetic data. Legacy recovery
and rejection of incompatible settings were checked separately.

## Prune and retrain (Exercises 1-5)

```bash
python Lab1/pruning.py --exercise all
```

Defaults: sparsity ratios 0, 0.2, 0.4, 0.6, 0.8 and 0.9; five fine-tuning epochs
at learning rate 0.01; five iterative steps pruning 20% of the remaining unmasked
entries each time. All Conv2d, BatchNorm2d and Linear weights **and biases** are
included. Independent sweeps reload the original baseline for every ratio.
Only iterative pruning carries the same model forward between steps.

Each exercise can also run separately, with a fresh output directory:

```bash
python Lab1/pruning.py --exercise 1 --output-dir Lab1/artifacts/exercise1
python Lab1/pruning.py --exercise 2 --output-dir Lab1/artifacts/exercise2
python Lab1/pruning.py --exercise 3 --output-dir Lab1/artifacts/exercise3
python Lab1/pruning.py --exercise 4 --global-dir Lab1/artifacts/exercise3 --output-dir Lab1/artifacts/exercise4
python Lab1/pruning.py --exercise 5 --output-dir Lab1/artifacts/exercise5
python Lab1/pruning.py --exercise bonus --output-dir Lab1/artifacts/bonus
```

Exercise 4 loads the masked global checkpoints saved by Exercise 3 and checks
that its baseline and split match. Fine-tuning retains masks so zeros cannot
regrow. The best validation state, including the state before fine-tuning if it
is better, is retained. Iterative requested cumulative sparsity is
`1 - (1 - amount) ** step`; integer rounding and pre-existing zeros can affect
actual sparsity, which is measured and reported separately.

The bonus removes whole **input-channel columns (dimension 1)**, ranked by L1
norm, from tensors with more than two dimensions. Existing zeros in the mask are
preserved; lower-dimensional tensors are unchanged. The original class docstring
is a placeholder and is preserved as supplied. This column interpretation
follows the assignment text; the friend's bonus instead removes output filters.

## Results and final evaluation

Each curve experiment writes intermediate checkpoints, a curve CSV/PNG, JSON
measurements, a measured-result answers file and a selection JSON. Every saved
checkpoint has a companion `*_layers.json` with its actual overall and per-layer
zero fractions. Exercise 4/5 also save per-epoch fine-tuning histories. Output
directories must be empty to prevent accidental replacement of experiment data.

The selection policy keeps the highest measured sparsity whose validation
accuracy is within **one percentage point** of the original baseline. Override
with `--accuracy-drop`. If none qualifies, the recommendation is the baseline.
Use the answers files and measured curves for your report; they do not claim
unmeasured intermediate performance.

Evaluate any saved checkpoint using the existing pruning script, including
masked fine-tuned checkpoints (no separate Python evaluation file is needed):

```bash
python Lab1/pruning.py --exercise evaluate --baseline Lab1/artifacts/pruning/finetuned_0.6.pt
```

Replace the filename with the selected checkpoint from `*_selection.json`.
`evaluate` prints validation/test loss and top-1 accuracy plus layer sparsities,
without retraining. Keep the same `--split`, `--seed` and data as training. Test
accuracy must not guide the choice of sparsity or hyperparameters.

`prune.remove` removes the mask/original-weight reparameterization. It does not
change tensor shapes or provide a sparse storage format. Exercise 1 and global
`*_dense.pt` exports use ordinary state dictionaries with the pruned zeros;
masked checkpoints are retained where retraining is needed.

## Verification and limits

CPU checks exercised two baseline training epochs on deterministic synthetic
images, an identical seeded rerun, all five pruning exercises, the bonus,
independent evaluation, saved global-checkpoint reuse, checkpoint reloads,
mask preservation, prune.remove equivalence, and split mismatch rejection.
The full CIFAR100 baseline and Exercises 1-5 have now completed. The optional
bonus implementation was checked on synthetic tensors but was not run as a
full CIFAR100 experiment.

Measured results (validation selects the models; testing is final evaluation):

| Model selected by validation | Actual sparsity | Validation accuracy | Test accuracy |
|---|---:|---:|---:|
| Baseline, best epoch 190 of 200 | 0% | 73.61% | 73.23% |
| Global L1 pruning | 40% | 73.42% | Not evaluated |
| Global L1 pruning + fine-tuning | 60% | 72.77% | 72.97% |
| Iterative pruning, step 4 | 59.04% | 72.65% | Not evaluated |

The selected 60% sparse model loses 0.26 percentage points of test accuracy
relative to the baseline. Its separate final evaluation reported 72.75%
validation accuracy, compared with 72.77% during the original GPU experiment;
the raw measurements are preserved in `artifacts/pruning/final_evaluation.log`
and `artifacts/pruning/finetuned_results.json`. Small evaluation differences
across execution environments should not be confused with a new training run.

The baseline was interrupted after epoch 65. The available best weights were
from epoch 54, so training recovered from those weights with optimizer momentum
and random states reset, then continued through epoch 200. The provenance is
recorded in `artifacts/baseline/recovery.json`; the pre-recovery weights, config
and history are retained in its backup directory. This run is reproducible as
a documented recovery procedure, not claimed identical to an uninterrupted run.

All saved model weights, the split, the final full training checkpoint, logs,
CSV/JSON measurements and curve images are included for review. In the GitHub
repository, `.pt` files use Git LFS. Run `git lfs install` and `git lfs pull`
after cloning. Downloaded CIFAR100 data, virtual environments and the friend's
reference folder are excluded. To evaluate saved weights on a fresh checkout,
add `--download` to the evaluation command to download the official dataset.
For a new pruning run, choose an empty `--output-dir` because the published
results already populate the default directory.

Seed 42, zero data-loader workers and deterministic PyTorch algorithms are used.
Repeat runs with the same software and hardware; bitwise identity across
versions, devices, or CPU/GPU execution is not promised. Preserve the saved
split, weights, configs and source together when submitting to the professor.

## Report

The completed report is available as
[PDF](report/Md_Zunayedul_Islam_Model_Pruning.pdf) and
[LaTeX source](report/main.tex). It uses Times New Roman and includes the measured
results, plots, and model-selection discussion.
See [report build instructions](report/README.md) to regenerate it.
