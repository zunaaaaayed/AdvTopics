# Advanced Topics in IPCVAI — Lab Assignments

## Lab 1: Model Pruning

See [Lab 1 instructions](Lab1/Instructions.md) and the
[reproduction guide and results](Lab1/README.md).

The lab includes the CIFAR100 baseline, pruning Exercises 1-5, source code,
saved train/validation split, checkpoints, logs, measurements and curve images.
The selected model has 60% zero parameters and 72.97% test accuracy, compared
with the baseline's 73.23% test accuracy.

Checkpoint `.pt` files are stored with **Git LFS**. After cloning:

```bash
git lfs install
git lfs pull
```

Use Python 3.13 and follow `Lab1/README.md` for installation, dataset download,
evaluation and reproduction. Downloaded data and virtual environments are
excluded. The friend's `cifar100-pruning-lab` reference folder is not included.

## Lab 1 report

[Read the report (PDF)](Lab1/report/Md_Zunayedul_Islam_Model_Pruning.pdf) or
[browse the LaTeX source](Lab1/report/main.tex).
The report uses Times New Roman, is titled **Lab 1: Pruning**, and ends with
its references. Figures, data-driven tables and build instructions accompany it.
