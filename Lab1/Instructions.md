# Lab 1 - Model Pruning

A skeleton of code as well as the exercises are in the source files above.

Note beforehand: Be careful when loading and saving your models, as all the pruning steps are performed on the same object: which means that you have to build a new model and load the pre-trained weights before doing another pruning step (except in the iterative pruning approach).

## Exercise 0: Building a baseline

This first exercise corresponds to the file “train.py”. The first part is to train a simple network using the PyTorch framework for the classification task on the Cifar100 dataset. This includes loading the data, building the training, validation and testing subsets, as well as building the network, and implementing the training loop.

Cifar100 not being as simple as MNIST or Cifar10, it is recommended to use a "large enough" architecture for your model. Something akin to VGG16 (with less max pooling layers), or ResNet18 (same remark) should be enough to get around 65-70% accuracy.

## Exercise 1 to 5: pruning this network

These exercises correspond to the file “pruning.py”. The goal of these exercises is to experiment with the pruning package in PyTorch and practically implement the pruning and iterative pruning procedures presented in the course.

It is advised to save every models to avoid running all the scripts every time, especially when doing the iterative pruning and training procedure. Also, try to have an independent evaluation script if you forgot to write the model accuracy so you don't have to re-run everything.