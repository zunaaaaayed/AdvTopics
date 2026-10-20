"""
Note: a pre-configured virtual environment is available on the CREMI computer here:
/net/ens/DeepLearning/python/tensorflow2
"""
import torch
import torchvision.transforms
from torchvision.datasets import CIFAR100
from torch.utils.data.dataloader import DataLoader
from torchmetrics.classification import MulticlassAccuracy

from model import ConvNet

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


# Hyper-parameters
n_epoch = 1
batch_size = 32
lr = 1e-4
device = "cuda" if torch.cuda.is_available() else "gpu"
best_model_params_path = 'best_model_params.pt'

# TODO - Step 1: Define Pre-processing or input data and target labels:
#   - Use the data augmentation classes from 'torchvision.transforms' package. The minimal augmentation required is 'ToTensor'
#   - Consider adding normalization (between -1 and +1), and other data augmentation is the performances are a bit low.

input_transform = None
target_transform = None

# TODO - Step 2: Load the data
#   - Load Cifar100 dataset from torchvision
#   - Retrieve both train and validation sets. We will use the validation set for our last model evaluation (test set)
#       and split the training set into two subsets (80%-20%): 1 for training, and a second one for validation.





# TODO - Step 3: Implement the model
#   - In "model.py" file, implement the BasicConvNet class
model = ConvNet().to(device=device)





# TODO - Step 4: Training configuration
#   - Define the loss function (cf "CrossEntropyLoss" class)
#   - Define the optimiser (SGD with learning rate of 10e-2)
#   - Define the metrics (Accuracy). You can use the package 'torchmetrics' if available.






# TODO - Step 5: Implement the training loop and the validation loop
#   - Fill the missing TODOs in the following training loop
torch.save(model.state_dict(), best_model_params_path)
best_acc = 0.

for epoch in range(1, n_epoch + 1):
    print(f'Epoch {epoch:02d}/{n_epoch:02d}')
    print('-' * 10)

    # TODO - 5.1: Training loop
    #   - Implement each of the following steps
    model.train()
    train_ma_loss = 0.
    train_ma_acc = 0.
    best_val_acc = 0.
    print("Training starts...")
    for step, (inp, out) in enumerate(train_loader):
        # TODO: put data to the correct device

        # TODO: Reset gradients stored in the optimiser

        # TODO: Perform forward pass

        # TODO: Compute loss function and accuracy between predictions and ground truth

        # TODO: Do backpropagation

        # TODO: Update optimiser internal parameters and metrics

        # TODO: Update moving average loss and accuracy
        print(f"\r{step+1:03d}/{len(train_loader):03d}: loss = {train_ma_loss:.4f} - acc (train) = {train_ma_acc:.4f}", end="")
    print()  # new line

    val_loss = 0.
    val_acc = 0.
    print("Validation starts...")
    model.eval()
    # TODO - 5.2: Validation loop
    for step, (inp, out) in enumerate(val_loader):
        # TODO: put data to the correct device

        # TODO: Perform forward pass

        # TODO: Compute loss function between predictions and ground truth

        # TODO: Update moving average loss and accuracy

        print(f"\r{step+1:03d}/{len(val_loader):03d}: loss = {val_loss:.4f} - acc (train) = {val_acc:.4f}", end="")
    print()  # new line

    if best_val_acc < val_acc:
        torch.save(model.state_dict(), best_model_params_path)
        print(f"Saving model with better perfs ({best_val_acc} -> {val_acc}")
        best_val_acc = val_acc
