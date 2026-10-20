
import torch
from torch.nn.utils import prune

from model import ConvNet
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

# TODO: load the-pretrained model
model = ConvNet()
filename = ""
model.load_state_dict(torch.load(filename, weights_only=True))
model.eval()

sparsity_ratio = 0.2




# TODO: implement the pruning loop:
#   tips: you can check which indexes of the original weights have been retained by printing 'model.named_buffers()'




# TODO: compress the model weight file.
#   Pruning using the prune package are performed on the original model, but the original weights are not removed. Basically,
#   it simply copies the original weights, and apply the pruning on these weights. To permanently remove the original weights,
#   you need to call on each layer the function 'prune.remove' that takes as input the layer and the weight to remove.




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








"""
Exercise 4: Re-training a pruned model

For each model saved in Exercise 3, re-train the network on the training set for a few epoch with a lower learning rate.

[QUESTION 1] Similarly, draw the accuracy / sparsity ratio curve and comments the obtained results. Which model would you keep and why ?
"""










"""
Exercise 5: Iterative training and pruning

Instead of doing 1 pruning and fine-tuning step, you can perform it many times.

[QUESTION 1] Similarly, draw the accuracy / sparsity ratio curve and comments the obtained results. Which model would you keep and why ?

[TIPS]
1) move exercise 4 into a function and iterate over it.  
"""







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

    def compute_mask(self, t, default_mask):
        """

        :param t: default tensor (weights)
        :param default_mask: mask obtained from a previous pruning procedure.
        :return: a mask with values in {0, 1} with the same dims as t.
        """
        mask = default_mask.clone()  # always clone to avoid memory corruption
        # TODO

        

        return mask