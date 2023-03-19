import numpy as np
import matplotlib.pyplot as plt
import os
import utils


def plot(args, y, comp1, label):
    plt.plot(np.arange(1, args.end_iter + 1), y, label=label)
    # plt.title(f"Loss Vs Iterations ({args.dataset},{args.arch_type})")
    plt.xlabel("Iterations")
    plt.ylabel(label)
    # plt.legend()
    plt.grid()
    utils.checkdir(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/")
    plt.savefig(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_{label}_{comp1}.png",
                dpi=1200)
    plt.close()

# methods suggested by chatGPT - to explore

# Histogram of weight magnitudes: 
#    A histogram of the magnitudes of the model weights before and after pruning can provide insight into the distribution of weights in the model, 
#    and how this distribution changes as weights are pruned. This can be useful in identifying any patterns or biases in the model's learned features, 
#    and in assessing the effectiveness of the pruning method in reducing the number of low-magnitude weights.

def weight_histogram(weights, bins=100):
    plt.hist(weights.flatten(), bins=bins)
    plt.xlabel('Weight Magnitude')
    plt.ylabel('Frequency')
    plt.title('Histogram of Weight Magnitudes')
    plt.show()

# Learning curves: 
#     Learning curves can show the relationship between the amount of training data used and the performance of the model, both before and after pruning. 
#     This can help identify any overfitting or underfitting issues that may arise as a result of pruning, and can also provide insight into the tradeoff 
#     between pruning and the amount of data needed to achieve good performance.

def learning_curves(train_losses, test_losses, train_accuracies, test_accuracies):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Performance')
    plt.title('Learning Curves')
    plt.legend()
    plt.show()

# Scatterplot of weight importance vs. magnitude: 
#     A scatterplot showing the importance (e.g., as measured by the change in loss or accuracy resulting from pruning a given weight) of each weight 
#     in the model against its magnitude can help identify any patterns or relationships between weight importance and magnitude. This can be useful in 
#     selecting an appropriate pruning threshold or method based on the distribution of weight importances and magnitudes.

def weight_importance_vs_magnitude(importances, magnitudes):
    plt.scatter(magnitudes.flatten(), importances.flatten())
    plt.xlabel('Weight Magnitude')
    plt.ylabel('Weight Importance')
    plt.title('Weight Importance vs. Magnitude')
    plt.show()

# Visualization of pruned vs. unpruned model: 
#     A visualization of the weights and activations in the model before and after pruning can help illustrate the effects of pruning on the structure of the model. 
#     This can be useful in identifying any changes in the model's learned features, and in assessing the interpretability of the pruned model.

def visualize_model_weights(model, layer_index):
    weights = model.layers[layer_index].get_weights()[0]
    pruned_weights = model.layers[layer_index].get_pruned_weights()[0]
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(weights)
    axs[0].set_title('Unpruned Weights')
    axs[1].imshow(pruned_weights)
    axs[1].set_title('Pruned Weights')
    plt.show()

# =====================end of chatGPT suggested plots ================================================================

def final_plot(args, bestacc, comp):
    a = np.arange(args.prune_iterations)
    plt.plot(a, bestacc, label="Winning tickets")
    # plt.title(f"Test Accuracy vs Unpruned Weights Percentage ({args.dataset},{args.arch_type})")
    plt.xlabel("Unpruned Weights Percentage")
    plt.ylabel("test accuracy")
    plt.xticks(a, comp, rotation ="vertical")
    plt.ylim(0,100)
    plt.legend()
    plt.grid()
    utils.checkdir(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/")
    plt.savefig(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_AccuracyVsWeights.png", dpi=1200)
    plt.close()
