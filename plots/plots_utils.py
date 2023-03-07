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
