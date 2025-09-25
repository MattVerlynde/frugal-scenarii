import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training results for a single run')
    parser.add_argument('--run_folder', type=str, required=True, help='Path to the run folder (e.g., results/conso-imagenet/run_xyz)')
    args = parser.parse_args()

    model = yaml.safe_load(open(os.path.join(args.run_folder, 'group_info.yaml'), 'r'))['parameters']['--model']
    print(model)

    res_train = pd.read_csv(os.path.join(args.run_folder, f'results_train_{model}.csv'))
    res_val = pd.read_csv(os.path.join(args.run_folder, f'results_val_{model}.csv'))
    res_test = pd.read_csv(os.path.join(args.run_folder, f'results_test_{model}.csv'))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(res_train['epoch'], res_train['accuracy'], label='Train Accuracy', color='orange')
    plt.plot(res_val['epoch'], res_val['accuracy'], label='Validation Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy over Epochs for {model}. Test Accuracy: {res_test["accuracy"].iloc[-1]:.4f}')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(res_train['epoch'], res_train['loss'], label='Train Loss', color='orange')
    plt.plot(res_val['epoch'], res_val['loss'], label='Validation Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss over Epochs for {model}. Test Loss: {res_test["loss"].iloc[-1]:.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.run_folder, f'training_plot_{model}.png'))
    plt.show()

    print(f'Plots saved to {os.path.join(args.run_folder, f"training_plot_{model}.png")}')
