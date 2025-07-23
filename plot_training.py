import pandas as pd
import numpy as np
import os

results = pd.DataFrame(columns=['Epoch','Model', 'Accuracy_train', 'Accuracy_val', 'Loss_train', 'Loss_val'])

for folder in os.listdir('results/conso-imagenet'):
    if folder[:3] == 'run':
        if folder[4:] not in ['0', '1', '2', '3', '4']:
            path = os.path.join('results/conso-imagenet', folder)
            for group in sorted(os.listdir(path)):
                if group[-4:] != 'yaml':
                    print(group[-4:])
                    path_group = os.path.join(path, group)
                    print(path_group)
                    split_paths = sorted([split for split in os.listdir(path_group) if split.endswith('.csv')])
                    path_splits = [os.path.join(path_group, split) for split in split_paths]
                    
                    train_data = pd.read_csv(path_splits[1])
                    val_data = pd.read_csv(path_splits[2])
                    train_acc = train_data.iloc[:,3]
                    train_loss = train_data.iloc[:,2]
                    val_acc = val_data.iloc[:,3]
                    val_loss = val_data.iloc[:,2]
                    
                    model_name = split_paths[0][:-4][13:]
                    results = pd.concat([results, pd.DataFrame({
                        'Epoch': train_data['epoch'],
                        'Model': [model_name]* len(train_acc),
                        'Accuracy_train': train_acc,
                        'Accuracy_val': val_acc,
                        'Loss_train': train_loss,
                        'Loss_val': val_loss
                    })], ignore_index=True)

results.to_csv('results/conso-imagenet/output_summary_trainings.csv', index=False)

results = pd.read_csv('results/conso-imagenet/output_summary_trainings.csv')

import matplotlib.pyplot as plt
import seaborn as sns

for model in results['Model'].unique():
    model_data = results[results['Model'] == model]
    print(f"Model: {model}")
    model_data_mean = model_data[['Epoch', 'Accuracy_train', 'Accuracy_val', 'Loss_train', 'Loss_val']].groupby('Epoch').mean().reset_index()
    model_data_std = model_data[['Epoch', 'Accuracy_train', 'Accuracy_val', 'Loss_train', 'Loss_val']].groupby('Epoch').std().reset_index()

    print(model_data_mean['Epoch'].shape)
    print(model_data_mean['Accuracy_train'].shape)
    print(model_data_mean['Loss_train'].shape)

    plt.plot(model_data_mean['Epoch'], model_data_mean['Accuracy_train'], label='Train Accuracy', color='orange', alpha=0.5)
    # plt.errorbar(model_data_mean['Epoch'], model_data_mean['Accuracy_train'], label='Train Accuracy', color='orange', alpha=0.5, yerr=model_data_std['Accuracy_train'])
    plt.plot(model_data_mean['Epoch'], model_data_mean['Accuracy_val'], label='Validation Accuracy', color='blue', alpha=0.5)
    # plt.errorbar(model_data_mean['Epoch'], model_data_mean['Accuracy_val'], label='Validation Accuracy', color='blue', alpha=0.5, yerr=model_data_std['Accuracy_val'])
    plt.title(f'Accuracy over Epochs for {model}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'results/conso-imagenet/plots/{model}_accuracy.png')
    plt.show()
    plt.close()

    plt.plot(model_data_mean['Epoch'], model_data_mean['Loss_train'], label='Train Loss', color='orange', alpha=0.5)
    # plt.errorbar(model_data_mean['Epoch'], model_data_mean['Loss_train'], label='Train Loss', color='orange', alpha=0.5, yerr=model_data_std['Loss_train'])
    plt.plot(model_data_mean['Epoch'], model_data_mean['Loss_val'], label='Validation Loss', color='blue', alpha=0.5)
    # plt.errorbar(model_data_mean['Epoch'], model_data_mean['Loss_val'], label='Validation Loss', color='blue', alpha=0.5, yerr=model_data_std['Loss_val'])
    plt.title(f'Loss over Epochs for {model}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'results/conso-imagenet/plots/{model}_loss.png')
    plt.show()
    plt.close()

