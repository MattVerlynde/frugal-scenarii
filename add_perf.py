import pandas as pd
import numpy as np
import os

results = pd.DataFrame(columns=['Model', 'Accuracy_train', 'Accuracy_val', 'Accuracy_test', 'Loss_train', 'Loss_test'])

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
                    train_acc = pd.read_csv(path_splits[1]).iloc[-1,3]
                    train_loss = pd.read_csv(path_splits[1]).iloc[-1,2]
                    val_acc = pd.read_csv(path_splits[2]).iloc[-1,3]
                    val_loss = pd.read_csv(path_splits[2]).iloc[-1,2]
                    test_acc = pd.read_csv(path_splits[0]).iloc[-1,3]
                    test_loss = pd.read_csv(path_splits[0]).iloc[-1,2]
                    model_name = split_paths[0][:-4][13:]
                    results = pd.concat([results, pd.DataFrame({
                        'Model': [model_name],
                        'Accuracy_train': [train_acc],
                        'Accuracy_val': [val_acc],
                        'Accuracy_test': [None],
                        'Loss_train': [train_loss],
                        'Loss_val': [val_loss],
                        'Loss_test': [None]
                    })], ignore_index=True)
                    results = pd.concat([results, pd.DataFrame({
                        'Model': [model_name],
                        'Accuracy_train': [None],
                        'Accuracy_val': [None],
                        'Accuracy_test': [test_acc],
                        'Loss_train': [None],
                        'Loss_val': [None],
                        'Loss_test': [test_loss]
                    })], ignore_index=True)

results.to_csv('results/conso-imagenet/perf.csv', index=False)
print('Results saved to results/conso-imagenet/perf.csv')

data = pd.read_csv('results/conso-imagenet/output_summary_copy.csv')
data = data.drop(columns=['Accuracy', 'Loss'], errors='ignore')
data = pd.concat([data, results[['Accuracy_train', 'Accuracy_val', 'Accuracy_test', 'Loss_train', 'Loss_test']]], axis=1)

print(data.head())
data.to_csv('results/conso-imagenet/output_summary_copy.csv', index=False)