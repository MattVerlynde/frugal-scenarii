import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training results for a single run')
    parser.add_argument('--run_folder', type=str, required=True, help='Path to the run folder (e.g., results/conso-imagenet/run_xyz)', nargs='+')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder for saving plots')
    args = parser.parse_args()

    if len(args.run_folder) == 1:
        model = yaml.safe_load(open(os.path.join(args.run_folder[0], 'group_info.yaml'), 'r'))['parameters']['--model']
        print(model)

        res_train_mean = pd.read_csv(os.path.join(args.run_folder[0], f'results_train_{model}.csv'))
        res_val_mean = pd.read_csv(os.path.join(args.run_folder[0], f'results_val_{model}.csv'))
        res_test_mean = pd.read_csv(os.path.join(args.run_folder[0], f'results_test_{model}.csv'))

    else:
        i = 0
        perf = pd.DataFrame(columns=['Accuracy_train','Accuracy_val','Accuracy_test','Loss_train','Loss_val','Loss_test'])
        for run in args.run_folder:
            model = yaml.safe_load(open(os.path.join(run, 'group_info.yaml'), 'r'))['parameters']['--model']
            print(model)

            res_train_i = pd.read_csv(os.path.join(run, f'results_train_{model}.csv'))
            res_val_i = pd.read_csv(os.path.join(run, f'results_val_{model}.csv'))
            res_test_i = pd.read_csv(os.path.join(run, f'results_test_{model}.csv'))
            res_train_i['run'] = i
            res_val_i['run'] = i
            res_test_i['run'] = i
            i += 1

            perf = pd.concat([perf, pd.DataFrame({
                'Accuracy_train': [res_train_i['accuracy'].iloc[-1]],
                'Accuracy_val': [res_val_i['accuracy'].iloc[-1]],
                'Accuracy_test': [res_test_i['accuracy'].iloc[-1]],
                'Loss_train': [res_train_i['loss'].iloc[-1]],
                'Loss_val': [res_val_i['loss'].iloc[-1]],
                'Loss_test': [res_test_i['loss'].iloc[-1]]})], ignore_index=True)



            if run == args.run_folder[0]:
                res_train = res_train_i
                res_val = res_val_i
                res_test = res_test_i
            else:
                res_train = pd.concat([res_train, res_train_i], ignore_index=True)
                res_val = pd.concat([res_val, res_val_i], ignore_index=True)
                res_test = pd.concat([res_test, res_test_i], ignore_index=True)
        print(perf)
        perf.to_csv(os.path.join(args.output_folder, 'performance_summary.csv'), index=False)
        
        res_train_mean = res_train.groupby('epoch').mean().reset_index()
        res_val_mean = res_val.groupby('epoch').mean().reset_index()
        res_test_mean = res_test.groupby('epoch').mean().reset_index()

        res_train_std = res_train.groupby('epoch').std().reset_index()*1.96/np.sqrt(len(args.run_folder))
        res_val_std = res_val.groupby('epoch').std().reset_index()*1.96/np.sqrt(len(args.run_folder))
        res_test_std = res_test.groupby('epoch').std().reset_index()*1.96/np.sqrt(len(args.run_folder))
        print(res_train)


    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(res_train_mean['epoch'], res_train_mean['accuracy'], label='Train Accuracy', color='orange')
    plt.plot(res_val_mean['epoch'], res_val_mean['accuracy'], label='Validation Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy over Epochs for {model}. Test Accuracy: {res_test_mean["accuracy"].iloc[-1]:.4f}')
    if len(args.run_folder) > 1:
        plt.fill_between(res_train_mean['epoch'], res_train_mean['accuracy'] - res_train_std['accuracy'], res_train_mean['accuracy'] + res_train_std['accuracy'], color='orange', alpha=0.2, label='95% CI')
        plt.fill_between(res_val_mean['epoch'], res_val_mean['accuracy'] - res_val_std['accuracy'], res_val_mean['accuracy'] + res_val_std['accuracy'], color='blue', alpha=0.2, label='95% CI')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(res_train_mean['epoch'], res_train_mean['loss'], label='Train Loss', color='orange')
    plt.plot(res_val_mean['epoch'], res_val_mean['loss'], label='Validation Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss over Epochs for {model}. Test Loss: {res_test_mean["loss"].iloc[-1]:.4f}')
    if len(args.run_folder) > 1:
        plt.fill_between(res_train_mean['epoch'], res_train_mean['loss'] - res_train_std['loss'], res_train_mean['loss'] + res_train_std['loss'], color='orange', alpha=0.2, label='95% CI')
        plt.fill_between(res_val_mean['epoch'], res_val_mean['loss'] - res_val_std['loss'], res_val_mean['loss'] + res_val_std['loss'], color='blue', alpha=0.2, label='95% CI')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_folder, f'training_plot_{model}.png'))
    plt.show()

    print(f'Plots saved to {os.path.join(args.output_folder, f"training_plot_{model}.png")}')
    print()
