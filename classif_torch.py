import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
import argparse

from tqdm import trange, tqdm



def create_model(name, num_classes, pretrained=True):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.fc.weight.data.normal_(0, 0.01)  # Initialize weights
        model.fc.bias.data.fill_(0.01)  # Initialize bias
        if pretrained:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
    elif name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.fc.weight.data.normal_(0, 0.01)  # Initialize weights
        model.fc.bias.data.fill_(0.01)  # Initialize bias
        if pretrained:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
    elif name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    elif name == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=pretrained)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif name == 'googlenet':
        model = models.googlenet(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif name == 'shufflenet_v2_x0_5':
        model = models.shufflenet_v2_x0_5(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif name == 'squeezenet1_0':
        model = models.squeezenet1_0(pretrained=pretrained)
        num_ftrs = model.classifier[1].in_channels
        model.classifier[1] = nn.Conv2d(num_ftrs, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
    else:
        raise ValueError("Model not supported")
    return model.to(device)

# def early_stopping(train_loss, val_loss, min_delta, patience, counter=0):
#     """
#     Check if early stopping criteria are met.
    
#     Parameters:
#     - train_loss: Current training loss
#     - val_loss: Current validation loss
#     - min_delta: Minimum change in the monitored quantity to qualify as an improvement
#     - patience: Number of epochs with no improvement after which training will be stopped
#     - counter: Counter for epochs with no improvement
    
#     Returns:
#     - bool: True if training should stop, False otherwise
#     """
#     finished = False
#     if abs(val_loss - train_loss) < min_delta:
#         counter += 1
#         if counter >= patience:
#             print("Early stopping triggered")
#             finished = True
#     return finished, counter

def train(trainloader, validloader, model, num_epochs=100, seed=42, testloader=None):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    results_train = []
    results_val = []
    counter = 0
    t = trange(num_epochs, desc="Epoch", position=0, leave=False)
    for epoch in t:
        model.train()

        train_loss = 0.0
        train_corrects = 0

        n = 0
        t_trainloader = tqdm(trainloader, desc="Training", position=1, leave=False)
        
        for inputs, labels in t_trainloader:
                    # with torch.profiler.record_function('h2d copy'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                loss.backward()
                optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)
            n += 1
        train_loss /= len(trainloader.dataset)
        train_corrects = train_corrects.double() / len(trainloader.dataset)
        results_train.append(EpochProgress(epoch, train_loss, train_corrects.item()))

        results_train_df = pd.DataFrame(results_train, columns=['epoch', 'loss', 'accuracy'])
        results_train_df.to_csv(os.path.join(args.storage_path, f'results_train_{args.model}.csv'), index=True)

        val_loss = 0.0
        val_corrects = 0
        model.eval()
        t_validloader = tqdm(validloader, desc="Validation", position=2, leave=False)
        n = 0
        for inputs, labels in t_validloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
            n += 1

        val_loss /= len(validloader.dataset)
        val_acc = val_corrects.double() / len(validloader.dataset)

        print(f'Epoch {epoch}/{num_epochs} \t\t Training Loss: {train_loss:.5f}, Acc: {train_corrects.item():.5f}, Validation Loss: {val_loss:.5f}, Acc: {val_acc.item():.5f}')
        results_val.append(EpochProgress(epoch, val_loss, val_acc.item()))

        scheduler.step()

        results_val_df = pd.DataFrame(results_val, columns=['epoch', 'loss', 'accuracy'])
        results_val_df.to_csv(os.path.join(args.storage_path, f'results_val_{args.model}.csv'), index=True)

        if testloader is not None and (epoch == num_epochs - 1):
            results_test = test(testloader=testloader, model=model, seed=args.seed)
            print(f'Test Loss: {results_test["loss"].values[-1]:.5f}, Acc: {results_test["accuracy"].values[-1]:.5f}')

    return results_train_df, results_val_df

def test(testloader, model, transform=None, seed=42):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    criterion = nn.CrossEntropyLoss()
    
    n = 0
    test_loss = 0.0
    test_corrects = 0
    model.eval()
    t_testloader = tqdm(testloader, desc="Test", position=1, leave=False)
    n = 0
    for inputs, labels in t_testloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

        test_loss += loss.item() * inputs.size(0)
        test_corrects += torch.sum(preds == labels.data)
        n += 1

    test_loss /= len(testloader.dataset)
    test_acc = test_corrects.double() / len(testloader.dataset)

    print(f'Test Loss: {test_loss:.5f}')

    return pd.DataFrame([[0, test_loss, test_acc.item()]], columns=['epoch', 'loss', 'accuracy'])



def plot_results(df, figsize=(10, 5)):
    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss', color='tab:red')
    ax1.plot(df['epoch'], df['loss'], color='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('accuracy', color='tab:blue')
    ax2.plot(df['epoch'], df['accuracy'], color='tab:blue')

    fig.tight_layout()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ImageNet Classification with PyTorch')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training and validation')
    parser.add_argument('--data_root', type=str, default='/media/HDD/ImageNet/ImageNet/imagenet', help='Root directory for ImageNet dataset')
    parser.add_argument('--model', type=str, default='resnet18', help='Model architecture to use (e.g., resnet18, resnet50, vgg16, etc.)')
    parser.add_argument('--pretrained', type=str, default='True', help='Use pretrained model weights')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cuda, cpu)')
    parser.add_argument('--storage_path', type=str, default='./results', help='Path to store results and logs')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=37, help='Random seed for reproducibility')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Number of batches to prefetch')
    args = parser.parse_args()

    args.pretrained = args.pretrained.lower() == 'true'  # Convert string to boolean

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.device.lower() == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your PyTorch installation or use CPU instead.")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    EpochProgress = namedtuple('EpochProgress', 'epoch, loss, accuracy')

    img_size = (256, 256)  # Image size for models like ResNet, VGG, etc.
    crop_size = (224, 224)  # Crop size for models like ResNet, VGG, etc.
    num_classes = 1000
    pretrained = True if args.pretrained else False  # Use pretrained weights if specified
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    from multiprocessing import cpu_count
    # num_workers = 0 # 2*cpu_count()  # Use the number of CPU cores for data loading
    num_workers = args.num_workers  # For debugging purposes, set to 0 to avoid multiprocessing issues
    prefetch_factor = args.prefetch_factor  # Number of batches to prefetch
    # data_root = '/media/HDD/ImageNet/ImageNet/imagenet'
    data_root = args.data_root

    class CustomCompose(transforms.Compose):
        def __call__(self, img):
            for t in self.transforms:
                with torch.profiler.record_function(t.__class__.__name__):
                    img = t(img)
            return img

    pretransform = CustomCompose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Resize the image to the specified size
        # transforms.CenterCrop(crop_size),  # Center crop the image to the specified size
        transforms.RandomResizedCrop(crop_size),  # Random crop the image to the specified size
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.ToTensor(),           # Convert the PIL image to a tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])  # Normalize the image
    ])

    transform = CustomCompose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Resize the image to the specified size
        transforms.CenterCrop(crop_size),  # Center crop the image to the specified size
        transforms.ToTensor(),           # Convert the PIL image to a tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])  # Normalize the image
    ])

    dataset_train = datasets.ImageNet(root=data_root, 
                                    split='train', 
                                    transform=pretransform
                                    )
    dataset_val = datasets.ImageNet(root=data_root, 
                                    split='val', 
                                    transform=transform
                                    )
    
    dataset_test = datasets.ImageNet(root=data_root, 
                                    split='test', 
                                    transform=transform
                                    )

    trainloader = DataLoader(dataset_train, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=num_workers, 
                            pin_memory=True, 
                            #  prefetch_factor=prefetch_factor,
                            #  multiprocessing_context='fork'
                            )
    validloader = DataLoader(dataset_val, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_workers, 
                            pin_memory=True,
                            #  prefetch_factor=prefetch_factor,
                            #  multiprocessing_context='fork'
                            )
    testloader = DataLoader(dataset_test, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_workers, 
                            pin_memory=True, 
                            #  prefetch_factor=prefetch_factor,
                            #  multiprocessing_context='fork'
                            )


    model = create_model(args.model, num_classes, pretrained=pretrained)
    results_train, results_val = train(trainloader=trainloader, validloader=validloader, model=model, num_epochs=num_epochs, seed=args.seed, testloader=testloader)
    results_test = test(testloader=testloader, model=model, seed=args.seed)


    results_train.to_csv(os.path.join(args.storage_path, f'results_train_{args.model}.csv'), index=True)
    results_val.to_csv(os.path.join(args.storage_path, f'results_val_{args.model}.csv'), index=True)
    results_test.to_csv(os.path.join(args.storage_path, f'results_test_{args.model}.csv'), index=True)