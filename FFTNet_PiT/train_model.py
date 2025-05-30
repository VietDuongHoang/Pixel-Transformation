import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import FFTNetViT
from checkpoint import load_checkpoint, save_checkpoint, checkpoint_path, checkpoint_path_best

from tqdm import tqdm

def train_model(model, train_loader, test_loader, optimizer, criterion
                , scheduler, start_epoch, num_epochs, device, model_name):

    best_acc = 0.0

    for epoch in range(start_epoch, num_epochs):
        # Training phase.
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        train_loader_tqdm = tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{num_epochs} - Training")
        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            train_loader_tqdm.set_postfix(loss=loss.item(), acc=100.*train_correct/train_total)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100. * train_correct / train_total
        print(f"\n{model_name} Epoch [{epoch+1}/{num_epochs}] Training Loss: {epoch_train_loss:.4f} | Accuracy: {epoch_train_acc:.2f}%")
        
        # Validation phase.
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        test_loader_tqdm = tqdm(test_loader, desc=f"{model_name} Epoch {epoch+1}/{num_epochs} - Validation")
        with torch.no_grad():
            for inputs, labels in test_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                test_loader_tqdm.set_postfix(loss=loss.item(), acc=100.*correct/total)
        
        val_loss = test_loss / len(test_loader.dataset)
        val_acc = 100. * correct / total

        scheduler.step(val_loss)
        
        print(f"{model_name} Epoch [{epoch+1}/{num_epochs}] Validation Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%\n")

        save_checkpoint(model, optimizer, epoch, loss, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1}, accuracy: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, loss, checkpoint_path_best)
            print(f"✅ Best checkpoint saved at epoch {epoch+1}, accuracy: {val_acc:.2f}%")
                
    print('Training Done')

def main():
    # Hyperparameters for CIFAR10.
    batch_size = 32
    num_epochs = 500
    learning_rate = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transforms for CIFAR10.
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load CIFAR10 dataset.
    traindata = torchvision.datasets.CIFAR10(root='./CIFAR_10', train=True,
                                                download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./CIFAR_10', train=False,
                                            download=True, transform=transform_test)

    trainset_size = int(len(traindata) * 0.8)
    validset_size = len(traindata) - trainset_size

    seed = torch.Generator().manual_seed(42)
    trainset, validset = data.random_split(traindata, [trainset_size, validset_size], generator=seed)

    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=2, shuffle=True, pin_memory=True)
    val_loader = DataLoader(validset, batch_size=batch_size, num_workers=2, shuffle=False, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, num_workers=2, shuffle=False, pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Instantiate the model.
    fftnet_model = FFTNetViT(img_size=32, patch_size=1, in_channels=3, num_classes=10,
                         embed_dim=192, depth=6, mlp_ratio=3.0, dropout=0.1,
                         num_heads=6, adaptive_spectral=True)
    fftnet_model.to(device)

    # Using multi GPUs
    print("Available GPUs:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        fftnet_model = nn.DataParallel(fftnet_model)

    # Number of Trainable parameters
    trainable_params = sum(p.numel() for p in fftnet_model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    criterion_fftnet = nn.CrossEntropyLoss()
    optimizer_fftnet = optim.AdamW(fftnet_model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler_fftnet = optim.lr_scheduler.CosineAnnealingLR(optimizer_fftnet, T_max=10)

    fftnet_model, optimizer_fftnet, start_epoch, last_loss = load_checkpoint(fftnet_model, optimizer_fftnet, checkpoint_path)

    # Train the model.
    train_model(fftnet_model, train_loader, test_loader, optimizer_fftnet, 
                criterion_fftnet, scheduler_fftnet, start_epoch, num_epochs, device, model_name = "FFTnet")

if __name__ == '__main__':
    main()