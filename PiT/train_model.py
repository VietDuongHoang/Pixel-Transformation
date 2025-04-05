import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from PiT_modules import PiT
from checkpoint import load_checkpoint, save_checkpoint, checkpoint_path, checkpoint_path_best

def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, start_epoch, num_epochs, device):
    best_acc = 0.0
    for epoch in range(start_epoch, num_epochs):
        model.train()

        losses = []
        running_loss = 0
        
        for i, inp in enumerate(train_loader):
            inputs, labels = inp
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
        
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 300 == 299:
                print(f'Loss [{epoch + 1}, {i}](epoch, minibatch): ', running_loss / 300)
                running_loss = 0.0

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
        accuracy = 100*(correct/total)
        
        print(f'End epoch[{epoch + 1}] - Accuracy: {accuracy:.2f}%')

        scheduler.step()

        if accuracy > best_acc:
            best_acc = accuracy
            save_checkpoint(model, optimizer, epoch, loss, checkpoint_path_best)
            print(f"✅ Best checkpoint saved at epoch {epoch+1}, accuracy: {accuracy:.2f}%")

        save_checkpoint(model, optimizer, epoch, loss, checkpoint_path)
                
    print('Training Done')

def main():
    # Hyperparameters for CIFAR10.
    batch_size = 32
    EPOCHS = 500
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

    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True)
    val_loader = DataLoader(validset, batch_size=batch_size, num_workers=0, shuffle=False, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, num_workers=0, shuffle=False, pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Instantiate the model.
    PiT_model = PiT(img_size = 32, patch_size = 1, embed_dim = 128, num_classes = 10,
                 num_layers = 8, num_heads = 8, dropout = 0.2).to(device)

    # Using multi GPUs
    print("Available GPUs:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        PiT_model = nn.DataParallel(PiT_model)

    # Number of Trainable parameters
    trainable_params = sum(p.numel() for p in PiT_model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    criterion_PiT = nn.CrossEntropyLoss()
    optimizer_PiT = optim.AdamW(PiT_model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler_PiT = optim.lr_scheduler.CosineAnnealingLR(optimizer_PiT, T_max=10)

    PiT_model, optimizer_PiT, start_epoch, last_loss = load_checkpoint(PiT_model, optimizer_PiT, checkpoint_path)

    # Train the model.
    train_model(PiT_model, train_loader, val_loader, optimizer_PiT, 
                criterion_PiT, scheduler_PiT, start_epoch, EPOCHS, device)

if __name__ == '__main__':
    main()