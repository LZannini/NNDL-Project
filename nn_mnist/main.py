import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from model import MyNetwork
from train_eval import train, evaluate

# Trasformazione
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Caricamento e split
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_dataset, _ = random_split(dataset, [10000, len(dataset) - 10000])
test_dataset, _ = random_split(
    datasets.MNIST(root='./data', train=False, download=True, transform=transform),
    [2500, 10000 - 2500]
)

# Parametri
batch_sizes = [1, 32, 10000]
hidden_sizes = [32, 64, 128]
num_epochs = 10
loss_function = nn.CrossEntropyLoss()
all_results = []

# Esperimenti
for hidden_size in hidden_sizes:
    for batch_size in batch_sizes:
        print(f'\nHidden Size: {hidden_size}, Batch Size: {batch_size}')
        model = MyNetwork(hidden_size)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64)

        train_losses = []
        test_accuracies = []

        for epoch in range(num_epochs):
            train_loss = train(model, train_loader, optimizer, loss_function)
            test_loss, test_acc = evaluate(model, test_loader, loss_function)
            train_losses.append(train_loss)
            test_accuracies.append(test_acc)
            print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Test Acc = {test_acc:.4f}')

        all_results.append({
            'hidden_size': hidden_size,
            'batch_size': batch_size,
            'train_losses': train_losses,
            'test_accuracies': test_accuracies
        })

# Visualizzazione dei grafici
num_results = len(all_results)
cols = 3
rows = (num_results + cols - 1) // cols

# Grafici Loss
fig_loss, axes_loss = plt.subplots(rows, cols, figsize=(15, 5 * rows))
axes_loss = axes_loss.flatten()

# Grafici Accuracy
fig_acc, axes_acc = plt.subplots(rows, cols, figsize=(15, 5 * rows))
axes_acc = axes_acc.flatten()

for i, result in enumerate(all_results):
    title = f"Hidden={result['hidden_size']}, Batch={result['batch_size']}"

    # Plot per la loss
    axes_loss[i].plot(result['train_losses'])
    axes_loss[i].set_title(f"Train Loss - {title}")
    axes_loss[i].set_xlabel("Epoch")
    axes_loss[i].set_ylabel("Loss")

    # Plot per l'accuracy
    axes_acc[i].plot(result['test_accuracies'])
    axes_acc[i].set_title(f"Test Accuracy - {title}")
    axes_acc[i].set_xlabel("Epoch")
    axes_acc[i].set_ylabel("Accuracy")

fig_loss.tight_layout()
fig_acc.tight_layout()
plt.show()
