import torch

def train(model, dataloader, optimizer, loss_function):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        optimizer.zero_grad()
        output = model(X)
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, loss_function):
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            output = model(X)
            loss = loss_function(output, y)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
    accuracy = correct / len(dataloader.dataset)
    return total_loss / len(dataloader), accuracy
