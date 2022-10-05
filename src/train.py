import json
import yaml
import os
from models import get_model
from dataset import get_dataloader


from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

params = yaml.safe_load(open("params.yaml"))["train"]


BATCH_SIZE = 32
train_loader, val_loader = get_dataloader('data/prepared/data_list.json', BATCH_SIZE, "train")


def test_epoch(network, loader):
    network.eval()
    correct = 0
    total = 0
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        output = network(inputs)
        labels = labels.unsqueeze(1)

        output = output.squeeze().data.cpu().numpy()
        labels = labels.squeeze().cpu().numpy()

        output[output > 0.5] = 1
        output[output <= 0.5] = 0

        correct += (output == labels).sum()
        try:
            total += labels.shape[0]
        except IndexError:
            total += 1

    accuracy = 100 * correct / total
    # print("Test {} || ACC: {:.4f}".format(str(i+1).zfill(4), accuracy))
    return accuracy


def train_epoch(network, loader, optimizer, criterion):
    network.train()
    batch_loss = 0
    i = 0
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        output = network(inputs)

        labels = labels.unsqueeze(1)
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

        batch_loss += loss
        i += 1
        # if i % 20 == 0:
        # print("Train Batch {} || Loss: {:.4f}".format(str(i).zfill(4), batch_loss/i))
    return batch_loss / i


def fit_network(network, criterion, optimizer, train_loader, val_loader, epochs_num):
    best_acc = 0.0
    epochs = range(1, epochs_num + 1)
    for epoch in epochs:
        epoch_loss = train_epoch(network, train_loader, optimizer, criterion)
        epoch_acc = test_epoch(network, val_loader)

        info_line = "Epoch {} || Loss: {:.4f} | Test Acc: {:.4f}".format(str(epoch).zfill(3), epoch_loss, epoch_acc)
        print(info_line)

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(network.state_dict(), "data/weights/Best_ACC.pth.gz")
    print(best_acc)


os.makedirs("data/weights", exist_ok=True)
network = get_model(params["network"])
network = network.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(network.parameters(), lr=0.001)
fit_network(network, criterion, optimizer, train_loader, val_loader, epochs_num=params["epochs"])
