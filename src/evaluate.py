import yaml
import json
from models import get_model
from dataset import get_dataloader
import torch
from dvclive import Live

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

params = yaml.safe_load(open("params.yaml"))["train"]
live = Live("evaluation")


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


test_dataloader = get_dataloader('data/prepared/data_list.json', 1, 'test')

network = get_model(params["network"])
network = network.to(device)
network.load_state_dict(torch.load('data/weights/Best_ACC.pth.gz', map_location=device))
with open("scores.json", 'w') as file:
    json.dump({"accuracy": test_epoch(network, test_dataloader)}, file)
