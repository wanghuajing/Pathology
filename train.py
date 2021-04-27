from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from dataset import imgdataset
import config
import os
import pandas as pd
import torchvision.models as models
import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = imgdataset(config.table, config.path)
    n_val = int(len(dataset) * config.val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = torch.utils.data.DataLoader(train, batch_size=config.batch_size, shuffle=True,
                                               num_workers=config.num_works)
    val_loader = torch.utils.data.DataLoader(val, batch_size=config.batch_size, shuffle=True,
                                             num_workers=config.num_works)

    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=config.lr)

    for epoch in range(config.epoch):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            for i, val_data in val_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / n_val
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / n_train, val_accurate))

        if epoch % 19 == 0:
            torch.save(net.state_dict(), config.save_path)

    print('Finished Training')

    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)
    dataset = imgdataset(config.table, config.path)
    # for i, (image, label) in enumerate(dataset):
    #     print(label)
    resnet18 = models.resnet18(pretrained=True)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
