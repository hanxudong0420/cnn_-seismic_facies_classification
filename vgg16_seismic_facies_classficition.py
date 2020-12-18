import time
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from obspy.io.segy.segy import _read_segy
from sklearn.model_selection import train_test_split

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

##########################
### SETTINGS
##########################

# Device
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print('Device:', DEVICE)

random_seed = 1
learning_rate = 0.001
num_epochs = 1
batch_size = 56
patch_size = 64
num_channels = 1
num_classes = 9
all_examples = 158812
num_examples = 7500
sampler = list(range(all_examples))
running_loss = 0.0
loss_list = []

##########################
### DATASET
##########################
filename = 'data/Seismic_data.sgy'
seismicdata = _read_segy(filename, headonly=True)
# train data and lable######
labeled_data = np.load('patchdata.npy')
labels = pd.read_csv('data/classification.ixz', delimiter=" ", names=["Inline", "Xline", "Time", "Class"])
labels["Xline"] -= 300 - 1
labels["Time"] = labels["Time"] // 4
# prediction data #########
inline_data = np.stack(t.data for t in seismicdata.traces
                       if t.header.for_3d_poststack_data_this_field_is_for_in_line_number == 500).T
train_data, test_data, train_samples, test_samples = train_test_split(labels, sampler, random_state=42)
print(train_data.shape, test_data.shape)


# 将地震数据分为64*64的矩阵
def patch_extractor2D(img, mid_x, mid_y, patch_size, dimensions=1):
    patch = img[mid_y:mid_y + patch_size, mid_x:mid_x + patch_size]
    return patch


class SeismicSequence(Dataset):
    def __init__(self, img, x_set, t_set, y_set, patch_size, batch_size, dimensions):
        self.slice = img
        self.X, self.t = x_set, t_set
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.dimensions = dimensions
        self.label = y_set

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        seis = [patch_extractor2D(self.slice, self.X[idx], self.t[idx], self.patch_size)]
        seis = torch.tensor(seis)
        labels = torch.tensor(self.label[idx])

        return seis, labels


dataset = SeismicSequence(labeled_data,
                          train_data["Xline"].values,
                          train_data["Time"].values,
                          train_data["Class"].values,
                          patch_size,
                          batch_size,
                          1)
train_loader = DataLoader(dataset=dataset,
                          batch_size=56,
                          num_workers=0)


##########################
### MODEL
##########################


class VGG16(torch.nn.Module):

    def __init__(self, num_features, num_classes):
        super(VGG16, self).__init__()

        # calculate same padding:
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=64,
                      kernel_size=(3, 3),
                      stride=(1, 1),  # (1(56-1)-56+3)/2=1
                      # (1(32-1)- 32 + 3)/2 = 1
                      padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2))
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2))
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2))
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2))
        )

        self.block_5 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2))
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(True),
            # nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            # nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.detach().zero_()

        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)

        return logits, probas


torch.manual_seed(random_seed)
model = VGG16(num_features=num_examples,
              num_classes=num_classes)

model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def compute_accuracy(model, data_loader):
    model.eval()
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
        features = torch.as_tensor(features, dtype=torch.float32)
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return torch.true_divide(correct_pred, num_examples * 100)


def compute_epoch_loss(model, data_loader):
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = torch.as_tensor(features, dtype=torch.float32)
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            logits, probas = model(features)
            loss = F.cross_entropy(logits, targets, reduction='sum')
            num_examples += targets.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss


start_time = time.time()
for epoch in range(num_epochs):

    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        features = torch.as_tensor(features, dtype=torch.float32)
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()

        cost.backward()

        ### UPDATE MODEL PARAMETERS
        optimizer.step()

        ### LOGGING
        if not batch_idx % 10:
            print('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                  % (epoch + 1, num_epochs, batch_idx,
                     len(train_loader), cost))

    model.eval()
    with torch.set_grad_enabled(False):  # save memory during inference
        print('Epoch: %03d/%03d | Train: %.3f%% |  Loss: %.3f' % (
            epoch + 1, num_epochs,
            compute_accuracy(model, train_loader),
            compute_epoch_loss(model, train_loader)))

    print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))

