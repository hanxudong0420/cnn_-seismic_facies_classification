import time
import torch
import numpy as np
import pandas as pd
from obspy.io.segy.segy import _read_segy
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

    ##########################
    # SETTINGS ###############
    ##########################

# Device
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print('Device:', DEVICE)

random_seed = 1
learning_rate = 0.001
num_epochs = 2
batch_size = 56
patch_size = 64
num_channels = 1
num_classes = 9
grayscale = 1
all_examples = 158812
num_examples = 7500
sampler = list(range(all_examples))
running_cost = 0.0
cost_list = []

##########################
# DATASET ################
##########################
filename = 'data/Seismic_data.sgy'
seismicdata = _read_segy(filename, headonly=True)
# train data and lable ###
labeled_data = np.load('patchdata.npy')
labels = pd.read_csv('data/classification.ixz', delimiter=" ", names=["Inline", "Xline", "Time", "Class"])
labels["Xline"] -= 300 - 1
labels["Time"] = labels["Time"] // 4
# prediction data ########
inline_data = np.stack(t.data for t in seismicdata.traces
                       if t.header.for_3d_poststack_data_this_field_is_for_in_line_number == 500).T
inline_patch = np.load('inline_patch.npy')
train_data, test_data, train_samples, test_samples = train_test_split(labels, sampler, random_state=42)
print(train_data.shape, test_data.shape)


###############################
# 将地震数据分为64*64的矩阵 #######
###############################
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


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Linear(512 * 4, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** .5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # because MNIST is already 1x1 here:
        # disable avg pooling
        # x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas


def resnet18(num_classes):
    """Constructs a ResNet-18 model."""
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   num_classes=num_classes,
                   grayscale=grayscale)
    return model


torch.manual_seed(random_seed)

model = resnet18(num_classes)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
        features = torch.as_tensor(features, dtype=torch.float32)
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return torch.true_divide(correct_pred * 100, num_examples)


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
        running_cost += cost.item()
        cost_list.append(running_cost)
        running_cost = 0.0

        ### LOGGING
        if not batch_idx % 10:
            print('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                  % (epoch + 1, num_epochs, batch_idx,
                     len(train_loader), cost))

    model.eval()
    with torch.set_grad_enabled(False):  # save memory during inference
        print('Epoch: %03d/%03d | Train: %.3f%%' % (
            epoch + 1, num_epochs,
            compute_accuracy(model, train_loader, device=DEVICE)))

    print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))

np.save('res18cost.npy', cost_list)
torch.save(model, 'res18model2.pkl')
torch.save(model.state_dict(), 'res18modelstd2.pkl')

t_max, y_max = inline_data.shape
with torch.no_grad():
    predx = np.full_like(inline_data, -1)
    for space in tqdm(range(y_max), desc='Space'):
        for depth in tqdm(range(t_max), leave=False, desc='Time'):
            inp = np.expand_dims(np.expand_dims(patch_extractor2D(inline_patch, space, depth, 64), axis=0), axis=0)
            inp = torch.as_tensor(inp, dtype=torch.float32)
            _, outp = model(inp)
            _, predx[depth, space] = torch.max(outp, dim=1)

np.save('res18predx.npy', predx)

# Epoch: 001/002 | Train: 99.761%   Time elapsed: 96.01 min
# Epoch:2    |  accuracy on test data : 99.841%
