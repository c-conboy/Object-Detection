import torch
import os
import cv2
import argparse
from KittiDataset import KittiDataset
from KittiAnchors import Anchors
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
from YodaDataset import YodaDataset

data_dir = '../datasets/Kitti8_ROIs/test'
label_file = '../datasets/Kitti8_ROIs/test/labels.txt'
roi_dataset = YodaDataset(label_file, data_dir)

model_ft = models.resnet18()
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft.load_state_dict(torch.load('model.pth'))
model_ft.eval()

Tp = 0
Fp = 0
Fn = 0
Tn = 0
correct = 0

for idx in range(len(roi_dataset)):
    input = roi_dataset[idx][0].unsqueeze(0)
    output = model_ft(input)[0]
    value, indices = torch.max(output, 0)
    # TP
    if ((indices == 1) and (roi_dataset[idx][1][1] == 1.)):
        Tp = Tp + 1
    # FP
    if ((indices == 1) and (roi_dataset[idx][1][1] == 0.)):
        Fp = Fp + 1

    # FN
    if ((indices == 0) and (roi_dataset[idx][1][1] == 1.)):
        Fn = Fn + 1

    # TN
    if ((indices == 0) and (roi_dataset[idx][1][1] == 0.)):
        Tn = Tn + 1

    if (roi_dataset[idx][1][indices] == 1.):
        correct = correct + 1

Accuracy = correct / len(roi_dataset)
print('Accuracy')
print(Accuracy)
print('Tp')
print(Tp)
print('Fp')
print(Fp)
print('Fn')
print(Fn)
print('Tn')
print(Tn)
