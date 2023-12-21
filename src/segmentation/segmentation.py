
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

model_ft = models.resnet18()
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft.load_state_dict(torch.load('model.pth'))
model_ft.eval()

data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = '../datasets/Kitti8_ROIs/test'
label_file = '../datasets/Kitti8_ROIs/test/labels.txt'
roi_dataset = YodaDataset(label_file, data_dir, imagemode=True)

argParser = argparse.ArgumentParser()
argParser.add_argument('-i', metavar='input_dir', type=str, help='input dir (./)')
argParser.add_argument('-idx', metavar='image_index', type=int, help='Image Index')
args = argParser.parse_args()

input_dir = None
if args.i != None:
    input_dir = args.i

input_idx = None
if args.idx != None:
    input_idx = args.idx

dataset = KittiDataset(input_dir, training=False)

#Select An Image
img = dataset[input_idx][0]
label = dataset[input_idx][1]
#Split up the image into ROI
anchors = Anchors()
anchor_centers = anchors.calc_anchor_centers(img.shape, anchors.grid)
ROIs, boxes = anchors.get_anchor_ROIs(img, anchor_centers, anchors.shapes)

print('Displaying each ROI classified as car')
#Pass each region of interest into model
isCar = [0]*len(boxes)
for k in range(len(boxes)):
    #input = roi_dataset[input_idx + k][0].unsqueeze(0)
    input = data_transform(ROIs[k]).unsqueeze(0)
    output = model_ft(input)[0]
    value, indices = torch.max(output, 0)
    if(indices == 1):
        cv2.imshow('window', roi_dataset[input_idx + k][0])
        isCar[k] = 1
        cv2.imshow('window', ROIs[k])
        cv2.waitKey(0)
           
#Calculate IOU against inital la
idx = dataset.class_label['Car']
car_ROIs = dataset.strip_ROIs(class_ID=idx, label_list=label)

ROI_IoUs = []
for idx in range(len(ROIs)):
    if(isCar[idx] == 1):
        ROI_IoUs += [anchors.calc_max_IoU(boxes[idx], car_ROIs)]
        box = boxes[idx]
        pt1 = (box[0][1],box[0][0])
        pt2 = (box[1][1],box[1][0])
        #Visualize outputs
        cv2.rectangle(img, pt1, pt2, color=(0, 255, 255))
        cv2.imshow('boxes', img)
        cv2.waitKey(0)
AverageIOU = sum(ROI_IoUs) / len(ROI_IoUs)
print(AverageIOU)