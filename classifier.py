# Data augmentation and normalization for training
# Just normalization for validation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from YodaDataset import YodaDataset
import cv2
cudnn.benchmark = True
plt.ioff()
data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

label_transfrom = transforms.Compose([
        transforms.ToTensor(),
])

#Load Training Data
data_dir = '../datasets/Kitti8_ROIs/train'
label_file = '../datasets/Kitti8_ROIs/train/labels.txt'
train_dataset = YodaDataset(label_file, data_dir, transform=data_transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

#Load Test Data
data_dir = '../datasets/Kitti8_ROIs/test'
label_file = '../datasets/Kitti8_ROIs/test/labels.txt'
test_dataset = YodaDataset(label_file, data_dir, transform=data_transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
            model.train()  # Set model to training mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in train_dataloader:
                sum = labels.sum(0)
                if(sum[1] != 2):
                    continue
                inputs = inputs.to(device)
                labels =labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)            
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    print(loss)
                # statistics
                running_loss += loss.item() * inputs.size(0)
                scheduler.step()


            epoch_loss = running_loss / 6000
            print('epoch')
            print(epoch_loss)
            state_dict = model_ft.state_dict()
            saveLocation = "./model" + str(epoch+1) + "(" + str(epoch_loss) + ")" ".pth"
            torch.save(state_dict, saveLocation)
        print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')
    return model

model_ft = models.resnet18()
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft.load_state_dict(torch.load('model0(5.879209007501602).pth', map_location=torch.device('cpu')))
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs 
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=10)

state_dict = model_ft.state_dict()
torch.save(state_dict, "./model.pth")