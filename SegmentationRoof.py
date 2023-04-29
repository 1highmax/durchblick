from utils import *
import os, cv2
import numpy as np
import pandas as pd
import random, tqdm
# import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as album

# !pip install -q -U segmentation-models-pytorch albumentations > /dev/null
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils

DATA_DIR = 'data'

x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'train_labels')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'val_labels')

class_dict = pd.read_csv("label_class_dict.csv")
print(class_dict)
# Get class names
class_names = class_dict['name'].tolist()
# Get class RGB values
class_rgb_values = class_dict[['r','g','b']].values.tolist()

print('All dataset classes and their corresponding RGB values in labels:')
print('Class Names: ', class_names)
print('Class RGB values: ', class_rgb_values)

select_classes = ['Roof','Ridge','Obstacle','Other']

# Get RGB values of required classes
select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]

select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]

print('Selected classes and their corresponding RGB values in labels:')
print('Class Names: ', class_names)
print('Class RGB values: ', class_rgb_values)


dataset = CustomDataset(x_train_dir, y_train_dir, class_rgb_values=select_class_rgb_values)
random_idx = random.randint(0, len(dataset)-1)
image, mask = dataset[0]

visualize(
    original_image = image,
    ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
    one_hot_encoded_mask = reverse_one_hot(mask)
)


augmented_dataset = CustomDataset(
    x_train_dir, y_train_dir, 
    augmentation=get_training_augmentation(),
    class_rgb_values=select_class_rgb_values,
)

random_idx = random.randint(0, len(augmented_dataset)-1)

# Different augmentations on a random image/mask pair (256*256 crop)
for i in range(1):
    image, mask = augmented_dataset[random_idx]
    visualize(
        original_image = image,
        ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
        one_hot_encoded_mask = reverse_one_hot(mask)
    )

ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['roof','ridge','obstacle','other']
ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multiclass segmentation

model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
    in_channels=3
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


train_dataset = CustomDataset(
    x_train_dir, y_train_dir, 
    augmentation=get_training_augmentation(),
    #preprocessing=get_preprocessing(preprocessing_fn=preprocessing_fn),
    preprocessing=get_preprocessing(preprocessing_fn=None),
    class_rgb_values=select_class_rgb_values,
)

valid_dataset = CustomDataset(
    x_valid_dir, y_valid_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn=preprocessing_fn),
    class_rgb_values=select_class_rgb_values,
)

# Get train and val data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

# create segmentation model with pretrained encoder

TRAINING = True

# Set num of epochs
EPOCHS = 100

# Set device: `cuda` or `cpu`
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define loss function
# loss = smp.utils.losses.DiceLoss()

class diceloss(torch.nn.Module):
    def init(self):
        super(diceloss, self).init()
    def forward(self,pred, target):
       smooth = 1.
       iflat = pred.contiguous().view(-1)
       tflat = target.contiguous().view(-1)
       intersection = (iflat * tflat).sum()
       A_sum = torch.sum(iflat * iflat)
       B_sum = torch.sum(tflat * tflat)
       return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )

loss_fn = diceloss()

# define metrics
metric_fn = smp.utils.metrics.IoU()


# define optimizer
optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.001),
])

# define learning rate scheduler (not used in this NB)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=1, T_mult=2, eta_min=5e-5,
)

# load best saved model checkpoint from previous commit (if present)
if os.path.exists('best_model.pth'):
    model = torch.load('best_model.pth', map_location=DEVICE)

if TRAINING:

    best_iou_score = 0.0

    model.to(DEVICE)
    loss_fn.to(DEVICE)
    metric_fn.to(DEVICE)

    loss_value_list = []
    metric_value_list = []

    loss_value_list_val = []
    metric_value_list_val = []

    for i in range(0, EPOCHS):

        model.train()

        with tqdm.tqdm(train_loader) as iterator:

            for x, y in iterator:
                x, y = x.to(DEVICE), y.to(DEVICE)

                optimizer.zero_grad()
                prediction = model.forward(x)
                loss = loss_fn(prediction, y)
                loss.backward()
                optimizer.step()

                # update loss logs
                loss_value_list.append(loss.cpu().detach().numpy())

                # update metrics logs
                metric_value_list.append(metric_fn(prediction, y).cpu().detach().numpy())

            print(f"Last Loss after epoch = {loss_value_list[-1]}")
            print(f"Last iou after epoch = {metric_value_list[-1]}")


        model.eval()

        for x, y in valid_loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                prediction = model.forward(x)
                loss = loss_fn(prediction, y)

            # update loss logs
            loss_value_list_val.append(loss.cpu().detach().numpy())

            # update metrics logs
            metric_value_list_val.append(metric_fn(prediction, y).cpu().detach().numpy())
        

        # Save model if a better val IoU score is obtained
        if best_iou_score < np.mean(metric_value_list_val):
            best_iou_score = np.mean(metric_value_list_val)
            torch.save(model, './best_model.pth')
            print('Model saved!')










