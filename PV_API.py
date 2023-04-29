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

#######configuration############
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_dict = pd.read_csv("label_class_dict.csv")
print(class_dict)
# Get class names
class_names = class_dict['name'].tolist()
# Get class RGB values
class_rgb_values = class_dict[['r','g','b']].values.tolist()

select_classes = ['Roof','Ridge','Obstacle','Other']

# Get RGB values of required classes
select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]

select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]

best_model = None
if os.path.exists('best_model.pth'):
    best_model = torch.load('best_model.pth', map_location=DEVICE)
    # print('Loaded Model.')

# preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

#RGB Image
image = cv2.cvtColor(cv2.imread("data/test/2.png"), cv2.COLOR_BGR2RGB)
image = to_tensor(image)


if best_model is not None:
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    # Predict test image
    pred_mask = best_model(x_tensor)
    pred_mask = pred_mask.detach().squeeze().cpu().numpy()
    # Convert pred_mask from `CHW` format to `HWC` format
    pred_mask = np.transpose(pred_mask,(1,2,0))
    pred_mask = colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values).astype("uint8")
    cv2.imwrite("predicted_mask.png", cv2.cvtColor(pred_mask,cv2.COLOR_RGB2BGR))


    # visualize(
    #     original_image = image.transpose(1,2,0).astype('uint8'),
    #     predicted_mask = pred_mask,
    # )