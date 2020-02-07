# Imports
import argparse
import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch

from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import seaborn as sb
import json

# Get arguments
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('image_path', default='./flowers/test/1/image_06743.jpg', type=str)
    parser.add_argument('checkpoint', default='image_classifier_checkpoint.pth', type=str)
    parser.add_argument('--top_k', default=5, type=int)
    parser.add_argument('--category_names', default='cat_to_name.json', type=str)
    parser.add_argument('--gpu', default=True, type=bool)

    return parser.parse_args()

# load saved model
def load_image_classifier(filepath, device):

    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)

    # Freeze Parameters
    for param in model.parameters():
        param.requires_grad = False
    #    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu1', nn.ReLU()),
                          ('drop1', nn.Dropout(0.6)),
                          ('fc2', nn.Linear(4096, 102)),
                           ('output', nn.LogSoftmax(dim=1))
                          ]))

    model.classifier = classifier
    model.classifier.optimizer = checkpoint['optimizer']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Process a PIL image for use in a PyTorch model
    # Open the image
    image = Image.open(image_path)

    # Resize the image
    image = image.resize((256,256))

    # Crop the image
    image = image.crop((0,0,224,224))

    # Get the color channels
    image = np.array(image)/255

    # Normalize the images
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    np_image = (image - means) / std

    # Transpose the colors
    np_image = np_image.transpose((2, 0, 1))

    return np_image

def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    model.to(device)
    # Predict the class from an image file
    image = process_image(image_path)
    torch_image = torch.from_numpy(np.expand_dims(image, axis=0)).float().to(device)


    with torch.no_grad():
        output = model(torch_image)

    probs, classes = output.topk(topk)
    probs = np.array(probs.exp().data)[0]
    classes = np.array(classes)[0]

    # Reverse the categories dictionary
    idx_to_class = {v:k for k, v in model.class_to_idx.items()}

    names = []
    for c in classes:
        names.append(cat_to_name[idx_to_class[c]])

    return probs, names

if __name__ == '__main__':
    # get arguments from command line
    args = get_args()

    image_path = args.image_path
    checkpoint = args.checkpoint
    topk = args.top_k
    cat_names = args.category_names
    gpu = args.gpu

    device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu) else "cpu")

    # load category names file
    with open(cat_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_image_classifier(checkpoint, device)

    probs, names = predict(image_path, model, topk, device)
    for i in range(topk):
        print("Flower {}, {:.4f}".format(names[i], probs[i]))
