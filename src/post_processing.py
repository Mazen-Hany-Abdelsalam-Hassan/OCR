import random
import matplotlib.pyplot as plt
import torch
from torch import nn
from dataset import DatasetOCR
from torch.utils.data import DataLoader,Dataset
from src.config import DEVICE



def greedy_decoding(x):
    """This is not True Method It will not work well with all the data"""
    compressed = []
    if x[0] != 0:
        compressed.append(x[0]-1)
    prev = x[0]
    for i in range( 1 , len(x)):
        if x[i] !=0 and x[i]!=prev:
            compressed.append(x[i]-1)
            prev = x[i]
        if x[i] ==0:
            prev=0
    #return compressed
    return ''.join(str(x) for x in compressed)


def plot_the_image(test:list , size:int , model:nn.Module ):
    sample = random.sample(test, k = size )
    titles = return_the_prediction(model=model,Sample=sample )

    images = [plt.imread(img) for img in sample]
    fig, axs = plt.subplots(4, 4, figsize=(10, 10))
    # Flatten the axs array for easy iteration
    axs = axs.flatten()

    for i in range(len(images)):

        axs[i].imshow(images[i])  # Display image
        axs[i].set_title(titles[i])  # Set title
        axs[i].axis('off')  # Hide axes for a cleaner view

    plt.tight_layout()  # Adjust layout
    plt.show()


def return_the_prediction(model:nn.Module,Sample:list):
    data = DataLoader(DatasetOCR(x=Sample,y = [5] * len(Sample) , for_what="Test") , batch_size=len(Sample))
    for images, _ in data:
        images = images.to(DEVICE)
    model.eval()
    with torch.no_grad():
        model.to(DEVICE)
        prob = model(images)
        res = prob.argmax(axis = 2).detach().numpy().tolist()
    dec = [greedy_decoding(i) for i in res]
    return dec


