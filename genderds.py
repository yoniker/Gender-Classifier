import os
import pandas as pd
import scipy.misc as sc
from torchvision import transforms
import torch
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt


from skimage import transform as sktransform
from PIL import Image
from torch.utils.data import Dataset

class MyRescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int))
        self.output_size = output_size

    def __call__(self, sample):
        # image, landmarks = sample['image'], sample['landmarks']

        # h, w = image.shape[:2]
        # if isinstance(self.output_size, int):
        #     if h > w:
        #         new_h, new_w = self.output_size * h / w, self.output_size
        #     else:
        #         new_h, new_w = self.output_size, self.output_size * w / h
        # else:
        new_h, new_w = self.output_size,self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img =sample.resize((new_h, new_w), Image.ANTIALIAS)

        return img


def imshow(img,normalized_pic=False,transposed_pic=False):
    if normalized_pic==True:
        img=img.float()
        img = img / 2 + 0.5     # unnormalize
    npimg = img
    if transposed_pic==True:
        npimg=np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)
    plt.show()



class MyColorPlay(object):
    """Jitter the color of an image

    """

    def __init__(self):
        pass

    def __call__(self, sample):
        # image, landmarks = sample['image'], sample['landmarks']

        # h, w = image.shape[:2]
        # if isinstance(self.output_size, int):
        #     if h > w:
        #         new_h, new_w = self.output_size * h / w, self.output_size
        #     else:
        #         new_h, new_w = self.output_size, self.output_size * w / h
        # else:


        np_sample=np.array(sample)
        np_hsv=matplotlib.colors.rgb_to_hsv(np_sample)

        #TODO: Play with the colors in some simple way

        np_sample=matplotlib.colors.hsv_to_rgb(np_hsv)
        sample = Image.fromarray(np.uint8(np_sample))

        return sample


            
class GenderDataset(Dataset):
    ALL='all'
    TRAINING='training'
    VALIDATION='validation'
    TEST='test'
    #DB_BASE_DIRECTORY='YoniDB' #default value for where the database is located in respect to the current directory
    GENDER='gender'
    IMAGE='image'
    #ratios of the training/validation/test subsets within the original database
    TRAINING_RATIO=0.6
    VALIDATION_RATIO=0.2
    TEST_RATIO=0.2
    CLASSES=['f','m']
    FEMALE=0
    MALE=1

    @staticmethod
    def label_to_index(gender):
        if gender==GenderDataset.CLASSES[0]:
            return 0
        return 1
    @staticmethod
    def sample_to_tensor(data):
        inputs=data[GenderDataset.IMAGE]
        char_labels=data[GenderDataset.GENDER]
        labels=[GenderDataset.label_to_index(l) for l in char_labels]
        labels=torch.LongTensor(labels)
        return inputs,labels

    def __init__(self,dbsInfo,the_type=ALL,transform=None):
        if the_type==self.TRAINING:
            self.info=dbsInfo.iloc[0:round(self.TRAINING_RATIO*len(dbsInfo))]
        elif the_type==self.VALIDATION:
            self.info=dbsInfo.iloc[round(self.TRAINING_RATIO*len(dbsInfo)):round(self.TRAINING_RATIO*len(dbsInfo))+round(self.VALIDATION_RATIO*len(dbsInfo))]
        elif the_type==self.TEST:
            self.info=dbsInfo.iloc[round(self.TRAINING_RATIO*len(dbsInfo))+round(self.VALIDATION_RATIO*len(dbsInfo)):len(dbsInfo)]
        else: #I'm going to assume the type is all (alternative is asserting it)
            self.info=dbsInfo
        self.the_type=the_type
        self.transform=transform
    
    def __len__(self):
        return len(self.info)
                

    def __getitem__(self, idx):

        d={}
        d[self.GENDER]=self.info.iloc[idx][self.GENDER]
        the_file=self.info.iloc[idx]['image']
        d[self.IMAGE]=sc.imread(the_file)
        if self.transform:
            d[self.IMAGE]=self.transform(d[self.IMAGE])

        return d
        
     
class UnionDatasets(Dataset):
    def __init__(self,dataset1,dataset2):
        self.dataset1=dataset1
        self.dataset2=dataset2
    
    def __len__(self):
        return len(self.dataset1)+len(self.dataset2)
                

    def __getitem__(self, idx):
        if idx<len(self.dataset1):
            return self.dataset1[idx]
        return self.dataset2[idx-len(self.dataset1)]

        
# dbsInfo=pd.read_csv('fbdb.csv',quotechar='"')      
# thedb=FBDB(dbsInfo)


#a convenience function

def show(thedb,index):
    sample=thedb[index]
    print('Index is {} gender is {}'.format(str(index),sample['gender']))
    sc.imshow(sample['image'])




# def count_gender():
    
    
#     males=0
#     females=0
    
#     for i in range(len(thedb)):
#         print('i is number {}\n'.format(i))
#         gender=thedb[i]['gender']
#         if gender=='m':
#             males+=1
#         elif gender=='f':
#             females+=1
#         #else:
#             #print('weird, gender is {}'.format(gender))
#             #import pdb
#             #pdb.set_trace()
#         print('number of males is {} while number of females is {}'.format(males,females))
#    