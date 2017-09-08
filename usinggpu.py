import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from genderds import GenderDataset,UnionDatasets,MyRescale,MyColorPlay
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

TRAIN_BATCH_SIZE=4
VALID_BATCH_SIZE=4
TEST_BATCH_SIZE=4

MAX_POOL='max'
PIC_SIZE=128
DATASETS_AVAILABLE=False
fb_transform=transforms.Compose([transforms.ToPILImage(),MyRescale(PIC_SIZE),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

if DATASETS_AVAILABLE==True:



    facebookInfo=pd.read_csv('fbdb.csv')
    fb_training=GenderDataset(facebookInfo,GenderDataset.TRAINING,transform=fb_transform)
    fb_validation=GenderDataset(facebookInfo,GenderDataset.VALIDATION,transform=fb_transform)
    fb_test=GenderDataset(facebookInfo,GenderDataset.TEST,transform=transform)

    facesInfo=pd.read_csv('normalized_facesdb.csv')
    faces_training=GenderDataset(facesInfo,GenderDataset.TRAINING,transform=fb_transform)
    faces_validation=GenderDataset(facesInfo,GenderDataset.VALIDATION,transform=fb_transform)
    faces_test=GenderDataset(facesInfo,GenderDataset.TEST,transform=fb_transform)



    wikiInfo=pd.read_csv('normalized_wiki.csv')
    wiki_training=GenderDataset(wikiInfo,GenderDataset.TRAINING,transform=fb_transform)
    wiki_validation=GenderDataset(wikiInfo,GenderDataset.VALIDATION,transform=fb_transform)
    wiki_test=GenderDataset(wikiInfo,GenderDataset.TEST,transform=fb_transform)

    fb_girls_info=pd.read_csv('fbgirls.csv')
    fb_girls_training=GenderDataset(fb_girls_info,GenderDataset.TRAINING,transform=fb_transform)
    fb_girls_validation=GenderDataset(fb_girls_info,GenderDataset.VALIDATION,transform=fb_transform)
    fb_girls_test=GenderDataset(fb_girls_info,GenderDataset.TEST,transform=fb_transform)



    trainingSet=UnionDatasets(fb_girls_training,UnionDatasets(UnionDatasets(fb_training,faces_training),wiki_training))
    validationSet=UnionDatasets(fb_girls_validation,UnionDatasets(UnionDatasets(fb_validation,faces_validation),wiki_validation))
    testSet=UnionDatasets(fb_girls_test,UnionDatasets(UnionDatasets(fb_test,faces_test),wiki_test))

    trainloader = torch.utils.data.DataLoader(trainingSet, batch_size=TRAIN_BATCH_SIZE,
                                          shuffle=True, num_workers=4)

    validationloader=torch.utils.data.DataLoader(validationSet, batch_size=VALID_BATCH_SIZE,
                                          shuffle=False, num_workers=4)
    testloader = torch.utils.data.DataLoader(testSet, batch_size=TEST_BATCH_SIZE,
                                         shuffle=False, num_workers=4)


def imshow(img,normalized_pic=True,transposed_pic=True):
    if normalized_pic==True:
        img=img.float()
        img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if transposed_pic==True:
        npimg=np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)
    plt.show()



# facebookLoader=torch.utils.data.DataLoader(facebookSet, batch_size=TEST_BATCH_SIZE,
#                                          shuffle=False, num_workers=4)





#Get validation accuracy on all three different datasets-another way of seeing if it overfitted a particular dataset

def val_by_ds(net):

    for val_set in [fb_validation,faces_validation,wiki_validation,fb_girls_validation]:
        validationloader=torch.utils.data.DataLoader(val_set, batch_size=VALID_BATCH_SIZE,
                                              shuffle=False, num_workers=4)
        (acc,loss,confusion_matrix)=get_validation_accuracy(net,dataLoader=validationloader)
        print('acc is {} and loss is {},confusion matrix is {}\n'.format(acc,loss,confusion_matrix))





import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self,convNumFiltersAndSizes):
        super(Net, self).__init__()
        print('initializing Net...')
        self.conv_description=convNumFiltersAndSizes
        self.convNets=torch.nn.ModuleList()
        PIC_CHANNELS=3
        #PIC_SIZE=32
        input_shape=(PIC_CHANNELS,PIC_SIZE,PIC_SIZE)
        prevNumFilters=PIC_CHANNELS
        for layer_description in self.conv_description:
            if isinstance(layer_description,tuple):
                (numFilters,filtersSize)=layer_description
                self.convNets.append(nn.Conv2d(prevNumFilters,numFilters,filtersSize,padding=int((filtersSize-1)/2)).cuda())
                prevNumFilters=numFilters
                input_shape=(numFilters,input_shape[1],input_shape[2])
            if isinstance(layer_description,str):
                if layer_description==MAX_POOL:
                    input_shape=(input_shape[0],input_shape[1]/2,input_shape[2]/2)
        final_input_size=int(input_shape[0]*input_shape[1]*input_shape[2])
        self.linearLayers=torch.nn.ModuleList()
        #For the time being, use a very particular feed forward net with 220 neurons,this will be optimized further
        self.linearLayers.append(nn.Linear(final_input_size,220))
        self.do1=torch.nn.Dropout(p=0.5)
        self.linearLayers.append(nn.Linear(220,2))
        self.do2=torch.nn.Dropout(p=0.5)
        self.train(True)

    def forward(self,x):
        conv_layer_number=0
        for layer_description in self.conv_description:
            if isinstance(layer_description,tuple):
                x=F.relu(self.convNets[conv_layer_number](x))
                conv_layer_number+=1
            if isinstance(layer_description,str):
                if layer_description==MAX_POOL:
                    x=F.max_pool2d(x,2)
        #For the time being,the linear part isn't part of the network's description
        x=x.view(-1,self.num_flat_features(x))   
        for i in range(len(self.linearLayers)-1):
            x=F.relu(self.linearLayers[i](x))
            x=self.do1(x)
        x=self.linearLayers[-1](x)
        x=self.do2(x) 

        return x
        
    def num_flat_features(self, x):
        sizes=x.size()[1:]
        total_size=1
        for size in sizes:
            total_size*=size
        return total_size



import torch.optim as optim
criterion= nn.CrossEntropyLoss()


#Inputs: predicted and actual labels
#output: a tuple (Correct M,Correct F,Mispredict M,Mispredict F)
def get_confusion_matrix(predicted,labels):
    predicted_f=(predicted==GenderDataset.FEMALE)
    labeled_f=(labels==GenderDataset.FEMALE)

    predicted_m=(predicted==GenderDataset.MALE)
    labeled_m=(labels==GenderDataset.MALE)


    correct_m=torch.sum(labeled_m*predicted_m)
    correct_f=torch.sum(labeled_f*predicted_f)

    mispredict_m=torch.sum(predicted_m*labeled_f)
    mispredict_f=torch.sum(predicted_f*labeled_m)

    return (correct_m,correct_f,mispredict_m,mispredict_f)


def get_validation_accuracy(net,l2_regularization,dataLoader):
        net.eval()
        correct = 0
        total = 0
        total_loss=0.0
        seen_examples=0.0
        confusion_matrix=np.array([0,0,0,0]) #(total correct_m,total correct_f,total mispredicted_m,total mispredicted_f)
        for (i,data) in enumerate(dataLoader,0):
            images,labels=GenderDataset.sample_to_tensor(data)
            outputs=net(Variable(images).cuda())
            _,predicted=torch.max(outputs,1)
            (correct_m,correct_f,mispredict_m,mispredict_f)=get_confusion_matrix(predicted.data,labels.cuda())
            confusion_matrix=confusion_matrix+(correct_m,correct_f,mispredict_m,mispredict_f)


            correct+=torch.sum(predicted.data==labels.cuda())
            total+=predicted.size()[0]
            loss=criterion(outputs,Variable(labels.cuda()))
            loss+=l2_regularization*regularization_loss(net)
            total_loss+=loss.data[0]
            seen_examples+=images.size()[0]
        net.train(True)
        return correct*1.0/total,total_loss/seen_examples,confusion_matrix






def regularization_loss(net):
        reg_loss=0
        for layer in net.linearLayers:
                for name, p in layer.named_parameters():
                        if not 'bias' in name:
                                reg_loss += torch.sum(p*p)
        return reg_loss

validation_loss=[]
validation_accuracy=[]
training_loss=[]
 

def save_net_train_info(net):
    torch.save(net.state_dict(), './net.model')
    log_file=open('train_log_file','w')
    log_file.write(str(validation_loss)+'\n')
    log_file.write(str(validation_accuracy)+'\n')
    log_file.write(str(training_loss))
    log_file.close()


#Loads the net saved
def load_net():
    net.load_state_dict(torch.load('./net.model'))

def train_net(net,epochs=5,lr=0.0003907119806717071,l2_regularization=1.2336278140431234e-09,record_stats=True):
        optimizer= optim.Adam(net.parameters(),lr=lr,weight_decay=0)
        seen_examples=0.0
        net.train(True)
        for epoch in range(epochs):
            running_loss=0.0
            for i,data in enumerate(trainloader):
                inputs,labels=GenderDataset.sample_to_tensor(data)
                inputs,labels=Variable(inputs.cuda()),Variable(labels.cuda())
                optimizer.zero_grad()
                outputs=net(inputs)
                seen_examples+=inputs.size()[0]
                loss=criterion(outputs,labels)
                loss+=l2_regularization*regularization_loss(net)
                loss.backward()
                optimizer.step()
                running_loss+=loss.data[0]
                if i%2000==1999:
                    running_loss=running_loss/(2000*TRAIN_BATCH_SIZE)
                    seen_examples=0.0
                    print('[%d,%5d] loss: %.3f' % (epoch+1,i+1,running_loss))
                    if record_stats==True:
                        (valid_acc,valid_loss,_)=get_validation_accuracy(net,l2_regularization)
                        validation_loss.append(valid_loss)
                        validation_accuracy.append(valid_acc)
                        training_loss.append(running_loss)
                        print('valid loss is {} and percents {}'.format(valid_loss,valid_acc))
                    running_loss=0.0
        print('Finished Training')




net=Net(convNumFiltersAndSizes=[(50,3),(40,3),MAX_POOL,(40,3),(39,3),MAX_POOL,(39,3),(39,3),MAX_POOL])

net.cuda()