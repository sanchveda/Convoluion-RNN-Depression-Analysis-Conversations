import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm


class Dataset_CRNN_new(data.Dataset):

    def __init__(self,data_path,folders1,folders2,labels,start,end, transform=None):
        "Initialization with the various parameters"
        self.data_path=data_path
        self.folders1=folders1
        self.folders2=folders2
        self.labels=labels
        self.transform=transform
        self.start=start
        self.end=end
        
        assert len(self.folders1) == len(self.folders2) == len (self.labels) == len(self.start) == len(self.end)

    def __len__(self):
        "Denotes the number of samples "
        return len(self.folders1)

    def read_images(self, path, selected_folder, use_transform,index):
        X = []
        start,end =self.start[index], self.end[index]
        

        
        for i in range(start,end,10):
           
            #print (i)
            #print (os.path.join(path,selected_folder,'%s_%05d.jpg' %(selected_folder,i)))
            
            image = Image.open(os.path.join(path,selected_folder,'%s_%s_%05d.jpg' %(selected_folder,"norm",i)))
            
            if use_transform is not None:
                image = use_transform(image)

            #print (image.shape)

            X.append(image)
        

        
        X = torch.stack(X, dim=0)

        #Reading 28 frames and not hundreds of frames
        
        return X

    def __getitem__(self,index):

        folder1=self.folders1[index]
        folder2=self.folders2[index]


        X1 = self.read_images(self.data_path, folder1, self.transform,index)
        X2 = self.read_images(self.data_path, folder2, self.transform,index) #Only to be used for a twin-network setup
        #print (X1.shape,X2.shape)
        #input ('')
        y = torch.LongTensor([self.labels[index]])

        return X1,X2,y
        #return X1,y


class Dataset_CRNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, data_path, folders1, folders2, labels, mother_length, child_length, frames, transform=None):
        "Initialization"
        self.data_path = data_path
        self.labels = labels
        #self.folders1 = folders1 #train_list becomes folder here
        #self.folders2 = folders2
        self.transform = transform
        self.frames = frames
        self.mother_length= mother_length
        self.child_length = child_length

        self.indices=np.empty((0,2))
        self.folders1=np.empty((0))
        self.folders2=np.empty((0))
        count=0

        for i in range (len(folders1)):

            indice=np.array(segment_real(mother_length[i],2.5,mode='slice',window_size=30))
            temp1= np.full(len(indice),folders1[i])
            temp2= np.full(len(indice),folders2[i])

          

            count=len(indice)+count
            #print(indice.shape,folders1[i],mother_length[i],child_length[i])
            
            self.indices=np.vstack([self.indices,indice]) if self.indices.size else indice
            self.folders1=np.hstack([self.folders2,temp1]) if self.folders1.size else temp1
            self.folders2=np.hstack([self.folders2,temp2]) if self.folders2.size else temp2
                      
        #self.indices=torch.Tensor(self.indices).long().to(device)
      
         
    def __len__(self):
        "Denotes the total number of samples"
        return len(self.indices)

    def read_images(self, path, selected_folder, use_transform,index):
        X = []
        
        print(selected_folder)

        for i in range(start,end):
            print (i)
            print (os.path.join(path,selected_folder,'%s_%05d.jpg' %(selected_folder,i)))
            input ('')
            image = Image.open(os.path.join(path,selected_folder,'%s_%05d.jpg' %(selected_folder,i)))

            if use_transform is not None:
                image = use_transform(image)

            print (image.shape)
            X.append(image)
        X = torch.stack(X, dim=0)
        #Reading 28 frames and not hundreds of frames
        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder1 = self.folders1[index]
        
        # Load data
        X = self.read_images(self.data_path, folder1, self.transform,index)     # (input) spatial images
        y = torch.LongTensor([self.labels[index]])                  # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return X, y