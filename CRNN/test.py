import os
import random
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

from cnn_rnn import *
from data_prepare import *


import time

data_path= "/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=raw_data/TPOT/Video_Data/normalized_raw_frames/"

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5],[0.1,0.1,0.1])])
# training parameters
k = 5             # number of target category
epochs = 120        # training epochs
batch_size = 10
learning_rate = 1e-4
log_interval = 10   # interval for displaying training info

# for CRNN
# EncoderCNN architecture
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 300      # latent dim extracted by 2D CNN
img_x, img_y = 200, 200  # resize video 2d frame size
dropout_p = 0.0          # dropout probability


# DecoderRNN architecture
RNN_hidden_layers = 1
RNN_hidden_nodes = 512
RNN_FC_dim = 256


#Siamese Architecture
siamese_fc1=256
# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda:0,1,2,3" if use_cuda else "cpu")   # use CPU or GPU

# Data loading parameters

params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}

#phq_data=np.load('phq_final.npy',allow_pickle=True).item()
train_data=np.load('train_data.npy',allow_pickle=True).item()
valid_data=np.load('valid_data.npy',allow_pickle=True).item()
test_data=np.load('test_data.npy',allow_pickle=True).item()

train_mother_name,train_child_name,train_label_list,train_start_list,train_end_list=train_data['mother_name'],\
                                                                                    train_data['child_name'],\
                                                                                    train_data['label'],\
                                                                                    train_data['start_list'],\
                                                                                    train_data['end_list']


valid_mother_name,valid_child_name,valid_label_list,valid_start_list,valid_end_list=valid_data['mother_name'],\
                                                                                    valid_data['child_name'],\
                                                                                    valid_data['label'],\
                                                                                    valid_data['start_list'],\
                                                                                    valid_data['end_list']
test_mother_name,test_child_name,test_label_list,test_start_list,test_end_list=test_data['mother_name'],\
                                                                                    test_data['child_name'],\
                                                                                    test_data['label'],\
                                                                                    test_data['start_list'],\
                                                                                    test_data['end_list']


print ("Testing samples",len(test_mother_name),len(test_child_name),len(test_label_list))

#print (test_mother_name[:5],test_child_name[:5],test_start_list[:5])




train_category=np.array(train_label_list)
test_category=np.array(test_label_list)




# Compute the weight vector  for each of the classes . Change 5 to the number of classes for the problem
position_vector=np.zeros(5)
for i in range(position_vector.size):
    size=np.sum(train_category==i)/(train_category.size)
    position_vector[i]=size

position_vector=torch.Tensor(position_vector).to(device)


test_set=Dataset_CRNN_new(data_path, test_mother_name, test_child_name, test_category, test_start_list, test_end_list, transform=transform)
test_loader=data.DataLoader(test_set, **params)
# Create model
cnn_encoder = EncoderCNN(img_x=img_x, img_y=img_y, fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2,
                         drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)

rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, 
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p).to(device)

siamese_net = Siamese ( cnn_rnn_embedding= RNN_FC_dim*2 ,hidden_layer=siamese_fc1, num_classes=k).to(device)



if torch.cuda.device_count()> 1:
    cnn_encoder = nn.DataParallel(cnn_encoder,device_ids=[0,1,2,3])
    rnn_decoder = nn.DataParallel(rnn_decoder,device_ids=[0,1,2,3])
    siamese_net = nn.DataParallel(siamese_net,device_ids=[0,1,2,3])

checkpoint_cnn=torch.load('./model_path/cnn_encoder_epoch1.pth')
cnn_encoder.load_state_dict(checkpoint_cnn)


checkpoint_rnn=torch.load('./model_path/rnn_decoder_epoch1.pth')
rnn_decoder.load_state_dict(checkpoint_rnn)

checkpoint_siamese=torch.load('./model_path/siamese_net_epoch1.pth')
siamese_net.load_state_dict(checkpoint_siamese)



#Testing time..................

total=0
cnn_encoder.eval()
rnn_decoder.eval()
siamese_net.eval()
final_list=[]
with torch.no_grad():


     for batch_idx, (X1,X2, y) in enumerate(test_loader):

        X1,X2, y = X1.to(device), X2.to(device),y.to(device).view(-1, )
       
       	print ("Score:",y)

        output_cnn_rnn_1,output_cnn_rnn_2= rnn_decoder(cnn_encoder(X1)), rnn_decoder(cnn_encoder(X2))
        output= siamese_net(output_cnn_rnn_1,output_cnn_rnn_2)

        print (output)
        _, predicted = torch.max(output.data,1)

        print (predicted)

        total +=X1.size(1)
        
        final_list=final_list+list(predicted.cpu().numpy().ravel())

assert len(test_label_list)  == len(final_list)


print ('-----Saving the segment level -----')

np.save('final_list_test0.npy',final_list)