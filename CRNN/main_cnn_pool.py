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
import shutil
import pickle

#from cnn_rnn_siamese import *
from cnn_pool import *
from data_prepare import *

import time
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5],[0.1,0.1,0.1])])


data_path= "/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=raw_data/TPOT/Video_Data/normalized_raw_final/"

save_result_dir = "/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=raw_data/TPOT/Video_Data/Sanchayan2/cnn_pool/result_test/cnn_pool1/"
# training parameters
k = 2             # number of target category
epochs = 7        # training epochs
batch_size = 32
learning_rate = [0.05,0.01,0.005,0.001,0.0005,0.0001,0.00001]
log_interval = 10   # interval for displaying training info

# for CRNN
# EncoderCNN architecture
CNN_fc_hidden1, CNN_fc_hidden2 = 1024 ,  768
CNN_embed_dim = 300      # latent dim extracted by 2D CNN
img_x, img_y = 200, 200  # resize video 2d frame size
dropout_p = 0.0          # dropout probability
dropout_list=[0.0,0.3,0.5,0.6,0.8]

# DecoderRNN architecture
RNN_hidden_layers = 1
RNN_hidden_nodes = 512
RNN_FC_dim = 256


#Siamese Architecture
siamese_fc1=256

def convert_bstring_to_normal(inputlist):

    outputlist=[]
    for elements in inputlist:

        if isinstance(elements,np.bytes_):
            outputlist.append(elements.decode('utf-8'))
        else:

            outputlist=inputlist
    
    return outputlist

def train(log_interval, model, device, train_loader, optimizer, epoch,position_vector):
    # set model as training mode
    #cnn_encoder, rnn_decoder = model
    cnn_encoder=model
    #cnn_encoder,rnn_decoder,siamese_net= model
    cnn_encoder.train()
    #rnn_decoder.train()
    #siamese_net.train()

    losses = []
    scores = []
    N_count = 0   # counting total trained sample in one epoch
    all_prob=[]
    
    for batch_idx, (X1, y) in enumerate(train_loader):
        # distribute data to device
        
        #X1, X2, y = X1.to(device), X2.to(device),y.to(device).view(-1, )
        X1, y = X1.to(device), y.to(device).view(-1,)
            
        N_count += X1.size(0)

        optimizer.zero_grad()
        
        #output_cnn_rnn_1,output_cnn_rnn_2= rnn_decoder(cnn_encoder(X1)), rnn_decoder(cnn_encoder(X2))
           
        #output= siamese_net(output_cnn_rnn_1,output_cnn_rnn_2)
        #output=rnn_decoder(cnn_encoder(X1))
       
        output= cnn_encoder(X1)
        #output = rnn_decoder(cnn_encoder(X))   # output has dim = (batch, number of classes)

        #p=torch.nn.functional.softmax(output, dim=1)
        #print (p)
        
        
        #print (output.shape)
        loss = F.cross_entropy(output, y)
        #print (loss)
        
        losses.append(loss.item())
        #print (loss.item())
        #print (loss)
        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        
        #print (y_pred.shape,y.shape)
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy()) if len(y_pred) > 1 else y_pred.cpu().data==y.cpu().data
        scores.append(step_score)         # computed on CPU
        
        loss.backward()
        optimizer.step()
        #print (all_prob)



        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))

   

    return losses, scores

def validation(model, device, optimizer, test_loader, epoch, save_model_path, position_vector):
    # set model as testing mode
    #cnn_encoder, rnn_decoder,siamese_net = model
    #cnn_encoder, rnn_decoder = model
    cnn_encoder=model
    cnn_encoder.eval()
    #rnn_decoder.eval()
    
    #siamese_net.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    all_prob=[]
    with torch.no_grad():
        for batch_idx, (X1, y) in enumerate(test_loader):
            # distribute data to device
            #X1,X2, y = X1.to(device), X2.to(device),y.to(device).view(-1, )
            X1, y = X1.to(device) , y.to (device). view (-1,)
            #output_cnn_rnn_1,output_cnn_rnn_2= rnn_decoder(cnn_encoder(X1)), rnn_decoder(cnn_encoder(X2))
           
            #output= siamese_net(output_cnn_rnn_1,output_cnn_rnn_2)

            #output = rnn_decoder(cnn_encoder(X1))
            output= cnn_encoder(X1)
            #-----Computing probabilty distribution across classes. Keep in mind this is not computing the loss 

            prob=torch.nn.functional.softmax(output, dim=1)
            
            
            loss = F.cross_entropy(output, y, reduction='sum') #This function directly calculates the log_probabailties and the classes and the loss from that. 
            
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability
           
           
            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)
            all_prob.extend (prob)

            

    test_loss /= len(test_loader.dataset)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    all_prob=torch.stack(all_prob,dim=0).cpu().numpy()


    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))
    

    
    
    #torch.save({'epoch':epoch,\
    #            'model_state_dict':cnn_encoder.state_dict(),\
    #            }, os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    #torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    
   
    
    # save Pytorch models of best record
    torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    #torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    #torch.save(siamese_net.state_dict(), os.path.join(save_model_path, 'siamese_net_epoch{}.pth'.format(epoch + 1)))
    torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    print("Epoch {} model saved!".format(epoch + 1))

    result={}
    result['valid_score']=test_score
    result['valid_loss']=test_loss
    result['probs']=all_prob

    #torch.save(test_score,os.path.join(save_result_path,'score{}.txt'.format(test_score)))
    #torch.save(test_loss, os.path.join(save_result_path,'loss{}.txt'.format(test_loss)))

    save_name=save_model_path+'result_{}.npy'.format(epoch+1)
    np.save(save_name,result)

    return test_loss, test_score

def siamese_pickup(list1,list2,length1,length2):

    list1=np.array(list1)
    list2=np.array(list2)

    task_list_1_child=np.empty((0))
    task_list_2_child=np.empty((0))
    task_list_1_mother=np.empty((0))
    task_list_2_mother=np.empty((0))

    labels=[]
    length1_task1=[]
    length2_task1=[]
    length1_task2=[]
    length2_task2=[]

    for i in range (len(list1)):
        if list2[i][6:8] == '01':
            task_list_1_child=np.hstack([task_list_1_child, list2[i]]) if task_list_1_child.size else list2[i]
            length2_task1.append(length2[i])
        else:
            task_list_2_child=np.hstack([task_list_2_child,list2[i]]) if task_list_2_child.size else list2[i] 
            length2_task2.append(length2[i])

        if list1[i][6:8] == '01':
            task_list_1_mother=np.hstack([task_list_1_mother, list1[i]]) if task_list_1_mother.size else list1[i]
            length1_task1.append(length1[i])
        else:
            task_list_2_mother=np.hstack([task_list_2_mother,list1[i]]) if task_list_2_mother.size else list1[i] 
            length1_task2.append(length1[i])
  
    assert len(task_list_1_child) ==  len(task_list_1_mother) and  (len(task_list_2_child) == len(task_list_2_mother)) 
    
    new_list=[]
    length1_list=[]
    length2_list=[]
    for i in range (len(task_list_2_mother)):
        new_list=new_list+[[task_list_2_mother,task_list_2_child[i]]]
        length1_list.append(length1_task2[i])
        length2_list.append(length2_task2[i])
        labels.append(0)
      

    for i in range (len(task_list_1_mother)):
        new_list=new_list+[[task_list_1_mother,task_list_1_child[i]]]
        labels.append(0)
        length1_list.append(length1_task1[i])
        length2_list.append(length2_task1[i])
        
    #new_list=np.array(new_list)

    list_task1=[]
    for i in range (len(task_list_1_mother)):
        x=random.randint(0,len(task_list_1_mother)-1)

        while True:
            if x != i:
                list_task1=list_task1+[[task_list_1_mother[i],task_list_1_child[x]]]
                labels.append(1)
                length1_list.append(length1_task1[i])
                length2_list.append(length2_task1[x])
                break
            x=random.randint(0,len(task_list_1_mother)-1)


    list_task2=[]
    for i in range (len(task_list_2_mother)):
        x=random.randint(0,len(task_list_2_mother)-1)
        
        while True:
            if x != i:
                list_task2=list_task2+[[task_list_2_mother[i],task_list_2_child[x]]]
                labels.append(1)
                length1_list.append(length1_task2[i])
                length2_list.append(length2_task2[x])
                break
            x=random.randint(0,len(task_list_2_mother)-1)
    new_list=new_list+list_task1+list_task2
        
    assert len(new_list) == len(labels) == len(length1_list) == len(length2_list)
    

    return new_list,labels,length1_list,length2_list
    


# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda:0,1" if use_cuda else "cpu")   # use CPU or GPU

cores=2

# Data loading parameters
train_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': cores, 'pin_memory': True} if use_cuda else {}

valid_params = {'batch_size': batch_size, 'shuffle': False,'num_workers': cores, 'pin_memory': True} if use_cuda else {}
#phq_data=np.load('phq_final.npy',allow_pickle=True).item()

train_data=np.load('./kfold_full/train_data_reliable_1.npy',allow_pickle=True).item()
valid_data=np.load('./kfold_full/test_data_reliable_1.npy',allow_pickle=True).item()
test_data=np.load('./kfold_full/test_data_reliable_0.npy',allow_pickle=True).item()


train_mother_name,train_child_name,train_label_list,train_start_list,train_end_list=np.array(train_data['mother_name']),\
                                                                                    np.array(train_data['child_name']),\
                                                                                    np.array(train_data['label']),\
                                                                                    np.array(train_data['start_list']),\
                                                                                    np.array(train_data['end_list'])


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



'''
mother,child,score,mother_length,child_length=phq_data['name1'],phq_data['name2'],\
                        phq_data['score'],phq_data['mother_length'],\
                        phq_data['child_length']
'''
#print (train_mother_name[:10],valid_mother_name[:10])

print ("Training samples",len(train_mother_name),len(train_child_name),len(train_label_list))
print ("Validaton samples",len(valid_mother_name),len(valid_child_name),len(valid_label_list))
print ("Testing samples",len(test_mother_name),len(test_child_name),len(test_label_list))

'''
train_indices=np.array([train_mother_name[i][6:8] == '01' for i in range (len(train_mother_name)) ])


train_mother_name,train_child_name,train_label_list,train_start_list,train_end_list=train_mother_name[train_indices],\
                                                                                train_child_name[train_indices],\
                                                                                train_label_list[train_indices],\
                                                                                train_start_list[train_indices],\
                                                                                train_end_list[train_indices]



assert (train_mother_name.shape) == (train_child_name.shape) == train_label_list.shape == train_start_list.shape == train_end_list.shape 

valid_indices=np.array([valid_mother_name[i][6:8] == '01' for i in range (len(valid_mother_name)) ])
valid_mother_name,valid_child_name,valid_label_list,valid_start_list,valid_end_list=valid_mother_name[valid_indices],\
                                                                                valid_child_name[valid_indices],\
                                                                                valid_label_list[valid_indices],\
                                                                                valid_start_list[valid_indices],\
                                                                                valid_end_list[valid_indices]


assert valid_mother_name.shape == valid_child_name.shape == valid_label_list.shape == valid_start_list.shape == valid_end_list.shape

test_indices=np.array([test_mother_name[i][6:8] == '01' for i in range (len(test_mother_name)) ])
test_mother_name,test_child_name,test_label_list,test_start_list,test_end_list=test_mother_name[test_indices],\
                                                                                test_child_name[test_indices],\
                                                                                test_label_list[test_indices],\
                                                                                test_start_list[test_indices],\
                                                                                test_end_list[test_indices]

assert test_mother_name.shape == test_child_name.shape == test_label_list.shape == test_start_list.shape == test_end_list.shape
'''


fail=[]
for names in train_mother_name:
    flag=0    
    if names in valid_mother_name:
        flag=1
    if flag == 1:
        fail.append(names)
   


#---------------Convert labels to 1 hot category-------------
# convert labels -> category

le = LabelEncoder()
le.fit(train_label_list)



# show how many classes there are
print(list(le.classes_))


# convert category -> 1-hot
train_category = le.transform(train_label_list).reshape(-1, 1)
enc = OneHotEncoder()
enc.fit(train_category)


valid_category=le.transform(valid_label_list).reshape(-1,1)
enc.fit(valid_category)
#print (valid_label_list[:10],valid_category[:10])

'''
test_category=le.transform(test_label_list).reshape(-1,1)
enc.fit(test_category)
'''
print (train_category)
#input ('')
#-----------------------------------------------------------------


# Compute the weight vector  for each of the classes . Change 5 to the number of classes for the problem
position_vector=np.zeros(5)
for i in range(position_vector.size):
    size=np.sum(train_category==i)/(train_category.size)
    position_vector[i]=size

position_vector=torch.Tensor(position_vector).to(device)
print (position_vector)
#--------------------------------------------------------------------------



#-----------Using the data loaders. This shsould be same for every configuration---------------------------------------
train_set= Dataset_CRNN_new(data_path, train_mother_name, train_child_name , train_category, train_start_list, train_end_list, transform=transform)
valid_set=  Dataset_CRNN_new(data_path, valid_mother_name, valid_child_name , valid_category, valid_start_list, valid_end_list, transform=transform)
#test_set=  Dataset_CRNN_new(data_path, test_mother_name, test_child_name, test_category, test_start_list, test_end_list, transform=transform)



train_loader = data.DataLoader(train_set, **train_params)
valid_loader = data.DataLoader(valid_set, **valid_params)

#-------------------------------------------------------







#------------------------------GEnerating all possible combinations -------------------------------
configuration_list=[]

for dropout_values in dropout_list:
    for learn_rate in learning_rate:

            config= str(dropout_values) + "_" + str(learn_rate)
            configuration_list.append(config)

np.save('configuration.npy',configuration_list)

configuration_list=np.load('configuration.npy',allow_pickle=True)

configuration_list=configuration_list[:20]


#configuration_list=['0.3_0.0005','0.0_0.0001','0.6_0.0005','0.3_0.00001','0.3_0.005'] # Uncomment this for siamese
configuration_list=[#'0.3_0.00001_0.0',
                    '0.3_0.0001_0.0'
                    #'0.0_0.00001_0.0',
                    #'0.0_0.0001_0.0',
                    #'0.3_0.00001_0.9',
                    #'0.3_0.0001_0.9',
                    #'0.0_0.00001_0.9',
                    #'0.0_0.0001_0.9',
                    #'0.5_0.0001_0.0'
                    #'0.8_0.0001_0.0',
                    #'0.8_0.00001_0.0'
                    ] # Uncomment this for siamese

configuration_list=configuration_list[0:]
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 


print (configuration_list)

#---------------------------------------------------------------------------------------
#------------------TRAIN and VALIDATION is done here ---------------------

for config in configuration_list:
   
    dropout,learn_rate,decay_rate = config.split('_')


    dropout = float(dropout)
    learn_rate= float(learn_rate)
    decay_rate= float(decay_rate)

    # Create model
    cnn_encoder = EncoderCNN(img_x=img_x, img_y=img_y, fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2,
                         drop_p=dropout, CNN_embed_dim=CNN_embed_dim, num_classes=k).to(device)
    
    '''
    rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, 
                         h_FC_dim=RNN_FC_dim, drop_p=dropout, num_classes=k).to(device)
    ''' 
    #siamese_net = Siamese ( cnn_rnn_embedding= RNN_FC_dim*2 ,hidden_layer=siamese_fc1, num_classes=k).to(device)

    # Parallelize model to multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        cnn_encoder = nn.DataParallel(cnn_encoder,device_ids=[0,1])
        #rnn_decoder = nn.DataParallel(rnn_decoder,device_ids=[0,1])
        #siamese_net = nn.DataParallel(siamese_net,device_ids=[0,1,2,3])

    crnn_params = list(cnn_encoder.parameters()) #+ list(rnn_decoder.parameters()) #+list(siamese_net.parameters())
    optimizer = torch.optim.Adam(crnn_params, lr=learn_rate,weight_decay=decay_rate)


    # record training process
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_test_losses = []
    epoch_test_scores = []

    save_folder=save_result_dir+config+'/'
    
    print (save_folder)
    
    #------------Detecting and creating folder of format 0.3_0.00001_0.0 (dropout,learning rate,weight decay)-----------#
    
    if os.path.isdir (save_folder):
        shutil.rmtree(save_folder)
    os.mkdir (save_folder)
    
    # start training
    for epoch in range(epochs):
        # train, test model
        model = cnn_encoder
        #model=[cnn_encoder,rnn_decoder]
        #model=[cnn_encoder,rnn_decoder,siamese_net]
        start=time.time()
        train_losses, train_scores = train(log_interval, model, device, train_loader, optimizer, epoch,position_vector)
        epoch_test_loss, epoch_test_score = validation(model, device, optimizer, valid_loader, epoch , save_folder,  position_vector)
        print ("Time elapsed for training the model: %4.2f" %(time.time()-start))
        # save results
        epoch_train_losses.append(train_losses)
        epoch_train_scores.append(train_scores)
        epoch_test_losses.append(epoch_test_loss)
        epoch_test_scores.append(epoch_test_score)

        # save all train test results
        A = np.array(epoch_train_losses)
        B = np.array(epoch_train_scores)
        C = np.array(epoch_test_losses)
        D = np.array(epoch_test_scores)

       



    # Have to change this part 
    np.save(os.path.join(save_folder,'CRNN_epoch_training_losses.npy'), A)
    np.save(os.path.join(save_folder,'CRNN_epoch_training_scores.npy'), B)
    np.save(os.path.join(save_folder,'CRNN_epoch_test_losses.npy'), C)
    np.save(os.path.join(save_folder,'CRNN_epoch_test_scores.npy'), D)

        
        
    
print ('Its done')




#-----------------------------------------------------Script for training is done----------------------------------------
