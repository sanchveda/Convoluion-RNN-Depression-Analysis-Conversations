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
import os
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from data_prepare import *
from cnn_rnn import *

raw_dir='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=raw_data/TPOT/Video_Data/Sanchayan/result_nov18_cnn_rnn/'

data_path= "/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=raw_data/TPOT/Video_Data/normalized_raw_final/"

save_path='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=raw_data/TPOT/Video_Data/Sanchayan/test/test_nov18_cnn_rnn/'

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5],[0.1,0.1,0.1])])


valid_data=np.load('./data/valid_data_reliable.npy',allow_pickle=True).item()
train_data=np.load('./data/train_data_reliable.npy', allow_pickle=True).item()
test_data=np.load('./data/test_data_reliable.npy',allow_pickle=True).item()

 #EncoderCNN architecture

CNN_fc_hidden1, CNN_fc_hidden2 = 1024 ,  768
CNN_embed_dim = 300      # latent dim extracted by 2D CNN
img_x, img_y = 200, 200  # resize video 2d frame size
#dropout_p = 0.0          # dropout probability
#dropout_list=[0.0,0.3,0.5,0.6,0.8]

# DecoderRNN architecture
RNN_hidden_layers = 1
RNN_hidden_nodes = 512
RNN_FC_dim = 256


batch_size=11
k=2 # Number of classes

valid_mother_name,valid_child_name,valid_label_list,valid_start_list,valid_end_list=valid_data['mother_name'],\
                                                                                    valid_data['child_name'],\
                                                                                    valid_data['label'],\
                                                                                    valid_data['start_list'],\
                                                                                    valid_data['end_list']

train_mother_name,train_child_name,train_label_list,train_start_list,train_end_list=np.array(train_data['mother_name']),\
                                                                                    np.array(train_data['child_name']),\
                                                                                    np.array(train_data['label']),\
                                                                                    np.array(train_data['start_list']),\
                                                                                    np.array(train_data['end_list'])
test_mother_name,test_child_name,test_label_list,test_start_list,test_end_list=test_data['mother_name'],\
                                                                                    test_data['child_name'],\
                                                                                    test_data['label'],\
                                                                                    test_data['start_list'],\
                                                                                    test_data['end_list']
'''
train_indices=np.array([train_mother_name[i][6:8] == '02' for i in range (len(train_mother_name)) ])


train_mother_name,train_child_name,train_label_list,train_start_list,train_end_list=train_mother_name[train_indices],\
                                                                                train_child_name[train_indices],\
                                                                                train_label_list[train_indices],\
                                                                                train_start_list[train_indices],\
                                                                                train_end_list[train_indices]



assert (train_mother_name.shape) == (train_child_name.shape) == train_label_list.shape == train_start_list.shape == train_end_list.shape 

valid_indices=np.array([valid_mother_name[i][6:8] == '02' for i in range (len(valid_mother_name)) ])
valid_mother_name,valid_child_name,valid_label_list,valid_start_list,valid_end_list=valid_mother_name[valid_indices],\
                                                                                valid_child_name[valid_indices],\
                                                                                valid_label_list[valid_indices],\
                                                                                valid_start_list[valid_indices],\
                                                                                valid_end_list[valid_indices]


assert valid_mother_name.shape == valid_child_name.shape == valid_label_list.shape == valid_start_list.shape == valid_end_list.shape

test_indices=np.array([test_mother_name[i][6:8] == '02' for i in range (len(test_mother_name)) ])
test_mother_name,test_child_name,test_label_list,test_start_list,test_end_list=test_mother_name[test_indices],\
                                                                                test_child_name[test_indices],\
                                                                                test_label_list[test_indices],\
                                                                                test_start_list[test_indices],\
                                                                                test_end_list[test_indices]

assert test_mother_name.shape == test_child_name.shape == test_label_list.shape == test_start_list.shape == test_end_list.shape
'''
print (train_mother_name.shape,valid_mother_name.shape,test_mother_name.shape)



def test(model, device, optimizer, test_loader, save_model_path):


	cnn_encoder, rnn_decoder = model
	
	cnn_encoder.eval()
	rnn_decoder.eval()
	
	total=0
	final_list=[]

	all_y = []
	all_y_pred = []
	all_prob=[]
	with torch.no_grad():


		for batch_idx, (X1, y) in enumerate(test_loader):

			X1, y = X1.to(device), y.to(device).view(-1, )
			#X1,X2,y= X1.to (device), X2.to (device), y.to(device).view(-1,)
			print ("Score:",y)
			
			output = rnn_decoder(cnn_encoder(X1))
			prob=torch.nn.functional.softmax(output, dim=1)

			y_pred = output.max(1, keepdim=True)[1]
			
			print (prob)
			
			all_prob.extend (prob)

			all_y.extend(y)
			all_y_pred.extend(y_pred)
	all_prob=torch.stack(all_prob,dim=0).cpu().numpy()

	 #compute accuracy
	all_y = torch.stack(all_y, dim=0)
	all_y_pred = torch.stack(all_y_pred, dim=0)

	print ('-----Saving the segment level -----')

	test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())


	return all_prob,test_score
	#np.save('final_list_test0.npy',final_list)


def fusion_metric(valid_score,valid_loss,probs):

	x=np.argmax(probs,1)

	flag=0
	i=0
	count=0

	current_subject=valid_mother_name[0]
	fused_pred=np.empty(0)
	fused_confidence=np.empty((0))
	#print  (valid_mother_name[:23])
	
	fused_real=np.empty(0)
	video_tracker=0
	while True:

		if i == probs.shape[0]:
			break

		
		if valid_mother_name[i] == current_subject:
			#print (valid_mother_name[i],probs[i],i)
			
			#chunk_probs=np.vstack([chunk_probs,probs[i]]) if chunk_probs.size else probs[i]
			pass
		else:
			chunk_probs=probs[video_tracker:i]
			pred=np.argmax(chunk_probs,1)
			#confidence=np.max(chunk_probs,1)
			mode=stats.mode(pred)[0]
			fused_pred=np.vstack([fused_pred,mode]) if fused_pred.size else mode
			fused_real=np.vstack([fused_real,valid_label_list[video_tracker]]) if fused_real.size else valid_label_list[video_tracker]
			

			#print (valid_mother_name[:i],np.argmax(probs[:i],1),valid_label_list[:i])
			video_tracker=i
			count+=1
			current_subject=valid_mother_name[i]
			continue

		i+=1

	fused_pred=fused_pred.ravel().reshape(-1,1)
	fused_real=fused_real.ravel().reshape(-1,1)
	#print (fused_pred,fused_real)
	

	
	return fused_pred,fused_real

def process_epochs(root_dir,filelist):

	filelist=[root_dir + string for string in filelist]
	
	result=[]
	for i in range (len(filelist)):
		#print (i,filelist[i].split('_')[-1:])
		data=np.load(filelist[i],allow_pickle=True).item()
		valid_score,valid_loss,probs=data['valid_score'],data['valid_loss'],data['probs']

		fused_score,real_score=fusion_metric(valid_score,valid_loss,probs)
		acc=accuracy_score(fused_score,real_score)
		c_matrix=confusion_matrix(fused_score,real_score)
		
		result.append(acc)

	return result


def remove_training_zeros(number):

	print (number,type(number))
	print (number.rstrip('0'))
	input ('')


results=os.listdir(raw_dir)


grid_list=[]
dropout_mat=[]
learn_rate_mat=[]
epoch_mat=[]
for i in range (len(results)):

	folder_dir=raw_dir+results[i]+'/'

	all_files=os.listdir(folder_dir)
	
	dropout_rate,learn_rate=results[i].split('_')
	dropout_rate,learn_rate=float(dropout_rate),float(learn_rate)
	
	result_files=sorted([j for j in all_files if j.startswith('result')]) #Stores the names for all the epochs


	loss_files=sorted ([j for j in all_files if j.startswith('CRNN')])
	try :
		train_losses=np.load(folder_dir+loss_files[2],allow_pickle=True)
	except:
		print ("All files are not there. Re-run this configuration")

	accuracies=process_epochs(folder_dir,result_files) 
	print (accuracies)
	
	dropouts=np.full(len(accuracies),dropout_rate)
	learn_rates=np.full(len(accuracies),learn_rate)
	epoch_array=np.array(list(range(len(accuracies))))+1

	grid_list.append(accuracies)
	dropout_mat.append(dropouts)
	learn_rate_mat.append(learn_rates)
	epoch_mat.append(epoch_array)

#input ('We are here now')
'''
grid_list=np.array(grid_list).ravel()
dropout_mat=np.array(dropout_mat).ravel()
learn_rate_mat=np.array(learn_rate_mat).ravel()
epoch_mat=np.array(epoch_mat).ravel()
indices=np.argsort(grid_list)
'''
grid_list=np.concatenate(grid_list)
learn_rate_mat=np.concatenate(learn_rate_mat)
dropout_mat=np.concatenate(dropout_mat)
epoch_mat=np.concatenate(epoch_mat)
indices=np.argsort(grid_list)
elements=indices[-5:]


grid_list=grid_list[elements]
learn_rate_mat=learn_rate_mat[elements]
dropout_mat=dropout_mat[elements]
epoch_mat=epoch_mat[elements]

print (grid_list,dropout_mat,learn_rate_mat,epoch_mat)

#input ('')
#input ('')

# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda:0,1" if use_cuda else "cpu")   # use CPU or GPU
cores=2

test_params = {'batch_size': batch_size, 'shuffle': False,'num_workers': cores, 'pin_memory': True} if use_cuda else {}

train_category=np.array(train_label_list)
test_category=np.array(test_label_list)


test_set=Dataset_CRNN_new(data_path, test_mother_name, test_child_name, test_category, test_start_list, test_end_list, transform=transform)
test_loader=data.DataLoader(test_set, **test_params)

#----- Test section of the code begins here---------------------#

for i in range (len(grid_list)):

	config=str(dropout_mat[i])+"_"+str("%0.8f" %learn_rate_mat[i]).rstrip('0')
	epoch_number=epoch_mat[i]
	folder=config+'/'

	# Create model
	cnn_encoder = EncoderCNN(img_x=img_x, img_y=img_y, fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2,drop_p=dropout_mat[i], CNN_embed_dim=CNN_embed_dim).to(device)

	rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, h_FC_dim=RNN_FC_dim, drop_p=dropout_mat[i], num_classes=k).to(device)

	print (config)
	#print (cnn_encoder,rnn_decoder)
	
	# Parallelize model to multiple GPUs
	if torch.cuda.device_count() > 1:
		print("Using", torch.cuda.device_count(), "GPUs!")
		cnn_encoder = nn.DataParallel(cnn_encoder,device_ids=[0,1])
		rnn_decoder = nn.DataParallel(rnn_decoder,device_ids=[0,1])

	crnn_params = list(cnn_encoder.parameters()) + list(rnn_decoder.parameters())# + list(siamese_net.parameters())
	optimizer = torch.optim.Adam(crnn_params, lr=learn_rate,weight_decay=0.01)

	cnn_path=raw_dir+folder+'cnn_encoder_epoch{}.pth'.format(int(epoch_number))
	rnn_path=raw_dir+folder+'rnn_decoder_epoch{}.pth'.format(int(epoch_number))

	optimizer_path=raw_dir+folder+'optimizer_epoch{}.pth'.format(int(epoch_number))

	checkpoint_cnn=torch.load(cnn_path)
	cnn_encoder.load_state_dict(checkpoint_cnn)

	checkpoint_rnn=torch.load(rnn_path)
	rnn_decoder.load_state_dict(checkpoint_rnn)

	checkpoint_optimizer=torch.load(optimizer_path)

	#model = [cnn_encoder,rnn_decoder,siamese_net]
	model = [cnn_encoder,rnn_decoder]
	test_probs,test_score = test(model, device, optimizer, test_loader, "")
	
	test_dict={}
	test_dict['probs']=test_probs
	test_dict['score']=test_score

	filename=save_path+config+'_'+str(epoch_number)+'.npy'
	
	print ('Saving to %s' %filename)
	np.save(filename,test_dict)
	

